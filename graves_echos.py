#!/usr/bin/env python3
"""
graves_echos.py — Étape 4 : détection et classification automatique des échos GRAVES

À partir d'une capture enregistrée avec  graves_capture.py --iq :

  1. Cascade FINE  (0,5 Hz, ~1 s)   → traces longues : avions (et raies stables)
  2. Cascade RAPIDE (8 Hz, ~60 ms)  → événements brefs : météores, satellites, interférences
  3. Corrélation ADS-B (si un journal adsb_AAAAMMJJ.jsonl existe) → indicatif des avions
  4. Classification par règles physiques :
       avion ADS-B · avion sans ADS-B · météore (traînée / tête / surdense ?) ·
       satellite (candidat) · raie stable · interférence large bande · saturation
  5. Sorties : tableau console, image annotée _echos.png, et ajout au journal echos.csv

Usage :
  python3 graves_echos.py graves_20261105_061134_iq_16666.npy
  python3 graves_echos.py fichier.npy --journal ~/graves --sans-adsb

Les fréquences sont exprimées par rapport à la porteuse directe de GRAVES (0 Hz).
Dépendances : numpy, scipy, matplotlib + graves_capture.py et graves_adsb.py dans le même dossier.
"""
import argparse
import csv
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graves_capture import cascade                                   # noqa: E402
import graves_adsb as adsb                                            # noqa: E402

# ----------------------------------------------------------------------------- paramètres
FIN_RES, FIN_LARG = 0.5, 600.0        # Hz, demi-largeur analysée en cascade fine
RAP_RES, RAP_LARG = 8.0, 5500.0       # Hz, demi-largeur utile (filtre de décimation au-delà)
SEUIL_FIN, SEUIL_RAP = 9.0, 10.0      # dB au-dessus du fond

COULEURS = {
    "avion ADS-B": "#ffffff",
    "avion sans ADS-B": "#ff9500",
    "météore": "#ff2dfe",
    "satellite": "#00e5ff",
    "raie stable": "#ffd60a",
    "objet rapide": "#7cff4f",
    "non classé": "#aaaaaa",
    "artefact": "#8888ff",
    "interférence": "#ff3b30",
    "saturation": "#ff3b30",
}


# ----------------------------------------------------------------------------- outils
def composantes(masque, fermeture):
    """Étiquette les composantes connexes après fermeture morphologique (tolère les trous)."""
    ferme = ndimage.binary_closing(masque, structure=np.ones(fermeture, bool)) | masque
    lab, n = ndimage.label(ferme, structure=np.ones((3, 3), bool))
    objets = ndimage.find_objects(lab)
    return lab, n, objets


def ajuster(t, f, w, deg):
    deg = min(deg, max(0, len(np.unique(t)) - 1))
    if deg == 0:
        return np.poly1d([np.average(f, weights=w)])
    return np.poly1d(np.polyfit(t, f, deg, w=np.sqrt(w)))


def pister(norm, f, tt, seuil=SEUIL_FIN, ecart_max_s=25.0):
    """Pisteur multi-cibles : pics > seuil tranche par tranche, reliés par prédiction linéaire.
    Gère les croisements de traces et les trous dus à la commutation des faisceaux."""
    from scipy.signal import find_peaks
    actives, finies = [], []
    for col in range(norm.shape[1]):
        t = tt[col]
        pics, props = find_peaks(norm[:, col], height=seuil, distance=4)
        pts = [(f[k], norm[k, col], col) for k in pics]
        # clôture des pistes trop anciennes
        for tr in [a for a in actives if t - a["t"][-1] > ecart_max_s]:
            actives.remove(tr)
            finies.append(tr)
        # prédiction et appariement glouton (plus petite erreur d'abord)
        couples = []
        for ia, a in enumerate(actives):
            gap = t - a["t"][-1]
            if len(a["t"]) >= 3:
                fen = np.array(a["t"][-12:])
                pente = np.polyfit(fen, a["f"][-12:], 1)[0] if np.ptp(fen) > 0 else 0.0
            else:
                pente = 0.0
            f_att = a["f"][-1] + pente * gap
            gate = min(1.6 + 0.1 * max(gap - (tt[1] - tt[0]), 0), 4.0)
            for ip, (fp, wp, _) in enumerate(pts):
                e = abs(fp - f_att)
                if e <= gate:
                    couples.append((e, ia, ip))
        couples.sort()
        pris_a, pris_p = set(), set()
        for e, ia, ip in couples:
            if ia in pris_a or ip in pris_p:
                continue
            pris_a.add(ia); pris_p.add(ip)
            fp, wp, c = pts[ip]
            a = actives[ia]
            a["t"].append(t); a["f"].append(fp); a["w"].append(wp); a["cols"].append(c)
        for ip, (fp, wp, c) in enumerate(pts):
            if ip not in pris_p:
                actives.append(dict(t=[t], f=[fp], w=[wp], cols=[c]))
    finies += actives
    segs = []
    for a in finies:
        if len(a["t"]) < 3:
            continue
        cols = np.array(a["cols"])
        # au moins 60 % des points dans des séries d'au moins 3 tranches consécutives (le bruit ne le fait pas)
        d = np.diff(cols)
        consec = np.zeros(len(cols), bool)
        run = 1
        for i in range(1, len(cols) + 1):
            if i < len(cols) and d[i - 1] == 1:
                run += 1
            else:
                if run >= 3:
                    consec[i - run:i] = True
                run = 1
        if consec.mean() < 0.6 or consec.sum() < 3:
            continue                     # les tronçons courts (faisceau balayé) seront reliés par fusionner()
        # on ne garde que les séries consécutives : les points isolés ramassés dans les trous sont du bruit
        segs.append({k: np.array(a[k])[consec] for k in ("t", "f", "w", "cols")})
    return segs


def _unir(a, b):
    """Union de deux fragments, en gardant le pixel le plus fort par tranche temporelle."""
    t = np.concatenate([a["t"], b["t"]]); f = np.concatenate([a["f"], b["f"]])
    w = np.concatenate([a["w"], b["w"]]); c = np.concatenate([a["cols"], b["cols"]])
    o = np.lexsort((-w, c))
    _, prem = np.unique(c[o], return_index=True)
    k = o[prem]
    return dict(t=t[k], f=f[k], w=w[k], cols=c[k])


def fusionner(segs, ecart_max_s=25.0, pente_max=3.0):
    """1) regroupe les fragments simultanés et proches (étalement d'une même bouffée d'écho) ;
    2) chaîne les bouffées successives (trous de la commutation des faisceaux) : la première liaison
    tolère une pente jusqu'à pente_max Hz/s, les suivantes doivent prolonger la pente de la chaîne."""
    segs = sorted(segs, key=lambda s: s["t"].min())
    # 1) fragments simultanés
    groupes = []
    for s in segs:
        for g in groupes:
            recouvre = s["t"].min() <= g["t"].max() + 1.5 and s["t"].max() >= g["t"].min() - 1.5
            if recouvre and abs(np.median(s["f"]) - np.median(g["f"])) <= 12:
                g.update(_unir(g, s))
                break
        else:
            groupes.append(dict(s))
    # 2) chaînage temporel
    groupes.sort(key=lambda s: s["t"].min())
    chaines = []
    for s in groupes:
        t_deb = s["t"].min()
        o = np.argsort(s["t"])
        f_deb = float(np.median(s["f"][o][:3]))
        meilleure, err_min = None, 1e9
        for c in chaines:
            gap = t_deb - c["t"].max()
            if gap < -1.0 or gap > ecart_max_s:
                continue
            oc = np.argsort(c["t"])
            f_fin = float(np.median(c["f"][oc][-3:]))
            recent = c["t"] >= c["t"].max() - 40
            if np.ptp(c["t"][recent]) >= 8:
                pente = np.polyfit(c["t"][recent], c["f"][recent], 1)[0]
                err = abs(f_deb - (f_fin + pente * gap))
                tol = 6 + 0.3 * max(gap, 0)
            else:
                err = abs(f_deb - f_fin)
                tol = 4 + pente_max * max(gap, 0)
            if err <= tol and err < err_min:
                meilleure, err_min = c, err
        if meilleure is None:
            chaines.append(dict(s))
        else:
            meilleure.update(_unir(meilleure, s))
    return chaines


def decrire(seg):
    """Caractérise une trace : courbe centrale lissée (médiane glissante), pente moyenne, dispersion."""
    o = np.argsort(seg["t"])
    t, f, w = seg["t"][o], seg["f"][o], seg["w"][o]
    lisse = np.empty_like(f)
    for i in range(len(t)):
        v = np.abs(t - t[i]) <= 10.0
        if np.ptp(t[v]) > 0 and v.sum() >= 3:
            c = np.polyfit(t[v] - t[i], f[v], 1, w=np.sqrt(w[v]))
            lisse[i] = c[1]
        else:
            lisse[i] = np.average(f[v], weights=w[v])
    fit = (lambda tq, t=t, l=lisse: np.interp(tq, t, l))
    duree = float(t[-1] - t[0])
    return dict(t0=float(t[0]), t1=float(t[-1]), duree=duree, f0=float(lisse[0]), f1=float(lisse[-1]),
                fmin=float(f.min()), fmax=float(f.max()),
                pente=float((lisse[-1] - lisse[0]) / duree) if duree > 0 else 0.0,
                snr=float(w.max()), snr_moy=float(w.mean()),
                rms=float(np.sqrt(np.mean((f - lisse) ** 2))), npix=len(t), fit=fit)


# ----------------------------------------------------------------------------- analyse
def analyser(args):
    iq, fs, t0 = adsb.charger_capture(args.iq, args.decalage_temps)
    debut = datetime.fromtimestamp(t0, timezone.utc)
    rx = adsb.position_recepteur(args, args.iq)

    # --- cascade fine (après correction de la dérive de l'oscillateur du récepteur)
    f, tt, S = cascade(iq, fs, FIN_RES)
    if not args.sans_derive:
        off0 = adsb.recaler_porteuse(f, S, args.porteuse, bavard=False)
        iq, derive = adsb.corriger_derive(iq, fs, f, tt, S, off0)
        f, tt, S = cascade(iq, fs, FIN_RES)
    off = adsb.recaler_porteuse(f, S, args.porteuse)
    f_rel = f - off
    m = np.abs(f_rel) <= FIN_LARG
    f_fin = f_rel[m]
    norm_fin = S[m] - np.median(S[m], axis=1, keepdims=True)
    # une élévation du fond sur toute la bande (parasite, saturation) ne doit pas créer de fausses traces
    norm_fin -= np.median(norm_fin, axis=0, keepdims=True)
    dt_fin = tt[1] - tt[0]
    nm_fin = adsb.masquer_porteuse(norm_fin, f_fin)

    # --- cascade rapide
    fr, tr, Sr = cascade(iq, fs, RAP_RES)
    fr_rel = fr - off
    mr = np.abs(fr_rel) <= RAP_LARG
    fr_rel = fr_rel[mr]
    norm_rap = Sr[mr] - np.median(Sr[mr], axis=1, keepdims=True)
    dt_rap = tr[1] - tr[0]
    # autour de la porteuse : référence = niveau « porteuse allumée » (90e centile), pour que la
    # commutation des faisceaux ne soit pas prise pour des échos, sans masquer un météore plus fort
    pc = np.abs(fr_rel) <= 1.5 * RAP_RES
    norm_rap[pc] = Sr[mr][pc] - np.percentile(Sr[mr][pc], 90, axis=1, keepdims=True)

    echos = []

    # --- saturation / interférence : élévation du fond sur toute la bande
    niveau = np.median(norm_rap, axis=0)
    norm_rap = norm_rap - niveau[None, :]               # même neutralisation dans la cascade rapide
    faible = ndimage.label(niveau > 1.5)[1]
    # hystérésis : déclenchement au seuil, prolongation tant que le fond reste > seuil − 1,5 dB,
    # et regroupement des épisodes séparés de moins d'une seconde
    coeur = niveau > args.seuil_saturation
    etendu = niveau > max(1.0, args.seuil_saturation - 1.5)
    etendu = ndimage.binary_closing(etendu, np.ones(max(3, int(1.0 / dt_rap)), bool)) | etendu
    lab_e, n_e = ndimage.label(etendu)
    garder = np.unique(lab_e[coeur])
    sat = np.isin(lab_e, garder[garder > 0])
    sat = ndimage.binary_dilation(sat, np.ones(3, bool))
    lab, n = ndimage.label(sat)
    for i in range(1, n + 1):
        idx = np.where(lab == i)[0]
        d = float(tr[idx[-1]] - tr[idx[0]] + dt_rap)
        bref = d <= 0.4
        pic = float(niveau[idx].max())
        echos.append(dict(classe="interférence" if bref else "saturation",
                          detail=(f"impulsion large bande, fond +{pic:.0f} dB" if bref else
                                  f"fond relevé de {pic:.1f} dB sur toute la bande"),
                          t0=float(tr[idx[0]]), t1=float(tr[idx[-1]]), duree=d,
                          f0=-RAP_LARG, f1=RAP_LARG, fmin=-RAP_LARG, fmax=RAP_LARG, pente=0.0,
                          snr=pic, vue="rapide"))
        # seuls les événements forts neutralisent la cascade fine (sinon on perdrait les avions)
        if pic >= args.seuil_masque:
            hors = (tt + dt_fin >= tr[idx[0]] - dt_fin) & (tt - dt_fin <= tr[idx[-1]] + dt_fin)
            nm_fin[:, hors] = -99.0

    # --- traces longues (cascade fine)
    segs = pister(nm_fin, f_fin, tt)
    for s in segs:
        s["dt"] = dt_fin
    segs = fusionner(segs)
    traces = []
    for s in segs:
        d = decrire(s)
        d["dt"] = dt_fin
        if d["duree"] < args.duree_min_trace or d["npix"] < 6:
            continue
        if d["rms"] > 3.0:                      # nuage de bruit, pas une trace
            continue
        traces.append(d)

    # --- corrélation ADS-B
    resultats, tous_resultats = [], []
    if not args.sans_adsb:
        pistes, fichiers = adsb.charger_journal(args.journal, t0 + tt[0], t0 + tt[-1])
        if fichiers:
            p = argparse.Namespace(largeur=FIN_LARG, portee=600, tolerance=3.0, seuil=SEUIL_FIN,
                                   score_min=0.04, z_min=5.0)
            tous_resultats = adsb.correler(norm_fin, f_fin, t0 + tt, pistes, rx, p)
            resultats = [r for r in tous_resultats if r["identifie"]]
            print(f"ADS-B : {len(pistes)} avions dans le journal, {len(resultats)} écho(s) identifié(s)")
        else:
            print("ADS-B : aucun journal pour cette période (les avions resteront « sans ADS-B »)")

    for d in traces:
        meilleur, ecart_min = None, 1e9
        tm = np.linspace(d["t0"], d["t1"], 25)
        for r in resultats:
            fd = np.interp(tm, tt, r["c"]["fd"], left=np.nan, right=np.nan)
            ok = ~np.isnan(fd)
            if ok.sum() < 5:
                continue
            e = float(np.median(np.abs(d["fit"](tm[ok]) - fd[ok])))
            if e < ecart_min:
                meilleur, ecart_min = r, e
        if abs(d["pente"]) > args.pente_max_avion:
            d.update(classe="satellite", detail="trace longue à forte pente (cascade fine)")
        elif d["duree"] > 0.8 * tt[-1] and abs(d["pente"]) < 0.05 and abs(d["f0"]) > 2:
            d.update(classe="raie stable", detail="présente toute la capture (Lune ? parasite ?)")
        elif meilleur is not None and ecart_min <= 4.0:
            d.update(classe="avion ADS-B", detail=f"{meilleur['flight']} — {meilleur['alt']:.0f} ft, "
                                                  f"{meilleur['dT']:.0f} km de GRAVES, écart {ecart_min:.1f} Hz")
        elif max(abs(d["fmin"]), abs(d["fmax"])) > args.fd_max_avion:
            v = max(abs(d["fmin"]), abs(d["fmax"])) * adsb.LAMBDA / 2
            d.update(classe="objet rapide", detail=f"Doppler hors de portée d'un avion (≥ {v:.0f} m/s) : "
                                                   f"satellite lointain, débris, ou raie parasite")
        else:
            prox = min(abs(d["fmin"]), abs(d["fmax"])) < 15 and max(abs(d["fmin"]), abs(d["fmax"])) < 30
            d.update(classe="avion sans ADS-B",
                     detail=("collé à la porteuse : avion en diffusion avant (sur l'axe GRAVES–récepteur) "
                             "ou trajet multiple du signal direct" if prox else
                             "trace d'avion sans vol ADS-B correspondant"))
        d["vue"] = "fine"
        echos.append(d)

    # vols identifiés par corrélation mais sans trace continue (échos hachés, faibles) : ajoutés tels quels
    vus = {e["detail"].split(" — ")[0] for e in echos if e.get("classe") == "avion ADS-B"}
    for r in resultats:
        if r["flight"] in vus or len(r["cols"]) == 0:
            continue
        fd = r["c"]["fd"]
        ok = ~np.isnan(fd)
        t_ok, fd_ok = tt[ok], fd[ok]
        fit = (lambda tq, a=t_ok, b=fd_ok: np.interp(tq, a, b))
        ta, tb = float(tt[r["cols"].min()]), float(tt[r["cols"].max()])
        d = dict(t0=ta, t1=tb, duree=tb - ta, f0=float(fit(ta)), f1=float(fit(tb)),
                 fmin=float(r["fpics"].min()), fmax=float(r["fpics"].max()),
                 pente=float((fit(tb) - fit(ta)) / max(tb - ta, 1e-3)), snr=float(r["z"]), fit=fit,
                 npix=len(r["cols"]), dt=dt_fin, vue="fine", classe="avion ADS-B",
                 detail=f"{r['flight']} — {r['alt']:.0f} ft, {r['dT']:.0f} km de GRAVES, écart {r['ecart']:.1f} Hz "
                        f"(par corrélation, z = {r['z']:.1f})")
        traces.append(d)
        echos.append(d)

    # --- événements brefs (cascade rapide)
    masque = norm_rap > SEUIL_RAP
    masque[:, sat] = False
    # zones à retirer : traces pistées + courbe ADS-B complète des vols identifiés (y compris entre les bouffées)
    for r in resultats:
        fd = r["c"]["fd"]
        ok = ~np.isnan(fd)
        if ok.sum() >= 2:
            traces.append(dict(t0=float(tt[ok][0]), t1=float(tt[ok][-1]), snr=99.0, masque=True,
                               fit=(lambda tq, a=tt[ok], b=fd[ok]: np.interp(tq, a, b))))
    for d in traces:
        jt = np.where((tr >= d["t0"] - 1) & (tr <= d["t1"] + 1))[0]
        if len(jt):
            fp = d["fit"](tr[jt])
            proche = np.abs(fr_rel[:, None] - fp[None, :]) <= 1.5 * RAP_RES
            masque[:, jt] &= ~proche
    lab, n, objets = composantes(masque, (1, 3))
    evs = []
    for i, sl in enumerate(objets, start=1):
        if sl is None:
            continue
        sous = (lab[sl] == i) & masque[sl]
        kk, cc = np.nonzero(sous)
        kk += sl[0].start
        cc += sl[1].start
        w = norm_rap[kk, cc]
        if not (w.max() >= args.seuil_pic or len(kk) >= 10):
            continue                                   # trop faible : probablement du bruit
        t_ev, f_ev = tr[cc], fr_rel[kk]
        fpic = float(f_ev[np.argmax(w)])
        tm = 0.5 * (t_ev.min() + t_ev.max())
        # fragment d'un avion déjà vu en cascade fine (plus faible ici que là-bas) ?
        long_ = (t_ev.max() - t_ev.min()) >= 1.0
        if any(d["t0"] - 2 <= tm <= d["t1"] + 2 and abs(d["fit"](tm) - fpic) <= 2 * RAP_RES
               and (long_ or w.max() < d["snr"]) for d in traces):
            continue
        evs.append(dict(t=t_ev, f=f_ev, w=w, t0=float(t_ev.min()), t1=float(t_ev.max()) + dt_rap))

    # chaînage des fragments alignés (satellite rapide, météore surdense qui s'évanouit…)
    evs.sort(key=lambda e: e["t0"])
    chaines = []
    for e in evs:
        f_e0 = float(e["f"][np.argmin(e["t"])])
        meilleure, err_min = None, 1e9
        for c in chaines:
            gap = e["t0"] - c["t1"]
            if gap < -0.2 or gap > 1.2:
                continue
            tc, fc = np.concatenate(c["t"]), np.concatenate(c["f"])
            if len(np.unique(tc)) >= 3 and tc.max() - tc.min() > 0.2:
                pente = np.polyfit(tc, fc, 1)[0]
                f_att = fc[np.argmax(tc)] + pente * (e["t0"] - tc.max())
                tol = 40 + 0.3 * abs(pente) * max(gap, dt_rap)
            else:
                f_att, tol = fc[np.argmax(tc)], 20 + 800 * max(gap, dt_rap)
            err = abs(f_e0 - f_att)
            if err <= tol and err < err_min:
                meilleure, err_min = c, err
        if meilleure is None:
            chaines.append(dict(t=[e["t"]], f=[e["f"]], w=[e["w"]], t0=e["t0"], t1=e["t1"]))
        else:
            for k in ("t", "f", "w"):
                meilleure[k].append(e[k])
            meilleure["t1"] = max(meilleure["t1"], e["t1"])

    if os.environ.get("GRAVES_DEBUG"):
        for c in chaines:
            print("chaîne", [(round(float(t.min()), 2), round(float(t.max()), 2), int(f.min()), int(f.max()))
                             for t, f in zip(c["t"], c["f"])])
    for c in chaines:
        t_ev, f_ev, w = (np.concatenate(c[k]) for k in ("t", "f", "w"))
        duree = float(c["t1"] - c["t0"])
        etendue = float(f_ev.max() - f_ev.min() + RAP_RES)
        fpic = float(f_ev[np.argmax(w)])
        pente = float(np.polyfit(t_ev, f_ev, 1)[0]) if len(np.unique(t_ev)) >= 3 else 0.0
        ev = dict(t0=c["t0"], t1=c["t1"], duree=duree, f0=float(f_ev[np.argmin(t_ev)]),
                  f1=float(f_ev[np.argmax(t_ev)]), fmin=float(f_ev.min()), fmax=float(f_ev.max()),
                  snr=float(w.max()), vue="rapide", pente=pente)
        if duree <= args.duree_max_meteore and 50 <= abs(pente) <= 2000 and duree >= 0.25 \
                and etendue <= 0.6 * abs(pente) * duree + 60:
            ev.update(classe="satellite", detail=f"fragment, pente {pente:+.0f} Hz/s")
        elif duree <= args.duree_max_meteore:
            if etendue > 150 or abs(pente) > 300:
                ev.update(classe="météore", detail=f"bref et large ({etendue:.0f} Hz) : écho de tête ou début de traînée")
            elif abs(fpic) <= args.fd_max_trainee:
                ev.update(classe="météore", detail=f"traînée sous-dense à {fpic:+.0f} Hz")
            else:
                # une traînée dérive avec le vent (quelques dizaines de Hz) : à ±kHz ce n'en est pas une
                ev.update(classe="non classé", detail=f"bref et étroit à {fpic:+.0f} Hz : fragment de satellite ? bruit ?")
        elif abs(pente) >= args.pente_max_avion or (abs(fpic) > 500 and duree >= 1.5):
            ev.update(classe="satellite", detail=f"pente {pente:+.0f} Hz/s")
        elif abs(fpic) <= 60 and duree <= 30:
            ev.update(classe="météore", detail=f"traînée surdense probable ({duree:.1f} s à {fpic:+.0f} Hz)")
        else:
            continue
        echos.append(ev)

    echos = identifier_satellites(echos, fr_rel, tr, norm_rap, t0, rx, args)
    echos = marquer_artefacts(echos, traces)
    echos = agreger_saturations(echos)
    echos = regrouper_recurrents(echos, tous_resultats, tt, args)
    echos = rattacher_continuite(echos)
    echos = fusionner_vols(echos)
    echos = regrouper_satellites(echos)
    echos.sort(key=lambda e: e["t0"])
    return dict(echos=echos, debut=debut, t0=t0, off=off, activite_faible=int(faible), f_fin=f_fin, tt=tt, norm_fin=norm_fin,
                fr=fr_rel, tr=tr, norm_rap=norm_rap, resultats=resultats)


def identifier_satellites(echos, fr, tr, norm, t0, rx, args):
    """Corrélation avec le catalogue orbital (graves_satellites.py) : les échos qui suivent la courbe
    Doppler recalée d'un objet identifié reçoivent son nom ; un objet identifié sans écho détecté
    par ailleurs est ajouté (le filtre adapté voit plus faible que la détection « à l'aveugle »)."""
    if args.sans_satellites:
        return echos
    import glob as _glob
    fichiers = sorted(_glob.glob(os.path.join(args.catalogues, "catalogue_*.json")))
    if not fichiers:
        print("Satellites : aucun catalogue (python3 graves_satellites.py catalogue) — étape ignorée")
        return echos
    import graves_satellites as gsat
    cat = gsat.charger_catalogue(args.catalogues)
    p = argparse.Namespace(tau_max=8.0, max=400, min_tranches=15, min_forts=4, seuil=8.0, z_min=6.0)
    res = gsat.identifier(cat, fr, tr, norm, t0, rx, p, bavard=False)
    ids = [r for r in res if r["identifie"]]
    print(f"Satellites : {len(res)} objets du catalogue traversent la fenêtre, {len(ids)} écho(s) identifié(s)")
    for r in ids:
        ok = ~np.isnan(r["F"])
        courbe = (lambda tq, a=tr[ok], b=r["F"][ok]: np.interp(tq, a, b, left=np.nan, right=np.nan))
        absorbes = []
        for e in echos:
            if e["classe"] in ("saturation", "interférence", "avion ADS-B"):
                continue
            tm = 0.5 * (e["t0"] + e["t1"])
            fc = courbe(tm)
            if not np.isnan(fc) and abs(fc - 0.5 * (e["fmin"] + e["fmax"])) <= 40 + 0.5 * (e["fmax"] - e["fmin"]):
                absorbes.append(e)
        forts = r["forts"] if len(r["forts"]) else np.where(ok)[0]
        ta, tb = float(tr[forts.min()]), float(tr[forts.max()])
        f_a, f_b = float(courbe(ta)), float(courbe(tb))
        echos = [e for e in echos if not any(e is x for x in absorbes)]
        echos.append(dict(classe="satellite", vue="rapide", t0=ta, t1=tb, duree=tb - ta, f0=f_a, f1=f_b,
                          fmin=min(f_a, f_b), fmax=max(f_a, f_b), pente=(f_b - f_a) / max(tb - ta, 1e-3),
                          snr=r["z"], courbe=(tr, r["F"]),
                          detail=f"{r['nom']} (NORAD {r['norad']}, {r['type']}, RCS {r['rcs']}) — recalage TLE "
                                 f"{r['tau']:+.2f} s, z = {r['z']:.1f}"
                                 + (f", {len(absorbes)} fragment(s) regroupé(s)" if absorbes else "")))
    return echos


def marquer_artefacts(echos, traces, fenetre_s=2.5):
    """Événements brefs simultanés, étalés symétriquement autour d'un signal fort (porteuse ou écho
    d'avion) : ce n'est pas un objet mais l'« éclaboussure » spectrale d'une discontinuité du signal
    (échantillons perdus, saut de gain, transitoire de l'émetteur…)."""
    brefs = sorted([e for e in echos if e.get("vue") == "rapide" and e["duree"] <= 0.6
                    and e["classe"] not in ("saturation", "interférence")], key=lambda e: e["t0"])
    # éclaboussure d'un écho d'avion très fort (saut d'amplitude à la commutation des faisceaux) :
    # un événement bref collé à cet écho, au même instant, n'est pas un météore
    forts = [d for d in traces if d.get("snr", 0) >= 25 and not d.get("masque")]
    for e in brefs:
        tm = 0.5 * (e["t0"] + e["t1"])
        fc = 0.5 * (e["fmin"] + e["fmax"])
        for d in forts:
            if d["t0"] - 1 <= tm <= d["t1"] + 1 and abs(float(d["fit"](tm)) - fc) <= 50:
                e.update(classe="artefact", detail=f"éclaboussure autour d'un écho d'avion très fort "
                                                   f"({d['snr']:.0f} dB, à {float(d['fit'](tm)):+.0f} Hz)")
                break
    brefs = [e for e in brefs if e["classe"] != "artefact"]
    groupes, g = [], []
    for e in brefs:
        if g and e["t0"] - max(x["t0"] for x in g) > fenetre_s:
            groupes.append(g); g = []
        g.append(e)
    if g:
        groupes.append(g)
    for g in groupes:
        fmin, fmax = min(e["fmin"] for e in g), max(e["fmax"] for e in g)
        centre, etendue = 0.5 * (fmin + fmax), fmax - fmin
        tm = g[0]["t0"]
        refs = [0.0] + [float(d["fit"](tm)) for d in traces if d["t0"] - 2 <= tm <= d["t1"] + 2]
        proche = min(abs(centre - r) for r in refs)
        englobe = any(fmin - 40 <= r <= fmax + 40 for r in refs)
        if (len(g) >= 2 and etendue >= 60 and englobe) or (len(g) == 1 and 80 <= etendue <= 600 and proche <= 25):
            for e in g:
                e.update(classe="artefact", detail=f"éclaboussure spectrale autour d'un signal fort ({len(g)} fragment(s) "
                                                   f"simultanés sur {etendue:.0f} Hz) : discontinuité du signal reçu")
    return echos


def agreger_saturations(echos, ecart_s=10.0, mini=3):
    """Saturations répétées à quelques secondes d'intervalle → une seule ligne."""
    sats = sorted([e for e in echos if e["classe"] == "saturation"], key=lambda e: e["t0"])
    autres = [e for e in echos if e["classe"] != "saturation"]
    paquets, p = [], []
    for e in sats:
        if p and e["t0"] - p[-1]["t1"] > ecart_s:
            paquets.append(p); p = []
        p.append(e)
    if p:
        paquets.append(p)
    for p in paquets:
        if len(p) < mini:
            autres += p
            continue
        actif = sum(e["duree"] for e in p)
        g = dict(p[0])
        g.update(t1=p[-1]["t1"], duree=p[-1]["t1"] - p[0]["t0"], snr=max(e["snr"] for e in p),
                 detail=f"{len(p)} épisodes répétés (fond relevé jusqu'à +{max(e['snr'] for e in p):.1f} dB, "
                        f"{100 * actif / max(p[-1]['t1'] - p[0]['t0'], 1e-3):.0f} % du temps)")
        autres.append(g)
    return autres


def regrouper_recurrents(echos, tous_resultats, tt, args):
    """Des « météores » longs (≥ 1 s) qui reviennent ≥ 3 fois à la même fréquence en quelques minutes
    sont en réalité l'écho haché d'un avion (faisceau balayé). On les regroupe et on cherche le vol."""
    cand = sorted([e for e in echos if e["classe"] == "météore" and e["duree"] >= 1.0], key=lambda e: e["t0"])
    groupes = []
    for e in cand:
        fe = 0.5 * (e["fmin"] + e["fmax"])
        for g in groupes:
            if e["t0"] - g[-1]["t1"] <= 60 and abs(fe - 0.5 * (g[-1]["fmin"] + g[-1]["fmax"])) <= 12:
                g.append(e)
                break
        else:
            groupes.append([e])
    retires = []
    for g in groupes:
        if len(g) < 3:
            continue
        retires += g
        tm = np.array([0.5 * (e["t0"] + e["t1"]) for e in g])
        fm = np.array([0.5 * (e["fmin"] + e["fmax"]) for e in g])
        nom, best = None, 1e9
        for r in tous_resultats:
            fd = np.interp(tm, tt, r["c"]["fd"], left=np.nan, right=np.nan)
            ok = ~np.isnan(fd)
            if ok.sum() >= 0.66 * len(g):
                err = float(np.median(np.abs(fd[ok] - fm[ok])))
                if err < best:
                    nom, best = r, err
        n_g = len(g)
        base = dict(t0=g[0]["t0"], t1=g[-1]["t1"], duree=g[-1]["t1"] - g[0]["t0"], f0=float(fm[0]), f1=float(fm[-1]),
                    fmin=min(e["fmin"] for e in g), fmax=max(e["fmax"] for e in g),
                    pente=float((fm[-1] - fm[0]) / max(tm[-1] - tm[0], 1e-3)), snr=max(e["snr"] for e in g),
                    vue="rapide", groupe=g)
        if nom is not None and best <= 8:
            base.update(classe="avion ADS-B", detail=f"{nom['flight']} — probable, {n_g} échos hachés, écart {best:.1f} Hz")
        else:
            base.update(classe="avion sans ADS-B", detail=f"{n_g} échos hachés récurrents vers {np.median(fm):+.0f} Hz")
        echos.append(base)
    return [e for e in echos if not any(e is x for x in retires)]


def rattacher_continuite(echos, ecart_max_s=60.0):
    """Une trace « sans ADS-B » qui précède (ou suit) un vol identifié et se trouve dans son prolongement
    est le même avion, vu avant que ses positions ADS-B soient reçues ou après une coupure (saturation…)."""
    vols = [e for e in echos if e["classe"] == "avion ADS-B" and e.get("vue") == "fine"]
    for e in [x for x in echos if x["classe"] == "avion sans ADS-B" and x.get("vue") == "fine"]:
        for v in vols:
            if 0 <= v["t0"] - e["t1"] <= ecart_max_s:          # e avant v
                gap, f_a, f_b, pente = v["t0"] - e["t1"], e["f1"], v["f0"], e["pente"]
            elif 0 <= e["t0"] - v["t1"] <= ecart_max_s:        # e après v
                gap, f_a, f_b, pente = e["t0"] - v["t1"], v["f1"], e["f0"], v["pente"]
            else:
                continue
            if abs(f_a + pente * gap - f_b) <= 6 + 0.6 * gap:
                nom = v["detail"].split(" — ")[0]
                e.update(classe="avion ADS-B", detail=f"{nom} — même avion (dans le prolongement de sa trace, "
                                                     f"hors positions ADS-B reçues)")
                break
    return echos


def fusionner_vols(echos):
    """Plusieurs tronçons attribués au même vol ADS-B → une seule ligne."""
    vols, autres = {}, []
    for e in echos:
        if e["classe"] == "avion ADS-B":
            vols.setdefault(e["detail"].split(" — ")[0], []).append(e)
        else:
            autres.append(e)
    for nom, lst in vols.items():
        lst.sort(key=lambda e: e["t0"])
        g = dict(lst[0])
        if len(lst) > 1:
            dern = lst[-1]
            g.update(t1=max(e["t1"] for e in lst), f1=dern["f1"], fmin=min(e["fmin"] for e in lst),
                     fmax=max(e["fmax"] for e in lst), snr=max(e["snr"] for e in lst))
            g["duree"] = g["t1"] - g["t0"]
            g["pente"] = (g["f1"] - g["f0"]) / max(g["duree"], 1e-3)
            g["troncons"] = lst
        autres.append(g)
    return autres


def regrouper_satellites(echos):
    """Les tronçons de satellite séparés par la commutation des faisceaux forment un seul passage."""
    a_grouper = lambda e: e["classe"] == "satellite" and e["vue"] == "rapide" and "courbe" not in e
    sats = sorted([e for e in echos if a_grouper(e)], key=lambda e: e["t0"])
    autres = [e for e in echos if not a_grouper(e)]
    groupes = []
    for e in sats:
        g = groupes[-1] if groupes else None
        if g and e["t0"] - g["t1"] <= 25:
            gap = e["t0"] - g["t1"]
            sens = np.sign(e["f0"] - g["f1"])
            tendance = np.sign(g["pente"] if abs(g["pente"]) > 1 else e["pente"])
            if sens == tendance and abs(e["f0"] - g["f1"]) <= 300 + 600 * gap:
                g.update(t1=e["t1"], f1=e["f1"], fmin=min(g["fmin"], e["fmin"]), fmax=max(g["fmax"], e["fmax"]),
                         snr=max(g["snr"], e["snr"]), duree=e["t1"] - g["t0"], n=g["n"] + 1)
                g["pente"] = (g["f1"] - g["f0"]) / max(g["duree"], 1e-3)
                continue
        groupes.append(dict(e, n=1))
    for g in groupes:
        g["detail"] = f"{g['n']} tronçon(s), {g['f0']:+.0f} → {g['f1']:+.0f} Hz, pente moyenne {g['pente']:+.0f} Hz/s"
    return autres + groupes


# ----------------------------------------------------------------------------- sorties
def afficher(r, args):
    print(f"\nCapture du {r['debut']:%Y-%m-%d %H:%M:%S} UTC — porteuse à {r['off']:+.1f} Hz (référence 0 Hz)")
    print(" N°  Heure UTC  Durée    Fréquence (Hz)        Pente     SNR   Classe             Détail")
    for i, e in enumerate(r["echos"], 1):
        h = datetime.fromtimestamp(r["t0"] + e["t0"], timezone.utc)
        print(f"{i:3d}  {h:%H:%M:%S}  {e['duree']:6.1f} s  {e['f0']:+7.0f} → {e['f1']:+7.0f}  {e['pente']:+7.1f} Hz/s "
              f"{e['snr']:5.1f}  {e['classe']:<18} {e.get('detail', '')}")
    bilan = {}
    for e in r["echos"]:
        bilan[e["classe"]] = bilan.get(e["classe"], 0) + 1
    print("\nBilan : " + ", ".join(f"{v} {k}" for k, v in sorted(bilan.items(), key=lambda x: -x[1])))
    if r.get("activite_faible"):
        print(f"Activité impulsive faible (+1,5 à {args.seuil_saturation:g} dB sur toute la bande) : "
              f"{r['activite_faible']} épisode(s), ignorés")


def journaliser(r, args):
    nouveau = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as fcsv:
        w = csv.writer(fcsv, delimiter=";")
        if nouveau:
            w.writerow(["capture", "heure_utc", "duree_s", "f_debut_hz", "f_fin_hz", "f_min_hz", "f_max_hz",
                        "pente_hz_s", "snr_db", "classe", "detail"])
        for e in r["echos"]:
            h = datetime.fromtimestamp(r["t0"] + e["t0"], timezone.utc)
            w.writerow([os.path.basename(args.iq), h.isoformat(timespec="seconds"), f"{e['duree']:.2f}",
                        f"{e['f0']:.1f}", f"{e['f1']:.1f}", f"{e['fmin']:.1f}", f"{e['fmax']:.1f}",
                        f"{e['pente']:.2f}", f"{e['snr']:.1f}", e["classe"], e.get("detail", "")])


def tracer(r, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 10), layout="constrained",
                                 gridspec_kw=dict(width_ratios=[1, 1.15]))
    # fine ±400 Hz
    f, tt, nf = r["f_fin"], r["tt"], r["norm_fin"]
    m = np.abs(f) <= 400
    vmin, vmax = np.percentile(nf[m], [20, 99.8])
    a1.imshow(nf[m].T, aspect="auto", origin="upper", cmap="viridis", interpolation="nearest",
              extent=[f[m][0], f[m][-1], tt[-1], tt[0]], vmin=vmin, vmax=max(vmax, vmin + 6))
    a1.set_title("Cascade fine ±400 Hz (0,5 Hz) — traces longues")
    a1.set_xlabel("Doppler / porteuse GRAVES (Hz)")
    a1.set_ylabel(f"Temps (s) depuis {r['debut']:%H:%M:%S} UTC")
    # rapide ±6 kHz (sous-échantillonnée pour l'affichage)
    fr, tr, nr = r["fr"], r["tr"], r["norm_rap"]
    pas_t = max(1, len(tr) // 1500)
    nr_aff = nr[:, : (nr.shape[1] // pas_t) * pas_t].reshape(nr.shape[0], -1, pas_t).max(axis=2)
    vmin, vmax = np.percentile(nr_aff, [30, 99.9])
    a2.imshow(nr_aff.T, aspect="auto", origin="upper", cmap="magma", interpolation="nearest",
              extent=[fr[0], fr[-1], tr[-1], tr[0]], vmin=vmin, vmax=max(vmax, vmin + 6))
    a2.set_title("Cascade rapide ±6 kHz (8 Hz, 60 ms) — événements brefs")
    a2.set_xlabel("Doppler / porteuse GRAVES (Hz)")

    for i, e in enumerate(r["echos"], 1):
        col = COULEURS.get(e["classe"], "white")
        if e["vue"] == "fine":
            if abs(e["f0"]) > 400 and abs(e["f1"]) > 400:
                continue
            for k, tr_ in enumerate(e.get("troncons", [e])):
                tl = np.linspace(tr_["t0"], tr_["t1"], 50)
                a1.plot(tr_["fit"](tl) + 10, tl, color=col, lw=1.2, linestyle=(0, (3, 4)))
                if k == 0:
                    a1.annotate(f"{i}", (tr_["fit"](tl[0]) + 12, tl[0]), color=col, fontsize=11, fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0))
        else:
            ax = a2
            if e["classe"] == "saturation":
                ax.axhspan(e["t0"], e["t1"] + 0.1, color=col, alpha=0.15)
                ax.annotate(f"{i} saturation", (fr[0] + 100, e["t0"]), color=col, fontsize=9)
                continue
            if "courbe" in e:
                tc, Fc = e["courbe"]
                ax.plot(Fc + 120, tc, color=col, lw=1.2, linestyle=(0, (4, 5)))
                k = np.where(~np.isnan(Fc))[0]
                if len(k):
                    j = k[len(k) // 3]
                    ax.annotate(f"{i} {e['detail'].split(' (')[0]}", (Fc[j] + 150, tc[j]), color=col, fontsize=9,
                                fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0))
                continue
            marge_f = 60 if e["classe"] != "interférence" else 0
            ax.add_patch(Rectangle((e["fmin"] - marge_f, e["t0"] - 0.6), e["fmax"] - e["fmin"] + 2 * marge_f,
                                   e["duree"] + 1.2, fill=False, ec=col, lw=1.2))
            ax.annotate(f"{i}", (e["fmax"] + marge_f + 30, e["t0"]), color=col, fontsize=10, fontweight="bold")
    poignees = [plt.Line2D([], [], color=c, lw=3, label=k) for k, c in COULEURS.items() if k != "saturation"]
    a2.legend(handles=poignees, loc="lower right", fontsize=9, facecolor="black", labelcolor="white", framealpha=0.6)
    if any("adsbfi" in x.get("sources", []) for x in r.get("resultats", [])):
        fig.text(0.01, 0.003, adsb.ADSBFI_MENTION, fontsize=8, color="0.4")
    fig.suptitle(f"GRAVES 143,050 MHz — {r['debut']:%Y-%m-%d %H:%M:%S} UTC — échos détectés et classés "
                 f"(numéros = lignes du tableau)", fontsize=13)
    fig.savefig(png, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iq", help="capture graves_..._iq_NNNNN.npy (avec son .json)")
    p.add_argument("--journal", default=".", help="dossier des journaux ADS-B adsb_AAAAMMJJ.jsonl")
    p.add_argument("--sans-adsb", dest="sans_adsb", action="store_true")
    p.add_argument("--catalogues", default=".", help="dossier des catalogues orbitaux catalogue_AAAAMMJJ.json")
    p.add_argument("--sans-satellites", dest="sans_satellites", action="store_true")
    p.add_argument("--csv", default="echos.csv", help="journal cumulatif des échos (CSV, ;)")
    p.add_argument("--lat", type=float, default=None, help="récepteur (défaut : position enregistrée, sinon station.ini)")
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--alt", type=float, default=None)
    p.add_argument("--porteuse", type=float, default=None, help="imposer la position de la porteuse (Hz)")
    p.add_argument("--decalage-temps", dest="decalage_temps", type=float, default=0.0)
    p.add_argument("--sans-derive", dest="sans_derive", action="store_true",
                   help="ne pas corriger la dérive de l'oscillateur du récepteur")
    p.add_argument("--duree-min-trace", dest="duree_min_trace", type=float, default=8.0,
                   help="durée minimale d'une trace d'avion (s)")
    p.add_argument("--pente-max-avion", dest="pente_max_avion", type=float, default=15.0,
                   help="au-delà (Hz/s), une trace longue est classée satellite")
    p.add_argument("--seuil-pic", dest="seuil_pic", type=float, default=15.0,
                   help="pic minimal (dB) d'un événement bref, sauf s'il couvre ≥ 10 pixels")
    p.add_argument("--fd-max-avion", dest="fd_max_avion", type=float, default=290.0,
                   help="Doppler max d'un avion (Hz) ; 290 Hz ≈ 300 m/s en géométrie la plus favorable")
    p.add_argument("--fd-max-trainee", dest="fd_max_trainee", type=float, default=200.0,
                   help="Doppler max d'une traînée de météore (Hz)")
    p.add_argument("--duree-max-meteore", dest="duree_max_meteore", type=float, default=1.5)
    p.add_argument("--etendue-interference", dest="etendue_interference", type=float, default=2500.0)
    p.add_argument("--seuil-saturation", dest="seuil_saturation", type=float, default=3.0,
                   help="élévation médiane du fond (dB) à partir de laquelle un événement est listé")
    p.add_argument("--seuil-masque", dest="seuil_masque", type=float, default=6.0,
                   help="élévation (dB) à partir de laquelle la cascade fine est neutralisée pendant l'événement")
    args = p.parse_args()

    r = analyser(args)
    afficher(r, args)
    png = os.path.splitext(args.iq)[0].split("_iq_")[0] + "_echos.png"
    tracer(r, png)
    journaliser(r, args)
    print(f"\nImage : {png}\nJournal cumulatif : {args.csv}")


if __name__ == "__main__":
    main()
