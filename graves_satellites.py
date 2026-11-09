#!/usr/bin/env python3
"""
graves_satellites.py — Étape 5 : identification des échos de satellites

Modes :

  1) catalogue : télécharge les éléments orbitaux (GP) de tous les objets en orbite
     python3 graves_satellites.py catalogue                 # Space-Track (≈ 30 000 objets)
     python3 graves_satellites.py catalogue --source celestrak   # sans compte, satellites actifs seulement
         → catalogue_AAAAMMJJ.json (réutilisé pendant 24 h ; Space-Track demande ≤ 1 requête GP/heure)

  2) passages : prévoit les objets observables (au-dessus de l'horizon de GRAVES et du récepteur,
     Doppler dans la fenêtre de la capture) pour les heures à venir, classés par détectabilité
     python3 graves_satellites.py passages --heures 6

  3) superpose : sur une capture IQ, cherche l'écho de chaque objet candidat par « filtre adapté » :
     l'énergie est additionnée le long de la courbe Doppler prévue, en cherchant le meilleur
     recalage temporel (erreur des éléments TLE), et comparée à des courbes leurres décalées en
     fréquence. Un objet est identifié si son score dépasse nettement celui des leurres.
     python3 graves_satellites.py superpose graves_20261106_001313_iq_16666.npy
         → tableau + graves_..._satellites.png

Identifiants Space-Track (jamais dans le code) :
   variables d'environnement SPACETRACK_USER et SPACETRACK_PASSWORD, ou fichier
   ~/.config/graves/spacetrack.ini (droits 600) :
       [spacetrack]
       identity = votre@adresse
       password = votre_mot_de_passe

Dépendances : numpy, scipy, matplotlib, sgp4 (installé avec skyfield)
               + graves_capture.py et graves_adsb.py dans le même dossier.
"""
import argparse
import configparser
import glob
import http.cookiejar
import json
import math
import os
import stat
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graves_adsb as adsb                                            # noqa: E402

LAMBDA = adsb.LAMBDA
OMEGA_TERRE = 7.292115e-5                       # rad/s
FENETRE_HZ = 5400.0                             # demi-largeur utile de la cascade rapide
RCS_M2 = {"SMALL": 0.05, "MEDIUM": 0.5, "LARGE": 5.0}   # ordres de grandeur des classes Space-Track

ST_BASE = "https://www.space-track.org"
ST_REQUETE = ("/basicspacedata/query/class/gp/decay_date/null-val/epoch/%3Enow-30"
              "/orderby/norad_cat_id/format/json")


# ============================================================================ catalogue
def identifiants():
    u, p = os.environ.get("SPACETRACK_USER"), os.environ.get("SPACETRACK_PASSWORD")
    if u and p:
        return u, p
    fn = os.path.expanduser("~/.config/graves/spacetrack.ini")
    if os.path.exists(fn):
        if os.stat(fn).st_mode & (stat.S_IRGRP | stat.S_IROTH):
            print(f"Attention : {fn} est lisible par d'autres utilisateurs (chmod 600 {fn})")
        c = configparser.ConfigParser(interpolation=None)     # mot de passe pris tel quel (%, #, ; permis)
        try:
            with open(fn, encoding="utf-8-sig") as f:           # tolère le BOM de certains éditeurs
                c.read_file(f)
        except PermissionError:
            sys.exit(f"{fn} : lecture refusée (fichier créé avec sudo ?). Corriger avec :\n"
                     f"  sudo chown $USER: {fn} && chmod 600 {fn}")
        except configparser.Error as e:
            sys.exit(f"{fn} illisible : {e}\n{MODELE_INI}")
        sec = next((s for s in c.sections() if s.strip().lower() in ("spacetrack", "space-track")), None)
        if sec is None:
            vues = ", ".join(f"[{s}]" for s in c.sections()) or "aucune"
            sys.exit(f"{fn} : section [spacetrack] absente (sections trouvées : {vues}).\n{MODELE_INI}")
        s = c[sec]
        u = s.get("identity") or s.get("user") or s.get("login")
        p = s.get("password") or s.get("mdp")
        if not u or not p:
            sys.exit(f"{fn} : clés 'identity' et/ou 'password' manquantes dans [{sec}].\n{MODELE_INI}")
        return u.strip(), p.strip()
    sys.exit("Identifiants Space-Track introuvables : définir SPACETRACK_USER / SPACETRACK_PASSWORD "
             f"ou créer {fn}.\n{MODELE_INI}")


MODELE_INI = ("Contenu attendu du fichier (puis chmod 600) :\n"
              "  [spacetrack]\n"
              "  identity = votre_adresse_de_connexion\n"
              "  password = votre_mot_de_passe")


def telecharger_spacetrack():
    user, mdp = identifiants()
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    op.addheaders = [("User-Agent", "graves-radar-amateur/1.0")]
    donnees = urllib.parse.urlencode({"identity": user, "password": mdp}).encode()
    with op.open(ST_BASE + "/ajaxauth/login", data=donnees, timeout=60) as r:
        rep = r.read().decode(errors="replace")
        if "Failed" in rep or r.status != 200:
            sys.exit(f"Connexion Space-Track refusée : {rep[:200]}")
    print("Connecté à Space-Track, téléchargement du catalogue GP (peut prendre une minute)…")
    with op.open(ST_BASE + ST_REQUETE, timeout=300) as r:
        brut = json.load(r)
    try:
        op.open(ST_BASE + "/ajaxauth/logout", timeout=30).read()
    except Exception:
        pass
    cat = []
    for o in brut:
        if not o.get("TLE_LINE1"):
            continue
        cat.append(dict(norad=int(o["NORAD_CAT_ID"]), nom=(o.get("OBJECT_NAME") or "").strip(),
                        type=(o.get("OBJECT_TYPE") or "?").strip(), rcs=(o.get("RCS_SIZE") or "?").strip(),
                        l1=o["TLE_LINE1"], l2=o["TLE_LINE2"], epoque=o.get("EPOCH", "")))
    return cat


def telecharger_celestrak(groupe="active"):
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={groupe}&FORMAT=tle"
    req = urllib.request.Request(url, headers={"User-Agent": "graves-radar-amateur/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        lignes = [l.rstrip() for l in r.read().decode().splitlines() if l.strip()]
    cat = []
    for i in range(0, len(lignes) - 2, 3):
        nom, l1, l2 = lignes[i:i + 3]
        cat.append(dict(norad=int(l1[2:7]), nom=nom.strip(), type="?", rcs="?", l1=l1, l2=l2, epoque=""))
    return cat


def mode_catalogue(args):
    nom = os.path.join(args.dossier, datetime.now(timezone.utc).strftime("catalogue_%Y%m%d.json"))
    if os.path.exists(nom) and time.time() - os.path.getmtime(nom) < 24 * 3600 and not args.force:
        print(f"{nom} a moins de 24 h : réutilisé (--force pour retélécharger)")
        return
    cat = telecharger_spacetrack() if args.source == "spacetrack" else telecharger_celestrak()
    with open(nom, "w") as f:
        json.dump(cat, f)
    types = {}
    for o in cat:
        types[o["type"]] = types.get(o["type"], 0) + 1
    print(f"{len(cat)} objets enregistrés dans {nom} : " +
          ", ".join(f"{v} {k}" for k, v in sorted(types.items(), key=lambda x: -x[1])))


def charger_catalogue(dossier, fichier=None):
    if fichier is None:
        fichiers = sorted(glob.glob(os.path.join(dossier, "catalogue_*.json")))
        if not fichiers:
            sys.exit("Aucun catalogue : lancer d'abord  python3 graves_satellites.py catalogue")
        fichier = fichiers[-1]
    cat = json.load(open(fichier))
    age = (time.time() - os.path.getmtime(fichier)) / 86400
    if age > 3:
        print(f"Attention : catalogue vieux de {age:.0f} jours, les prévisions se dégradent (Starlink manœuvrent)")
    return Catalogue(cat, fichier)


class Catalogue:
    def __init__(self, objets, fichier=""):
        from sgp4.api import Satrec, SatrecArray
        sats, garde = [], []
        for o in objets:
            try:
                sats.append(Satrec.twoline2rv(o["l1"], o["l2"]))
                garde.append(o)
            except Exception:
                continue
        self.objets = garde
        self.sats = SatrecArray(sats)
        self.satrecs = sats
        self.fichier = fichier
        self.epoque_jd = np.array([s.jdsatepoch + s.jdsatepochF for s in sats])

    def __len__(self):
        return len(self.objets)

    def sous_ensemble(self, idx):
        c = Catalogue.__new__(Catalogue)
        from sgp4.api import SatrecArray
        c.objets = [self.objets[i] for i in idx]
        c.satrecs = [self.satrecs[i] for i in idx]
        c.sats = SatrecArray(c.satrecs)
        c.fichier = self.fichier
        c.epoque_jd = self.epoque_jd[idx]
        return c


# ============================================================================ propagation
def jd_unix(t):
    jd = np.asarray(t, float) / 86400.0 + 2440587.5
    j = np.floor(jd)
    return j, jd - j


def gmst(jd_ut1):
    T = (jd_ut1 - 2451545.0) / 36525.0
    s = 67310.54841 + (876600 * 3600 + 8640184.812866) * T + 0.093104 * T ** 2 - 6.2e-6 * T ** 3
    return np.mod(s, 86400.0) / 86400.0 * 2 * np.pi


def propager(cat, t_unix):
    """Positions / vitesses terrestres (repère fixe lié à la Terre, m et m/s), forme (nsat, nt, 3)."""
    j, fr = jd_unix(t_unix)
    err, r, v = cat.sats.sgp4(j, fr)            # TEME, km et km/s
    th = gmst(j + fr)
    c, s = np.cos(th), np.sin(th)
    # rotation TEME → terrestre (pseudo-ITRF, mouvement du pôle négligé)
    rx = c[None, :] * r[..., 0] + s[None, :] * r[..., 1]
    ry = -s[None, :] * r[..., 0] + c[None, :] * r[..., 1]
    vx = c[None, :] * v[..., 0] + s[None, :] * v[..., 1]
    vy = -s[None, :] * v[..., 0] + c[None, :] * v[..., 1]
    R = np.stack([rx, ry, r[..., 2]], axis=-1) * 1000.0
    V = np.stack([vx, vy, v[..., 2]], axis=-1) * 1000.0
    V[..., 0] += OMEGA_TERRE * R[..., 1]        # v_terrestre = v - ω × r
    V[..., 1] -= OMEGA_TERRE * R[..., 0]
    R[err != 0] = np.nan
    V[err != 0] = np.nan
    return R, V


def geometrie(R, V, rx=adsb.RX_EXEMPLE, tx=adsb.GRAVES):
    """Doppler bistatique (Hz), élévations (°) vues de GRAVES et du récepteur, distances (km)."""
    T = adsb.ecef(*tx)
    Rr = adsb.ecef(*rx)
    dT, dR = R - T, R - Rr
    nT = np.linalg.norm(dT, axis=-1)
    nR = np.linalg.norm(dR, axis=-1)
    fd = -np.sum(V * (dT / nT[..., None] + dR / nR[..., None]), axis=-1) / LAMBDA
    upT = T / np.linalg.norm(T)                 # verticale géocentrique (écart < 0,2° : suffisant ici)
    upR = Rr / np.linalg.norm(Rr)
    elT = np.degrees(np.arcsin(np.sum(dT * upT, axis=-1) / nT))
    elR = np.degrees(np.arcsin(np.sum(dR * upR, axis=-1) / nR))
    return fd, elT, elR, nT / 1000, nR / 1000


def indice_detectabilite(obj, dT_km, dR_km):
    """10·log10(σ / (R_T² R_R²)) relatif à 1 m² à 1000 km × 1000 km (plus c'est haut, plus l'écho est fort)."""
    sigma = RCS_M2.get(obj["rcs"], 0.5)
    return 10 * np.log10(sigma) - 20 * np.log10(dT_km / 1000) - 20 * np.log10(dR_km / 1000)


def candidats(cat, t0, t1, rx, pas=10.0, el_min=0.0, fenetre=FENETRE_HZ, par_bloc=4000):
    """Objets au-dessus de l'horizon des deux sites avec un Doppler dans la fenêtre, entre t0 et t1."""
    tq = np.arange(t0 - pas, t1 + 2 * pas, pas)
    garde = []
    for d in range(0, len(cat), par_bloc):
        sous = cat.sous_ensemble(list(range(d, min(d + par_bloc, len(cat)))))
        R, V = propager(sous, tq)
        fd, elT, elR, nT, nR = geometrie(R, V, rx)
        ok = (elT > el_min) & (elR > el_min) & (np.abs(fd) <= fenetre + 500)
        for i in np.where(ok.any(axis=1))[0]:
            j = np.where(ok[i])[0]
            garde.append((d + i, float(indice_detectabilite(sous.objets[i], nT[i, j].min(), nR[i, j].min()))))
    garde.sort(key=lambda x: -x[1])
    return garde


# ============================================================================ corrélation
def courbes_fines(cat, idx, t_abs, rx, pas=1.0):
    """Doppler prévu (Hz) aux instants t_abs pour les objets idx (propagation à 1 s puis interpolation)."""
    sous = cat.sous_ensemble(idx)
    tq = np.arange(t_abs[0] - 15, t_abs[-1] + 15 + pas, pas)
    R, V = propager(sous, tq)
    fd, elT, elR, nT, nR = geometrie(R, V, rx)
    visible = (elT > 0) & (elR > 0)
    fd = np.where(visible, fd, np.nan)
    return tq, fd, nT, nR


def mode_superpose(args):
    from graves_capture import cascade

    cat = charger_catalogue(args.dossier, args.catalogue)
    iq, fs, t0 = adsb.charger_capture(args.iq, args.decalage_temps)
    rx = adsb.position_recepteur(args, args.iq)

    # cascade fine pour la porteuse et la dérive, puis cascade rapide
    f, tt, S = cascade(iq, fs, 0.5)
    off = adsb.recaler_porteuse(f, S, args.porteuse, bavard=False)
    if not args.sans_derive:
        iq, _ = adsb.corriger_derive(iq, fs, f, tt, S, off)
    fr, tr, Sr = cascade(iq, fs, args.resolution)
    fr = fr - off
    m = np.abs(fr) <= FENETRE_HZ
    fr, Sr = fr[m], Sr[m]
    norm = Sr - np.median(Sr, axis=1, keepdims=True)
    norm -= np.median(norm, axis=0, keepdims=True)
    debut = datetime.fromtimestamp(t0, timezone.utc)
    print(f"Capture du {debut:%Y-%m-%d %H:%M:%S} UTC, {tr[-1]:.0f} s — catalogue {os.path.basename(cat.fichier)} "
          f"({len(cat)} objets)")

    resultats = identifier(cat, fr, tr, norm, t0, rx, args)
    if resultats is None:
        return
    afficher(resultats, args)
    png = os.path.splitext(args.iq)[0].split("_iq_")[0] + "_satellites.png"
    tracer(png, fr, tr, norm, resultats, debut, args)
    print(f"\nImage : {png}")
    return resultats


def identifier(cat, fr, tr, norm, t0, rx, args, bavard=True):
    """Filtre adapté le long des courbes Doppler prévues. norm : cascade rapide normalisée (dB au-dessus
    du fond), fr : fréquences relatives à la porteuse, tr : temps depuis t0. Retourne la liste des objets testés."""
    from scipy.ndimage import maximum_filter1d
    P = 10 ** (np.clip(norm, -30, 40) / 10)                    # rapport de puissance au fond
    P[np.abs(fr) <= 20, :] = np.median(P)                      # porteuse et ses abords neutralisés
    Pm = maximum_filter1d(P, size=3, axis=0)                   # tolérance ±1 case
    df = fr[1] - fr[0]
    t_abs = t0 + tr
    t_calc = time.time()
    cands = candidats(cat, t_abs[0], t_abs[-1], rx)
    if bavard:
        print(f"{len(cands)} objets au-dessus de l'horizon de GRAVES et du récepteur avec un Doppler dans ±{FENETRE_HZ/1000:.1f} kHz "
              f"(calcul {time.time() - t_calc:.1f} s)")
    cands = cands[: args.max]
    if not cands:
        return []
    idx = [c[0] for c in cands]
    tq, FD, NT, NR = courbes_fines(cat, idx, t_abs, rx)

    taus = np.arange(-args.tau_max, args.tau_max + 1e-9, 0.25)
    decal_null = np.array([-461, -337, -211, -149, 151, 223, 347, 457], float)
    resultats = []
    for n, (i_cat, idet) in enumerate(cands):
        fd_q = FD[n]
        if np.all(np.isnan(fd_q)):
            continue
        # matrice (τ, tranche) des fréquences prévues : écho observé à t  ↔  prévision à t + τ
        F = np.array([np.interp(tr + t0 + tau, tq, fd_q, left=np.nan, right=np.nan) for tau in taus])
        valide = ~np.isnan(F) & (np.abs(F) <= FENETRE_HZ - 500) & (np.abs(F) > 25)
        if valide.sum(axis=1).max() < args.min_tranches:
            continue
        scores = np.full((len(decal_null) + 1, len(taus)), np.nan)
        for s, dec in enumerate(np.r_[0.0, decal_null]):
            K = np.round((F + dec - fr[0]) / df)
            ok = valide & (K >= 0) & (K < len(fr))
            for a in range(len(taus)):
                j = np.where(ok[a])[0]
                if len(j) >= args.min_tranches:
                    scores[s, a] = Pm[K[a, j].astype(int), j].mean()
        if np.all(np.isnan(scores[0])):
            continue
        a_best = int(np.nanargmax(scores[0]))
        meilleur = scores[0, a_best]
        # affinage du recalage au 1/20 s autour du meilleur pas grossier
        # (puissance lue exactement sur la courbe, interpolée entre cases : pas de plateau de tolérance)
        tau_fin, F_fin, sc_fin = taus[a_best], None, -np.inf
        for tau in np.arange(taus[a_best] - 0.5, taus[a_best] + 0.5001, 0.05):
            Fi = np.interp(tr + t0 + tau, tq, fd_q, left=np.nan, right=np.nan)
            vi = ~np.isnan(Fi) & (np.abs(Fi) <= FENETRE_HZ - 500) & (np.abs(Fi) > 25)
            x = (Fi - fr[0]) / df
            vi &= (x >= 0) & (x < len(fr) - 1)
            j = np.where(vi)[0]
            if len(j) >= args.min_tranches:
                k0 = np.floor(x[j]).astype(int); w = x[j] - k0
                sc = ((1 - w) * P[k0, j] + w * P[k0 + 1, j]).mean()
                if sc > sc_fin:
                    sc_fin, tau_fin, F_fin = sc, tau, Fi
        if F_fin is not None:
            F = F.copy(); F[a_best] = F_fin
            valide = valide.copy(); valide[a_best] = ~np.isnan(F_fin) & (np.abs(F_fin) <= FENETRE_HZ - 500) & (np.abs(F_fin) > 25)
        # décalages nuls dont la courbe sort entièrement de la fenêtre : aucune valeur, on les ignore
        fini = np.isfinite(scores[1:]).any(axis=1)
        nulls = np.nanmax(scores[1:][fini], axis=1) if fini.any() else np.array([])
        if len(nulls) < 4:
            continue
        mu, sd = float(nulls.mean()), float(max(nulls.std(ddof=1), 1e-3))
        z = (meilleur - mu) / sd
        # trace retenue : tranches où le pixel le long de la courbe dépasse nettement le fond
        K = np.round((F[a_best] - fr[0]) / df)
        ok = valide[a_best] & (K >= 0) & (K < len(fr))
        j = np.where(ok)[0]
        forts = j[10 * np.log10(Pm[K[j].astype(int), j]) >= args.seuil]
        o = cat.objets[i_cat]
        jj = ~np.isnan(fd_q)
        resultats.append(dict(i=i_cat, nom=o["nom"], norad=o["norad"], type=o["type"], rcs=o["rcs"],
                              idet=idet, tau=float(tau_fin), z=float(z), score=float(meilleur), mu=mu, sd=sd,
                              valide=valide[a_best],
                              ntr=int(len(j)), forts=forts, F=F[a_best],
                              dT=float(np.nanmin(NT[n][jj])), dR=float(np.nanmin(NR[n][jj])),
                              age=float(t_abs[0] / 86400 + 2440587.5 - cat.epoque_jd[i_cat])))

    # attribution exclusive : les objets sont acceptés par z décroissant ; les pixels d'un objet accepté
    # sont retirés, et le score des suivants est recalculé sur ce qui reste. Un objet dont le score ne
    # tenait qu'à l'écho d'un autre (courbe voisine, trains de Starlink…) retombe sous le seuil.
    resultats.sort(key=lambda r: -r["z"])
    Pr = Pm.copy()
    fond = float(np.median(P))
    proprio = np.full(Pm.shape, -1, dtype=np.int32)
    acceptes = []
    for r in resultats:
        r["identifie"] = False
        if r["z"] < args.z_min or len(r["forts"]) < args.min_forts:
            continue
        K = np.round((r["F"] - fr[0]) / df)
        ok = r["valide"] & (K >= 0) & (K < len(fr)) & ~np.isnan(r["F"])
        j = np.where(ok)[0]
        k = K[j].astype(int)
        z_reste = (Pr[k, j].mean() - r["mu"]) / r["sd"] if len(j) else -np.inf
        if z_reste < args.z_min:
            pris = proprio[k, j]
            pris = pris[pris >= 0]
            if len(pris):
                r["confondu"] = acceptes[np.bincount(pris).argmax()]["nom"]
            continue
        r["identifie"] = True
        r["z_reste"] = float(z_reste)
        for d in range(-2, 3):
            kk = np.clip(k + d, 0, len(fr) - 1)
            Pr[kk, j] = fond
            proprio[kk, j] = len(acceptes)
        acceptes.append(r)

    return resultats


def afficher(resultats, args):
    print("\nObjet                     NORAD  Type         RCS     Dist. GRAVES/RX    Indice   Âge TLE  Recalage   z     Verdict")
    for r in sorted(resultats, key=lambda r: (-r["identifie"], -r["z"]))[: args.lignes]:
        verdict = "ÉCHO IDENTIFIÉ" if r["identifie"] else (f"confondu avec {r['confondu']}" if r.get("confondu") else "-")
        print(f"{r['nom'][:25]:<25} {r['norad']:6d}  {r['type'][:11]:<11}  {r['rcs'][:6]:<6} "
              f"{r['dT']:6.0f} / {r['dR']:5.0f} km  {r['idet']:+6.1f} dB  {r['age']:5.1f} j  {r['tau']:+5.2f} s  "
              f"{r['z']:5.1f}  {verdict}")
    n = sum(r["identifie"] for r in resultats)
    print(f"\n{n} satellite(s) identifié(s) sur {len(resultats)} objet(s) dont la courbe traverse la fenêtre.")
    print("Recalage = décalage temporel entre la prévision TLE et l'écho observé ; z = écart aux courbes leurres (σ).")


def tracer(png, fr, tr, norm, resultats, debut, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pas_t = max(1, len(tr) // 1500)
    aff = norm[:, : (norm.shape[1] // pas_t) * pas_t].reshape(norm.shape[0], -1, pas_t).max(axis=2)
    fig, ax = plt.subplots(figsize=(13, 10), layout="constrained")
    vmin, vmax = np.percentile(aff, [30, 99.9])
    im = ax.imshow(aff.T, aspect="auto", origin="upper", cmap="magma", interpolation="nearest",
                   extent=[fr[0], fr[-1], tr[-1], tr[0]], vmin=vmin, vmax=max(vmax, vmin + 6))
    couleurs = ["#00e5ff", "#7cff4f", "#ffd60a", "#ff2dfe", "#ffffff", "#ff9500"]
    k = 0
    for r in resultats:
        if r["identifie"]:
            col = couleurs[k % len(couleurs)]; k += 1
            ax.plot(r["F"] + 120, tr, color=col, lw=1.3, linestyle=(0, (4, 5)))
            ok = np.where(~np.isnan(r["F"]))[0]
            j = ok[len(ok) // 3]
            ax.annotate(f"{r['nom']} ({r['norad']})  τ={r['tau']:+.1f} s", (r["F"][j] + 150, tr[j]), color=col,
                        fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, lw=0))
        elif args.tous and r["z"] > 0:
            ax.plot(r["F"] + 120, tr, ":", color="0.7", lw=0.7)
    ax.set_xlabel("Doppler bistatique (Hz) — 0 = porteuse directe GRAVES")
    ax.set_ylabel(f"Temps (s) depuis {debut:%H:%M:%S} UTC")
    ax.set_title(f"GRAVES 143,050 MHz — {debut:%Y-%m-%d %H:%M:%S} UTC — satellites identifiés\n"
                 f"(tirets = Doppler prévu recalé, tracé 120 Hz à droite de l'écho)")
    fig.colorbar(im, ax=ax, label="dB au-dessus du fond", pad=0.01, shrink=0.8)
    fig.savefig(png, dpi=100)
    plt.close(fig)


# ============================================================================ prévisions
def mode_passages(args):
    cat = charger_catalogue(args.dossier, args.catalogue)
    rx = adsb.position_recepteur(args)
    t0 = time.time() if args.debut is None else datetime.fromisoformat(args.debut).replace(tzinfo=timezone.utc).timestamp()
    t1 = t0 + args.heures * 3600
    pas = 20.0
    tq = np.arange(t0, t1, pas)
    passages = []
    t_calc = time.time()
    for d in range(0, len(cat), 2000):
        sous = cat.sous_ensemble(list(range(d, min(d + 2000, len(cat)))))
        R, V = propager(sous, tq)
        fd, elT, elR, nT, nR = geometrie(R, V, rx)
        ok = (elT > 0) & (elR > 0) & (np.abs(fd) <= FENETRE_HZ)
        for i in np.where(ok.any(axis=1))[0]:
            j = np.where(ok[i])[0]
            ruptures = np.where(np.diff(j) > 1)[0]
            for a, b in zip(np.r_[0, ruptures + 1], np.r_[ruptures, len(j) - 1]):
                jj = j[a:b + 1]
                o = sous.objets[i]
                passages.append(dict(t=tq[jj[0]], duree=(jj[-1] - jj[0] + 1) * pas, nom=o["nom"], norad=o["norad"],
                                     type=o["type"], rcs=o["rcs"], f0=fd[i, jj[0]], f1=fd[i, jj[-1]],
                                     dT=nT[i, jj].min(), dR=nR[i, jj].min(),
                                     idet=indice_detectabilite(o, nT[i, jj].min(), nR[i, jj].min())))
    passages.sort(key=lambda p: -p["idet"])
    print(f"{len(passages)} passages dans la fenêtre Doppler ±{FENETRE_HZ/1000:.1f} kHz sur {args.heures:g} h "
          f"(calcul {time.time() - t_calc:.0f} s). Les {args.lignes} plus favorables :")
    print("Début (UTC)          Durée   Objet                     NORAD  Type         RCS     Doppler (Hz)       GRAVES/RX (km)  Indice")
    for p in sorted(passages[: args.lignes], key=lambda p: p["t"]):
        print(f"{datetime.fromtimestamp(p['t'], timezone.utc):%Y-%m-%d %H:%M:%S}  {p['duree']:4.0f} s  {p['nom'][:25]:<25} "
              f"{p['norad']:6d}  {p['type'][:11]:<11}  {p['rcs'][:6]:<6} {p['f0']:+6.0f} → {p['f1']:+6.0f}   "
              f"{p['dT']:6.0f} / {p['dR']:5.0f}   {p['idet']:+5.1f} dB")


# ============================================================================ principal
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="mode", required=True)

    def communs(q):
        q.add_argument("--dossier", default=".", help="dossier des catalogues")
        q.add_argument("--catalogue", default=None, help="fichier catalogue précis (défaut : le plus récent)")
        q.add_argument("--lat", type=float, default=None, help="récepteur (défaut : position enregistrée, sinon station.ini)")
        q.add_argument("--lon", type=float, default=None)
        q.add_argument("--alt", type=float, default=None)
        q.add_argument("--lignes", type=int, default=30, help="nombre de lignes affichées")

    c = sp.add_parser("catalogue", help="télécharger les éléments orbitaux")
    c.add_argument("--source", choices=["spacetrack", "celestrak"], default="spacetrack")
    c.add_argument("--dossier", default=".")
    c.add_argument("--force", action="store_true")

    q = sp.add_parser("passages", help="prévoir les objets observables")
    communs(q)
    q.add_argument("--heures", type=float, default=3)
    q.add_argument("--debut", default=None, help="début UTC, ex. 2026-11-06T22:00 (défaut : maintenant)")

    s = sp.add_parser("superpose", help="identifier les échos de satellites d'une capture IQ")
    communs(s)
    s.add_argument("iq")
    s.add_argument("--resolution", type=float, default=8.0, help="résolution de la cascade rapide (Hz)")
    s.add_argument("--tau-max", dest="tau_max", type=float, default=8.0, help="recalage temporel cherché (± s)")
    s.add_argument("--max", type=int, default=400, help="nombre max de candidats testés (les plus détectables)")
    s.add_argument("--min-tranches", dest="min_tranches", type=int, default=15)
    s.add_argument("--min-forts", dest="min_forts", type=int, default=4,
                   help="nombre minimal de tranches où l'écho dépasse le seuil")
    s.add_argument("--seuil", type=float, default=8.0, help="seuil (dB) d'une tranche « forte »")
    s.add_argument("--z-min", dest="z_min", type=float, default=6.0)
    s.add_argument("--porteuse", type=float, default=None)
    s.add_argument("--decalage-temps", dest="decalage_temps", type=float, default=0.0)
    s.add_argument("--sans-derive", dest="sans_derive", action="store_true")
    s.add_argument("--tous", action="store_true", help="tracer aussi les candidats non identifiés")
    args = p.parse_args()
    {"catalogue": mode_catalogue, "passages": mode_passages, "superpose": mode_superpose}[args.mode](args)


if __name__ == "__main__":
    main()
