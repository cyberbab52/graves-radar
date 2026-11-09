#!/usr/bin/env python3
"""
graves_adsb.py — Étape 3 : identification des échos d'avions par l'ADS-B

Deux modes :

  1) journal : enregistre en continu les positions ADS-B (readsb local et/ou API adsb.fi)
     python3 graves_adsb.py journal                              # readsb local seul
     python3 graves_adsb.py journal --source local --source adsbfi   # + avions lointains et MLAT (adsb.fi)
     python3 graves_adsb.py journal --source adsbfi              # sur le terrain, sans récepteur ADS-B local
         → adsb_AAAAMMJJ.jsonl (un fichier par jour UTC), dans le dossier courant
     À laisser tourner dans un terminal pendant les captures.

  2) superpose : pour une capture enregistrée avec --iq, calcule le Doppler
     bistatique GRAVES → avion → récepteur de chaque avion ADS-B, mesure si un
     écho suit cette courbe dans la cascade, et trace les courbes + indicatifs.
     python3 graves_adsb.py superpose graves_20261105_054754_iq_16666.npy
         → graves_..._adsb.png + tableau des vols identifiés

Doppler bistatique :  f_D = -(1/λ) · v⃗ · (û_GRAVES→avion + û_RX→avion)
(v⃗ = vitesse sol + taux de montée de l'ADS-B, λ = c / 143,050 MHz)

Les fréquences de la cascade sont recalées sur la porteuse directe de GRAVES
(raie la plus forte à ±200 Hz), ce qui corrige l'erreur d'oscillateur du RSPdx.

Dépendances : numpy, scipy, matplotlib (+ graves_capture.py dans le même dossier).
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

C = 299_792_458.0
F0 = 143.050e6
LAMBDA = C / F0
KT = 0.514444          # nœud -> m/s
FT = 0.3048            # pied -> m

GRAVES = (47.3480, 5.5151, 200.0)
RX_EXEMPLE = (48.8566, 2.3522, 35.0)             # Paris : position d'exemple, à remplacer par station.ini
FICHIER_STATION = os.path.expanduser("~/.config/graves/station.ini")


def station_defaut():
    """Position par défaut du récepteur : ~/.config/graves/station.ini si présent, sinon une position d'exemple (Paris).
       [station]
       nom = Maison
       lat = 48.1000
       lon = 5.1000
       alt = 300          ; altitude de l'antenne (m), sol + hauteur du mât
    """
    if os.path.exists(FICHIER_STATION):
        import configparser
        c = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        c.read(FICHIER_STATION)
        try:
            st = c["station"]
            return (float(st["lat"]), float(st["lon"]), float(st.get("alt", "0"))), st.get("nom", "station")
        except Exception as e:
            print(f"Attention : {FICHIER_STATION} illisible ({e}) — position d'exemple (Paris) utilisée")
    print(f"Attention : aucune station configurée ({FICHIER_STATION}) — position d'exemple (Paris) utilisée, "
          f"les Doppler prévus seront faux pour votre récepteur")
    return RX_EXEMPLE, "Paris (exemple)"


STATION, NOM_STATION = station_defaut()


# --------------------------------------------------------------------------- géométrie
def ecef(lat, lon, h):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    la, lo = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    return np.stack([(N + h) * np.cos(la) * np.cos(lo),
                     (N + h) * np.cos(la) * np.sin(lo),
                     (N * (1 - e2) + h) * np.sin(la)], axis=-1)


def enu_vers_ecef(lat, lon, ve, vn, vu):
    la, lo = np.radians(lat), np.radians(lon)
    sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    vx = -so * ve - sl * co * vn + cl * co * vu
    vy = co * ve - sl * so * vn + cl * so * vu
    vz = cl * vn + sl * vu
    return np.stack([vx, vy, vz], axis=-1)


def doppler_bistatique(lat, lon, alt_m, gs_kt, track_deg, vr_ftmin, rx=RX_EXEMPLE, tx=GRAVES):
    P = ecef(lat, lon, alt_m)
    T = ecef(*tx)
    R = ecef(*rx)
    ve = gs_kt * KT * np.sin(np.radians(track_deg))
    vn = gs_kt * KT * np.cos(np.radians(track_deg))
    vu = vr_ftmin * FT / 60.0
    V = enu_vers_ecef(lat, lon, ve, vn, vu)
    uT = (P - T) / np.linalg.norm(P - T, axis=-1, keepdims=True)
    uR = (P - R) / np.linalg.norm(P - R, axis=-1, keepdims=True)
    fd = -np.sum(V * (uT + uR), axis=-1) / LAMBDA
    dT = np.linalg.norm(P - T, axis=-1) / 1000
    dR = np.linalg.norm(P - R, axis=-1) / 1000
    return fd, dT, dR


def az_el(depuis, lat, lon, alt_m):
    """Azimut / élévation d'un point vu depuis 'depuis' (lat, lon, h)."""
    O = ecef(*depuis)
    P = ecef(lat, lon, alt_m)
    d = P - O
    la, lo = math.radians(depuis[0]), math.radians(depuis[1])
    e = -math.sin(lo) * d[..., 0] + math.cos(lo) * d[..., 1]
    n = -math.sin(la) * math.cos(lo) * d[..., 0] - math.sin(la) * math.sin(lo) * d[..., 1] + math.cos(la) * d[..., 2]
    u = math.cos(la) * math.cos(lo) * d[..., 0] + math.cos(la) * math.sin(lo) * d[..., 1] + math.sin(la) * d[..., 2]
    return np.degrees(np.arctan2(e, n)) % 360, np.degrees(np.arctan2(u, np.hypot(e, n)))


# --------------------------------------------------------------------------- journal ADS-B
ADSBFI_URL = "https://opendata.adsb.fi/api/v3/lat/{lat:.4f}/lon/{lon:.4f}/dist/{nm:.0f}"
ADSBFI_MENTION = "Données ADS-B complémentaires : adsb.fi (https://adsb.fi)"
CENTRE_DEFAUT = GRAVES[:2]          # zone éclairée par GRAVES


def lire_aircraft_json(source, timeout=5):
    """Lit un aircraft.json readsb (fichier ou URL) ou une réponse compatible ADSBexchange v2.
    Retourne (instant de la réponse en s Unix, liste des avions)."""
    if source.startswith("http"):
        import urllib.request
        req = urllib.request.Request(source, headers={"User-Agent": "graves-radar-amateur/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    else:
        with open(source) as f:
            d = json.load(f)
    now = d.get("now", time.time())
    if now > 1e11:                    # ADSBexchange v2 : millisecondes
        now /= 1000.0
    return float(now), d.get("aircraft", d.get("ac", [] )) or []


class Source:
    """Une source de positions : readsb local (1 s) ou API adsb.fi (période ≥ 1 s, recul en cas d'erreur)."""

    def __init__(self, nom, url, periode):
        self.nom, self.url, self.periode = nom, url, periode
        self.prochain, self.recul, self.n_avions, self.erreur = 0.0, 0.0, 0, ""

    def interroger(self):
        if time.time() < self.prochain:
            return None
        try:
            now, avions = lire_aircraft_json(self.url)
            self.recul, self.erreur, self.n_avions = 0.0, "", len(avions)
            self.prochain = time.time() + self.periode
            return now, avions
        except Exception as e:
            code = getattr(e, "code", None)
            # adsb.fi bloque temporairement les IP qui enchaînent les erreurs 4xx : on recule franchement
            self.recul = min(max(self.recul * 2, 30.0 if code in (400, 401, 403, 404, 429) else 5.0), 600.0)
            self.erreur = f"{code or ''} {e}".strip()
            self.prochain = time.time() + self.recul
            print(f"  [{self.nom}] lecture impossible ({self.erreur}) — nouvel essai dans {self.recul:.0f} s", flush=True)
            return None


def construire_sources(args):
    sources = []
    for s in args.source:
        if s == "local":
            sources.append(Source("local", args.local, 1.0))
        elif s == "adsbfi":
            url = (args.url_adsbfi or ADSBFI_URL).format(lat=args.centre[0], lon=args.centre[1],
                                                          nm=min(args.rayon_nm, 250))
            sources.append(Source("adsbfi", url, max(1.0, args.periode_adsbfi)))
        else:
            sources.append(Source(os.path.basename(s) or s, s, 1.0))
    return sources


def mode_journal(args):
    sources = construire_sources(args)
    dernier = {}
    types = {}                                  # hex → type OACI (A320, B738…), mémorisé quelle que soit la source
    n_ecrits, n_par_src, t_affiche = 0, {}, 0
    print("Journal ADS-B → adsb_AAAAMMJJ.jsonl (Ctrl-C pour arrêter). Sources : " +
          ", ".join(f"{s.nom} ({s.url}, toutes les {s.periode:g} s)" for s in sources))
    if any(s.nom == "adsbfi" for s in sources):
        print("  " + ADSBFI_MENTION + " — usage personnel non commercial, 1 requête/s maximum")
    while True:
        lignes, now_max = [], time.time()
        for src in sources:
            r = src.interroger()
            if r is None:
                continue
            now, avions = r
            now_max = max(now_max, now)
            for a in avions:
                if "lat" not in a or "lon" not in a or a.get("seen_pos", 99) > 3 or "hex" not in a:
                    continue
                alt = a.get("alt_geom", a.get("alt_baro"))
                if alt is None or alt == "ground":
                    continue
                t = now - a["seen_pos"]
                h = a["hex"].lower().lstrip("~")
                if (a.get("t") or "").strip():          # adsb.fi, ou readsb avec --db-file
                    types[h] = a["t"].strip()
                if t <= dernier.get(h, 0) + 0.5:        # fusion : on garde la position la plus fraîche
                    continue
                dernier[h] = t
                lignes.append(json.dumps(dict(
                    t=round(t, 2), hex=h, flight=(a.get("flight") or "").strip(), typ=types.get(h, ""),
                    lat=a["lat"], lon=a["lon"], alt_ft=alt, gs=a.get("gs"), track=a.get("track"),
                    vr=a.get("geom_rate", a.get("baro_rate", 0)) or 0,
                    src=src.nom, pos=a.get("type", ""))))
                n_par_src[src.nom] = n_par_src.get(src.nom, 0) + 1
        if lignes:
            nom = datetime.fromtimestamp(now_max, timezone.utc).strftime("adsb_%Y%m%d.jsonl")
            with open(os.path.join(args.dossier, nom), "a") as f:
                f.write("\n".join(lignes) + "\n")
            n_ecrits += len(lignes)
        if time.time() - t_affiche > 60:
            print(f"  {datetime.now(timezone.utc):%H:%M:%S} UTC — " +
                  ", ".join(f"{s.nom} : {s.n_avions} avions{' (' + s.erreur + ')' if s.erreur else ''}"
                            for s in sources) +
                  f" — {n_ecrits} positions enregistrées (" +
                  ", ".join(f"{k} {v}" for k, v in n_par_src.items()) + ")", flush=True)
            t_affiche = time.time()
        time.sleep(0.5)


def completer_vitesses(piste):
    """Positions MLAT ou incomplètes sans vitesse/cap : on les déduit des positions successives."""
    manque = [i for i, r in enumerate(piste) if r.get("gs") is None or r.get("track") is None]
    if not manque:
        return piste
    lat = np.array([r["lat"] for r in piste]); lon = np.array([r["lon"] for r in piste])
    t = np.array([r["t"] for r in piste])
    for i in manque:
        a, b = max(0, i - 1), min(len(piste) - 1, i + 1)
        if b == a or t[b] - t[a] <= 0:
            continue
        dn = math.radians(lat[b] - lat[a]) * 6371000
        de = math.radians(lon[b] - lon[a]) * 6371000 * math.cos(math.radians(lat[i]))
        v = math.hypot(dn, de) / (t[b] - t[a])
        piste[i]["gs"] = v / KT
        piste[i]["track"] = math.degrees(math.atan2(de, dn)) % 360
    garde = [r for r in piste if r.get("gs") is not None and r.get("track") is not None]
    return garde


def charger_journal(dossier, t0, t1, marge=60):
    pistes = {}
    jours = {datetime.fromtimestamp(x, timezone.utc).strftime("%Y%m%d") for x in (t0 - marge, t1 + marge)}
    fichiers = [f for j in jours for f in glob.glob(os.path.join(dossier, f"adsb_{j}.jsonl"))]
    for fn in fichiers:
        with open(fn) as f:
            for ligne in f:
                try:
                    r = json.loads(ligne)
                except ValueError:
                    continue
                if t0 - marge <= r["t"] <= t1 + marge:
                    pistes.setdefault(r["hex"], []).append(r)
    for h in list(pistes):
        pistes[h].sort(key=lambda r: r["t"])
        pistes[h] = completer_vitesses(pistes[h])
        if not pistes[h]:
            del pistes[h]
    return pistes, fichiers


# --------------------------------------------------------------------------- corrélation
def courbe_predite(piste, temps_abs, rx):
    """Doppler prédit aux instants temps_abs (NaN hors de la piste ADS-B, ±5 s toléré)."""
    t = np.array([r["t"] for r in piste])
    if len(t) < 2:
        return None
    def interp(cle, unwrap=False):
        v = np.array([float(r[cle]) for r in piste])
        if unwrap:
            v = np.degrees(np.unwrap(np.radians(v)))
        return np.interp(temps_abs, t, v)
    lat, lon = interp("lat"), interp("lon")
    alt = interp("alt_ft") * FT
    gs, trk, vr = interp("gs"), interp("track", True), interp("vr")
    fd, dT, dR = doppler_bistatique(lat, lon, alt, gs, trk, vr, rx=rx)
    hors = (temps_abs < t[0] - 5) | (temps_abs > t[-1] + 5)
    # horizon radio (Terre « 4/3 ») : élévation apparente ≈ géométrique + d / (8 R)
    _, elT = az_el(GRAVES, lat, lon, alt)
    _, elR = az_el(rx, lat, lon, alt)
    elT = elT + np.degrees(dT / (8 * 6371.0))
    elR = elR + np.degrees(dR / (8 * 6371.0))
    hors |= (elT < -0.3) | (elR < -0.3)
    fd[hors] = np.nan
    return dict(fd=fd, dT=dT, dR=dR, lat=lat, lon=lon, alt=alt)


def presence(norm, f, fd_path, tol_hz, seuil_db):
    """Fraction des tranches temporelles où un pic > seuil existe à ±tol autour de la courbe."""
    det, n = detections(norm, f, fd_path, tol_hz, seuil_db)
    if n == 0:
        return np.nan, 0
    return len(det[0]) / n, n


def detections(norm, f, fd_path, tol_hz, seuil_db):
    """Pics > seuil à ±tol de la courbe : (colonnes, fréquence du pic, résidu pic - prédiction), n tranches."""
    ok = ~np.isnan(fd_path)
    vide = (np.array([], int), np.array([]), np.array([]))
    if ok.sum() < 5:
        return vide, 0
    df = f[1] - f[0]
    k = np.round((fd_path[ok] - f[0]) / df).astype(int)
    w = int(np.ceil(tol_hz / df))
    cols = np.where(ok)[0]
    dedans = (k - w >= 0) & (k + w < len(f))
    if dedans.sum() < 5:
        return vide, 0
    k, cols = k[dedans], cols[dedans]
    fen = np.stack([norm[k + j, cols] for j in range(-w, w + 1)])
    jmax = fen.argmax(axis=0)
    vmax = fen.max(axis=0)
    hit = vmax > seuil_db
    kp = k[hit] + jmax[hit] - w
    return (cols[hit], f[kp], f[kp] - fd_path[cols[hit]]), int(dedans.sum())


def masquer_porteuse(norm, f_rel, largeur_hz=1.5):
    """Copie de la cascade où la porteuse directe (0 Hz) ne peut pas être prise pour un écho."""
    m = norm.copy()
    m[np.abs(f_rel) <= largeur_hz, :] = -99.0
    return m


def recaler_porteuse(f, S, impose=None, bavard=True):
    moy = S.mean(axis=1)
    zone = np.abs(f) <= 200
    i_p = np.argmax(np.where(zone, moy, -np.inf))
    snr_p = moy[i_p] - np.median(moy)
    if impose is not None:
        off = impose
        msg = f"Porteuse imposée à {off:+.1f} Hz"
    elif snr_p >= 6:
        off = f[i_p]
        msg = f"Porteuse directe GRAVES détectée à {off:+.1f} Hz (+{snr_p:.1f} dB) → référence 0 Hz"
    else:
        off = 0.0
        msg = "Porteuse directe non détectée : pas de recalage (option --porteuse pour l'imposer)"
    if bavard:
        print(msg)
    return off


def suivre_porteuse(f, tt, S, off, fenetre=25.0, pas_max=1.0, seuil_db=8.0):
    """Suit la fréquence de la porteuse directe tranche par tranche (continuité imposée : la dérive
    d'un oscillateur est lente, un avion qui croise la porteuse ne peut pas la « capturer »)."""
    m = np.abs(f - off) <= fenetre
    fz = f[m]
    Z = S[m] - np.median(S, axis=0, keepdims=True)
    fc = np.full(len(tt), np.nan)
    forts = Z.max(axis=0)
    if forts.max() < seuil_db:
        return fc
    j0 = int(np.argmax(np.where(np.arange(len(tt)) >= 2 * len(tt) // 3, forts, -np.inf)))
    if forts[j0] < seuil_db:
        j0 = int(np.argmax(forts))
    fc[j0] = fz[np.argmax(Z[:, j0])]
    dt = tt[1] - tt[0]
    for sens in (1, -1):
        f_prev, j_prev = fc[j0], j0
        j = j0 + sens
        while 0 <= j < len(tt):
            tol = 0.6 + pas_max * abs(j - j_prev) * dt
            v = np.abs(fz - f_prev) <= tol
            if v.any():
                k = np.argmax(np.where(v, Z[:, j], -np.inf))
                if Z[k, j] >= seuil_db:
                    fc[j] = fz[k]
                    f_prev, j_prev = fc[j], j
            j += sens
    return fc


def corriger_derive(iq, fs, f, tt, S, off, bavard=True):
    """Corrige la dérive de l'oscillateur du récepteur en ramenant la porteuse directe à une fréquence
    constante (sa valeur stabilisée, en fin de capture). Retourne (iq corrigé, dérive par tranche)."""
    from scipy.ndimage import median_filter
    moy = S.mean(axis=1)
    i_p = int(np.argmin(np.abs(f - off)))
    if moy[max(0, i_p - 1):i_p + 2].max() - np.median(moy) < 6:
        if bavard:
            print("Dérive de l'oscillateur : porteuse directe non détectée, pas de correction")
        return iq, np.zeros(len(tt))
    fc = suivre_porteuse(f, tt, S, off)
    ok = ~np.isnan(fc)
    if ok.mean() < 0.3:
        if bavard:
            print("Dérive de l'oscillateur : porteuse trop intermittente pour la suivre, pas de correction")
        return iq, np.zeros(len(tt))
    lisse = median_filter(fc[ok], size=min(9, ok.sum() | 1), mode="nearest")
    courbe = np.interp(tt, tt[ok], lisse)
    lent = median_filter(courbe, size=min(31, len(courbe) | 1), mode="nearest")
    if np.abs(np.gradient(lent, tt)).max() > 0.6:
        # repli : ajustement polynomial robuste sur le pic le plus proche de la porteuse, tranche par
        # tranche ; les échos d'avions qui croisent la porteuse sont rejetés comme points aberrants
        m = np.abs(f - off) <= 15
        Z = S[m] - np.median(S, axis=0, keepdims=True)
        k = Z.argmax(axis=0)
        fp, forts = f[m][k], Z.max(axis=0) >= 6
        garde = forts.copy()
        poly = None
        for _ in range(5):
            if garde.sum() < 0.25 * len(tt):
                break
            poly = np.polyfit(tt[garde], fp[garde], 3)
            garde = forts & (np.abs(fp - np.polyval(poly, tt)) <= 1.5)
        if poly is None or garde.sum() < 0.25 * len(tt) or np.abs(np.polyval(np.polyder(poly), tt)).max() > 0.6:
            if bavard:
                print("Dérive de l'oscillateur : porteuse trop faible ou trop perturbée par des échos, pas de correction")
            return iq, np.zeros(len(tt))
        courbe = np.polyval(poly, tt)
        if bavard:
            print(f"Dérive de l'oscillateur : suivi direct perturbé par des échos, ajustement robuste "
                  f"({100 * garde.mean():.0f} % des tranches retenues)")
    ref = float(np.median(courbe[2 * len(tt) // 3:]))
    derive = courbe - ref
    if np.abs(derive).max() < 0.5:
        if bavard:
            print(f"Dérive de l'oscillateur négligeable (< 0,5 Hz)")
        return iq, derive
    ts = np.arange(len(iq)) / fs
    d = np.interp(ts, tt, derive)
    phase = -2 * np.pi * np.cumsum(d) / fs
    if bavard:
        j = int(np.argmax(np.abs(derive)))
        t_stab = tt[np.where(np.abs(derive) > 0.5)[0][-1]]
        fin = (f"stabilisée après ~{t_stab:.0f} s" if t_stab < 0.85 * tt[-1]
               else "dérive lente encore en cours en fin de capture")
        print(f"Dérive de l'oscillateur corrigée : jusqu'à {derive[j]:+.1f} Hz (à t = {tt[j]:.0f} s), {fin}")
    return (iq * np.exp(1j * phase).astype(np.complex64)).astype(np.complex64), derive


def correler(norm, f_rel, temps_abs, pistes, rx, p):
    """Corrèle chaque piste ADS-B avec la cascade. Attribution exclusive : un pic d'écho
    revendiqué par plusieurs vols va au vol dont la prédiction est la plus proche."""
    nm = masquer_porteuse(norm, f_rel)
    cands = []
    for h, piste in pistes.items():
        c = courbe_predite(piste, temps_abs, rx)
        if c is None or np.all(np.isnan(c["fd"])):
            continue
        if np.nanmin(np.abs(c["fd"])) > p.largeur or np.nanmin(c["dT"]) > p.portee:
            continue
        det, n = detections(nm, f_rel, c["fd"], p.tolerance, p.seuil)
        if n == 0:
            continue
        nulls = [presence(nm, f_rel, c["fd"] + d, p.tolerance, p.seuil)[0]
                 for d in (-61.0, -37.0, -23.0, 29.0, 43.0, 71.0)]
        nulls = [x for x in nulls if not np.isnan(x)]
        cands.append(dict(hex=h, flight=next((r["flight"] for r in reversed(piste) if r.get("flight")), h),
                          sources=sorted({r.get("src", "local") for r in piste}), c=c, det=det, n=n,
                          p0=float(np.mean(nulls)) if nulls else 0.0))

    # attribution exclusive, tranche par tranche
    reclam = {}
    for ia, a in enumerate(cands):
        for col, fp, res in zip(*a["det"]):
            reclam.setdefault(int(col), []).append((abs(res), ia, fp, res))
    gardes = [[] for _ in cands]
    for col, lst in reclam.items():
        lst.sort()
        acceptes = []
        for ares, ia, fp, res in lst:
            if any(abs(fp - fa) <= 2.0 for fa in acceptes):
                continue          # même écho déjà attribué à un vol plus proche
            acceptes.append(fp)
            gardes[ia].append((col, fp, res))

    resultats = []
    for a, g in zip(cands, gardes):
        n, p0 = a["n"], a["p0"]
        pres = len(g) / n
        score = pres - p0
        p0c = min(max(p0, 1.0 / n), 0.99)
        z = score / math.sqrt(p0c * (1 - p0c) / n)
        c = a["c"]
        valid = ~np.isnan(c["fd"])
        i_mid = np.where(valid)[0][valid.sum() // 2]
        az, el = az_el(GRAVES, c["lat"][i_mid], c["lon"][i_mid], c["alt"][i_mid])
        ecart = float(np.median(np.abs([x[2] for x in g]))) if g else np.nan
        resultats.append(dict(hex=a["hex"], flight=a["flight"], sources=a["sources"], c=c, pres=pres, p0=p0, score=score, z=z, n=n,
                              cols=np.array([x[0] for x in g], int), fpics=np.array([x[1] for x in g]),
                              ecart=ecart, alt=float(np.nanmedian(c["alt"])) / FT,
                              dT=float(np.nanmin(c["dT"])), dR=float(np.nanmin(c["dR"])),
                              az=float(az), el=float(el),
                              identifie=(score >= p.score_min and z >= p.z_min)))
    resultats.sort(key=lambda r: -r["score"])
    return resultats


def position_recepteur(args, fichier_iq=None, bavard=True):
    """Position du récepteur : --lat/--lon/--alt si donnés, sinon celle enregistrée avec la capture
    (graves_capture.py la sauvegarde dans le .json), sinon station.ini, sinon la position d'exemple."""
    if getattr(args, "lat", None) is not None and getattr(args, "lon", None) is not None:
        rx, src = (args.lat, args.lon, args.alt if args.alt is not None else 0.0), "ligne de commande"
    else:
        rx, src = STATION, NOM_STATION + (" — station.ini" if os.path.exists(FICHIER_STATION) else "")
        if fichier_iq:
            meta_fn = os.path.splitext(fichier_iq)[0] + ".json"
            if os.path.exists(meta_fn):
                m = json.load(open(meta_fn))
                if m.get("rx"):
                    rx, src = tuple(m["rx"]), "enregistrée avec la capture"
    if bavard:
        print(f"Récepteur : {rx[0]:.4f} N, {rx[1]:.4f} E, {rx[2]:.0f} m ({src})")
    return rx


def charger_capture(fichier_iq, decalage_temps=0.0):
    meta_fn = os.path.splitext(fichier_iq)[0] + ".json"
    if not os.path.exists(meta_fn):
        sys.exit(f"Fichier {meta_fn} introuvable (capturer avec la nouvelle version de graves_capture.py --iq)")
    meta = json.load(open(meta_fn))
    return np.load(fichier_iq), meta["fs"], meta["t0_unix"] + decalage_temps


def mode_superpose(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from graves_capture import cascade

    iq, fs, t0 = charger_capture(args.iq, args.decalage_temps)
    rx = position_recepteur(args, args.iq)
    f, tt, S = cascade(iq, fs, args.resolution)
    off = recaler_porteuse(f, S, args.porteuse, bavard=False)
    if not args.sans_derive:
        iq, _ = corriger_derive(iq, fs, f, tt, S, off)
        f, tt, S = cascade(iq, fs, args.resolution)
    norm = S - np.median(S, axis=1, keepdims=True)
    norm -= np.median(norm, axis=0, keepdims=True)      # élévations du fond sur toute la bande neutralisées
    temps_abs = t0 + tt
    off = recaler_porteuse(f, S, args.porteuse)
    f_rel = f - off

    pistes, fichiers = charger_journal(args.journal, temps_abs[0], temps_abs[-1])
    if not fichiers:
        sys.exit(f"Aucun journal ADS-B adsb_AAAAMMJJ.jsonl pour cette période dans {args.journal}")
    print(f"{len(pistes)} avions ADS-B dans le journal sur la période")
    resultats = correler(norm, f_rel, temps_abs, pistes, rx, args)

    debut = datetime.fromtimestamp(t0, timezone.utc)
    print(f"\nCapture du {debut:%Y-%m-%d %H:%M:%S} UTC, {tt[-1]:.0f} s")
    print("Vol       Hex     Alt (ft)  Dist. GRAVES  Az/El vu de GRAVES  Doppler prédit (Hz)  Présence  Bruit  Score     z  Écart  Verdict")
    for r in resultats:
        fd = r["c"]["fd"]
        ec = f"{r['ecart']:4.1f} Hz" if not np.isnan(r["ecart"]) else "   -   "
        print(f"{r['flight']:<9} {r['hex']}  {r['alt']:7.0f}   {r['dT']:6.0f} km     {r['az']:5.0f}° / {r['el']:4.1f}°      "
              f"{fd[~np.isnan(fd)][0]:+6.0f} → {fd[~np.isnan(fd)][-1]:+6.0f}     {r['pres']:5.2f}   {r['p0']:5.2f}  "
              f"{r['score']:+.2f} {r['z']:5.1f}  {ec}  {'ÉCHO IDENTIFIÉ' if r['identifie'] else '-'}")

    sortie = os.path.splitext(args.iq)[0].replace("_iq_" + str(int(fs)), "") + "_adsb.png"
    tracer(sortie, f_rel, tt, norm, resultats, args, debut, off)
    print(f"\nGraphique : {sortie}")
    n_id = sum(r["identifie"] for r in resultats)
    print(f"{n_id} écho(s) identifié(s) sur {len(resultats)} avion(s) dont la courbe passe dans la fenêtre ±{args.largeur:g} Hz")
    print("Écart = écart médian (Hz) entre les pics d'écho attribués et la courbe prédite.")


def tracer(png, f, tt, norm, resultats, args, debut, off):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = np.abs(f) <= args.largeur
    fig, ax = plt.subplots(figsize=(13, 10), layout="constrained")
    vmin, vmax = np.percentile(norm[m], [20, 99.8])
    im = ax.imshow(norm[m].T, aspect="auto", origin="upper", cmap="viridis",
                   extent=[f[m][0], f[m][-1], tt[-1], tt[0]], vmin=vmin, vmax=max(vmax, vmin + 6),
                   interpolation="nearest")
    couleurs = ["#ff3b30", "#ffffff", "#ff9500", "#ff2dfe", "#00e5ff", "#ffd60a"]
    i_c = 0
    for r in resultats:
        fd = r["c"]["fd"]
        if r["identifie"]:
            col = couleurs[i_c % len(couleurs)]; i_c += 1
            # courbe décalée de +8 Hz pour ne pas masquer l'écho lui-même
            ax.plot(fd + 8, tt, color=col, lw=1.3, alpha=0.9, linestyle=(0, (4, 6)))
            ok = np.where(~np.isnan(fd) & (np.abs(fd) <= args.largeur))[0]
            if len(ok):
                j = ok[int(len(ok) * (0.15 + 0.17 * ((i_c - 1) % 5)))]
                ax.annotate(f"{r['flight']}  {r['alt']/100:.0f}00 ft", (fd[j] + 8, tt[j]), xytext=(10, 0),
                            textcoords="offset points", color=col, fontsize=10, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, lw=0))
        elif args.tous:
            ax.plot(fd + 8, tt, ":", color="0.85", lw=0.8)
    ax.set_xlim(-args.largeur, args.largeur)
    ax.set_xlabel(f"Doppler bistatique (Hz) — 0 = porteuse directe GRAVES (mesurée à {off:+.1f} Hz)")
    ax.set_ylabel(f"Temps (s) depuis {debut:%H:%M:%S} UTC")
    ax.set_title(f"GRAVES 143,050 MHz — {debut:%Y-%m-%d %H:%M:%S} UTC — échos et vols ADS-B "
                 f"\n(tirets = Doppler prédit des vols identifiés, tracé 8 Hz à droite de l'écho)")
    fig.colorbar(im, ax=ax, label="dB au-dessus du fond", pad=0.01, shrink=0.8)
    if any("adsbfi" in r.get("sources", []) for r in resultats):
        fig.text(0.01, 0.003, ADSBFI_MENTION, fontsize=8, color="0.4")
    fig.savefig(png, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- principal
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="mode", required=True)

    j = sp.add_parser("journal", help="enregistrer les positions ADS-B en continu")
    j.add_argument("--source", action="append", default=None,
                   help="source de positions, répétable : local (readsb), adsbfi (API adsb.fi), ou un fichier/URL "
                        "aircraft.json. Défaut : local. Exemple : --source local --source adsbfi")
    j.add_argument("--local", default="/run/readsb/aircraft.json", help="aircraft.json du readsb local")
    j.add_argument("--centre", type=float, nargs=2, default=CENTRE_DEFAUT, metavar=("LAT", "LON"),
                   help="centre de la zone interrogée sur adsb.fi (défaut : GRAVES)")
    j.add_argument("--rayon-nm", dest="rayon_nm", type=float, default=250, help="rayon adsb.fi (NM, max 250)")
    j.add_argument("--periode-adsbfi", dest="periode_adsbfi", type=float, default=2.0,
                   help="secondes entre deux requêtes adsb.fi (min 1, limite du service)")
    j.add_argument("--url-adsbfi", dest="url_adsbfi", default=None, help=argparse.SUPPRESS)
    j.add_argument("--dossier", default=".")

    s = sp.add_parser("superpose", help="identifier les échos d'une capture IQ")
    s.add_argument("iq", help="fichier graves_..._iq_NNNNN.npy (avec son .json)")
    s.add_argument("--journal", default=".", help="dossier des fichiers adsb_AAAAMMJJ.jsonl")
    s.add_argument("--lat", type=float, default=None, help="récepteur (défaut : position enregistrée, sinon station.ini)")
    s.add_argument("--lon", type=float, default=None)
    s.add_argument("--alt", type=float, default=None)
    s.add_argument("--largeur", type=float, default=400, help="demi-largeur affichée (Hz)")
    s.add_argument("--resolution", type=float, default=0.5, help="résolution de la cascade (Hz)")
    s.add_argument("--tolerance", type=float, default=3.0, help="tolérance autour de la courbe prédite (Hz)")
    s.add_argument("--seuil", type=float, default=9.0, help="seuil de détection (dB au-dessus du fond)")
    s.add_argument("--score-min", dest="score_min", type=float, default=0.04)
    s.add_argument("--z-min", dest="z_min", type=float, default=5.0, help="significativité minimale (écarts-types)")
    s.add_argument("--portee", type=float, default=600, help="distance max à GRAVES (km)")
    s.add_argument("--porteuse", type=float, default=None, help="imposer la position de la porteuse (Hz)")
    s.add_argument("--decalage-temps", dest="decalage_temps", type=float, default=0.0,
                   help="correction d'horloge de la capture (s)")
    s.add_argument("--sans-derive", dest="sans_derive", action="store_true",
                   help="ne pas corriger la dérive de l'oscillateur du récepteur")
    s.add_argument("--tous", action="store_true", help="tracer aussi (en pointillés) les vols non identifiés")
    args = p.parse_args()
    if args.mode == "journal" and not args.source:
        args.source = ["local"]
    (mode_journal if args.mode == "journal" else mode_superpose)(args)


if __name__ == "__main__":
    main()
