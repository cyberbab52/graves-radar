#!/usr/bin/env python3
"""
graves_gps.py — Phase 2 : position et heure précises par GPS, avec l'entrée A du RSPdx

Récepteur GPS logiciel complet (signal L1 C/A, 1575,42 MHz). Le RSPdx enregistre quelques dizaines de
secondes d'IQ sur l'entrée A (antenne active MikroTik ACGPSA alimentée par un bias-T externe), puis :
  1. acquisition des satellites (recherche parallèle code × Doppler par FFT) ;
  2. poursuite de chaque satellite (boucle de Costas assistée en fréquence, DLL early/late assistée
     par la porteuse) ;
  3. synchronisation bit (50 bit/s) et sous-trame, contrôle des parités, décodage des éphémérides,
     des paramètres ionosphériques et du nombre de secondes intercalaires ;
  4. position (WGS84), heure GPS / UTC de chaque échantillon IQ, vitesse ;
  5. écart de fréquence de l'oscillateur (TCXO) du RSPdx, en ppm : le même oscillateur pilote l'OL à
     143,050 MHz, d'où la correction à appliquer aux Doppler GRAVES ; écart de l'horloge du PC à l'UTC.

Le RSPdx n'a qu'un seul tuner : on ne peut pas recevoir GPS (entrée A) et GRAVES (entrée B) en même temps.
Ce module s'utilise donc en « session de calage », avant ou après une session GRAVES.

Usage :
  python3 graves_gps.py --liste                    # inventaire du RSPdx
  python3 graves_gps.py --test-antenne             # l'antenne active est-elle alimentée ? (~20 s)
  python3 graves_gps.py                            # capture + solution (48 s, ou 20 s si éphémérides en cache)
  python3 graves_gps.py --duree 600 --position 48.8566 2.3522 85   # 10 min, traitées par tranches de 60 s
  python3 graves_gps.py --duree 60 --garder        # garde aussi l'IQ brut (.cs8 + .json) pour retraitement
  python3 graves_gps.py --fichier gps_20261109_142233.cs8           # retraitement d'un enregistrement gardé
  python3 graves_gps.py --fichier sim.cs8 --format cs8 --fs 2048000 # fichier gps-sdr-sim (essai sans matériel)
  python3 graves_gps.py --regler-horloge           # met en plus l'horloge système à l'heure GPS (terrain sans
                                                   # réseau ; sudo n'est demandé que pour la mise à l'heure)
  python3 graves_gps.py --position 48.8566 2.3522 85   # ciel masqué : heure et oscillateur dès 1 satellite
  (les coordonnées ci-dessus sont un exemple : Paris)

L'IQ est écrit en continu sur le disque (8 bits, 4,1 Mo/s : 1,2 Go pour 5 min) : la durée n'est limitée que
par la place libre. Au-delà de 90 s, le traitement se fait par tranches de 60 s, avec ré-acquisition à chaque
tranche (un satellite qui se lève ou qui décroche sous un toit est repris) ; les sous-trames de toutes les
tranches sont réunies, ce qui complète les éphémérides des satellites faibles. Sans --garder, le fichier est
effacé après un traitement réussi.

Résultat : graves_gps.json (position, heure, écart d'horloge du PC, ppm de l'oscillateur, satellites),
lisible par les autres modules avec graves_gps.charger_reference(). Les éphémérides décodées sont gardées
dans gps_ephemerides.json : relancé dans les 2 h, le calage ne demande plus que ~20 s de signal.

Branchement (antenne active ACGPSA : 3 V ± 0,3 V, 8 mA) :
  antenne ──> bias-T « RF+DC » ;  bias-T « RF » ──> RSPdx entrée A ;  bias-T « DC » <── 3,0 à 3,3 V.
  Le Bias-T interne du RSPdx (4,7 V, entrée B seulement) ne convient pas : tension hors tolérance de l'antenne.

Dépendances : SoapySDR + SoapySDRPlay3, numpy, scipy ; numba fortement conseillé (pip install numba) :
poursuite ~15× plus rapide et répartie sur les cœurs ; la 1re exécution compile (quelques secondes, mis en cache).
Utilise ouvrir_rsp / choisir_antenne / choisir_debit de graves_capture.py (même dossier).
Fermer SDRconnect / graves_v2.py avant de lancer : un seul programme peut ouvrir le RSPdx.

Essai sans matériel : gps-sdr-sim (github.com/osqzss/gps-sdr-sim) produit un signal GPS réaliste, ex.
  gps-sdr-sim -e brdc0010.22n -l 48.8566,2.3522,85 -d 50 -s 2048000 -b 8 -o sim.cs8
  python3 graves_gps.py --fichier sim.cs8 --fs 2048000
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

# ------------------------------------------------------------------------------------------- constantes
C = 299_792_458.0                  # m/s
F_L1 = 1575.42e6                   # Hz
F_CA = 1.023e6                     # chips/s
LONG_CA = 1023
LAMBDA_L1 = C / F_L1
MU = 3.986005e14                   # constante gravitationnelle WGS84 (IS-GPS-200)
OMEGA_E = 7.2921151467e-5          # rotation terrestre (rad/s)
PI_GPS = 3.1415926535898           # valeur de π imposée par IS-GPS-200
F_REL = -4.442807633e-10           # terme relativiste (s/√m)
GPS_UNIX = 315_964_800             # 1980-01-06 00:00:00 UTC en temps Unix
SEMAINE = 604_800
F_GRAVES = 143.050e6
PLANCHER_RSPDX_DBFS = -34.0        # bruit propre du RSPdx à 1575 MHz, LNA 0, IFGR 20, sans antenne active (mesuré)
WGS_A, WGS_F = 6_378_137.0, 1 / 298.257223563
PREAMBULE = np.array([1, 0, 0, 0, 1, 0, 1, 1], np.int8)
SECONDES_INTERCALAIRES_DEFAUT = 18  # GPS − UTC depuis 2017 ; remplacé par la valeur diffusée si décodée
CORRECTION_TROPO = True

try:                                # numba : facultatif, accélère la poursuite (même code, compilé)
    from numba import njit
except Exception:
    njit = None


# ------------------------------------------------------------------------------------------- code C/A
_G2 = {1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10), 7: (1, 8), 8: (2, 9),
       9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6), 13: (6, 7), 14: (7, 8), 15: (8, 9), 16: (9, 10),
       17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6),
       25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7), 31: (3, 8), 32: (4, 9)}


def code_ca(prn):
    """Code C/A (Gold, 1023 chips) du satellite prn, en ±1 (bit 0 → +1)."""
    a, b = _G2[prn]
    g1, g2 = [1] * 10, [1] * 10
    out = np.empty(LONG_CA, np.float64)
    for i in range(LONG_CA):
        out[i] = 1.0 - 2.0 * (g1[9] ^ g2[a - 1] ^ g2[b - 1])
        f1 = g1[2] ^ g1[9]
        f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [f1] + g1[:9]
        g2 = [f2] + g2[:9]
    return out


CODES = {p: code_ca(p) for p in range(1, 33)}


# ------------------------------------------------------------------------------------------- données IQ
class Donnees:
    """Échantillons IQ entrelacés (int8, int16 ou float32), en mémoire ou projetés depuis un fichier."""

    def __init__(self, brut, fs, dc=None):
        self.brut = brut                       # tableau 1-D I0 Q0 I1 Q1 …
        self.fs = float(fs)
        self.n = len(brut) // 2
        if dc is None:
            m = min(self.n, int(fs))           # composante continue estimée sur 1 s
            seg = np.asarray(brut[: 2 * m], np.float64)
            dc = (seg[0::2].mean(), seg[1::2].mean())
        self.dc = dc

    def complexe(self, debut, nb):
        seg = np.asarray(self.brut[2 * debut: 2 * (debut + nb)], np.float32)
        return (seg[0::2] - self.dc[0]) + 1j * (seg[1::2] - self.dc[1])

    def niveau_dbfs(self, pleine_echelle):
        x = self.complexe(0, min(self.n, int(self.fs)))
        return 10 * np.log10(np.mean(np.abs(x) ** 2) / pleine_echelle ** 2 + 1e-30)


def ouvrir_fichier(chemin, fmt=None, fs=None):
    """Fichier IQ brut entrelacé : cs8 (gps-sdr-sim -b 8), cs16 (ce module, --garder) ou cf32.
    Le .json associé (même nom) donne débit et horodatages PC s'il existe."""
    meta = {}
    base = os.path.splitext(chemin)[0]
    if os.path.exists(base + ".json"):
        with open(base + ".json") as f:
            meta = json.load(f)
    fmt = (fmt or meta.get("format") or os.path.splitext(chemin)[1].lstrip(".") or "cs16").lower()
    types = {"cs8": np.int8, "cs16": np.int16, "cf32": np.float32}
    if fmt not in types:
        sys.exit(f"Format {fmt} inconnu (cs8, cs16 ou cf32)")
    fs = fs or meta.get("fs")
    if not fs:
        sys.exit("Débit d'échantillonnage inconnu : préciser --fs (ex. --fs 2048000)")
    brut = np.memmap(chemin, dtype=types[fmt], mode="r")
    return Donnees(brut, fs), meta


# ------------------------------------------------------------------------------------------- acquisition
def acquerir(d, prns, doppler_max=7000.0, pas=500.0, n_nc=80, pfa=1e-6, debut=0):
    """Recherche parallèle sur le code (FFT) : 1 ms cohérente, n_nc ms sommées en puissance (80 ms : ~5 dB de
    mieux qu'avec 20 ms, utile sous un toit). Renvoie {prn: (indice d'échantillon d'un début de code,
    Doppler grossier, rapport pic/fond)} et le seuil de détection."""
    from scipy.stats import gamma
    fs = d.fs
    N = int(round(fs * 1e-3))
    n_nc = int(min(n_nc, (d.n - debut) // N))
    if n_nc < 5:                                   # pas assez de données dans la tranche
        return {}, float("inf")
    x = d.complexe(debut, N * n_nc).reshape(n_nc, N).astype(np.complex64)
    t = np.arange(N) / fs
    freqs = np.arange(-doppler_max, doppler_max + 1, pas)
    X = np.empty((len(freqs), n_nc, N), np.complex64)
    for i, f in enumerate(freqs):
        X[i] = np.fft.fft(x * np.exp(-2j * np.pi * f * t).astype(np.complex64), axis=1)
    seuil = gamma.isf(pfa / (len(freqs) * N), n_nc) / n_nc    # fond normalisé : loi Gamma(n_nc)/n_nc
    idx = np.floor(np.arange(N) * F_CA / fs).astype(int) % LONG_CA
    trouves = {}
    for prn in prns:
        Cf = np.conj(np.fft.fft(CODES[prn][idx])).astype(np.complex64)
        P = np.empty((len(freqs), N))
        for i in range(len(freqs)):
            P[i] = (np.abs(np.fft.ifft(X[i] * Cf[None, :], axis=1)) ** 2).sum(axis=0)
        P /= P.mean()
        # seuil adaptatif : avec des satellites forts, leurs corrélations croisées (qui ne s'atténuent pas en
        # intégrant plus longtemps) épaississent la queue du fond au-delà du modèle de bruit pur ; on exige
        # que le pic dépasse nettement le 99,99e centile du fond mesuré (sans effet sur du bruit seul)
        med, q = np.percentile(P, [50, 99.99])
        i_f, i_c = np.unravel_index(np.argmax(P), P.shape)
        if P[i_f, i_c] > max(seuil, q + 0.6 * (q - med)):
            trouves[prn] = (debut + int(i_c), float(freqs[i_f]), float(P[i_f, i_c]))
    return trouves, seuil


def affiner_doppler(d, prn, depuis, f0, duree_ms=200):
    """Doppler fin : sous-cases ±250 Hz puis FFT du carré des corrélations à 1 ms (le carré efface les bits)."""
    fs = d.fs
    N = int(round(fs * 1e-3))
    M = duree_ms
    if depuis + N * M > d.n:
        M = (d.n - depuis) // N
    x = d.complexe(depuis, N * M)
    n = np.arange(N * M)

    def correlations(f, m):
        fc = F_CA * (1 + f / F_L1)
        rep = CODES[prn][np.floor(n[: N * m] * fc / fs).astype(int) % LONG_CA]
        return (x[: N * m] * rep * np.exp(-2j * np.pi * f * n[: N * m] / fs)).reshape(m, N).sum(axis=1)

    candidats = [f0 + df for df in (-250, -125, 0, 125, 250)]
    f1 = max(candidats, key=lambda f: np.sum(np.abs(correlations(f, min(M, 20))) ** 2))
    p = correlations(f1, M)
    Z = np.abs(np.fft.fft(p ** 2, 16384))
    ff = np.fft.fftfreq(16384, 1e-3)
    m = np.abs(ff) <= 300
    k = np.flatnonzero(m)[np.argmax(Z[m])]
    if 0 < k < len(Z) - 1:                                   # interpolation parabolique du pic
        a, b, c = Z[k - 1], Z[k], Z[k + 1]
        dk = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    else:
        dk = 0.0
    return f1 + 0.5 * (ff[k] + dk * (ff[1] - ff[0]))


# ------------------------------------------------------------------------------------------- poursuite
def _correler_numpy(brut, s, blk, dc_i, dc_q, code, rem_code, pas, rem_carr, dphi):
    """Corrélateurs E/P/L sur un bloc (version numpy vectorisée, utilisée sans numba)."""
    t = np.arange(blk).astype(np.float64)
    ph = rem_code + t * pas
    i_p = np.floor(ph).astype(np.int64) % 1023
    i_e = np.floor(ph + 0.5).astype(np.int64) % 1023
    i_l = (np.floor(ph - 0.5).astype(np.int64) + 1023) % 1023
    xi = brut[2 * s: 2 * (s + blk): 2].astype(np.float64) - dc_i
    xq = brut[2 * s + 1: 2 * (s + blk): 2].astype(np.float64) - dc_q
    arg = rem_carr + dphi * t
    co = np.cos(arg)
    si = np.sin(arg)
    bi = xi * co + xq * si
    bq = xq * co - xi * si
    cp = code[i_p]
    ce = code[i_e]
    cl = code[i_l]
    return (np.sum(cp * bi), np.sum(cp * bq), np.sum(ce * bi), np.sum(ce * bq), np.sum(cl * bi), np.sum(cl * bq))


def _correler_boucle(brut, s, blk, dc_i, dc_q, code, rem_code, pas, rem_carr, dphi):
    """Même calcul en boucle scalaire (compilée par numba) : porteuse par rotateur, aucun tableau temporaire."""
    co = math.cos(rem_carr)
    si = math.sin(rem_carr)
    cd = math.cos(dphi)
    sd = math.sin(dphi)
    ip = qp = ie = qe = il = ql = 0.0
    ph = rem_code
    for t in range(blk):
        xi = brut[2 * (s + t)] - dc_i
        xq = brut[2 * (s + t) + 1] - dc_q
        bi = xi * co + xq * si
        bq = xq * co - xi * si
        k = int(ph)
        ke = int(ph + 0.5)
        if ke >= 1023:
            ke -= 1023
        kl = int(math.floor(ph - 0.5))
        if kl < 0:
            kl += 1023
        cp = code[k]
        ce = code[ke]
        cl = code[kl]
        ip += cp * bi
        qp += cp * bq
        ie += ce * bi
        qe += ce * bq
        il += cl * bi
        ql += cl * bq
        n = co * cd - si * sd
        si = si * cd + co * sd
        co = n
        ph += pas
    return ip, qp, ie, qe, il, ql


if njit:
    _correler = njit(cache=True, nogil=True)(_correler_boucle)
else:
    _correler = _correler_numpy


def _poursuite(brut, dc_i, dc_q, fs, code, s0, f_carr0, n_ms, bw_pll, bw_fll, bw_dll, ms_fll):
    """Boucles de poursuite d'un canal, une période de code (≈1 ms) par itération.
    Compilée par numba si disponible (sinon exécutée telle quelle, corrélateur numpy).
    Sorties par période : corrélateurs E/P/L, fréquence porteuse, cycles de porteuse cumulés au début de la
    période, échantillon exact (fractionnaire) du début de la période de code, échantillon de début du bloc."""
    ip = np.zeros(n_ms); qp = np.zeros(n_ms)
    ie = np.zeros(n_ms); qe = np.zeros(n_ms)
    il = np.zeros(n_ms); ql = np.zeros(n_ms)
    fca = np.zeros(n_ms); cyc = np.zeros(n_ms)
    epoque = np.zeros(n_ms); debut = np.zeros(n_ms)
    n_tot = brut.shape[0] // 2
    T = 1e-3
    w0 = bw_pll / 0.53                         # PLL 2e ordre (ζ = 0,707)
    s = s0
    rem_code = 0.0
    rem_carr = 0.0
    cycles = 0.0
    f_int = f_carr0
    f_carr = f_carr0
    f_code = F_CA * (1.0 + f_carr0 / F_L1)
    w0d = bw_dll / 0.53                        # DLL 2e ordre : l'intégrateur annule toute erreur statique
    code_int = 0.0                             # (ex. écart OL / horloge d'échantillonnage du synthétiseur)
    ip_prec = 0.0
    qp_prec = 0.0
    k_fin = n_ms
    for k in range(n_ms):
        pas = f_code / fs
        blk = int(math.ceil((1023.0 - rem_code) / pas))
        if s + blk + 2 > n_tot:
            k_fin = k
            break
        dphi = 2.0 * math.pi * f_carr / fs
        IP, QP, IE, QE, IL, QL = _correler(brut, s, blk, dc_i, dc_q, code, rem_code, pas, rem_carr, dphi)
        ip[k] = IP; qp[k] = QP; ie[k] = IE; qe[k] = QE; il[k] = IL; ql[k] = QL
        fca[k] = f_carr
        cyc[k] = cycles
        epoque[k] = s - rem_code / pas
        debut[k] = s
        # avance des NCO sur ce bloc
        cycles += f_carr * blk / fs
        rem_carr = (rem_carr + dphi * blk) % (2.0 * math.pi)
        rem_code = rem_code + blk * pas - 1023.0
        s += blk
        # boucle de porteuse : Costas (insensible aux bits) + FLL au démarrage
        if IP != 0.0:
            e = math.atan(QP / IP)
        else:
            e = 0.0
        if k > 0 and k < ms_fll:
            dot = ip_prec * IP + qp_prec * QP
            cross = ip_prec * QP - IP * qp_prec
            if dot != 0.0:
                f_int += 4.0 * bw_fll * T * math.atan(cross / dot) / (2.0 * math.pi * T)
        f_int += w0 * w0 * T * e / (2.0 * math.pi)
        f_carr = f_int + 1.414 * w0 * e / (2.0 * math.pi)
        ip_prec = IP
        qp_prec = QP
        # boucle de code : enveloppe E−L normalisée (écart en chips), assistée par la porteuse, 2e ordre
        ae = math.sqrt(IE * IE + QE * QE)
        al = math.sqrt(IL * IL + QL * QL)
        delta = 0.0
        if ae + al > 0.0:
            delta = 0.5 * (al - ae) / (ae + al)
        code_int += w0d * w0d * T * delta
        f_code = F_CA * (1.0 + f_carr / F_L1) - (code_int + 1.414 * w0d * delta)
    return ip, qp, ie, qe, il, ql, fca, cyc, epoque, debut, k_fin


_poursuite_rapide = njit(cache=True, nogil=True)(_poursuite) if njit else _poursuite


def poursuivre(d, prn, s0, f0, fin=None, bw_pll=18.0, bw_fll=8.0, bw_dll=1.0, ms_fll=600):
    """Poursuite du satellite prn de l'échantillon s0 jusqu'à fin (défaut : fin des données).
    Seules les sorties utiles sont gardées (mémoire bornée pour les longues captures)."""
    fin = d.n if fin is None else min(int(fin), d.n)
    n_ms = int((fin - s0) / d.fs * 1000) - 2
    brut = np.asarray(d.brut)
    r = _poursuite_rapide(brut, float(d.dc[0]), float(d.dc[1]), d.fs, CODES[prn], int(s0), float(f0),
                          max(n_ms, 0), bw_pll, bw_fll, bw_dll, ms_fll)
    k = r[-1]
    can = dict(ip=r[0][:k], qp=r[1][:k], cycles=r[7][:k], epoque=r[8][:k], debut=r[9][:k])
    can["prn"] = prn
    return can


def cn0_par_seconde(ip, qp, T=1e-3):
    """C/N0 (dB-Hz) par bloc de 1 s, estimateur des moments : Ps = (moyenne|I|)², Pn = puissance − Ps."""
    n = len(ip) // 1000
    out = np.full(n, np.nan)
    for j in range(n):
        i = ip[j * 1000:(j + 1) * 1000]
        q = qp[j * 1000:(j + 1) * 1000]
        ps = np.mean(np.abs(i)) ** 2
        pt = np.mean(i * i + q * q)
        if pt > ps > 0:
            out[j] = 10 * np.log10(ps / (pt - ps) / T)
    return out


# ------------------------------------------------------------------------------------------- message de navigation
def synchro_bits(ip, debut_ms=800):
    """Bord de bit par la méthode énergétique : parmi les 20 calages possibles, celui qui maximise la somme des
    |intégrales sur 20 ms| (un bit intégré à cheval sur une transition perd de l'amplitude). Robuste jusque vers
    25 dB-Hz, là où le comptage des changements de signe ms par ms échoue dès ~35 dB-Hz."""
    seg = ip[debut_ms:]
    nb = len(seg) // 20 - 1
    if nb < 50:
        return None
    scores = np.array([np.sum(np.abs(seg[b:b + 20 * nb].reshape(nb, 20).sum(axis=1))) for b in range(20)])
    b = int(np.argmax(scores))
    if scores[b] < 1.04 * np.median(scores):       # pas de calage net : bruit, ou porteuse non accrochée
        return None
    k0 = debut_ms + b
    nb = (len(ip) - k0) // 20
    sommes_i = ip[k0:k0 + 20 * nb].reshape(nb, 20).sum(axis=1)
    return k0, sommes_i


def _parite(w, d29, d30):
    """Contrôle de parité d'un mot de 30 bits (IS-GPS-200 § 20.3.5.2). Renvoie (ok, 24 bits de données)."""
    d = [int(w[i]) ^ d30 for i in range(24)]

    def x(*ix):
        v = 0
        for i in ix:
            v ^= d[i - 1]
        return v
    p = (d29 ^ x(1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23),
         d30 ^ x(2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24),
         d29 ^ x(1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22),
         d30 ^ x(2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23),
         d30 ^ x(1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24),
         d29 ^ x(3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24))
    return all(int(w[24 + i]) == p[i] for i in range(6)), d


def sous_trame(bits, d29, d30):
    """300 bits reçus → 300 bits « décodés » (données corrigées de D30*, parité conservée), ou None."""
    sf = np.zeros(300, np.int8)
    for w in range(10):
        seg = bits[30 * w: 30 * w + 30]
        ok, d = _parite(seg, d29, d30)
        if not ok:
            return None
        sf[30 * w: 30 * w + 24] = d
        sf[30 * w + 24: 30 * w + 30] = seg[24:30]
        d29, d30 = int(seg[28]), int(seg[29])
    return sf


def _u(sf, a, n):
    v = 0
    for b in sf[a - 1: a - 1 + n]:
        v = (v << 1) | int(b)
    return v


def _s(sf, a, n):
    v = _u(sf, a, n)
    return v - (1 << n) if v >> (n - 1) else v


def _u2(sf, a1, n1, a2, n2):
    return (_u(sf, a1, n1) << n2) | _u(sf, a2, n2)


def _s2(sf, a1, n1, a2, n2):
    v, n = _u2(sf, a1, n1, a2, n2), n1 + n2
    return v - (1 << n) if v >> (n - 1) else v


def decoder_sf(sf):
    """Champs utiles d'une sous-trame (IS-GPS-200, tableaux 20-I, 20-III, 20-IV, 20-V)."""
    out = {"id": _u(sf, 50, 3), "tow": _u(sf, 31, 17) * 6}   # TOW du début de la sous-trame SUIVANTE
    i = out["id"]
    if i == 1:
        out.update(wn=_u(sf, 61, 10), ura=_u(sf, 73, 4), sante=_u(sf, 77, 6),
                   iodc=(_u(sf, 83, 2) << 8) | _u(sf, 211, 8), tgd=_s(sf, 197, 8) * 2.0 ** -31,
                   toc=_u(sf, 219, 16) * 16.0, af2=_s(sf, 241, 8) * 2.0 ** -55,
                   af1=_s(sf, 249, 16) * 2.0 ** -43, af0=_s(sf, 271, 22) * 2.0 ** -31)
    elif i == 2:
        out.update(iode=_u(sf, 61, 8), crs=_s(sf, 69, 16) * 2.0 ** -5,
                   dn=_s(sf, 91, 16) * 2.0 ** -43 * PI_GPS, m0=_s2(sf, 107, 8, 121, 24) * 2.0 ** -31 * PI_GPS,
                   cuc=_s(sf, 151, 16) * 2.0 ** -29, e=_u2(sf, 167, 8, 181, 24) * 2.0 ** -33,
                   cus=_s(sf, 211, 16) * 2.0 ** -29, sqrta=_u2(sf, 227, 8, 241, 24) * 2.0 ** -19,
                   toe=_u(sf, 271, 16) * 16.0)
    elif i == 3:
        out.update(cic=_s(sf, 61, 16) * 2.0 ** -29, omega0=_s2(sf, 77, 8, 91, 24) * 2.0 ** -31 * PI_GPS,
                   cis=_s(sf, 121, 16) * 2.0 ** -29, i0=_s2(sf, 137, 8, 151, 24) * 2.0 ** -31 * PI_GPS,
                   crc=_s(sf, 181, 16) * 2.0 ** -5, omega=_s2(sf, 197, 8, 211, 24) * 2.0 ** -31 * PI_GPS,
                   omegadot=_s(sf, 241, 24) * 2.0 ** -43 * PI_GPS, iode=_u(sf, 271, 8),
                   idot=_s(sf, 279, 14) * 2.0 ** -43 * PI_GPS)
    elif i == 4 and _u(sf, 63, 6) == 56:                   # page 18 : ionosphère + UTC
        out.update(page18=True,
                   alpha=[_s(sf, 69, 8) * 2.0 ** -30, _s(sf, 77, 8) * 2.0 ** -27,
                          _s(sf, 91, 8) * 2.0 ** -24, _s(sf, 99, 8) * 2.0 ** -24],
                   beta=[_s(sf, 107, 8) * 2.0 ** 11, _s(sf, 121, 8) * 2.0 ** 14,
                         _s(sf, 129, 8) * 2.0 ** 16, _s(sf, 137, 8) * 2.0 ** 16],
                   a1=_s(sf, 151, 24) * 2.0 ** -50, a0=_s2(sf, 181, 24, 211, 8) * 2.0 ** -30,
                   tot=_u(sf, 219, 8) * 2.0 ** 12, wnt=_u(sf, 227, 8), dtls=_s(sf, 241, 8),
                   wnlsf=_u(sf, 249, 8), dn=_u(sf, 257, 8), dtlsf=_s(sf, 271, 8))
    return out


def lire_navigation(can):
    """Bits → sous-trames valides. Ajoute au canal : k_ref/tow_ref (période de code de début d'une sous-trame
    et son temps d'émission), la liste des sous-trames décodées et le verrouillage par bit."""
    sb = synchro_bits(can["ip"])
    if sb is None:
        return False
    k0, sommes = sb
    bits = (sommes > 0).astype(np.int8)
    can["k0_bits"] = k0
    nb = len(bits)
    sous, k_ref, tow_ref = [], None, None
    i = 0
    while i + 300 <= nb:
        trouve = False
        for pol in (0, 1):
            b = bits[i:i + 8] ^ pol
            if not np.array_equal(b, PREAMBULE):
                continue
            seg = bits[i:i + 300] ^ pol
            d29 = int(bits[i - 2] ^ pol) if i >= 2 else 0
            d30 = int(bits[i - 1] ^ pol) if i >= 1 else 0
            sf = sous_trame(seg, d29, d30)
            if sf is None and i < 2:                       # mot précédent inconnu : essai avec D29*=D30*=0
                sf = sous_trame(seg, 0, 0)
            if sf is None:
                continue
            champs = decoder_sf(sf)
            if not 1 <= champs["id"] <= 5:
                continue
            k_sf = k0 + 20 * i
            tow_debut = (champs["tow"] - 6) % SEMAINE
            if k_ref is not None:                          # cohérence avec les sous-trames déjà lues
                attendu = (tow_ref + (k_sf - k_ref) * 1e-3) % SEMAINE
                if abs(((tow_debut - attendu) + SEMAINE / 2) % SEMAINE - SEMAINE / 2) > 1e-6:
                    continue
            else:
                k_ref, tow_ref = k_sf, tow_debut
            champs["k"] = k_sf
            sous.append(champs)
            trouve = True
            break
        i += 300 if trouve else 1
    can["sous_trames"] = sous
    can["k_ref"], can["tow_ref"] = k_ref, tow_ref
    # verrouillage de phase par bit (I² − Q²)/(I² + Q²), lissé sur 10 bits
    sq = can["qp"][k0:k0 + 20 * nb].reshape(nb, 20).sum(axis=1)
    lk = (sommes ** 2 - sq ** 2) / (sommes ** 2 + sq ** 2 + 1e-30)
    can["verrou_bits"] = np.convolve(lk, np.ones(10) / 10, mode="same")
    return k_ref is not None


def assembler_ephemerides(can):
    """Sous-trames 1, 2, 3 de même IODE → éphéméride ; page 18 → ionosphère / UTC."""
    par_id = {}
    page18 = None
    for s in can.get("sous_trames", []):
        if s.get("page18"):
            page18 = s
        elif s["id"] in (1, 2, 3):
            par_id.setdefault(s["id"], []).append(s)
    eph = None
    for s1 in reversed(par_id.get(1, [])):
        for s2 in reversed(par_id.get(2, [])):
            for s3 in reversed(par_id.get(3, [])):
                if s2["iode"] == s3["iode"] == (s1["iodc"] & 0xFF):
                    eph = {k: v for s in (s1, s2, s3) for k, v in s.items() if k not in ("id", "tow", "k")}
                    eph["prn"] = can["prn"]
                    break
            if eph:
                break
        if eph:
            break
    return eph, page18


# ------------------------------------------------------------------------------------------- orbites et corrections
def _dt_semaine(t):
    return (t + SEMAINE / 2) % SEMAINE - SEMAINE / 2


def etat_satellite(eph, t):
    """Position ECEF (m) du satellite et correction d'horloge (s, relativité et T_GD L1 compris) à l'instant
    d'émission t (secondes de la semaine GPS) — IS-GPS-200 tableau 20-IV."""
    A = eph["sqrta"] ** 2
    n = math.sqrt(MU / A ** 3) + eph["dn"]
    tk = _dt_semaine(t - eph["toe"])
    M = eph["m0"] + n * tk
    e = eph["e"]
    E = M
    for _ in range(12):
        E = E - (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    v = math.atan2(math.sqrt(1 - e * e) * math.sin(E), math.cos(E) - e)
    phi = v + eph["omega"]
    s2, c2 = math.sin(2 * phi), math.cos(2 * phi)
    u = phi + eph["cus"] * s2 + eph["cuc"] * c2
    r = A * (1 - e * math.cos(E)) + eph["crs"] * s2 + eph["crc"] * c2
    i = eph["i0"] + eph["idot"] * tk + eph["cis"] * s2 + eph["cic"] * c2
    om = eph["omega0"] + (eph["omegadot"] - OMEGA_E) * tk - OMEGA_E * eph["toe"]
    xp, yp = r * math.cos(u), r * math.sin(u)
    pos = np.array([xp * math.cos(om) - yp * math.cos(i) * math.sin(om),
                    xp * math.sin(om) + yp * math.cos(i) * math.cos(om),
                    yp * math.sin(i)])
    dt = _dt_semaine(t - eph["toc"])
    dts = eph["af0"] + eph["af1"] * dt + eph["af2"] * dt * dt + F_REL * e * eph["sqrta"] * math.sin(E) - eph["tgd"]
    return pos, dts


def ecef_vers_geo(p):
    x, y, z = p
    e2 = WGS_F * (2 - WGS_F)
    lon = math.atan2(y, x)
    r = math.hypot(x, y)
    lat = math.atan2(z, r * (1 - e2))
    h = 0.0
    for _ in range(10):
        N = WGS_A / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        h = r / math.cos(lat) - N
        lat = math.atan2(z, r * (1 - e2 * N / (N + h)))
    return math.degrees(lat), math.degrees(lon), h


def geo_vers_ecef(lat, lon, h):
    e2 = WGS_F * (2 - WGS_F)
    la, lo = math.radians(lat), math.radians(lon)
    N = WGS_A / math.sqrt(1 - e2 * math.sin(la) ** 2)
    return np.array([(N + h) * math.cos(la) * math.cos(lo), (N + h) * math.cos(la) * math.sin(lo),
                     (N * (1 - e2) + h) * math.sin(la)])


def az_el(rx, sat):
    lat, lon, _ = ecef_vers_geo(rx)
    la, lo = math.radians(lat), math.radians(lon)
    d = sat - rx
    e = -math.sin(lo) * d[0] + math.cos(lo) * d[1]
    n = -math.sin(la) * math.cos(lo) * d[0] - math.sin(la) * math.sin(lo) * d[1] + math.cos(la) * d[2]
    u = math.cos(la) * math.cos(lo) * d[0] + math.cos(la) * math.sin(lo) * d[1] + math.sin(la) * d[2]
    return math.atan2(e, n), math.atan2(u, math.hypot(e, n))


def klobuchar(alpha, beta, lat, lon, az, el, t):
    """Retard ionosphérique L1 (m), modèle diffusé (IS-GPS-200 § 20.3.3.5.2.5). lat, lon, az, el en radians."""
    E = el / math.pi
    psi = 0.0137 / (E + 0.11) - 0.022
    phi = lat / math.pi + psi * math.cos(az)
    phi = max(-0.416, min(0.416, phi))
    lam = lon / math.pi + psi * math.sin(az) / math.cos(phi * math.pi)
    phim = phi + 0.064 * math.cos((lam - 1.617) * math.pi)
    tl = (4.32e4 * lam + t) % 86400.0
    F = 1.0 + 16.0 * (0.53 - E) ** 3
    amp = max(0.0, sum(a * phim ** k for k, a in enumerate(alpha)))
    per = max(72000.0, sum(b * phim ** k for k, b in enumerate(beta)))
    x = 2 * math.pi * (tl - 50400.0) / per
    d = F * (5e-9 + amp * (1 - x * x / 2 + x ** 4 / 24)) if abs(x) < 1.57 else F * 5e-9
    return d * C


def tropo(h, el):
    """Retard troposphérique (m) : zénith 2,3 m réduit avec l'altitude, projection de Black-Eisner."""
    if not CORRECTION_TROPO:
        return 0.0
    return 2.3 * math.exp(-1.16e-4 * max(h, 0.0)) * 1.001 / math.sqrt(0.002001 + math.sin(el) ** 2)


def _tourner(p, theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1], p[2]])


def resoudre_point(mes, t_rx, ephs, iono=None, x0=None, masque_deg=10.0, pos_fixe=None):
    """Moindres carrés : mes = [(prn, t_emission)], t_rx = heure de réception (approchée, même référence).
    Inconnues : position ECEF et biais d'horloge b (m) — ou b seul si pos_fixe est donnée.
    Renvoie dict(pos, b, residus, gdop, prns, elev) ou None."""
    x = np.array(pos_fixe if pos_fixe is not None else (x0 if x0 is not None else np.zeros(3)), float)
    b = 0.0
    exclus = set()
    for passe in range(4):
        for it in range(15):
            H, y, prns, el_l, az_l = [], [], [], [], []
            pret = np.linalg.norm(x) > 6.0e6
            if pret:
                lat, lon, h = ecef_vers_geo(x)
            for prn, t_tx in mes:
                if prn in exclus:
                    continue
                eph = ephs[prn]
                rho = C * (t_rx - t_tx)
                _, dts = etat_satellite(eph, t_tx)
                ps, dts = etat_satellite(eph, t_tx - dts)
                tau = np.linalg.norm(ps - x) / C if pret else 0.075
                ps = _tourner(ps, OMEGA_E * tau)
                r = np.linalg.norm(ps - x)
                u = (ps - x) / r
                corr = 0.0
                if pret:
                    az, el = az_el(x, ps)
                    masque_actif = it >= 3 or passe > 0 or x0 is not None or pos_fixe is not None
                    if masque_actif and math.degrees(el) < masque_deg:
                        continue
                    if iono:
                        corr += klobuchar(iono["alpha"], iono["beta"], math.radians(lat), math.radians(lon),
                                          az, el, t_rx)
                    corr += tropo(h, el)
                    el_l.append(math.degrees(el))
                    az_l.append(math.degrees(az))
                y.append(rho + C * dts - corr - (r + b))
                H.append([-u[0], -u[1], -u[2], 1.0])
                prns.append(prn)
            if len(y) < (1 if pos_fixe is not None else 4):
                return None
            H, y = np.array(H), np.array(y)
            if pos_fixe is not None:
                H = H[:, 3:]
            dx = np.linalg.lstsq(H, y, rcond=None)[0]
            if pos_fixe is None:
                x = x + dx[:3]
                b += dx[3]
            else:
                b += dx[0]
            if np.linalg.norm(dx) < 1e-4:
                break
        res = y - H @ dx
        # contrôle d'intégrité simple : avec ≥ 6 satellites, on écarte un résidu aberrant et on recommence
        if len(y) >= H.shape[1] + 2 and np.max(np.abs(res)) > 30.0:
            exclus.add(prns[int(np.argmax(np.abs(res)))])
            continue
        break
    try:
        Q = np.linalg.inv(H.T @ H)
        gdop = float(np.sqrt(np.trace(Q)))
    except np.linalg.LinAlgError:
        gdop = float("nan")
    return dict(pos=x, b=b, residus=res, gdop=gdop, prns=prns, elev=el_l, azim=az_l, exclus=sorted(exclus))


def resoudre_vitesse(dops, t_rx, b, ephs, x, statique=True):
    """Doppler (Hz, en temps récepteur) → vitesse du récepteur et écart relatif de fréquence ε de l'oscillateur.
    Modèle : −λ·f = (v_sat − v_rx)·u + c·ε − c·ḋt_sat  (u : vecteur unitaire récepteur → satellite).
    ε > 0 : l'oscillateur du RSPdx est trop rapide. Statique : v_rx = 0, seule ε est estimée."""
    H, y = [], []
    t = t_rx - b / C
    for prn, f in dops:
        eph = ephs[prn]
        pa, ca = etat_satellite(eph, t - 0.075 - 0.5)
        pb, cb = etat_satellite(eph, t - 0.075 + 0.5)
        p0 = 0.5 * (pa + pb)
        tau = np.linalg.norm(p0 - x) / C
        th = OMEGA_E * tau
        pa, pb = _tourner(pa, th), _tourner(pb, th)
        vs = pb - pa
        ps = 0.5 * (pa + pb)
        u = (ps - x) / np.linalg.norm(ps - x)
        _, el = az_el(x, ps)
        if math.degrees(el) < 5:
            continue
        y.append(-LAMBDA_L1 * f - vs @ u + C * (cb - ca))
        H.append([-u[0], -u[1], -u[2], C])
    if len(y) < (1 if statique else 4):
        return None
    H, y = np.array(H), np.array(y)
    if statique:
        eps = float(np.median(y) / C)
        return dict(v=np.zeros(3), eps=eps, residus=y - C * eps)
    s = np.linalg.lstsq(H, y, rcond=None)[0]
    return dict(v=s[:3], eps=float(s[3]), residus=y - H @ s)


# ------------------------------------------------------------------------------------------- chaîne complète
def _traiter_tranche(d, a, b, prns, doppler_max, cn0_min, journal, titre, detail):
    """Acquisition, poursuite et lecture du message sur les échantillons [a, b).
    Renvoie (canaux dont l'heure d'émission est connue, nombre de satellites détectés)."""
    from concurrent.futures import ThreadPoolExecutor
    fs = d.fs
    acq, seuil = acquerir(d, prns, doppler_max=doppler_max, debut=a + int(0.05 * fs))
    if not acq:
        journal(f"{titre}aucun satellite au-dessus du seuil ({seuil:.2f})")
        return [], 0
    if detail:
        journal(f"Acquisition : {len(acq)} satellite(s) au-dessus du seuil ({seuil:.2f}) — "
                + ", ".join(f"G{p:02d} {f:+.0f} Hz ({r:.1f})" for p, (_, f, r) in sorted(acq.items())))
    ordre = sorted(acq.items(), key=lambda kv: -kv[1][2])
    fins = {prn: affiner_doppler(d, prn, s0, f0) for prn, (s0, f0, _) in ordre}

    def un(kv):
        return poursuivre(d, kv[0], kv[1][0], fins[kv[0]], fin=b)
    tp = time.time()
    if njit and len(ordre) > 1:                    # numba libère le GIL : un canal par cœur
        with ThreadPoolExecutor(max_workers=max(1, min(len(ordre), os.cpu_count() or 1))) as ex:
            tous = list(ex.map(un, ordre))
    else:
        tous = [un(kv) for kv in ordre]
    if detail:
        journal(f"Poursuite de {len(tous)} satellite(s) : {time.time() - tp:.1f} s")
    canaux, resume = [], []
    for can in tous:
        prn = can["prn"]
        cn = cn0_par_seconde(can["ip"], can["qp"])
        can["cn0_moy"] = float(np.nanmedian(cn[1:])) if len(cn) > 1 else float("nan")
        ok = lire_navigation(can) if can["cn0_moy"] >= cn0_min - 3 else False
        n_sf = len(can.get("sous_trames", []))
        if detail:
            journal(f"  G{prn:02d} : Doppler {fins[prn]:+8.1f} Hz, C/N0 {can['cn0_moy']:4.1f} dB-Hz, "
                    f"{n_sf} sous-trame(s) valide(s)"
                    + ("  (non confirmé : C/N0 au niveau du bruit)" if can["cn0_moy"] < 28 else ""))
        resume.append(f"G{prn:02d} {can['cn0_moy']:.0f} dB-Hz/{n_sf} sf")
        del can["ip"], can["qp"]                   # inutiles une fois le message lu
        can["tranche"] = (a, b)
        if ok:
            canaux.append(can)
    if not detail:
        journal(titre + ", ".join(resume))
    return canaux, len(acq)


def traiter(d, ephs_cache=None, prns=range(1, 33), doppler_max=7000.0, masque=10.0, cn0_min=30.0,
            pas_s=0.5, statique=True, pos_connue=None, tranche_s=60.0, coupures=None, journal=print):
    """Échantillons IQ → solution complète. Renvoie un dict de résultats (voir sortie JSON).
    Au-delà de 1,5 × tranche_s, les données sont traitées par tranches (ré-acquisition à chaque tranche :
    un satellite qui se lève ou décroche est repris) ; sous-trames et mesures de toutes les tranches sont
    ensuite réunies (éphémérides complétées d'une tranche à l'autre, une seule solution d'heure)."""
    t0 = time.time()
    fs = d.fs
    duree = d.n / fs
    n_tr = 1 if duree <= 1.5 * tranche_s else int(round(duree / tranche_s))
    # bornes des tranches : régulières, plus les coupures (échantillons perdus) qui doivent en être une
    retenues = []
    for c in sorted(int(c) for c in (coupures or []) if 0 < c < d.n):
        if not retenues or c - retenues[-1] > 10 * fs:   # coupures rapprochées : la première suffit
            retenues.append(c)
    coupures = retenues
    bornes = [0, d.n] + coupures
    for b in np.linspace(0, d.n, n_tr + 1).astype(np.int64)[1:-1]:
        if all(abs(int(b) - c) > 8 * fs for c in bornes):
            bornes.append(int(b))
    limites = sorted(set(bornes))
    n_tr = len(limites) - 1
    canaux, n_detectes = [], 0
    for i in range(n_tr):
        a, b = int(limites[i]), int(limites[i + 1])
        if b - a < 5 * fs:                         # bout de segment trop court entre deux coupures
            continue
        titre = "" if n_tr == 1 else f"Tranche {i + 1}/{n_tr} [{a / fs:.0f}–{b / fs:.0f} s] : "
        c, n = _traiter_tranche(d, a, b, prns, doppler_max, cn0_min, journal, titre, detail=(n_tr == 1))
        canaux += c
        n_detectes += n
    if not n_detectes:
        return dict(erreur="aucun satellite détecté (antenne alimentée ? ciel visible ? niveau du signal ?)")
    if not canaux:
        return dict(erreur="aucun satellite poursuivi jusqu'au décodage du message (signal trop faible ?)")
    par_prn = {}
    for can in canaux:
        par_prn.setdefault(can["prn"], []).append(can)

    # éphémérides : décodées maintenant, sinon cache récent
    ephs, iono, page18 = {}, None, None
    for prn, cs in par_prn.items():
        e, p18 = assembler_ephemerides({"prn": prn, "sous_trames": [st for c in cs for st in c["sous_trames"]]})
        if e:
            ephs[prn] = e
        if p18:
            page18 = p18
    cache_utilise = []
    tow_mil = np.median([can["tow_ref"] for can in canaux])
    # instant de référence (modulo 1024 semaines) : semaine diffusée si une sous-trame 1 a été lue,
    # sinon horloge du PC si elle est plausible ; une éphéméride du cache n'est gardée qu'à ±2 h de cet instant
    cycle = 1024 * SEMAINE
    w_pc = int((time.time() - GPS_UNIX + 18) // SEMAINE)
    wn_dec = next((e["wn"] for e in ephs.values()), None)
    if wn_dec is not None:
        t_ref = (wn_dec * SEMAINE + tow_mil) % cycle
    elif w_pc >= 2048:
        t_ref = (time.time() - GPS_UNIX + 18) % cycle
    else:
        t_ref = None
    for p in par_prn:
        if p not in ephs and ephs_cache and p in ephs_cache:
            e = ephs_cache[p]
            if t_ref is None:
                ok = abs(_dt_semaine(tow_mil - e["toe"])) < 7200
            else:
                ecart = (e.get("wn", -10) * SEMAINE + e["toe"] - t_ref + cycle / 2) % cycle - cycle / 2
                ok = abs(ecart) < 7200
            if ok:
                ephs[p] = e
                cache_utilise.append(p)
    if cache_utilise and t_ref is None:
        journal("  attention : horloge du PC non plausible, semaine des éphémérides du cache non vérifiable")
    if page18:
        iono = dict(alpha=page18["alpha"], beta=page18["beta"])
    elif ephs_cache and ephs_cache.get("_iono"):
        iono = ephs_cache["_iono"]
    iono_defaut = iono is None                     # sans paramètres diffusés : terme constant de nuit (5 ns)
    if iono_defaut:
        iono = dict(alpha=[0.0] * 4, beta=[0.0] * 4)
    utiles = [c for c in canaux if c["prn"] in ephs and ephs[c["prn"]]["sante"] == 0]
    n_utiles = len({c["prn"] for c in utiles})
    if n_tr > 1:
        journal("Satellites lus au moins une fois : " + ", ".join(
            f"G{p:02d} ({len(cs)}/{n_tr} tranche(s))" for p, cs in sorted(par_prn.items())))
    journal(f"Éphémérides : {len(ephs)} satellite(s)" + (f" (dont {len(cache_utilise)} du cache)" if cache_utilise else "")
            + f", sains et utilisables : {n_utiles}"
            + (" ; ionosphère : modèle par défaut" if iono_defaut else " ; paramètres ionosphériques diffusés"))
    besoin = 1 if pos_connue is not None else 4
    if n_utiles < besoin:
        if len(par_prn) < besoin:
            cause = (f"seulement {len(par_prn)} satellite(s) reçu(s) : ciel trop masqué (toit, isolant) — "
                     "rapprocher l'antenne d'une fenêtre ou la sortir"
                     + ("" if pos_connue is not None else
                        ", ou donner la position (--position LAT LON H) : 1 satellite suffit alors "
                        "pour l'heure et l'oscillateur"))
        else:
            cause = ("éphémérides incomplètes : capture trop courte pour lire les sous-trames 1 à 3 "
                     "(relancer avec --duree 48 ou plus)")
        return dict(erreur=f"{n_utiles} satellite(s) utilisable(s), il en faut {besoin} — {cause}.",
                    canaux=canaux, ephs=ephs)

    # époques de mesure : grille commune en échantillons, là où tous les canaux retenus sont verrouillés
    def valide(can, n):
        e = can["epoque"]
        k = np.searchsorted(e, n) - 1
        if k < can["k0_bits"] + 20 or k + 300 >= len(e):
            return None
        vb = can["verrou_bits"]
        ib = (k - can["k0_bits"]) // 20
        if ib >= len(vb) or vb[ib] < 0.5:
            return None
        return k
    pas = pas_s * fs                                   # époques communes : multiples de pas_s
    epoques = {}
    for can in utiles:
        e = can["epoque"]
        if len(e) < can["k0_bits"] + 400:
            continue
        for j in range(math.ceil(e[can["k0_bits"] + 40] / pas), int(e[-300] / pas) + 1):
            n_m = j * pas
            k = valide(can, n_m)
            if k is None:
                continue
            frac = (n_m - e[k]) / (e[k + 1] - e[k])
            t_tx = (can["tow_ref"] + (k - can["k_ref"]) * 1e-3 + frac * 1e-3) % SEMAINE
            k1, k2 = max(k - 250, 0), min(k + 250, len(e) - 1)     # Doppler moyen sur ±0,25 s
            dt = (can["debut"][k2] - can["debut"][k1]) / fs
            m = epoques.setdefault(j, ([], []))
            m[0].append((can["prn"], t_tx))
            m[1].append((can["prn"], (can["cycles"][k2] - can["cycles"][k1]) / dt))
    def resoudre_epoque(j, pos_fixe, x0):
        """Solution d'une époque : complète (≥ 4 satellites), ou heure seule à pos_fixe (1 à 3 satellites)."""
        n_m = j * pas
        mes, dops = epoques[j]
        if len(mes) < besoin:
            return None
        t_rx = max(t for _, t in mes) + 0.075
        fixe = pos_fixe is not None and len(mes) < 4
        if fixe:
            r = resoudre_point(mes, t_rx, ephs, iono, pos_fixe=pos_fixe, masque_deg=masque)
        else:
            r = resoudre_point(mes, t_rx, ephs, iono, x0=x0, masque_deg=masque)
            if r is not None and len(r["prns"]) < 4 and pos_fixe is not None:
                fixe = True
                r = resoudre_point(mes, t_rx, ephs, iono, pos_fixe=pos_fixe, masque_deg=masque)
        if r is None or len(r["prns"]) < besoin:
            return None
        v = resoudre_vitesse([dp for dp in dops if dp[0] in r["prns"]], t_rx, r["b"], ephs, r["pos"], statique)
        vm = resoudre_vitesse([dp for dp in dops if dp[0] in r["prns"]], t_rx, r["b"], ephs, r["pos"], False) \
            if statique and len(r["prns"]) >= 4 else v
        return dict(n=n_m, j=j, fixe=fixe, tow=(t_rx - r["b"] / C) % SEMAINE, pos=r["pos"], gdop=r["gdop"],
                    prns=r["prns"], elev=r["elev"], azim=r["azim"], rms=float(np.sqrt(np.mean(r["residus"] ** 2))),
                    eps=v["eps"] if v else np.nan, vit=vm["v"] if vm else np.full(3, np.nan))

    def position_robuste(P):
        med = np.median(P, axis=0)
        ok = np.linalg.norm(P - med, axis=1) < max(30.0, 4 * np.median(np.linalg.norm(P - med, axis=1)))
        return P[ok].mean(axis=0), P[ok]

    pos_donnee = geo_vers_ecef(*pos_connue) if pos_connue is not None else None
    sol, x0 = [], pos_donnee
    for j in sorted(epoques):
        r = resoudre_epoque(j, pos_donnee, x0)
        if r is not None:
            sol.append(r)
            if not r["fixe"]:
                x0 = r["pos"]
    if not sol:
        return dict(erreur="aucune solution de position (géométrie ou verrouillage insuffisants)")
    completes = [s for s in sol if not s["fixe"]]
    ecart_position = None
    if completes and pos_donnee is not None:
        pos_mes, _ = position_robuste(np.array([s["pos"] for s in completes]))
        ecart_position = float(np.linalg.norm(pos_mes - pos_donnee))
        if ecart_position > 50.0:
            journal(f"  attention : la position donnée est à {ecart_position:.0f} m de la position mesurée "
                    f"({len(completes)} époques à ≥ 4 satellites) — c'est cette dernière qui sert pour l'heure")
        if any(s["fixe"] for s in sol):           # époques à < 4 satellites : recalculées à la position mesurée
            sol = [s if not s["fixe"] else resoudre_epoque(s["j"], pos_mes, None) for s in sol]
            sol = [s for s in sol if s is not None]

    # synthèse : position sur les solutions complètes (≥ 4 satellites) ; heure et ε sur toutes
    completes = [s for s in sol if not s["fixe"]]
    position_mesuree = bool(completes)
    if completes:
        pos, P = position_robuste(np.array([s["pos"] for s in completes]))
    else:
        journal("  moins de 4 satellites : heure et oscillateur calculés à la position donnée (non mesurée)")
        pos = pos_donnee
        P = pos[None, :]
    garde = np.array([s["fixe"] or np.linalg.norm(s["pos"] - pos) < 100.0 for s in sol])
    lat, lon, h = ecef_vers_geo(pos)
    la, lo = math.radians(lat), math.radians(lon)
    R = np.array([[-math.sin(lo), math.cos(lo), 0],
                  [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                  [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
    enu = (P - pos) @ R.T
    n_s = np.array([s["n"] for s in sol])[garde]
    tow = np.array([s["tow"] for s in sol])[garde]
    tow = tow[0] + _dt_semaine(tow - tow[0])           # continuité au passage de semaine
    eps = np.array([s["eps"] for s in sol])[garde]
    eps_ok = eps[np.isfinite(eps)]
    eps_med = float(np.median(eps_ok)) if len(eps_ok) else 0.0
    # Heure GPS de l'échantillon n : droite tow(n), une pente commune (elle donne aussi ε par l'horloge) mais
    # une origine par segment continu. Le RSPdx peut perdre des échantillons sans que le pilote le signale :
    # l'heure saute alors d'autant. Segments = coupures repérées par l'USB, puis sauts de l'heure GPS.
    seg = np.searchsorted(np.asarray(coupures, float), n_s, side="right")
    # pente locale mesurée sur l'heure GPS elle-même (médiane des pas entre époques voisines : insensible aux
    # quelques sauts), pour ne pas supposer que l'échantillonnage suit exactement l'oscillateur de la porteuse
    dn, dtw = np.diff(n_s), np.diff(tow)
    voisins = (seg[1:] == seg[:-1]) & (dn > 0) & (dn < 3 * pas)
    pente_rob = float(np.median(dtw[voisins] / dn[voisins])) if voisins.sum() >= 5 else 1.0 / (fs * (1.0 + eps_med))
    r = tow - (n_s - n_s[0]) * pente_rob
    sauts = []
    if len(r) >= 12:
        dif = np.zeros(len(r))
        for i in range(6, len(r) - 5):
            if seg[i - 6] == seg[i + 5]:
                dif[i] = np.median(r[i:i + 6]) - np.median(r[i - 6:i])
        i = 0
        while i < len(r):
            if abs(dif[i]) > 1e-6:
                # médianes pour détecter (robustes), moyennes pour placer : leur écart culmine pile sur la marche
                cand = [j for j in range(max(1, i - 6), min(len(r) - 1, i + 12)) if seg[j - 1] == seg[j]]
                j = max(cand, key=lambda j: abs(np.mean(r[j:j + 6]) - np.mean(r[max(0, j - 6):j]))) if cand else i
                sauts.append(j)
                i = j + 12
            else:
                i += 1
    for j in sauts:
        seg[j:] += 1
    etiquettes = list(np.unique(seg))
    if len(n_s) >= 3 and n_s[-1] > n_s[0]:
        A = np.zeros((len(n_s), 1 + len(etiquettes)))
        A[:, 0] = n_s - n_s[0]
        for c, e in enumerate(etiquettes):
            A[seg == e, 1 + c] = 1.0
        bon = np.ones(len(tow), bool)
        for _ in range(3):                         # rejet itératif des époques aberrantes
            coef = np.linalg.lstsq(A[bon], tow[bon], rcond=None)[0]
            res_t = tow - A @ coef
            mad = 1.4826 * np.median(np.abs(res_t[bon] - np.median(res_t[bon])))
            nouveau = np.abs(res_t) < max(5 * mad, 2e-7)
            if (nouveau == bon).all() or nouveau.sum() < 3:
                break
            bon = nouveau
        pente, origines = coef[0], dict(zip(etiquettes, coef[1:]))
        res_t = res_t[bon]
        eps_temps = 1.0 / (pente * fs) - 1.0
    else:
        pente, origines, res_t, eps_temps = 1.0 / (fs * (1.0 + eps_med)), {seg[0]: tow[0]}, np.zeros(1), np.nan
    # discontinuités mesurées par le GPS entre segments successifs
    discontinuites = []
    for a, b in zip(etiquettes[:-1], etiquettes[1:]):
        n_b = float(n_s[seg == b][0])
        discontinuites.append(dict(t_s=n_b / fs, perte_ms=(origines[b] - origines[a]) * 1e3))
    # segment de référence : celui qui contient le 1er échantillon s'il a des solutions
    e_ref = 0 if 0 in origines else etiquettes[0]
    t0_fiable = e_ref == 0
    n_ref = float(n_s[seg == e_ref][0])
    origine = origines[e_ref] + pente * (n_ref - n_s[0])
    limites_n = sorted(set([float(c) for c in coupures] + [float(n_s[j]) for j in sauts]))
    seg_bornes = (max([c for c in limites_n if c <= n_ref], default=0.0),
                  min([c for c in limites_n if c > n_ref], default=float(d.n)))
    t_sol = (n_s - n_s[0]) / fs
    derive = np.polyfit(t_sol[np.isfinite(eps)], eps_ok, 1)[0] * 60 if len(eps_ok) >= 3 else np.nan
    # semaine GPS (10 bits diffusés) : levée de l'ambiguïté de 1024 semaines avec l'horloge du PC
    wn10 = next((ephs[c["prn"]]["wn"] for c in utiles if "wn" in ephs[c["prn"]]), None)
    w_pc = int((time.time() - GPS_UNIX + 18) // SEMAINE)
    if wn10 is None:
        semaine = w_pc
    else:
        cand = [wn10 + 1024 * k for k in range(4)]
        semaine = min(cand, key=lambda w: abs(w - w_pc)) if w_pc >= 2048 else wn10 + 2048
    dtls = page18["dtls"] if page18 else (ephs_cache.get("_dtls") if ephs_cache else None)
    dtls_src = "diffusé" if page18 else ("cache" if dtls is not None else "valeur par défaut")
    if dtls is None:
        dtls = SECONDES_INTERCALAIRES_DEFAUT
    vit = np.array([s["vit"] for s in sol])[garde]
    vus = {}                                       # satellites utilisés : dernière élévation / azimut connus
    for so in sol:
        vus.update({p: (e, a) for p, e, a in zip(so["prns"], so["elev"], so["azim"])})
    journal(f"Solutions : {garde.sum()}/{len(sol)} époques retenues, traitement {time.time() - t0:.0f} s")
    return dict(
        n_ref=n_ref, tow_ref=float(origine), pente=float(pente), semaine=int(semaine),
        t0_fiable=t0_fiable, discontinuites=discontinuites, seg_bornes=seg_bornes,
        dtls=int(dtls), dtls_src=dtls_src, page18=page18,
        lat=lat, lon=lon, h=h, ecef=pos.tolist(), position_mesuree=position_mesuree,
        sigma_enu=enu.std(axis=0).tolist() if len(enu) > 1 else [np.nan] * 3,
        gdop=float(np.median([s["gdop"] for s in (completes or sol)])),
        rms=float(np.median([s["rms"] for s in (completes or sol)])),
        n_completes=len(completes), n_epoques=len(sol), ecart_position=ecart_position,
        prns_completes=sorted({p for s in completes for p in s["prns"]}),
        eps=float(np.median(eps_ok)) if len(eps_ok) else np.nan,
        eps_sigma=float(np.std(eps_ok)) if len(eps_ok) > 1 else np.nan,
        eps_temps=float(eps_temps), derive_ppb_min=float(derive * 1e9) if np.isfinite(derive) else np.nan,
        res_temps_ns=float(np.std(res_t) * 1e9), vitesse=np.nanmedian(vit, axis=0).tolist(),
        satellites=[dict(prn=p, el=round(e, 1), az=round(a % 360, 1),
                         cn0=round(float(np.nanmedian([c["cn0_moy"] for c in par_prn[p]])), 1))
                    for p, (e, a) in sorted(vus.items())],
        canaux=canaux, ephs=ephs, iono=None if iono_defaut else iono, cache_utilise=cache_utilise, n_solutions=int(garde.sum()))


def unix_de_l_echantillon(res, n):
    """Heure UTC (s Unix) de l'échantillon n, d'après la solution GPS."""
    tow = res["tow_ref"] + res["pente"] * (n - res["n_ref"])
    return GPS_UNIX + res["semaine"] * SEMAINE + tow - res["dtls"]


def ecart_horloge_pc(res, blocs, fs_vrai):
    """Écart horloge PC − UTC (s) à partir des instants d'arrivée des blocs USB : (n_fin_bloc, t_ns).
    L'arrivée d'un bloc suit toujours sa numérisation (latence USB/pilote) : on prend l'enveloppe inférieure
    t_arrivée − n/fs, qui approche la latence minimale. Le reliquat de latence (quelques ms) n'est pas mesurable."""
    b = np.asarray(blocs, np.float64)
    if "seg_bornes" in res:                        # blocs du segment continu de référence seulement
        b = b[(b[:, 0] > res["seg_bornes"][0]) & (b[:, 0] <= res["seg_bornes"][1])]
    # au plus ±10 s autour de l'instant de référence : l'horloge système peut être en cours de correction NTP
    proches = b[np.abs(b[:, 0] - res["n_ref"]) <= 10 * fs_vrai]
    if len(proches) >= 10:
        b = proches
    if len(b) < 10:
        return None
    n, t = b[:, 0], b[:, 1] * 1e-9
    L = t - n / fs_vrai
    l0 = np.min(L)
    gigue = np.percentile(L, 90) - l0
    n_m = res["n_ref"]
    t_pc = l0 + n_m / fs_vrai
    return dict(ecart=float(t_pc - unix_de_l_echantillon(res, n_m)), gigue=float(gigue))


def etat_antenne(exces_db):
    """Verdict sur l'alimentation de l'antenne active d'après l'excès de bruit au-dessus du plancher du RSPdx."""
    if exces_db is None:
        return "inconnu"
    if exces_db >= 8.0:
        return "alimentée"
    if exces_db >= 3.0:
        return "douteuse"
    return "non alimentée"


def detecter_pertes(blocs, fs, seuil=1.5e-3):
    """Échantillons perdus sans signalement du pilote, repérés par l'horodatage des blocs USB : la latence
    apparente (arrivée − n/fs) a une enveloppe inférieure stable ; une perte la fait monter D'UN COUP.
    On utilise l'horloge matérielle brute (CLOCK_MONOTONIC_RAW) quand elle a été enregistrée : l'horloge
    système, elle, peut être accélérée ou ralentie par NTP pendant la capture (rampes de ±100 ms observées),
    ce qu'il ne faut pas prendre pour des pertes. Par prudence, seuls les sauts brusques par rapport à la
    tendance locale sont retenus. Renvoie [(indice d'échantillon de la coupure, durée perdue en s)]."""
    b = np.asarray(blocs, np.float64)
    if len(b) < 200:
        return []
    n = b[:, 0]
    t = (b[:, 2] if b.shape[1] >= 3 else b[:, 1]) * 1e-9
    L = t - n / fs
    k = 16                                         # fenêtres de ~0,5 s : enveloppe inférieure de chacune
    nw = len(L) // k
    E = L[: nw * k].reshape(nw, k).min(axis=1)
    nE = n[: nw * k].reshape(nw, k)[:, 0]
    d = np.diff(E)
    pertes, i = [], 8
    while i < len(d) - 8:
        avant, apres = np.median(d[i - 8:i - 1]), np.median(d[i + 2:i + 9])   # pentes de part et d'autre
        tendance = 0.5 * (avant + apres)
        saut = d[i] - tendance
        persiste = (E[min(i + 4, len(E) - 1)] - E[i - 3]) - 7 * tendance
        # une perte est une marche : même pente avant et après ; le début d'une rampe NTP change la pente
        marche = abs(apres - avant) < 0.25 * seuil
        if saut > seuil and persiste > seuil and marche:
            pertes.append((int(nE[i + 1]), float(saut)))
            i += 4
        else:
            i += 1
    return pertes


def instabilite_horloge_pc(blocs, fs):
    """Amplitude (s) des variations de l'horloge système par rapport à l'horloge matérielle pendant la capture
    (corrections NTP), ou None si l'horloge brute n'a pas été enregistrée."""
    b = np.asarray(blocs, np.float64)
    if len(b) < 10 or b.shape[1] < 3:
        return None
    ecart = (b[:, 1] - b[:, 2]) * 1e-9
    ecart -= ecart[0]
    return float(np.ptp(ecart))


# ------------------------------------------------------------------------------------------- cache et sortie
def charger_cache(chemin):
    if not os.path.exists(chemin):
        return {}
    try:
        with open(chemin) as f:
            j = json.load(f)
        c = {int(k): v for k, v in j.get("ephemerides", {}).items()}
        if j.get("iono"):
            c["_iono"] = j["iono"]
        if j.get("dtls") is not None:
            c["_dtls"] = j["dtls"]
        c["_semaine"] = j.get("semaine")
        return c
    except Exception as e:
        print(f"Cache d'éphémérides illisible ({e}) : ignoré")
        return {}


def ecrire_cache(chemin, res, ancien):
    eph = {str(k): v for k, v in ancien.items() if isinstance(k, int)}
    for p, e in res["ephs"].items():
        if p not in res.get("cache_utilise", []):
            eph[str(p)] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in e.items()}
    with open(chemin, "w") as f:
        json.dump(dict(ecrit_utc=datetime.now(timezone.utc).isoformat(), semaine=res["semaine"],
                       dtls=res["dtls"] if res["dtls_src"] != "valeur par défaut" else None,
                       iono=res["iono"], ephemerides=eph), f, indent=1)


def _propre(o):
    """Valeurs JSON strictes : NaN → null, types numpy → types Python."""
    if isinstance(o, dict):
        return {k: _propre(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_propre(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def charger_reference(chemin="graves_gps.json"):
    """Pour les autres modules : dernière référence GPS (dict) ou None.
    Champs utiles : position.lat/lon/h_ellipsoide_m, oscillateur.epsilon (sans unité),
    oscillateur.correction_hz_graves (à AJOUTER aux fréquences mesurées autour de 143,050 MHz),
    horloge_pc.ecart_s (PC − UTC)."""
    try:
        with open(chemin) as f:
            return json.load(f)
    except Exception:
        return None


def regler_horloge(ecart):
    """Corrige l'horloge système de −ecart. Sans être root, seule cette opération passe par sudo : le mot de passe
    est demandé AVANT (sudo -v), puis la correction est appliquée de façon relative par un processus root
    (time.time() − ecart évalué au moment même), pour que l'attente du mot de passe ne la fausse pas.
    Le script reste ainsi dans le venv (numba) et les fichiers écrits restent à l'utilisateur."""
    import shutil
    import subprocess
    actifs = set()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/comm") as f:
                actifs.add(f.read().strip())
        except OSError:
            pass
    ntp = [p for p in ("chronyd", "ntpd", "systemd-timesyn") if p in actifs]
    if ntp:
        print(f"  attention : {', '.join(ntp)} actif — il ramènera l'horloge à sa propre référence ; à la maison,"
              " laisser NTP faire ; sur le terrain sans réseau : sudo timedatectl set-ntp false avant de régler")
    if os.geteuid() == 0:
        time.clock_settime(time.CLOCK_REALTIME, time.time() - ecart)
    else:
        if not shutil.which("sudo"):
            print(f"  impossible sans sudo : en root, date -s @{time.time() - ecart:.3f}")
            return False
        print("  mise à l'heure : sudo demande votre mot de passe")
        if subprocess.run(["sudo", "-v"]).returncode != 0:
            print("  mise à l'heure abandonnée")
            return False
        code = f"import time; time.clock_settime(time.CLOCK_REALTIME, time.time() - ({ecart!r}))"
        if subprocess.run(["sudo", "-n", sys.executable, "-c", code]).returncode != 0:
            print("  échec de la mise à l'heure")
            return False
    print(f"  horloge système corrigée de {-ecart * 1e3:+.1f} ms (latence USB comprise : quelques ms à ~25 ms)")
    return True


# ------------------------------------------------------------------------------------------- matériel
def capturer(args):
    """Enregistre args.duree secondes d'IQ (CS16) sur l'entrée A, en mémoire, avec l'heure PC d'arrivée de
    chaque bloc USB. Renvoie (Donnees, meta)."""
    import graves_capture as gc
    SoapySDR, dev, _ = gc.ouvrir_rsp()
    RX = SoapySDR.SOAPY_SDR_RX
    ant = gc.choisir_antenne(dev, RX, args.antenne)
    fs = gc.choisir_debit(dev, RX, args.debit)
    dev.setAntenna(RX, 0, ant)
    dev.setSampleRate(RX, 0, fs)
    try:                                          # bande IF la plus large ≤ débit (1,536 MHz à 2,048 MS/s)
        bws = [b for b in dev.listBandwidths(RX, 0) if b <= fs]
        if bws:
            dev.setBandwidth(RX, 0, max(bws))
        bw = dev.getBandwidth(RX, 0)
    except Exception:
        bw = None
    dev.setGainMode(RX, 0, False)
    gains = list(dev.listGains(RX, 0))
    rg = dev.getGainRange(RX, 0, "RFGR") if "RFGR" in gains else None
    lna = args.lna if rg is None else int(min(max(args.lna, rg.minimum()), rg.maximum()))
    ifgr = 40 if args.ifgr is None else args.ifgr
    if "RFGR" in gains:
        dev.setGain(RX, 0, "RFGR", lna)
    if "IFGR" in gains:
        dev.setGain(RX, 0, "IFGR", ifgr)
    for cle in ("biasT_ctrl", "rfnotch_ctrl", "dabnotch_ctrl"):   # bias-T interne : entrée B seulement → off
        try:
            dev.writeSetting(cle, "false")
        except Exception:
            pass
    dev.setFrequency(RX, 0, F_L1)
    f_lue = dev.getFrequency(RX, 0)
    if abs(f_lue - F_L1) > 1:
        print(f"ATTENTION : fréquence relue {f_lue/1e6:.6f} MHz")
    st = dev.setupStream(RX, SoapySDR.SOAPY_SDR_CS16)
    dev.activateStream(st)
    buf = np.empty(2 * 65536, np.int16)

    def lire_niveau(duree):
        """Puissance du bruit en dBFS, composante continue retirée (elle ne dépend pas de l'antenne)."""
        acc, n = 0.0, 0
        fin = time.time() + duree
        while time.time() < fin:
            sr = dev.readStream(st, [buf], 65536, timeoutUs=1_000_000)
            if sr.ret > 0:
                v = buf[: 2 * sr.ret].astype(np.float64)
                i, q = v[0::2] - v[0::2].mean(), v[1::2] - v[1::2].mean()
                acc += np.sum(i * i + q * q)
                n += sr.ret
        return 10 * np.log10(acc / max(n, 1) / 32768.0 ** 2 + 1e-30)

    print(f"RSPdx : {ant}, {F_L1/1e6:.3f} MHz, fs = {fs} S/s" + (f", bande IF {bw/1e6:.3f} MHz" if bw else "")
          + f", LNA {lna}, IFGR {ifgr} dB, AGC off")
    if args.test_antenne:
        args.prechauffage = min(args.prechauffage, 5.0)
    if args.prechauffage > 0:
        print(f"Mise en température du récepteur : {args.prechauffage:.0f} s…")
        fin = time.time() + args.prechauffage
        while time.time() < fin:
            dev.readStream(st, [buf], 65536, timeoutUs=1_000_000)
    if args.ifgr is None and "IFGR" in gains:     # réglage du niveau : bruit vers −30 dBFS (bruit ≈ signal GPS)
        for _ in range(4):
            niv = lire_niveau(0.3)
            if abs(niv - args.niveau) < 3:
                break
            ifgr = int(min(59, max(20, ifgr + round(niv - args.niveau))))
            dev.setGain(RX, 0, "IFGR", ifgr)
    niv = lire_niveau(0.5)
    print(f"Niveau : {niv:.1f} dBFS avec IFGR {ifgr} dB")
    # Alimentation de l'antenne : son préampli ajoute son propre bruit (~+13 dB) au plancher du RSPdx. On ramène
    # le niveau mesuré au gain maximal (IFGR 20 : 1 dB d'IFGR = 1 dB de niveau) et on le compare au plancher.
    exces = niv + (ifgr - 20) - args.plancher if lna == 0 else None
    etat = etat_antenne(exces)
    if exces is None:
        print("Antenne : vérification impossible avec --lna différent de 0 (plancher mesuré au LNA 0)")
    else:
        print(f"Antenne : {etat.upper()} — bruit {exces:+.1f} dB au-dessus du plancher propre du RSPdx"
              + {"alimentée": "",
                 "douteuse": " (piles faibles ? faux contact ?)",
                 "non alimentée": " (piles, bias-T 3,0–3,3 V, sens RF/RF+DC, connecteur SMA)"}[etat])
    if args.test_antenne or (etat == "non alimentée" and not args.forcer):
        dev.deactivateStream(st)
        dev.closeStream(st)
        if args.test_antenne:
            sys.exit(0 if etat == "alimentée" else 1)
        sys.exit("Capture annulée : antenne non alimentée (--forcer pour enregistrer quand même)")

    # enregistrement en flux vers le disque, sur 8 bits : pour un signal noyé dans le bruit comme le GPS,
    # 8 bits (écart-type ≈ 20 pas) ne coûtent rien en sensibilité et divisent le volume par deux (4 Mo/s)
    import queue
    import shutil
    import threading
    ecart_type = 32768.0 * 10 ** (niv / 20) / math.sqrt(2)          # par voie I ou Q
    echelle = 20.0 / max(ecart_type, 1.0)
    chemin = datetime.now(timezone.utc).strftime("gps_%Y%m%d_%H%M%S") + ".cs8"
    n_vise = int(args.duree * fs)
    libre = shutil.disk_usage(os.path.dirname(os.path.abspath(chemin))).free
    if libre < 2 * n_vise * 1.1 + 200e6:
        dev.deactivateStream(st)
        dev.closeStream(st)
        sys.exit(f"Place disque insuffisante : {2 * n_vise / 1e9:.1f} Go nécessaires, {libre / 1e9:.1f} Go libres")
    file = queue.Queue()
    sortie = open(chemin, "wb")

    def ecrire():                                 # fil d'écriture : absorbe les à-coups du disque
        while True:
            bloc = file.get()
            if bloc is None:
                return
            sortie.write(bloc.tobytes())
    fil = threading.Thread(target=ecrire, daemon=True)
    fil.start()
    blocs, n, debord, ecretes = [], 0, 0, 0
    niveaux, acc_p, acc_n, t_sec = [], 0.0, 0, 0     # niveau relatif (dB) par seconde : suivi de l'antenne
    ref_p = 2 * 20.0 ** 2                              # puissance du bruit au départ, en pas de 8 bits
    alerte = False
    print(f"Enregistrement GPS : {args.duree:.0f} s → {chemin} ({2 * n_vise / 1e9:.2f} Go)"
          + (" — Ctrl-C pour arrêter plus tôt (le déjà-capturé est traité)" if args.duree > 60 else "") + "…")
    t0 = time.time()
    t_affiche = t0
    try:
        while n < n_vise:
            sr = dev.readStream(st, [buf], 65536, timeoutUs=1_000_000)
            t_ns = time.time_ns()                                   # horloge système (peut être corrigée par NTP)
            t_brut = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)  # horloge matérielle, jamais corrigée
            if sr.ret > 0:
                m = min(sr.ret, n_vise - n)
                v = np.rint(buf[: 2 * m] * echelle)
                ecretes += int(np.count_nonzero(np.abs(v) > 127))
                file.put(np.clip(v, -127, 127).astype(np.int8))
                n += m
                blocs.append((n, t_ns, t_brut))
                acc_p += float(np.mean(v[::32] ** 2)) * 2      # I² + Q², sur un sous-échantillon
                acc_n += 1
                if n >= (t_sec + 1) * fs:                       # une valeur par seconde
                    rel = 10 * np.log10(acc_p / acc_n / ref_p + 1e-12)
                    niveaux.append(round(rel, 2))
                    acc_p, acc_n, t_sec = 0.0, 0, t_sec + 1
                    if rel < -6 and not alerte:
                        print(f"  ⚠ {n / fs:.0f} s : niveau tombé de {-rel:.1f} dB — alimentation de l'antenne coupée ?")
                        alerte = True
                    elif rel > -3 and alerte:
                        print(f"  {n / fs:.0f} s : niveau revenu ({rel:+.1f} dB)")
                        alerte = False
            elif sr.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                debord += 1
            if args.duree > 60 and time.time() - t_affiche >= 30:
                t_affiche = time.time()
                print(f"  … {n / fs:.0f}/{args.duree:.0f} s, niveau {niveaux[-1] if niveaux else 0:+.1f} dB"
                      + (" (antenne ?)" if alerte else "") + f", file d'écriture {file.qsize()} bloc(s)"
                      + (f", débordements {debord}" if debord else ""))
    except KeyboardInterrupt:
        print("Arrêt demandé : traitement de ce qui a été capturé")
    finally:
        dev.deactivateStream(st)
        dev.closeStream(st)
        file.put(None)
        fil.join()
        sortie.close()
    coupures_ant = []
    for i, r in enumerate(niveaux):
        if r < -6 and (not coupures_ant or coupures_ant[-1][1] < i - 1):
            coupures_ant.append([i, i])
        elif r < -6:
            coupures_ant[-1][1] = i
    if coupures_ant:
        print("ANTENNE : niveau effondré (alimentation coupée ?) entre "
              + ", ".join(f"{a} et {b + 1} s" for a, b in coupures_ant) + " — ces passages ne donneront rien")
    print(f"Capturé {n/fs:.1f} s en {time.time()-t0:.1f} s" + (f", DÉBORDEMENTS USB : {debord}" if debord else "")
          + (f", {ecretes / max(2 * n, 1) * 100:.3f} % d'échantillons écrêtés" if ecretes else ""))
    if debord:
        print("  Des échantillons ont été perdus : la continuité temporelle est rompue, la solution sera dégradée."
              " Fermer les autres programmes ou baisser --debit.")
    meta = dict(format="cs8", fs=fs, fc=f_lue, antenne=args.antenne, lna=lna, ifgr=ifgr, bande_if=bw,
                niveau_dbfs=niv, exces_antenne_db=exces, etat_antenne=etat, niveaux_relatifs_db=niveaux,
                echelle_cs16_vers_cs8=echelle, debordements=debord,
                t0_unix=blocs[0][1] * 1e-9 if blocs else None, blocs=blocs)
    with open(os.path.splitext(chemin)[0] + ".json", "w") as f:
        json.dump(meta, f)
    return Donnees(np.memmap(chemin, dtype=np.int8, mode="r")[: 2 * n], fs), meta, chemin


# ------------------------------------------------------------------------------------------- principal
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--liste", action="store_true", help="inventaire du RSPdx puis sortie")
    p.add_argument("--antenne", default="A", help="entrée du RSPdx (défaut A)")
    p.add_argument("--debit", type=int, default=2_048_000, help="débit souhaité (S/s, défaut 2048000)")
    p.add_argument("--lna", type=int, default=0, help="état LNA (0 = gain max, défaut 0)")
    p.add_argument("--ifgr", type=int, default=None, help="réduction de gain IF (dB) ; défaut : réglage automatique")
    p.add_argument("--niveau", type=float, default=-30.0, help="niveau visé par le réglage automatique (dBFS)")
    p.add_argument("--test-antenne", dest="test_antenne", action="store_true",
                   help="vérifie seulement que l'antenne active est alimentée (~20 s), puis s'arrête")
    p.add_argument("--forcer", action="store_true", help="enregistrer même si l'antenne semble non alimentée")
    p.add_argument("--plancher", type=float, default=PLANCHER_RSPDX_DBFS,
                   help=f"bruit propre du RSPdx sans antenne active, LNA 0, IFGR 20 (dBFS, défaut {PLANCHER_RSPDX_DBFS})")
    p.add_argument("--duree", type=float, default=None,
                   help="durée d'enregistrement (s) ; défaut 48, ou 20 si le cache d'éphémérides est récent")
    p.add_argument("--prechauffage", type=float, default=30.0,
                   help="secondes de flux ignorées avant l'enregistrement (défaut 30)")
    p.add_argument("--fichier", help="traiter un enregistrement IQ au lieu de capturer")
    p.add_argument("--format", choices=["cs8", "cs16", "cf32"], help="format du fichier (défaut : .json ou extension)")
    p.add_argument("--fs", type=float, help="débit du fichier (S/s) si pas de .json")
    p.add_argument("--garder", action="store_true",
                   help="garder l'IQ brut capturé (.cs8 + .json, 4,1 Mo/s), effacé sinon (gardé aussi en cas d'échec)")
    p.add_argument("--tranche", type=float, default=60.0,
                   help="au-delà de 1,5 × cette durée (s), traitement par tranches avec ré-acquisition (défaut 60)")
    p.add_argument("--doppler-max", dest="doppler_max", type=float, default=7000.0,
                   help="recherche Doppler ± (Hz, défaut 7000 : Doppler satellite + tolérance de l'oscillateur)")
    p.add_argument("--masque", type=float, default=10.0, help="masque d'élévation (°, défaut 10)")
    p.add_argument("--mobile", action="store_true", help="récepteur en mouvement (sinon statique : ε plus précis)")
    p.add_argument("--position", type=float, nargs=3, metavar=("LAT", "LON", "H"), default=None,
                   help="position connue (°, °, hauteur ellipsoïdale en m ≈ altitude + 50 m en France) : "
                        "avec moins de 4 satellites, l'heure et l'oscillateur restent mesurables (1 satellite suffit)")
    p.add_argument("--ondulation", type=float, default=None,
                   help="ondulation du géoïde N (m) pour convertir la hauteur ellipsoïdale en altitude")
    p.add_argument("--sortie", default="graves_gps.json", help="fichier de résultat (défaut graves_gps.json)")
    p.add_argument("--cache", default="gps_ephemerides.json", help="cache des éphémérides")
    p.add_argument("--regler-horloge", dest="regler", action="store_true",
                   help="corrige l'horloge système d'après le GPS (root ; utile sur le terrain sans Internet)")
    args = p.parse_args()

    if args.liste:
        import graves_capture as gc
        gc.lister()
        return
    if os.geteuid() == 0 and os.environ.get("SUDO_USER"):
        print("Lancé avec sudo : inutile pour --regler-horloge (le script demande lui-même sudo pour la seule mise à"
              " l'heure) ; sous sudo, le Python du venv (numba) n'est pas utilisé et les fichiers écrits appartiendront"
              " à root")
    if args.regler and args.fichier:
        sys.exit("--regler-horloge ne s'applique qu'à une capture en direct : l'écart mesuré dans un fichier date de"
                 " son enregistrement")
    if njit is None:
        print("numba absent : poursuite en numpy pur, nettement plus lente (pip install numba)")
    cache = charger_cache(args.cache)
    chemin_iq = None
    if args.fichier:
        d, meta = ouvrir_fichier(args.fichier, args.format, args.fs)
        print(f"Fichier {args.fichier} : {d.n / d.fs:.1f} s à {d.fs:.0f} S/s")
        source = f"fichier {os.path.basename(args.fichier)}"
    else:
        if args.duree is None:
            maint = (time.time() - GPS_UNIX) % SEMAINE
            frais = [e for k, e in cache.items() if isinstance(k, int) and abs(_dt_semaine(maint - e["toe"])) < 6000]
            args.duree = 20.0 if len(frais) >= 5 else 48.0
            if len(frais) >= 5:
                print(f"{len(frais)} éphémérides récentes en cache : capture courte ({args.duree:.0f} s)")
        d, meta, chemin_iq = capturer(args)
        source = f"RSPdx entrée {args.antenne}"

    fs_nominal = d.fs
    ref = charger_reference(args.sortie)
    pos_connue = None
    if args.position:
        pos_connue = tuple(args.position)
    elif ref and ref.get("position") and ref["position"].get("mesuree"):
        pos_connue = (ref["position"]["lat"], ref["position"]["lon"], ref["position"]["h_ellipsoide_m"])
    position_reprise = bool(pos_connue) and not args.position
    pertes_usb = detecter_pertes(meta["blocs"], fs_nominal) if meta.get("blocs") else []
    if pertes_usb:
        print("ATTENTION : échantillons perdus par le RSPdx / l'USB sans signalement du pilote, vers "
              + ", ".join(f"{n / fs_nominal:.1f} s (~{p * 1e3:.1f} ms)" for n, p in pertes_usb)
              + " — traitement par segments continus")
    res = traiter(d, ephs_cache=cache, doppler_max=args.doppler_max, masque=args.masque,
                  statique=not args.mobile, pos_connue=pos_connue, tranche_s=args.tranche,
                  coupures=[n for n, _ in pertes_usb])
    if chemin_iq:                                  # enregistrement de cette session : gardé si demandé ou si échec
        base = os.path.splitext(chemin_iq)[0]
        if args.garder or "erreur" in res:
            print(f"IQ brut gardé : {chemin_iq} (+ .json)" + ("" if args.garder else " — pour analyse de l'échec"))
        else:
            del d
            for ext in (".cs8", ".json"):
                try:
                    os.remove(base + ext)
                except OSError:
                    pass
    if "erreur" in res:
        sys.exit("Échec : " + res["erreur"])
    if res["cache_utilise"] or res["ephs"]:
        try:
            ecrire_cache(args.cache, res, cache)
        except PermissionError:
            print(f"Cache {args.cache} non modifiable (appartient à root ?) : sudo chown $USER: {args.cache}")

    eps = res["eps"]
    fs_vrai = fs_nominal * (1 + eps) if np.isfinite(eps) else fs_nominal
    t_utc0 = unix_de_l_echantillon(res, 0 if res["t0_fiable"] else res["n_ref"])
    horloge = ecart_horloge_pc(res, meta.get("blocs", []), fs_vrai) if meta.get("blocs") else None
    se, sn, su = res["sigma_enu"]
    alt = res["h"] - args.ondulation if args.ondulation is not None else None

    print()
    if not res["position_mesuree"]:
        print("Position NON mesurée (moins de 4 satellites) : position donnée (--position) ou référence précédente")
    print(f"Position  : {res['lat']:.6f}° N, {res['lon']:.6f}° E, hauteur ellipsoïdale {res['h']:.1f} m"
          + (f" (altitude {alt:.1f} m)" if alt is not None else ""))
    if res["position_mesuree"]:
        print(f"            dispersion E/N/H : {se:.1f} / {sn:.1f} / {su:.1f} m, GDOP {res['gdop']:.1f}, "
              f"résidu {res['rms']:.1f} m — {res['n_completes']}/{res['n_epoques']} époques à ≥ 4 satellites ("
              + ", ".join(f"G{p:02d}" for p in res["prns_completes"]) + ")")
    print("            satellites utilisés : "
          + ", ".join(f"G{s['prn']:02d} ({s['el']:.0f}°, {s['cn0']:.0f} dB-Hz)" for s in res["satellites"]))
    quel = "1er échantillon" if res["t0_fiable"] else f"échantillon n° {res['n_ref']:.0f} ({res['n_ref'] / fs_nominal:.1f} s)"
    print(f"Heure     : {quel} à {datetime.fromtimestamp(t_utc0, timezone.utc):%Y-%m-%d %H:%M:%S.%f} UTC"
          f"  (GPS − UTC = {res['dtls']} s, {res['dtls_src']}) ; "
          f"dispersion {res['res_temps_ns']:.0f} ns")
    if res["discontinuites"]:
        print("            discontinuités de l'échantillonnage (heure GPS) : "
              + ", ".join(f"{q['t_s']:.1f} s ({q['perte_ms']:+.3f} ms)" for q in res["discontinuites"])
              + " — chaque segment a sa propre origine de temps")
    print(f"Oscillateur du RSPdx : ε = {eps * 1e6:+.4f} ppm (Doppler)   {res['eps_temps'] * 1e6:+.4f} ppm (horloge)"
          + (f", dérive {res['derive_ppb_min']:+.2f} ppb/min" if np.isfinite(res['derive_ppb_min']) else ""))
    print(f"            → à 143,050 MHz : OL décalé de {F_GRAVES * eps:+.1f} Hz, la porteuse GRAVES doit apparaître "
          f"vers {-F_GRAVES * eps:+.1f} Hz ; fréquence vraie = fréquence mesurée {F_GRAVES * eps:+.1f} Hz")
    if not args.mobile:
        v = np.linalg.norm(res["vitesse"])
        print(f"Contrôle  : vitesse estimée {v:.2f} m/s (≈ 0 attendu pour une antenne fixe)")
    instab = instabilite_horloge_pc(meta["blocs"], fs_nominal) if meta.get("blocs") else None
    if instab is not None and instab > 0.005:
        print(f"Horloge PC : corrigée par NTP pendant la capture (variations de {instab * 1e3:.0f} ms) — "
              "écart mesuré à l'instant de référence")
    if horloge:
        ec = horloge["ecart"]
        txt = f"{ec * 1e3:+.1f} ms" if abs(ec) < 10 else f"{ec:+.3f} s ({ec / 86400:+.2f} j)"
        print(f"Horloge PC : {'en avance' if ec >= 0 else 'en retard'} de {txt} sur l'UTC "
              f"(gigue USB {horloge['gigue'] * 1e3:.1f} ms ; latence résiduelle inconnue, quelques ms)")
        if args.regler and abs(horloge["ecart"]) > 0.02:
            regler_horloge(horloge["ecart"])
    elif args.regler:
        print("Réglage d'horloge impossible : pas d'horodatage PC pour ces échantillons")

    position_sortie = dict(mesuree=res["position_mesuree"], lat=round(res["lat"], 7), lon=round(res["lon"], 7),
                           h_ellipsoide_m=round(res["h"], 2), altitude_m=round(alt, 2) if alt is not None else None,
                           ecart_position_donnee_m=res["ecart_position"],
                           sigma_enu_m=[round(v, 2) for v in res["sigma_enu"]], gdop=round(res["gdop"], 2),
                           ecef_m=[round(v, 2) for v in res["ecef"]],
                           mesure_position_utc=datetime.fromtimestamp(t_utc0, timezone.utc).isoformat())
    if not res["position_mesuree"] and position_reprise:
        # calage « heure seule » : la dernière position mesurée est conservée telle quelle (avec sa date), pour que
        # les calages suivants et graves_capture.py continuent de s'en servir
        position_sortie = dict(ref["position"])
        position_sortie.setdefault("mesure_position_utc", ref.get("mesure_utc"))
        position_sortie["reprise"] = True
        print(f"Position  : dernière position mesurée conservée (mesurée le {position_sortie['mesure_position_utc'][:16]} UTC)")
    sortie = dict(
        version=1, source=source, mesure_utc=datetime.fromtimestamp(t_utc0, timezone.utc).isoformat(),
        position=position_sortie,
        temps=dict(semaine_gps=res["semaine"], gps_moins_utc_s=res["dtls"], source_gps_moins_utc=res["dtls_src"],
                   t0_unix_premier_echantillon=t_utc0 if res["t0_fiable"] else None,
                   discontinuites=res["discontinuites"]),
        horloge_pc=dict(ecart_s=horloge["ecart"], gigue_s=horloge["gigue"]) if horloge else None,
        oscillateur=dict(epsilon=eps, epsilon_ppm=eps * 1e6, epsilon_horloge_ppm=res["eps_temps"] * 1e6,
                         derive_ppb_par_min=res["derive_ppb_min"],
                         correction_hz_graves=F_GRAVES * eps, debit_reel_sps=fs_vrai),
        satellites=res["satellites"])
    try:
        with open(args.sortie, "w") as f:
            json.dump(_propre(sortie), f, indent=1)
    except PermissionError:
        sys.exit(f"Impossible d'écrire {args.sortie} (appartient à root ?) : sudo chown $USER: {args.sortie}")
    print(f"Référence écrite : {args.sortie}")


if __name__ == "__main__":
    main()
