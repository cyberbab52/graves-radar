#!/usr/bin/env python3
"""
graves_capture.py — Étape 1 : acquisition IQ du RSPdx et cascade GRAVES

Ouvre le RSPdx via SoapySDR, enregistre N secondes autour de 143,050 MHz,
ramène le signal en bande de base (GRAVES = 0 Hz), décime vers ~15,6 kHz
et produit une cascade PNG (axe horizontal = décalage Doppler en Hz).

Usage :
  python3 graves_capture.py --liste                 # inventaire du RSPdx (antennes, gains, débits)
  python3 graves_capture.py --duree 120             # capture 2 min et cascade
  python3 graves_capture.py --duree 300 --iq        # idem + sauvegarde IQ décimé (.npy) pour analyse ultérieure
  python3 graves_capture.py --simulation            # test sans matériel (signaux synthétiques)
  python3 graves_capture.py --duree 300 --iq --direct   # cascades défilantes en temps réel pendant la capture
  python3 graves_capture.py --duree 300 --iq --analyse  # idem + avions / satellites / météores identifiés en direct
  python3 graves_capture.py --live                  # cascade en direct seule, sans limite de durée, aucun fichier écrit
  python3 graves_capture.py --live --analyse        # idem + identification en direct
  python3 graves_capture.py --freq 98.5e6 --decalage 0 --duree 10   # test antenne sur une station FM

Réglages par défaut (ceux qui conviennent à l'installation) : entrée B, LNA 2, filtre coupe-bande FM actif,
cascade fine affichée de −300 à +400 Hz.
Options : --antenne A|B|C, --sans-notch-fm (désactive le filtre FM), --biast (4,7 V sur l'entrée B)
Options gain : --lna N (état LNA, 0 = gain max, défaut 2) --ifgr N (réduction de gain IF en dB, 20–59)
Attention : gain maximal (--lna 0) sans filtre FM, les émetteurs FM saturent le RSPdx et les échos disparaissent.
L'AGC est toujours désactivée (elle déformerait les échos).

Dépendances : SoapySDR + SoapySDRPlay3 (apt/compilation), numpy, scipy, matplotlib.
Fermer SDRconnect avant de lancer : un seul programme peut ouvrir le RSPdx.
"""
import argparse
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy import signal

F_GRAVES = 143.050e6


# --------------------------------------------------------------------------- matériel
def ouvrir_rsp():
    import SoapySDR
    res = SoapySDR.Device.enumerate("driver=sdrplay")
    if not res:
        sys.exit("Aucun RSP trouvé. Vérifier : service sdrplay actif ? SDRconnect fermé ? "
                 "(SoapySDRUtil --find=\"driver=sdrplay\")")
    return SoapySDR, SoapySDR.Device(res[0]), res[0]


def lister():
    SoapySDR, dev, info = ouvrir_rsp()
    RX = SoapySDR.SOAPY_SDR_RX
    print("Périphérique :", dict(info))
    print("Antennes     :", list(dev.listAntennas(RX, 0)))
    print("Gains        :", {g: str(dev.getGainRange(RX, 0, g)) for g in dev.listGains(RX, 0)})
    print("Débits (S/s) :", [int(r) for r in dev.listSampleRates(RX, 0)])
    try:
        print("Réglages     :", [a.key for a in dev.getSettingInfo()])
    except Exception:
        pass


def choisir_antenne(dev, RX, lettre):
    noms = list(dev.listAntennas(RX, 0))
    for n in noms:
        if n.strip().upper().endswith(lettre.upper()):
            return n
    sys.exit(f"Antenne {lettre} introuvable parmi {noms}")


def choisir_debit(dev, RX, cible):
    debits = sorted(int(r) for r in dev.listSampleRates(RX, 0))
    ok = [r for r in debits if r >= cible]
    return ok[0] if ok else debits[-1]


def capturer(args, traiter_bloc):
    SoapySDR, dev, _ = ouvrir_rsp()
    RX = SoapySDR.SOAPY_SDR_RX
    ant = choisir_antenne(dev, RX, args.antenne)
    fs = choisir_debit(dev, RX, args.debit)
    fc = args.freq - args.decalage

    dev.setAntenna(RX, 0, ant)
    dev.setSampleRate(RX, 0, fs)
    dev.setGainMode(RX, 0, False)                      # AGC off
    gains = list(dev.listGains(RX, 0))
    if "RFGR" in gains:
        dev.setGain(RX, 0, "RFGR", args.lna)
    if "IFGR" in gains:
        dev.setGain(RX, 0, "IFGR", args.ifgr)
    for cle, val in (("biasT_ctrl", "true" if args.biast else "false"),
                     ("rfnotch_ctrl", "true" if args.notch_fm else "false")):
        try:
            dev.writeSetting(cle, val)
        except Exception as e:
            print(f"  réglage {cle} non appliqué : {e}")
    dev.setFrequency(RX, 0, fc)
    try:                                               # contrôle : fréquence réellement programmée
        f_lue = dev.getFrequency(RX, 0)
        if abs(f_lue - fc) > 1000:
            print(f"ATTENTION : fréquence demandée {fc/1e6:.6f} MHz, relue {f_lue/1e6:.6f} MHz")
    except Exception:
        f_lue = None

    print(f"RSPdx : {ant}, fc = {fc/1e6:.6f} MHz" + (f" (relue {f_lue/1e6:.6f})" if f_lue else "") + f", fs = {fs} S/s, LNA = {args.lna}, IFGR = {args.ifgr} dB, "
          f"Bias-T = {'ON' if args.biast else 'off'}, filtre FM = {'ON' if args.notch_fm else 'off'}")
    st = dev.setupStream(RX, SoapySDR.SOAPY_SDR_CF32)
    dev.activateStream(st)
    buf = np.empty(65536, np.complex64)
    if args.prechauffage > 0:
        # le tuner chauffe dès que le flux démarre et l'oscillateur dérive (~10 Hz en 1–2 min) :
        # on laisse tourner à vide avant d'enregistrer
        print(f"Mise en température du récepteur : {args.prechauffage:.0f} s (données ignorées)…")
        fin = time.time() + args.prechauffage
        args.etat = "mise en température"
        while time.time() < fin and not args.arret.is_set():
            sr = dev.readStream(st, [buf], len(buf), timeoutUs=1_000_000)
            if sr.ret > 0 and getattr(traiter_bloc, "apercu", None):
                traiter_bloc.apercu(buf[: sr.ret].copy(), fs)       # visible en direct, non enregistré
    n_tot, n_vise, debordements = 0, args.duree * fs, 0              # durée infinie possible (--live)
    args.etat = "en direct" if args.live else "enregistrement"
    t0 = time.time()
    try:
        while n_tot < n_vise and not args.arret.is_set():
            sr = dev.readStream(st, [buf], len(buf), timeoutUs=1_000_000)
            if sr.ret > 0:
                if n_tot == 0:
                    args.t_premier = time.time() - sr.ret / fs   # horodatage du 1er échantillon
                traiter_bloc(buf[: sr.ret].copy(), fs)
                n_tot += sr.ret
            elif sr.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                debordements += 1
                args.debordements = debordements
            elif sr.ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                print("  délai dépassé (pas de données)")
    except KeyboardInterrupt:
        print("Interrompu.")
    finally:
        dev.deactivateStream(st)
        dev.closeStream(st)
    print(f"Capturé {n_tot/fs:.1f} s en {time.time()-t0:.1f} s, débordements USB : {debordements}")
    args.etat = "terminé"
    return fs


def simuler(args, traiter_bloc):
    """Signaux synthétiques : bruit + porteuse directe hachée + avion + météore + satellite."""
    fs = 250_000
    rng = np.random.default_rng(1)
    n_vise = args.duree * fs
    bloc = 65536
    n = 0
    print(f"SIMULATION : fs = {fs} S/s, " + (f"{args.duree:g} s" if np.isfinite(args.duree) else "sans limite de durée"))
    args.etat = "en direct" if args.live else "enregistrement"
    t_depart = time.time()
    while n < n_vise and not args.arret.is_set():
        if args.direct:                                  # cadence temps réel pour l'affichage
            retard = n / fs - (time.time() - t_depart)
            if retard > 0:
                time.sleep(retard)
        m = int(min(bloc, n_vise - n))
        t = (n + np.arange(m)) / fs
        x = (rng.standard_normal(m) + 1j * rng.standard_normal(m)).astype(np.complex64)
        def ton(f_inst_phase, amp):
            return amp * np.exp(1j * f_inst_phase)
        hache = ((t % 19.2) < 3.2)                     # faisceau qui balaie
        f0 = args.decalage                             # GRAVES à +decalage Hz en bande de base RSP
        x += ton(2*np.pi*f0*t, 0.02) * hache           # porteuse directe faible
        # avion : Doppler de +180 à -60 Hz en 90 s, à partir de t = 10 s
        ta = t - 10
        fa = 180 - 240 * np.clip(ta / 90, 0, 1)
        x += ton(2*np.pi*(f0*t + np.cumsum(fa)/fs), 0.08) * ((ta > 0) & (ta < 90)) * hache
        # météore : chirp 3 kHz -> 0 en 0,2 s, puis traînée 0,8 s qui décroît, à t = 40 s
        tm = t - 40
        fm = np.where(tm < 0.2, 3000 * (1 - tm/0.2), 5)
        env = np.where(tm < 0.2, 0.1, 0.6*np.exp(-(tm-0.2)/0.3)) * ((tm > 0) & (tm < 1.5))
        x += ton(2*np.pi*(f0*t + np.cumsum(fm)/fs), env)
        # satellite : S de +5 kHz à -4 kHz en 60 s, à t = 55 s
        ts_ = t - 55
        fsat = 500 - 4500*np.tanh((ts_-30)/12)
        x += ton(2*np.pi*(f0*t + np.cumsum(fsat)/fs), 0.05) * ((ts_ > 0) & (ts_ < 60)) * hache
        traiter_bloc(x.astype(np.complex64), fs)
        n += m
    args.etat = "terminé"
    return fs


# --------------------------------------------------------------------------- traitement
class Decimateur:
    """Translation de fréquence (GRAVES -> 0 Hz) + filtre passe-bas + décimation, sans discontinuité entre blocs."""

    def __init__(self, decalage, fs_cible=16_000):
        self.decalage = decalage
        self.fs_cible = fs_cible
        self.pret = False
        self.sortie = []
        self.garder = True
        self.ecouteur = None

    def __call__(self, x, fs):
        if not self.pret:
            self.fs = fs
            self.D = max(1, int(fs // self.fs_cible))
            self.fs_out = fs / self.D
            self.h = signal.firwin(10 * self.D + 1, 0.9 / self.D).astype(np.float32)
            self.zi = np.zeros(len(self.h) - 1, np.complex64)
            self.n = 0
            self.phase_reste = 0
            self.pret = True
        k = self.n + np.arange(len(x))
        osc = np.exp(-2j * np.pi * self.decalage * k / self.fs).astype(np.complex64)
        y, self.zi = signal.lfilter(self.h, 1.0, x * osc, zi=self.zi)
        debut = (-self.n) % self.D
        bloc = y[debut:: self.D].astype(np.complex64)
        if self.garder:
            self.sortie.append(bloc)
        if self.ecouteur is not None:
            self.ecouteur(bloc, self.fs_out)
        self.n += len(x)

    def apercu(self, x, fs):
        """Bloc traité pour l'affichage seulement (mise en température : rien n'est enregistré)."""
        garder, self.garder = self.garder, False
        self(x, fs)
        self.garder = garder

    def resultat(self):
        return np.concatenate(self.sortie), self.fs_out


def _empiler(tab, lignes):
    """Insère des lignes en tête de tab (la plus récente en premier), en une seule copie."""
    k = len(lignes)
    if k == 0:
        return tab
    nouv = np.asarray(lignes[::-1], dtype=tab.dtype)
    if k >= len(tab):
        return nouv[:len(tab)].copy()
    return np.concatenate([nouv, tab[:-k]])


class Afficheur:
    """Cascades défilantes en temps réel : fine (±400 Hz, 0,5 Hz, ~1 s/ligne) et rapide (±5,5 kHz, 8 Hz,
    ~60 ms/ligne). Chaque ligne est normalisée par sa médiane : le fond est plat, porteuse et échos ressortent.

    Rendu optimisé : les lignes sont mises en couleur une seule fois à leur arrivée (table de couleurs
    256 niveaux), l'image est rafraîchie au plus `ips` fois par seconde, et par « blitting » : axes,
    graduations et légende sont dessinés une fois, seuls les images, repères et titre sont redessinés."""

    def __init__(self, titre, duree_fine=300.0, duree_rapide=60.0, dyn_db=20.0, analyse=None, horloge=time.time,
                 plage_fine=(-300.0, 400.0), taille=(1920, 1080), ips=5.0, blit=True, porteuse=None):
        self.taille, self.blit_permis = taille, blit
        self.porteuse_memo = porteuse                 # dernière position connue de la porteuse directe (Hz)
        self.porteuse_ref = porteuse                  # position donnée par --porteuse : zone de recherche
        self.titre, self.duree_fine, self.duree_rapide, self.dyn = titre, duree_fine, duree_rapide, dyn_db
        self.analyse, self.horloge = analyse, horloge
        # partie affichée de la cascade fine ; le calcul (porteuse, corrélation ADS-B) reste sur ±400 Hz
        self.plage_fine = (max(-400.0, min(plage_fine)), min(400.0, max(plage_fine)))
        self.periode = 1.0 / ips
        self.t_image = -1e9
        self.pret = False
        self.file = __import__("queue").Queue()

    def recevoir(self, bloc, fs):                     # appelé depuis le fil d'acquisition
        self.file.put((bloc, fs))

    def _init(self, fs):
        from matplotlib import colormaps
        self.fs = fs
        self.n_fin, self.n_rap = 32768, 2048
        self.h_fin, self.h_rap = self.n_fin // 2, self.n_rap // 2
        self.w_fin = np.hanning(self.n_fin).astype(np.float32)
        self.w_rap = np.hanning(self.n_rap).astype(np.float32)
        f_fin = np.fft.fftshift(np.fft.fftfreq(self.n_fin, 1 / fs))
        f_rap = np.fft.fftshift(np.fft.fftfreq(self.n_rap, 1 / fs))
        self.m_fin, self.m_rap = np.abs(f_fin) <= 400, np.abs(f_rap) <= 5500
        self.f_fin, self.f_rap = f_fin[self.m_fin], f_rap[self.m_rap]
        self.m_aff = (self.f_fin >= self.plage_fine[0]) & (self.f_fin <= self.plage_fine[1])   # colonnes affichées
        self.l_fin = int(self.duree_fine / (self.h_fin / fs))
        self.l_rap = int(self.duree_rapide / (self.h_rap / fs))
        self.img_fin = np.zeros((self.l_fin, self.m_fin.sum()), np.float32)     # dB au-dessus du fond (analyse)
        self.img_rap = np.zeros((self.l_rap, self.m_rap.sum()), np.float32)
        self.lut_fin = colormaps["viridis"](np.linspace(0, 1, 256), bytes=True)   # tables de couleurs RGBA
        self.lut_rap = colormaps["magma"](np.linspace(0, 1, 256), bytes=True)
        self.rgb_fin = np.broadcast_to(self.lut_fin[0], (self.l_fin, self.m_aff.sum(), 4)).copy()   # affichage
        self.rgb_rap = np.broadcast_to(self.lut_rap[0], (self.l_rap, self.m_rap.sum(), 4)).copy()
        self.t_fin = np.full(self.l_fin, np.nan)       # instant (s Unix) du centre de chaque ligne
        self.t_rap = np.full(self.l_rap, np.nan)
        self.niv_rap = np.full(self.l_rap, np.nan)     # niveau brut (médiane) de chaque ligne rapide
        self.n_lignes_rap = 0
        self.t_origine = None
        self.tampon = np.zeros(0, np.complex64)
        self.p_fin = self.p_rap = 0                   # positions (échantillons) des prochaines FFT
        self.base = 0                                 # indice absolu du début du tampon
        self._porteuse_cache, self._porteuse_n, self.n_lignes_fin = None, 0, 0
        self.pret = True

    def porteuse(self):
        """Porteuse directe : (décalage Hz, texte).
        Repérée par la MÉDIANE dans le temps de chaque case de la cascade fine (jusqu'à 5 min) : la
        porteuse est là presque tout le temps, même faiblement, alors qu'un écho d'avion — même fort et
        lent, presque immobile en fréquence pendant une minute — n'occupe une case qu'une minorité du
        temps et ne déplace pas la médiane. Recalculée toutes les 2 lignes (~2 s). Si elle n'est pas trouvée, on
        garde la dernière position connue (ou celle de --porteuse)."""
        if not self.pret:
            return (self.porteuse_memo or 0.0), ""
        if self._porteuse_cache is not None and self.n_lignes_fin - self._porteuse_n < 2:
            return self._porteuse_cache              # recalcul toutes les 2 lignes fines (~2 s)
        self._porteuse_n = self.n_lignes_fin
        ok = ~np.isnan(self.t_fin)
        if ok.sum() < 30:                              # ~30 s : trop peu de lignes pour une médiane fiable
            res = ((self.porteuse_memo or 0.0), "")
        else:
            med = np.median(self.img_fin[ok], axis=0)  # dB au-dessus du fond, médiane temporelle par case
            if self.porteuse_ref is not None:          # --porteuse : on ne la cherche qu'autour (±15 Hz,
                zone = np.abs(self.f_fin - self.porteuse_ref) <= 15   # dérive de mise en température comprise)
            else:
                zone = np.abs(self.f_fin) <= 200
            k = int(np.argmax(np.where(zone, med, -np.inf)))
            # bruit seul : médiane par case ≈ 0 dB, à ±0,4 dB près. Sur ±200 Hz (800 cases), un pic de
            # bruit peut atteindre ~1,5 dB : seuil 3 dB. Avec --porteuse, la zone ne fait que 60 cases :
            # un pic de 1,5 dB qui dépasse nettement ses voisines suffit — c'est le cas de la porteuse
            # faible de la Yagi relevée, qui continue de dériver lentement avec la température.
            seuil = 1.5 if self.porteuse_ref is not None else 3.0
            relief = med[k] - np.median(med[zone])
            if med[k] >= seuil and relief >= 1.2:
                self.porteuse_memo = float(self.f_fin[k])
                faible = " faible" if med[k] < 3.0 else ""
                res = (self.porteuse_memo,
                       f" — porteuse GRAVES à {self.porteuse_memo:+.1f} Hz (+{med[k]:.1f} dB{faible})")
            elif self.porteuse_memo is None:
                res = (0.0, " — porteuse GRAVES non détectée (option --porteuse pour donner sa position)")
            else:
                res = (self.porteuse_memo,
                       f" — porteuse GRAVES non vue, référence conservée à {self.porteuse_memo:+.1f} Hz")
        self._porteuse_cache = res
        return res

    def _couleur(self, l, lut):
        return lut[np.clip(l * (255.0 / self.dyn), 0, 255).astype(np.uint8)]

    def _ligne(self, seg, w, m):
        s = np.fft.fftshift(np.fft.fft(seg * w))[m]
        p = 10 * np.log10(np.abs(s) ** 2 + 1e-20)
        med = np.median(p)
        self._niveau = med
        return (p - med).astype(np.float32)

    def traiter_file(self):
        """Calcule les nouvelles lignes ; les tableaux sont décalés une seule fois par appel."""
        rap, t_rap, niv, fin, t_fin = [], [], [], [], []
        while True:
            try:
                bloc, fs = self.file.get_nowait()
            except Exception:
                break
            if not self.pret:
                self._init(fs)
            if self.t_origine is None:
                self.t_origine = self.horloge() - len(bloc) / fs
            self.tampon = np.concatenate([self.tampon, bloc])
            fin_abs = self.base + len(self.tampon)
            while self.p_rap + self.n_rap <= fin_abs:
                i = self.p_rap - self.base
                rap.append(self._ligne(self.tampon[i:i + self.n_rap], self.w_rap, self.m_rap))
                t_rap.append(self.t_origine + (self.p_rap + self.n_rap / 2) / self.fs)
                niv.append(self._niveau)
                self.p_rap += self.h_rap
                self.n_lignes_rap += 1
            while self.p_fin + self.n_fin <= fin_abs:
                i = self.p_fin - self.base
                l = self._ligne(self.tampon[i:i + self.n_fin], self.w_fin, self.m_fin)
                fin.append(l)
                self.n_lignes_fin += 1
                t_fin.append(self.t_origine + (self.p_fin + self.n_fin / 2) / self.fs)
                self.p_fin += self.h_fin
            garde = min(self.p_fin, self.p_rap) - self.base
            if garde > 0:
                self.tampon = self.tampon[garde:]
                self.base += garde
        if rap:
            rap = rap[-self.l_rap:]; t_rap = t_rap[-self.l_rap:]; niv = niv[-self.l_rap:]
            self.img_rap = _empiler(self.img_rap, rap)
            self.rgb_rap = _empiler(self.rgb_rap, [self._couleur(l, self.lut_rap) for l in rap])
            self.t_rap = _empiler(self.t_rap, t_rap)
            self.niv_rap = _empiler(self.niv_rap, niv)
        if fin:
            fin = fin[-self.l_fin:]; t_fin = t_fin[-self.l_fin:]
            self.img_fin = _empiler(self.img_fin, fin)
            self.rgb_fin = _empiler(self.rgb_fin, [self._couleur(l[self.m_aff], self.lut_fin) for l in fin])
            self.t_fin = _empiler(self.t_fin, t_fin)
        return bool(rap or fin)

    def ouvrir(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        w, h = self.taille
        self.fig, (self.a1, self.a2) = plt.subplots(1, 2, figsize=(w / 100, h / 100), dpi=100, layout="constrained",
                                                     gridspec_kw=dict(width_ratios=[1.4, 1]))   # la fine est la plus utile
        self.fig.canvas.manager.set_window_title("GRAVES — cascade en direct")
        try:
            self.fig.canvas.manager.resize(w, h)       # taille de la fenêtre en pixels (Tk, Qt, GTK)
        except Exception:
            pass
        self.im1 = self.im2 = None
        # deux lignes réservées dès le départ : le résumé de l'analyse s'affiche sans recalcul de la mise en page
        self.txt = self.fig.suptitle(self.titre + ("\n " if self.analyse is not None else ""))
        self.ferme = False
        self.fond = None                               # image des parties fixes, pour le blitting
        self.blit = self.blit_permis and bool(getattr(self.fig.canvas, "supports_blit", False))
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "ferme", True))
        self.fig.canvas.mpl_connect("draw_event", self._sur_dessin_complet)
        plt.show(block=False)                          # pas de plt.ion() : il redessinerait tout à chaque changement

    def _animes(self):
        arts = [self.im1, self.im2, self.mark]
        if self.analyse is not None:
            arts += list(self.analyse.artistes)
        return arts

    def _sur_dessin_complet(self, evt):
        """Après un dessin complet (ouverture, redimensionnement) : mémorise le fond, puis pose les parties mobiles."""
        if not self.blit or self.im1 is None:
            return
        self.fond = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for a in self._animes():
            a.axes.draw_artist(a)
        self.fig.draw_artist(self.txt)

    def _evenements(self, attente=0.02):
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
        time.sleep(attente)

    def _creer_images(self):
        import matplotlib as mpl
        d_fin = self.l_fin * self.h_fin / self.fs
        d_rap = self.l_rap * self.h_rap / self.fs
        fa = self.f_fin[self.m_aff]
        self.im1 = self.a1.imshow(self.rgb_fin, aspect="auto", origin="upper", interpolation="nearest",
                                  extent=[fa[0], fa[-1], d_fin, 0])
        self.im2 = self.a2.imshow(self.rgb_rap, aspect="auto", origin="upper", interpolation="nearest",
                                  extent=[self.f_rap[0], self.f_rap[-1], d_rap, 0])
        pf = self.plage_fine
        self.a1.set_title(f"Fine {pf[0]:+.0f} à {pf[1]:+.0f} Hz (0,5 Hz) — avions, porteuse".replace("-", "−"))
        self.a1.set_xlim(*pf)
        self.a2.set_xlim(self.f_rap[0], self.f_rap[-1])
        self.a2.set_title("Rapide ±5,5 kHz (8 Hz, 60 ms) — météores, satellites")
        for a in (self.a1, self.a2):
            a.set_xlabel("Décalage / 143,050 MHz (Hz)")
            a.set_autoscale_on(False)                  # les tracés de l'analyse ne doivent pas bouger les axes
        self.a1.set_ylabel("Il y a… (s)")
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, self.dyn), cmap="magma")
        cb = self.fig.colorbar(sm, ax=self.a2, location="bottom", shrink=0.5, aspect=40, pad=0.01)
        cb.set_label("dB au-dessus du fond", fontsize=7)
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        self.mark = self.a1.axvline(0, color="w", lw=0.6, alpha=0.5, ls=":")
        if self.blit:
            for a in (self.im1, self.im2, self.mark, self.txt):
                a.set_animated(True)
        self.fig.canvas.draw()                         # dessin complet unique → _sur_dessin_complet mémorise le fond

    def dessiner(self, etat, ecoule, total, debordements, forcer=False):
        """Rafraîchit l'affichage (au plus `ips` fois par seconde, sauf forcer=True) ; sinon traite les
        événements de la fenêtre et rend la main après une courte pause."""
        if not self.pret or self.ferme:
            if not self.ferme and hasattr(self, "fig"):
                self._evenements()
            else:
                time.sleep(0.02)
            return
        maintenant = time.monotonic()
        if not forcer and maintenant - self.t_image < self.periode:
            self._evenements()
            return
        self.t_image = maintenant
        if self.im1 is None:
            self._creer_images()
        self.im1.set_data(self.rgb_fin)
        self.im2.set_data(self.rgb_rap)
        off, info = self.porteuse()
        if info:
            self.mark.set_xdata([off] * 2)
        ligne2 = ""
        if self.analyse is not None:
            try:
                self.analyse.maj_avions(self.img_fin, self.t_fin, self.f_fin, off)
                self.analyse.maj_satellites(self.img_rap, self.t_rap, self.f_rap, off)
                self.analyse.maj_meteores(self.img_rap, self.t_rap, self.f_rap, self.niv_rap, off, self.n_lignes_rap)
                self.analyse.dessiner(self.a1, self.a2, self.t_fin, self.t_rap, off)
                if self.blit:
                    for a in self.analyse.artistes:
                        a.set_animated(True)
                ligne2 = "\n" + self.analyse.resume()
            except Exception as e:
                ligne2 = f"\nanalyse en direct indisponible : {e}"
        duree = f"{ecoule:.0f}/{total:.0f} s" if np.isfinite(total) else f"{ecoule:.0f} s"
        self.txt.set_text(f"{self.titre} — {etat} {duree}"
                          f"{' — débordements USB : ' + str(debordements) if debordements else ''}{info}{ligne2}")
        cv = self.fig.canvas
        if self.blit and self.fond is not None:
            cv.restore_region(self.fond)
            for a in self._animes():
                a.axes.draw_artist(a)
            self.fig.draw_artist(self.txt)
            cv.blit(self.fig.bbox)
        else:
            cv.draw_idle()
        self._evenements(0.0)


def cascade(iq, fs, resolution_hz, recouvrement=0.5):
    nperseg = int(2 ** np.round(np.log2(fs / resolution_hz)))
    f, t, S = signal.spectrogram(iq, fs=fs, window="hann", nperseg=nperseg,
                                 noverlap=int(nperseg * recouvrement), detrend=False,
                                 return_onesided=False, scaling="density", mode="psd")
    f = np.fft.fftshift(f)
    S = np.fft.fftshift(S, axes=0)
    return f, t, 10 * np.log10(S + 1e-20)


def rapport_raies(f, S_db, n=8):
    """Raies fixes / dominantes : moyenne temporelle, au-dessus du fond médian."""
    moy = S_db.mean(axis=1)
    fond = np.median(moy)
    pics, _ = signal.find_peaks(moy, height=fond + 6, distance=5)
    ordre = pics[np.argsort(moy[pics])[::-1][:n]]
    return fond, [(f[i], moy[i] - fond) for i in sorted(ordre, key=lambda i: f[i])]


def tracer(png, f, t, S_db, titre, fs_out, debut_utc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # aplatissement du fond : on retire la médiane temporelle de chaque case de fréquence
    norm = S_db - np.median(S_db, axis=1, keepdims=True)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, layout="constrained",
                                 gridspec_kw=dict(height_ratios=[1, 4]))
    a0.plot(f, S_db.mean(axis=1), lw=0.7, color="C0")
    a0.set_ylabel("Moyenne (dB rel.)")
    a0.grid(alpha=0.3)
    a0.set_title(titre)
    vmin, vmax = np.percentile(norm, [20, 99.8])
    im = a1.imshow(norm.T, aspect="auto", origin="upper", cmap="viridis",
                   extent=[f[0], f[-1], t[-1], t[0]], vmin=vmin, vmax=max(vmax, vmin + 6),
                   interpolation="nearest")
    a1.set_xlabel("Décalage par rapport à 143,050 MHz (Hz)")
    a1.set_ylabel(f"Temps (s) depuis {debut_utc:%H:%M:%S} UTC")
    fig.colorbar(im, ax=[a0, a1], label="dB au-dessus du fond", pad=0.01, shrink=0.8)
    fig.savefig(png, dpi=110)
    plt.close(fig)


def tracer_zoom(png, f, t, S_db, plage, titre, debut_utc):
    """plage : (min, max) des décalages affichés, en Hz."""
    m = (f >= min(plage)) & (f <= max(plage))
    tracer(png, f[m], t, S_db[m], titre, None, debut_utc)


# --------------------------------------------------------------------------- principal
def creer_parseur(description=__doc__):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--liste", action="store_true", help="inventaire du RSPdx puis sortie")
    p.add_argument("--simulation", action="store_true", help="signaux synthétiques, sans matériel")
    p.add_argument("--freq", type=float, default=F_GRAVES, help="fréquence d'intérêt (Hz)")
    p.add_argument("--decalage", type=float, default=10_000,
                   help="on accorde le RSP à freq - decalage pour éviter la raie DC (Hz, défaut 10000)")
    p.add_argument("--duree", type=float, default=None,
                   help="durée de capture (s) ; défaut 120, ou sans limite avec --live")
    p.add_argument("--debit", type=int, default=250_000, help="débit RSP souhaité (S/s)")
    p.add_argument("--antenne", default="B", help="entrée du RSPdx : A, B ou C")
    p.add_argument("--lna", type=int, default=2,
                   help="état LNA (0 = gain max ; défaut 2, validé avec la Yagi et le filtre FM)")
    p.add_argument("--ifgr", type=int, default=40, help="réduction de gain IF en dB (20–59)")
    p.add_argument("--prechauffage", type=float, default=120,
                   help="secondes de flux ignorées avant l'enregistrement, le temps que l'oscillateur se stabilise "
                        "(défaut 120, 0 pour désactiver)")
    p.add_argument("--biast", action="store_true", help="active le Bias-T 4,7 V (entrée B uniquement)")
    p.add_argument("--notch-fm", dest="notch_fm", action="store_true", default=True,
                   help="filtre coupe-bande FM 88–108 MHz du RSPdx (actif par défaut)")
    p.add_argument("--sans-notch-fm", dest="notch_fm", action="store_false",
                   help="désactive le filtre coupe-bande FM")
    p.add_argument("--resolution", type=float, default=4.0, help="résolution de la cascade large (Hz)")
    p.add_argument("--iq", action="store_true", help="sauvegarder l'IQ décimé (.npy)")
    p.add_argument("--direct", action="store_true",
                   help="afficher les cascades en temps réel pendant l'acquisition (fenêtre graphique)")
    p.add_argument("--analyse", action="store_true",
                   help="avec --direct : identifier en direct avions (journal ADS-B), satellites (catalogue) et météores")
    p.add_argument("--porteuse", type=float, default=None, metavar="HZ",
                   help="position connue de la porteuse directe (Hz, ex. 64), utilisée tant qu'elle n'est pas "
                        "détectée (Yagi relevée : porteuse faible). Sinon : dernière position détectée")
    p.add_argument("--plage-fine", dest="plage_fine", type=float, nargs=2, default=[-300.0, 400.0],
                   metavar=("MIN", "MAX"),
                   help="décalages affichés dans la cascade fine (en direct et PNG _fin), en Hz "
                        "(défaut −300 400, limites ±400)")
    def taille_px(s):
        try:
            w, h = (int(x) for x in s.lower().replace("×", "x").split("x"))
            assert w >= 320 and h >= 240
            return w, h
        except Exception:
            raise argparse.ArgumentTypeError("format attendu : LARGEURxHAUTEUR, par ex. 1920x1080")
    p.add_argument("--taille", type=taille_px, default=(1920, 1080), metavar="LxH",
                   help="taille de la fenêtre en direct, en pixels (défaut 1920x1080)")
    p.add_argument("--ips", type=float, default=5.0,
                   help="images par seconde maximum de l'affichage en direct (défaut 5 ; 15 avec graves_v2.py)")
    p.add_argument("--sans-blit", dest="sans_blit", action="store_true",
                   help="affichage en direct sans blitting (plus lent) : en cas d'image incomplète ou décalée")
    p.add_argument("--live", action="store_true",
                   help="cascade en direct uniquement : aucun fichier écrit (ni IQ, ni PNG), rien n'est gardé en mémoire ; "
                        "tourne jusqu'à fermeture de la fenêtre ou Ctrl-C (sauf --duree). Combinable avec --analyse")
    try:
        import graves_adsb as _a
        st, nom_st = _a.STATION, _a.NOM_STATION
    except Exception:
        st, nom_st = (48.8566, 2.3522, 35.0), "Paris (exemple)"
    p.add_argument("--lat", type=float, default=st[0], help=f"latitude du récepteur (défaut : {nom_st})")
    p.add_argument("--lon", type=float, default=st[1], help="longitude du récepteur")
    p.add_argument("--alt", type=float, default=st[2], help="altitude de l'antenne (m)")
    p.add_argument("--journal", default=".", help="dossier du journal ADS-B (adsb_AAAAMMJJ.jsonl)")
    p.add_argument("--catalogues", default=".", help="dossier des catalogues orbitaux (catalogue_AAAAMMJJ.json)")
    p.add_argument("--prefixe", default="graves", help="préfixe des fichiers de sortie")
    return p


def preparer(args):
    """Cohérence des options après lecture de la ligne de commande."""
    if args.analyse or args.live:
        args.direct = True
    if args.duree is None:
        args.duree = float("inf") if args.live else 120.0
    if args.live and args.iq:
        print("--iq ignoré : le mode --live n'écrit aucun fichier")
        args.iq = False
    print(f"Récepteur : {args.lat:.4f} N, {args.lon:.4f} E, {args.alt:.0f} m")
    return args


def creer_analyse(args):
    if not args.analyse:
        return None
    try:
        import graves_direct
        return graves_direct.AnalyseDirecte(args.journal, args.catalogues, rx=(args.lat, args.lon, args.alt))
    except Exception as e:
        print(f"Analyse en direct indisponible ({e}) : cascades seules")
        return None


def titre_direct(args):
    return f"GRAVES {args.freq/1e6:.3f} MHz — {'SIMULATION' if args.simulation else 'RSPdx ant. ' + args.antenne}"


def main():
    args = preparer(creer_parseur().parse_args())

    if args.liste:
        lister()
        return

    import threading
    debut = datetime.now(timezone.utc)
    args.t_premier = debut.timestamp()
    args.arret, args.etat, args.debordements = threading.Event(), "démarrage", 0
    dec = Decimateur(args.decalage)
    dec.garder = not args.live                          # --live : l'IQ décimé n'est pas accumulé en mémoire
    acquisition = simuler if args.simulation else capturer
    if args.direct:
        titre = titre_direct(args)
        analyse = creer_analyse(args)
        aff = Afficheur(titre, analyse=analyse, plage_fine=args.plage_fine, taille=args.taille,
                        ips=args.ips, blit=not args.sans_blit, porteuse=args.porteuse)
        dec.ecouteur = aff.recevoir
        fil = threading.Thread(target=acquisition, args=(args, dec), daemon=True)
        t_dep = time.time()
        fil.start()
        try:
            aff.ouvrir()
        except Exception as e:
            if args.live:
                print(f"Affichage impossible ({e}) : rien à faire en mode --live, arrêt")
                args.arret.set()
                fil.join()
                return
            print(f"Affichage impossible ({e}) : l'acquisition continue sans fenêtre")
            aff = None
        total = args.duree + (0 if args.simulation else args.prechauffage)
        if args.live:
            print("Mode --live : aucun fichier ne sera écrit. Fermer la fenêtre ou Ctrl-C pour arrêter.")
        try:
            while fil.is_alive():
                if aff is not None and not aff.ferme:
                    aff.traiter_file()
                    aff.dessiner(args.etat, time.time() - t_dep, total, args.debordements)
                elif args.live:                        # fenêtre fermée : plus rien à montrer
                    args.arret.set()
                    fil.join()
                else:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            print("Arrêt demandé" + (" …" if args.live else " : fin de l'enregistrement en cours…"))
            args.arret.set()
            fil.join()
        if aff is not None and not aff.ferme:
            aff.traiter_file()
            aff.dessiner(args.etat, time.time() - t_dep, total, args.debordements, forcer=True)
        if analyse is not None:
            analyse.fermer()
        if args.live:
            print("Session --live terminée, aucun fichier écrit.")
            return
    else:
        acquisition(args, dec)
    sauvegarder(args, dec)


def sauvegarder(args, dec):
    """Fin d'une capture enregistrée : IQ décimé (--iq), cascades PNG et relevé des raies."""
    iq, fs_out = dec.resultat()
    print(f"IQ décimé : {len(iq)} échantillons à {fs_out:.1f} S/s ({len(iq)/fs_out:.1f} s)")

    debut = datetime.fromtimestamp(args.t_premier, timezone.utc)
    tag = debut.strftime("%Y%m%d_%H%M%S")
    if args.iq:
        import json
        base = f"{args.prefixe}_{tag}_iq_{int(fs_out)}"
        np.save(base + ".npy", iq)
        with open(base + ".json", "w") as fj:
            json.dump(dict(t0_unix=args.t_premier, t0_utc=debut.isoformat(), fs=fs_out,
                           rx=[args.lat, args.lon, args.alt],
                           freq_hz=args.freq, antenne=args.antenne, lna=args.lna, ifgr=args.ifgr,
                           notch_fm=args.notch_fm, simulation=args.simulation), fj, indent=1)
        print(f"IQ sauvegardé : {base}.npy (+ .json)")

    # cascade large (satellites, météores) et cascade fine (avions, porteuse)
    f, t, S = cascade(iq, fs_out, args.resolution)
    titre = f"GRAVES {args.freq/1e6:.4f} MHz — {debut:%Y-%m-%d %H:%M:%S} UTC — {'SIMULATION' if args.simulation else 'RSPdx ant. ' + args.antenne}"
    tracer(f"{args.prefixe}_{tag}_large.png", f, t, S, titre + f" — ±{fs_out/2/1000:.1f} kHz, {args.resolution:g} Hz", fs_out, debut)

    f2, t2, S2 = cascade(iq, fs_out, 0.5)
    fond, raies = rapport_raies(f2, S2)          # raies mesurées à 0,5 Hz : une porteuse n'est pas diluée
    pf = (max(-400.0, min(args.plage_fine)), min(400.0, max(args.plage_fine)))
    tracer_zoom(f"{args.prefixe}_{tag}_fin.png", f2, t2, S2, pf,
                titre + f" — {pf[0]:+.0f} à {pf[1]:+.0f} Hz, 0,5 Hz".replace("-", "−"), debut)

    print(f"Fond de bruit (densité à 0,5 Hz, dB relatifs) : {fond:.1f}")
    if raies:
        print("Raies au-dessus du fond (moyenne sur la capture) :")
        for fr, snr in raies:
            print(f"  {fr:+9.1f} Hz   +{snr:4.1f} dB")
    else:
        print("Aucune raie permanente au-dessus du fond (+6 dB).")
    print(f"Cascades : {args.prefixe}_{tag}_large.png  et  {args.prefixe}_{tag}_fin.png")


if __name__ == "__main__":
    main()
