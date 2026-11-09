#!/usr/bin/env python3
"""
graves_direct.py — analyse en temps réel affichée sur les cascades de graves_capture.py --direct

Utilisé automatiquement par :
    python3 graves_capture.py --direct --analyse [--journal DOSSIER] [--catalogues DOSSIER] …

Trois analyses tournent pendant l'acquisition, avec les fonctions validées hors ligne :
  • AVIONS     : les positions du journal ADS-B (graves_adsb.py journal, lancé à côté) donnent le Doppler
                 prévu de chaque vol ; la corrélation de graves_adsb.correler() est refaite toutes les
                 3 s sur les 5 dernières minutes de la cascade fine → courbe + indicatif des vols confirmés.
  • SATELLITES : (dans un processus séparé, pour ne pas ralentir l'affichage)
                 toutes les 60 s, les objets du catalogue qui traversent la fenêtre sont présélectionnés ;
                 toutes les ~15 s, graves_satellites.identifier() cherche leur écho (filtre adapté +
                 recalage TLE) sur la dernière minute de la cascade rapide → courbe + nom des objets vus.
  • MÉTÉORES   : événements brefs et intenses de la cascade rapide, hors avions/satellites et hors
                 parasites large bande → repère « M » et taux par minute.

C'est un affichage indicatif (latence de quelques secondes) ; la classification de référence reste
celle de graves_echos.py sur la capture enregistrée.
"""
import glob
import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graves_adsb as adsb                                            # noqa: E402

COUL_AVION, COUL_SAT, COUL_MET = "#ffffff", "#00e5ff", "#ff2dfe"


class Namespace:
    def __init__(self, **k):
        self.__dict__.update(k)


def distance_sol_km(lat1, lon1, lat2, lon2):
    """Distance au sol (grand cercle) en km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


# ---------------------------------------------------------------- calcul satellites (processus séparé)
_SAT = {}


def _sat_init(catalogues):
    import multiprocessing as mp
    if mp.parent_process() is not None:           # processus de calcul : Ctrl-C est géré par le programme principal
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    import graves_satellites as gsat
    _SAT.update(gsat=gsat, cat=gsat.charger_catalogue(catalogues), presel=None, t_presel=0.0, prevus=[])


def _sat_travail(img_rap, t_rap, f_rap, off, maintenant, rx):
    """Présélection (toutes les 60 s) puis filtre adapté sur la dernière minute de la cascade rapide.
    Retourne (prévisions à afficher ou None si inchangées, objets identifiés)."""
    gsat, cat = _SAT["gsat"], _SAT["cat"]
    prevus = None
    if _SAT["presel"] is None or maintenant - _SAT["t_presel"] > 60:
        c = gsat.candidats(cat, maintenant - 70, maintenant + 70, rx)
        idx = [i for i, _ in c[:300]]
        _SAT["presel"] = cat.sous_ensemble(idx) if idx else None
        _SAT["t_presel"] = maintenant
        prevus = []
        if _SAT["presel"] is not None:
            n = min(8, len(idx))
            tq, FD, _, _ = gsat.courbes_fines(_SAT["presel"], list(range(n)),
                                              np.array([maintenant - 70, maintenant + 70]), rx)
            prevus = [(_SAT["presel"].objets[k]["nom"], tq, FD[k]) for k in range(n)]
    presel = _SAT["presel"]
    if presel is None:
        return prevus, []
    ok = ~np.isnan(t_rap)
    o = np.argsort(t_rap[ok])                     # identifier() veut des temps croissants
    tr_abs = t_rap[ok][o]
    norm = img_rap[ok][o].T
    p = Namespace(tau_max=8.0, max=300, min_tranches=15, min_forts=4, seuil=8.0, z_min=6.0)
    res = gsat.identifier(presel, f_rap - off, tr_abs - tr_abs[0], norm, tr_abs[0], rx, p, bavard=False)
    ids = []
    for r in res:
        if r["identifie"]:
            r["t_abs"] = tr_abs
            ids.append(r)
    return prevus, ids


def etiquette_avion(r, rx=adsb.STATION):
    """Trois lignes, d'après la dernière position ADS-B :
         indicatif  type
         altitude (ft) · vitesse sol (kt)
         cap suivi (°) · distance au sol au récepteur (km)"""
    d = r.get("dernier") or {}
    l1 = f"{r['flight']}  {r['typ']}" if r.get("typ") else r["flight"]
    l2, l3 = [], []
    if d.get("alt_ft") is not None:
        l2.append(f"{float(d['alt_ft']):,.0f} ft".replace(",", " "))
    if d.get("gs") is not None:
        l2.append(f"{float(d['gs']):.0f} kt")
    if d.get("track") is not None:
        l3.append(f"{float(d['track']) % 360:03.0f}°")
    if d.get("lat") is not None and d.get("lon") is not None:
        l3.append(f"{distance_sol_km(rx[0], rx[1], d['lat'], d['lon']):.0f} km")
    return "\n".join([l1] + [" · ".join(x) for x in (l2, l3) if x])


class AnalyseDirecte:
    def __init__(self, journal=".", catalogues=".", rx=adsb.STATION, horloge=time.time, bavard=True):
        self.journal, self.catalogues, self.rx, self.horloge = journal, catalogues, rx, horloge
        self.bavard = bavard
        # ADS-B
        self.positions = {}                   # hex -> deque de positions (format du journal)
        self.fichier, self.position_fichier = None, 0
        self.t_adsb = 0.0
        self.avions = []                      # résultats de adsb.correler() (identifiés)
        self.n_avions_journal = 0
        # satellites
        self.cat = None
        self.cat_charge = False
        self.sats = []                        # résultats identifiés (gsat.identifier)
        self.sats_prevus = []                 # présélection (nom, tq, fd) pour l'affichage discret
        self.t_sat = 0.0
        self.n_commutations = 0                       # « flashs » de commutation écartés (diagnostic)
        self.t_debut_met = None
        self.exec_sat, self.fut_sat = None, None
        self.fil_av = None
        self.verrou = threading.Lock()
        # météores
        self.meteores = []                    # (t_abs, f_rel, snr)
        self.ligne_analysee = -1
        self.artistes = []

    # ------------------------------------------------------------------ ADS-B
    def _lire_journal(self, maintenant):
        nom = os.path.join(self.journal, datetime.fromtimestamp(maintenant, timezone.utc).strftime("adsb_%Y%m%d.jsonl"))
        if nom != self.fichier:
            self.fichier, self.position_fichier = nom, 0
            if os.path.exists(nom):             # on ne relit que les 10 dernières minutes
                taille = os.path.getsize(nom)
                self.position_fichier = max(0, taille - 4_000_000)
        if not os.path.exists(nom):
            return
        with open(nom) as f:
            f.seek(self.position_fichier)
            if self.position_fichier:
                f.readline()                      # ligne éventuellement coupée
            for ligne in f:
                try:
                    r = json.loads(ligne)
                except ValueError:
                    continue
                if r["t"] < maintenant - 420:
                    continue
                self.positions.setdefault(r["hex"], deque(maxlen=600)).append(r)
            self.position_fichier = f.tell()
        for h in list(self.positions):
            d = self.positions[h]
            while d and d[0]["t"] < maintenant - 420:
                d.popleft()
            if not d:
                del self.positions[h]

    def maj_avions(self, img_fin, t_fin, f_fin, off):
        """img_fin : lignes (la plus récente en premier) × fréquences ; t_fin : instants des lignes.
        Lecture du journal ici (incrémentale, rapide) ; corrélation dans un fil d'exécution, pour ne pas
        figer l'affichage quand le journal compte des centaines d'avions."""
        maintenant = self.horloge()
        if self.fil_av is not None and self.fil_av.is_alive():
            return
        if maintenant - self.t_adsb < 3.0:
            return
        self.t_adsb = maintenant
        self._lire_journal(maintenant)
        ok = ~np.isnan(t_fin)
        if ok.sum() < 30 or not self.positions:
            self.n_avions_journal = len(self.positions)
            return
        instantane = {h: list(d) for h, d in self.positions.items()}
        self.fil_av = threading.Thread(target=self._travail_avions,
                                       args=(instantane, img_fin[ok], t_fin[ok], f_fin - off), daemon=True)
        self.fil_av.start()

    def _travail_avions(self, instantane, img_ok, t_ok, f_rel):
        try:
            pistes = {}
            for h, d in instantane.items():
                p = adsb.completer_vitesses([dict(r) for r in d])
                if len(p) >= 2:
                    pistes[h] = p
            p = Namespace(largeur=400, portee=600, tolerance=3.0, seuil=9.0, score_min=0.04, z_min=5.0)
            res = adsb.correler(img_ok.T, f_rel, t_ok, pistes, self.rx, p)
            o = np.argsort(t_ok)
            avions = []
            for r in res:
                if not r["identifie"]:
                    continue
                r["t_abs_c"], r["fd_c"] = t_ok[o], r["c"]["fd"][o]   # rangée par temps croissant (interpolation)
                r["dernier"] = pistes[r["hex"]][-1]               # dernière position ADS-B : vitesse, altitude, cap
                r["typ"] = next((q["typ"] for q in reversed(instantane[r["hex"]]) if q.get("typ")), "")
                avions.append(r)
            self.n_avions_journal = len(pistes)
            self.avions = avions                                  # remplacement d'un bloc : lecture sûre ailleurs
        except Exception as e:
            if self.bavard:
                print(f"  [direct] corrélation ADS-B : {e}")

    # ------------------------------------------------------------------ satellites
    def _demarrer_satellites(self):
        """Prépare le calcul satellites dans un processus séparé (aucun partage du GIL avec l'affichage) ;
        à défaut, dans un fil d'exécution."""
        self.cat_charge = True
        if not sorted(glob.glob(os.path.join(self.catalogues, "catalogue_*.json"))):
            if self.bavard:
                print("  [direct] pas de catalogue orbital : satellites non analysés")
            return
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        try:
            import multiprocessing as mp
            self.exec_sat = ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"),
                                                initializer=_sat_init, initargs=(self.catalogues,))
        except Exception as e:
            if self.bavard:
                print(f"  [direct] processus satellites impossible ({e}) : calcul dans un fil")
            self.exec_sat = ThreadPoolExecutor(1, initializer=_sat_init, initargs=(self.catalogues,))
        self.cat = True

    def _recuperer_satellites(self):
        f, self.fut_sat = self.fut_sat, None
        try:
            prevus, ids = f.result()
        except Exception as e:
            if self.bavard:
                print(f"  [direct] satellites : {e}")
            return
        with self.verrou:
            if prevus is not None:
                self.sats_prevus = prevus
            self.sats = ids

    def maj_satellites(self, img_rap, t_rap, f_rap, off):
        if not self.cat_charge:
            self._demarrer_satellites()
        if self.cat is None:
            return
        if self.fut_sat is not None:
            if not self.fut_sat.done():
                return
            self._recuperer_satellites()
        if self.horloge() - self.t_sat < 15 or np.sum(~np.isnan(t_rap)) < 300:
            return
        self.t_sat = self.horloge()
        self.fut_sat = self.exec_sat.submit(_sat_travail, img_rap, t_rap, f_rap, off, self.t_sat, self.rx)

    def attendre(self):
        """Attend la fin des calculs en cours et range leurs résultats (rejeu de fichiers, tests)."""
        if self.fil_av is not None:
            self.fil_av.join()
        if self.fut_sat is not None:
            self.fut_sat.result()
            self._recuperer_satellites()

    def fermer(self):
        if self.exec_sat is not None:
            try:
                self.exec_sat.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    # ------------------------------------------------------------------ météores
    def maj_meteores(self, img_rap, t_rap, f_rap, niv_rap, off, n_lignes):
        """Analyse les lignes âgées de 1,6 à 6 s (l'événement est terminé), une seule fois chacune."""
        dt = np.nanmedian(np.diff(t_rap[:50])) if np.sum(~np.isnan(t_rap[:50])) > 5 else None
        if dt is None or dt == 0:
            return
        if self.t_debut_met is None:
            self.t_debut_met = self.horloge()
        if self.horloge() - self.t_debut_met < 40:    # position de la porteuse pas encore établie (~30 s)
            return
        dt = abs(dt)
        a, b = int(1.6 / dt), int(6.0 / dt)
        if b >= len(t_rap) or np.isnan(t_rap[b]):
            return
        bloc = img_rap[a:b].copy()                # lignes récentes en premier
        tb = t_rap[a:b]
        f_rel = f_rap - off
        # autour de la porteuse : référence = niveau « porteuse présente » (90e centile sur la minute),
        # pour que ses fluctuations (battement avec un avion qui la croise) ne passent pas pour des météores
        pc = np.abs(f_rel) <= 12
        ok = ~np.isnan(t_rap)
        if pc.any() and ok.sum() > 50:
            bloc[:, pc] -= np.percentile(img_rap[ok][:, pc], 90, axis=0)[None, :]
        df = abs(f_rap[1] - f_rap[0])
        masque = bloc >= 10.0
        # lignes perturbées (parasite large bande) : ignorées
        fond = np.nanmedian(niv_rap[~np.isnan(niv_rap)]) if np.any(~np.isnan(niv_rap)) else 0
        masque[(niv_rap[a:b] - fond) > 3.0] = False
        # avions et satellites identifiés : retirés
        with self.verrou:
            sats = list(self.sats)
        for r in self.avions:
            fd = np.interp(tb, r["t_abs_c"], r["fd_c"], left=np.nan, right=np.nan)
            masque &= ~(np.abs(f_rel[None, :] - np.nan_to_num(fd, nan=1e9)[:, None]) <= 16)
        for r in sats:
            fd = np.interp(tb, r["t_abs"], r["F"], left=np.nan, right=np.nan)
            masque &= ~(np.abs(f_rel[None, :] - np.nan_to_num(fd, nan=1e9)[:, None]) <= 40)
        lab, n = ndimage.label(ndimage.binary_closing(masque, np.ones((3, 1), bool)) | masque,
                               structure=np.ones((3, 3), bool))
        for sl in ndimage.find_objects(lab):
            if sl is None:
                continue
            zone = bloc[sl] * (lab[sl] > 0)
            pic = zone.max()
            if pic < 15:
                continue
            duree = (sl[0].stop - sl[0].start) * dt
            if duree > 1.5 or sl[0].start == 0:   # trop long, ou pas encore terminé
                continue
            if sl[0].stop >= len(bloc):            # commencé avant la fenêtre : seule sa fin est visible
                continue                           # (un écho de 3,2 s tronqué passerait pour un éclair bref)
            i, j = np.unravel_index(np.argmax(zone), zone.shape)
            t_ev, f_ev = tb[sl[0].start + i], f_rel[sl[1].start + j]
            etendue = (sl[1].stop - sl[1].start) * abs(f_rap[1] - f_rap[0])
            if abs(f_ev) > 200 and etendue < 150:  # ni traînée (près de 0 Hz) ni tête (large) : on s'abstient
                continue
            # commutation de faisceau : GRAVES change de direction toutes les 3,2 s ; tout signal fort qu'il
            # éclaire (porteuse directe, écho d'avion proche) change alors brutalement de niveau et de phase,
            # ce qui étale son spectre pendant une ligne, SYMÉTRIQUEMENT de part et d'autre de ce signal.
            # Un « flash » voisin d'un signal fort qui saute de niveau au même instant, et présent aussi à
            # la fréquence miroir, n'est pas un météore (une traînée, elle, n'est que d'un côté).
            r = a + sl[0].start + i                           # ligne de l'événement dans img_rap
            # durée mesurée sur la cascade brute, à la fréquence du pic : un écho d'avion non identifié qui
            # traverse la zone de la porteuse est découpé en morceaux par les traitements ci-dessus ; sur
            # le signal brut il dure, lui, plusieurs secondes
            if self._duree_brute(img_rap, r, f_rel, f_ev, df) * dt > 1.5:
                continue
            if self._eclaboussure(img_rap, r, f_rel, f_ev, df):
                self.n_commutations += 1
                continue
            # même fréquence dans les 8 s : la même traînée qui s'évanouit et revient (surdense)
            if any(abs(t_ev - m[0]) < 8 and abs(f_ev - m[1]) < 40 for m in self.meteores):
                continue
            self.meteores.append((float(t_ev), float(f_ev), float(pic)))
        self.meteores = [m for m in self.meteores if m[0] > self.horloge() - 3600]

    @staticmethod
    def _duree_brute(img_rap, r, f_rel, f_ev, df):
        """Nombre de lignes consécutives, autour de la ligne r, où le signal reste fort à la fréquence f_ev."""
        cases = np.abs(f_rel - f_ev) <= 1.5 * df
        col = img_rap[:, cases].max(axis=1)
        seuil = max(8.0, col[r] - 10.0)
        haut = r
        while haut > 0 and col[haut - 1] >= seuil:
            haut -= 1
        bas = r
        while bas < len(col) - 1 and col[bas + 1] >= seuil:
            bas += 1
        return bas - haut + 1

    @staticmethod
    def _eclaboussure(img_rap, r, f_rel, f_ev, df, portee=300.0):
        """Vrai si l'événement (ligne r, fréquence f_ev) est l'étalement spectral d'un signal fort voisin
        au moment où celui-ci change brusquement de niveau (commutation de faisceau de GRAVES)."""
        lignes = img_rap[max(0, r - 1): r + 2]
        ligne = img_rap[r]
        ici = ligne[np.abs(f_rel - f_ev) <= 1.5 * df].max()
        zone = (np.abs(f_rel - f_ev) <= portee) & (np.abs(f_rel - f_ev) > 1.5 * df)
        if not zone.any():
            return False
        # signaux « parents » possibles : la porteuse directe, et le signal le plus fort du voisinage
        parents = [int(np.argmin(np.abs(f_rel)))]
        parents.append(int(np.where(zone)[0][np.argmax(lignes[:, zone].max(axis=0))]))
        for j in parents:
            if abs(f_rel[j] - f_ev) <= 1.5 * df or abs(f_rel[j] - f_ev) > portee:
                continue
            voisins = np.abs(f_rel - f_rel[j]) <= 1.5 * df
            niv = img_rap[max(0, r - 2): r + 3][:, voisins].max(axis=1)    # son niveau autour de l'instant
            if niv.size < 2 or np.ptp(niv) < 6.0 or niv.max() < ici - 3.0:
                continue
            miroir = ligne[np.abs(f_rel - (2 * f_rel[j] - f_ev)) <= 1.5 * df]
            if miroir.size and miroir.max() >= ici - 6.0:
                return True
        return False

    def taux_meteores(self, minutes=10):
        t = self.horloge()
        n = sum(1 for m in self.meteores if m[0] > t - 60 * minutes)
        return n / minutes

    # ------------------------------------------------------------------ dessin
    def elements(self, t_fin, t_rap, off):
        """Repères à superposer aux cascades, indépendants de la bibliothèque graphique.
        Chaque élément : dict(type="ligne"|"texte"|"cercle", axe="fin"|"rap", x, y, couleur, …) ;
        y = âge en secondes (0 = maintenant, en haut)."""
        el = []
        if np.isnan(t_fin[0]) and np.isnan(t_rap[0]):
            return el
        haut_fin, haut_rap = t_fin[0], t_rap[0]
        # avions : courbe prévue (8 Hz à droite de l'écho) + étiquette
        if not np.isnan(haut_fin):
            ok = ~np.isnan(t_fin)
            ty = t_fin[ok]
            for n_av, r in enumerate(self.avions):
                fd = np.interp(ty, r["t_abs_c"], r["fd_c"], left=np.nan, right=np.nan)
                y = haut_fin - ty
                x = fd + off + 8
                v = ~np.isnan(x) & (np.abs(x) <= 400)
                if v.sum() < 3:
                    continue
                el.append(dict(type="ligne", axe="fin", x=x[v], y=y[v], couleur=COUL_AVION, largeur=1.1,
                               tirets=(4, 5)))
                iv = np.where(v)[0]
                k = iv[min(len(iv) - 1, n_av * 25)]          # étiquettes décalées dans le temps : pas de chevauchement
                el.append(dict(type="texte", axe="fin", x=x[k] + 6, y=y[k], texte=etiquette_avion(r, self.rx),
                               couleur=COUL_AVION, taille=6.5, gras=True, va="top", fond=True))
        # satellites
        with self.verrou:
            sats, prevus = list(self.sats), list(self.sats_prevus)
        if not np.isnan(haut_rap):
            ok = ~np.isnan(t_rap)
            ty = t_rap[ok]
            for nom, tq, fd in prevus:                # prévisions discrètes (où regarder)
                x = np.interp(ty, tq, fd, left=np.nan, right=np.nan) + off
                v = ~np.isnan(x) & (np.abs(x) <= 5500)
                if v.sum() > 3:
                    el.append(dict(type="ligne", axe="rap", x=x[v] + 120, y=(haut_rap - ty)[v], couleur=COUL_SAT,
                                   largeur=0.7, tirets=(1, 2), alpha=0.5))
            for n_s, r in enumerate(sats):            # objets confirmés
                x = np.interp(ty, r["t_abs"] + 0.0, r["F"], left=np.nan, right=np.nan) + off
                v = ~np.isnan(x)
                if v.sum() > 3:
                    el.append(dict(type="ligne", axe="rap", x=x[v] + 120, y=(haut_rap - ty)[v], couleur=COUL_SAT,
                                   largeur=1.3, tirets=(4, 4)))
                    iv = np.where(v)[0]
                    k = iv[min(len(iv) - 1, n_s * 80)]
                    el.append(dict(type="texte", axe="rap", x=x[k] + 150, y=haut_rap - ty[k],
                                   texte=f"{r['nom']} ({r['norad']})", couleur=COUL_SAT, taille=6.5, gras=True,
                                   va="top", fond=True))
            # météores
            for t_ev, f_ev, pic in self.meteores:
                age = haut_rap - t_ev
                if 0 <= age <= haut_rap - np.nanmin(t_rap):
                    el.append(dict(type="cercle", axe="rap", x=f_ev + off, y=age, couleur=COUL_MET))
                    el.append(dict(type="texte", axe="rap", x=f_ev + off + 150, y=age, texte="M", couleur=COUL_MET,
                                   taille=7, gras=True, va="center", fond=False))
        return el

    def dessiner(self, a1, a2, t_fin, t_rap, off):
        """Version matplotlib (graves_capture.py) des repères."""
        for art in self.artistes:
            try:
                art.remove()
            except Exception:
                pass
        self.artistes = []
        for e in self.elements(t_fin, t_rap, off):
            ax = a1 if e["axe"] == "fin" else a2
            if e["type"] == "ligne":
                ls = ":" if e["tirets"] == (1, 2) else (0, e["tirets"])
                self.artistes += ax.plot(e["x"], e["y"], color=e["couleur"], lw=e["largeur"], ls=ls,
                                         alpha=e.get("alpha", 1.0))
            elif e["type"] == "texte":
                bbox = dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0) if e["fond"] else None
                self.artistes.append(ax.annotate(e["texte"], (e["x"], e["y"]), color=e["couleur"], fontsize=e["taille"],
                                                 fontweight="bold" if e["gras"] else "normal", va=e["va"], bbox=bbox))
            else:
                self.artistes += ax.plot([e["x"]], [e["y"]], "o", mfc="none", mec=e["couleur"], ms=9, mew=1.2)

    def resume(self):
        with self.verrou:
            ns = len(self.sats)
        avions = ", ".join(r["flight"] for r in self.avions[:6]) or "—"
        sat = (" | satellites : " + (", ".join(r["nom"] for r in self.sats[:3]) if ns else "—")) if self.cat else ""
        return (f"avions ({self.n_avions_journal} au journal) : {avions}{sat} | "
                f"météores : {sum(1 for m in self.meteores if m[0] > self.horloge() - 600)} en 10 min "
                f"({self.taux_meteores():.1f}/min)")
