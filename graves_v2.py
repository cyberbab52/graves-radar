#!/usr/bin/env python3
"""
graves_v2.py — V2 de l'affichage en direct, avec pyqtgraph (Qt 6)

Même acquisition, mêmes options et même analyse que graves_capture.py ; seul l'affichage change :
pyqtgraph est conçu pour les cascades temps réel (images mises à jour en quelques ms, repères
réutilisés d'une image à l'autre), là où matplotlib redessine tout. Cadence par défaut : 15 images/s.

Usage (options identiques à graves_capture.py) :
  python3 graves_v2.py --live --analyse                # cascade en direct + identification, aucun fichier
  python3 graves_v2.py --duree 300 --iq --analyse      # idem, avec enregistrement (IQ + PNG en fin de capture)
  python3 graves_v2.py --simulation --live --analyse   # essai sans matériel
Options propres à la V2 : --ips (défaut 15), --opengl (rendu OpenGL, nécessite PyOpenGL)

Bouton « Capture » (ou F12) : image PNG de la fenêtre, nommée graves_ecran_AAAAMMJJ_HHMMSS.png (UTC),
dans le dossier courant — aussi en mode --live.
Fermer la fenêtre ou Ctrl-C arrête l'acquisition ; en mode enregistrement, ce qui a été capté
jusque-là est sauvegardé normalement.

Dépendances supplémentaires (venv) : PyQt6, pyqtgraph ; système : libxcb-cursor0.
"""
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graves_capture as gc                                            # noqa: E402

TITRE_FENETRE = "GRAVES — cascade en direct (V2)"


# ------------------------------------------------------------------------------- repères (pyqtgraph)
class Calque:
    """Repères d'une cascade (courbes, étiquettes, cercles), réutilisés d'une image à l'autre :
    créer et détruire des objets Qt à chaque image coûterait plus cher que de les déplacer."""

    def __init__(self, pg, QtCore, QtGui, plot):
        self.pg, self.QtCore, self.QtGui, self.plot = pg, QtCore, QtGui, plot
        self.lignes, self.textes = {}, {}             # style -> liste d'objets
        self.cercles = pg.ScatterPlotItem(pxMode=True)
        self.cercles.setZValue(20)
        plot.addItem(self.cercles)

    def _stylo(self, e):
        c = self.QtGui.QColor(e["couleur"])
        c.setAlphaF(float(e.get("alpha", 1.0)))
        p = self.pg.mkPen(c, width=e["largeur"])
        p.setDashPattern([float(v) for v in e["tirets"]])
        p.setCosmetic(True)
        return p

    def afficher(self, elements):
        utilises_l, utilises_t, cercles = {}, {}, []
        for e in elements:
            if e["type"] == "ligne":
                cle = (e["couleur"], e["largeur"], tuple(e["tirets"]), e.get("alpha", 1.0))
                k = utilises_l.get(cle, 0)
                pool = self.lignes.setdefault(cle, [])
                if k == len(pool):
                    it = self.pg.PlotCurveItem(pen=self._stylo(e))
                    it.setZValue(10)
                    self.plot.addItem(it)
                    pool.append(it)
                pool[k].setData(np.asarray(e["x"], float), np.asarray(e["y"], float))
                pool[k].setVisible(True)
                utilises_l[cle] = k + 1
            elif e["type"] == "texte":
                cle = (e["couleur"], e["taille"], e["gras"], e["va"], e["fond"])
                k = utilises_t.get(cle, 0)
                pool = self.textes.setdefault(cle, [])
                if k == len(pool):
                    it = self.pg.TextItem(color=e["couleur"], anchor=(0, 0 if e["va"] == "top" else 0.5),
                                          fill=self.pg.mkBrush(0, 0, 0, 160) if e["fond"] else None)
                    f = self.QtGui.QFont()
                    f.setPixelSize(max(8, round(e["taille"] * 1.45)))
                    f.setBold(e["gras"])
                    it.setFont(f)
                    it.setZValue(30)
                    self.plot.addItem(it)
                    pool.append(it)
                pool[k].setText(e["texte"])
                pool[k].setPos(float(e["x"]), float(e["y"]))
                pool[k].setVisible(True)
                utilises_t[cle] = k + 1
            elif e["type"] == "cercle":
                cercles.append(dict(pos=(float(e["x"]), float(e["y"])), size=13, symbol="o",
                                    pen=self.pg.mkPen(e["couleur"], width=1.5), brush=None))
        for cle, pool in self.lignes.items():         # objets inutilisés : masqués, gardés pour plus tard
            for it in pool[utilises_l.get(cle, 0):]:
                it.setVisible(False)
        for cle, pool in self.textes.items():
            for it in pool[utilises_t.get(cle, 0):]:
                it.setVisible(False)
        self.cercles.setData(cercles)


# ------------------------------------------------------------------------------- fenêtre
def creer_fenetre(args, moteur, analyse, fil, t_dep, total):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    class Fenetre(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(TITRE_FENETRE)
            self.resize(*args.taille)
            centre = QtWidgets.QWidget()
            centre.setStyleSheet("background-color: black;")
            vbox = QtWidgets.QVBoxLayout(centre)
            vbox.setContentsMargins(4, 4, 4, 4)
            vbox.setSpacing(2)
            self.titre = QtWidgets.QLabel(gc.titre_direct(args))
            self.titre.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.titre.setStyleSheet("color: #e8e8e8; font-size: 13px;")
            # bandeau du haut : titre centré, bouton de capture d'écran à droite
            bandeau = QtWidgets.QHBoxLayout()
            self.bouton = QtWidgets.QPushButton("Capture")
            self.bouton.setToolTip("Enregistre une image de la fenêtre (raccourci : F12)")
            self.bouton.setStyleSheet(
                "QPushButton { color: #e8e8e8; background-color: #303030; border: 1px solid #606060;"
                " border-radius: 4px; padding: 4px 12px; font-size: 12px; }"
                "QPushButton:hover { background-color: #454545; }"
                "QPushButton:pressed { background-color: #606060; }")
            self.bouton.clicked.connect(self.capturer_ecran)
            QtGui.QShortcut(QtGui.QKeySequence("F12"), self, activated=self.capturer_ecran)
            self.message = QtWidgets.QLabel("")       # confirmation d'enregistrement, à gauche
            self.message.setStyleSheet("color: #8fd18f; font-size: 11px;")
            self.message.setFixedWidth(300)
            droite = QtWidgets.QWidget()               # même largeur à droite : le titre reste centré
            droite.setFixedWidth(300)
            hd = QtWidgets.QHBoxLayout(droite)
            hd.setContentsMargins(0, 0, 0, 0)
            hd.addStretch(1)
            hd.addWidget(self.bouton)
            bandeau.addWidget(self.message)
            bandeau.addWidget(self.titre, 1)
            bandeau.addWidget(droite)
            vbox.addLayout(bandeau)
            self.glw = pg.GraphicsLayoutWidget()
            vbox.addWidget(self.glw, 1)
            self.setCentralWidget(centre)

            self.p1 = self.glw.addPlot(row=0, col=0)
            self.p2 = self.glw.addPlot(row=0, col=1)
            lay = self.glw.ci.layout
            lay.setColumnStretchFactor(0, 14)          # la cascade fine (avions) est la plus utile
            lay.setColumnStretchFactor(1, 10)
            pf = moteur.plage_fine
            self.p1.setTitle(f"Fine {pf[0]:+.0f} à {pf[1]:+.0f} Hz (0,5 Hz) — avions, porteuse".replace("-", "−"),
                             size="10pt")
            self.p2.setTitle("Rapide ±5,5 kHz (8 Hz, 60 ms) — météores, satellites", size="10pt")
            for p in (self.p1, self.p2):
                p.setLabel("bottom", "Décalage / 143,050 MHz (Hz)")
                p.invertY(True)                        # 0 s (maintenant) en haut, comme une cascade
                p.setMouseEnabled(False, False)
                p.setMenuEnabled(False)
                p.hideButtons()
                p.enableAutoRange(False)
            self.p1.setLabel("left", "Il y a… (s)")
            # légende des couleurs : petite, sous la cascade rapide
            self.barre = pg.ColorBarItem(values=(0, moteur.dyn), colorMap=pg.colormap.get("magma"),
                                         orientation="horizontal", interactive=False, width=10,
                                         label="dB au-dessus du fond")
            sous = self.glw.addLayout(row=1, col=1)   # barre centrée, sur la moitié de la largeur
            sous.addLabel("", col=0)
            sous.addItem(self.barre, col=1)
            sous.addLabel("", col=2)
            for c, f in ((0, 1), (1, 2), (2, 1)):
                sous.layout.setColumnStretchFactor(c, f)
            lay.setRowStretchFactor(0, 30)
            lay.setRowStretchFactor(1, 1)
            self.im1 = self.im2 = None
            self.analyse = analyse
            self.fini = False
            self.minuterie = QtCore.QTimer()
            self.minuterie.timeout.connect(self.tick)
            self.minuterie.start(max(10, int(1000 / args.ips)))

        def _creer_images(self):
            d_fin = moteur.l_fin * moteur.h_fin / moteur.fs
            d_rap = moteur.l_rap * moteur.h_rap / moteur.fs
            fa = moteur.f_fin[moteur.m_aff]
            self.im1 = pg.ImageItem(axisOrder="row-major")
            self.im2 = pg.ImageItem(axisOrder="row-major")
            self.p1.addItem(self.im1)
            self.p2.addItem(self.im2)
            self.im1.setImage(moteur.rgb_fin, autoLevels=False)   # l'image doit exister avant setRect
            self.im2.setImage(moteur.rgb_rap, autoLevels=False)
            self.im1.setRect(QtCore.QRectF(fa[0], 0, fa[-1] - fa[0], d_fin))
            self.im2.setRect(QtCore.QRectF(moteur.f_rap[0], 0, moteur.f_rap[-1] - moteur.f_rap[0], d_rap))
            self.p1.setXRange(*moteur.plage_fine, padding=0)
            self.p1.setYRange(0, d_fin, padding=0)
            self.p2.setXRange(moteur.f_rap[0], moteur.f_rap[-1], padding=0)
            self.p2.setYRange(0, d_rap, padding=0)
            self.mark = pg.InfiniteLine(0, angle=90, pen=pg.mkPen((255, 255, 255, 130), width=1,
                                                                   style=QtCore.Qt.PenStyle.DotLine))
            self.mark.setZValue(5)
            self.p1.addItem(self.mark)
            self.calques = (Calque(pg, QtCore, QtGui, self.p1), Calque(pg, QtCore, QtGui, self.p2))

        def tick(self):
            moteur.traiter_file()
            termine = not fil.is_alive()
            if moteur.pret:
                if self.im1 is None:
                    self._creer_images()
                self.im1.setImage(moteur.rgb_fin, autoLevels=False)
                self.im2.setImage(moteur.rgb_rap, autoLevels=False)
                off, info = moteur.porteuse()
                if info:
                    self.mark.setValue(off)
                ligne2 = ""
                if self.analyse is not None:
                    try:
                        a = self.analyse
                        a.maj_avions(moteur.img_fin, moteur.t_fin, moteur.f_fin, off)
                        a.maj_satellites(moteur.img_rap, moteur.t_rap, moteur.f_rap, off)
                        a.maj_meteores(moteur.img_rap, moteur.t_rap, moteur.f_rap, moteur.niv_rap, off,
                                       moteur.n_lignes_rap)
                        el = a.elements(moteur.t_fin, moteur.t_rap, off)
                        self.calques[0].afficher([e for e in el if e["axe"] == "fin"])
                        self.calques[1].afficher([e for e in el if e["axe"] == "rap"])
                        ligne2 = "\n" + a.resume()
                    except Exception as e:
                        ligne2 = f"\nanalyse en direct indisponible : {e}"
                ecoule = time.time() - t_dep
                duree = f"{ecoule:.0f}/{total:.0f} s" if np.isfinite(total) else f"{ecoule:.0f} s"
                deb = f" — débordements USB : {args.debordements}" if args.debordements else ""
                self.titre.setText(f"{gc.titre_direct(args)} — {args.etat} {duree}{deb}{info}{ligne2}")
            if termine and not self.fini:              # acquisition finie (durée atteinte) : on ferme
                self.fini = True
                self.minuterie.stop()
                QtCore.QTimer.singleShot(300, self.close)

        def capturer_ecran(self):
            """Image PNG de la fenêtre entière, nommée d'après l'heure UTC (comme les captures IQ)."""
            t = datetime.now(timezone.utc)
            nom = f"{args.prefixe}_ecran_{t:%Y%m%d_%H%M%S}.png"
            ok = self.grab().save(nom, "PNG")
            texte = f"Capture enregistrée : {nom}" if ok else f"Échec de l'enregistrement de {nom}"
            print(texte)
            self.message.setText(texte)
            self.message.setStyleSheet(f"color: {'#8fd18f' if ok else '#ff7070'}; font-size: 11px;")
            QtCore.QTimer.singleShot(5000, lambda: self.message.setText(""))

        def closeEvent(self, evt):
            args.arret.set()                           # fermer la fenêtre arrête l'acquisition
            self.minuterie.stop()
            evt.accept()

    return pg, QtWidgets, Fenetre


def main():
    p = gc.creer_parseur(__doc__)
    p.set_defaults(ips=15.0)
    p.add_argument("--opengl", action="store_true",
                   help="rendu OpenGL de pyqtgraph (nécessite PyOpenGL ; à essayer si l'affichage reste lent)")
    args = gc.preparer(p.parse_args())
    if args.liste:
        gc.lister()
        return

    debut = datetime.now(timezone.utc)
    args.t_premier = debut.timestamp()
    args.arret, args.etat, args.debordements = threading.Event(), "démarrage", 0
    dec = gc.Decimateur(args.decalage)
    dec.garder = not args.live
    acquisition = gc.simuler if args.simulation else gc.capturer
    if not args.direct:                                # sans fenêtre : comportement de la V1
        acquisition(args, dec)
        gc.sauvegarder(args, dec)
        return

    try:
        import pyqtgraph  # noqa: F401
        from pyqtgraph.Qt import QtWidgets  # noqa: F401
    except Exception as e:
        sys.exit(f"pyqtgraph / PyQt6 indisponibles ({e}) : pip install PyQt6 pyqtgraph, ou utiliser graves_capture.py")

    analyse = gc.creer_analyse(args)
    moteur = gc.Afficheur(gc.titre_direct(args), analyse=analyse, plage_fine=args.plage_fine,
                          porteuse=args.porteuse)   # calcul des cascades
    dec.ecouteur = moteur.recevoir
    fil = threading.Thread(target=acquisition, args=(args, dec), daemon=True)
    total = args.duree + (0 if args.simulation else args.prechauffage)

    pg, QtWidgets, Fenetre = creer_fenetre(args, moteur, analyse, fil, time.time(), total)
    pg.setConfigOptions(antialias=False, imageAxisOrder="row-major", useOpenGL=args.opengl)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    # Qt adopte la locale du système (fr_FR : virgule décimale) pour tout le processus ; les bibliothèques C
    # (SoapySDR, API SDRplay) qui convertissent des nombres en texte risqueraient alors de mal lire les
    # fréquences et réglages. On rétablit la convention « C » (point décimal) pour les nombres.
    import locale
    locale.setlocale(locale.LC_NUMERIC, "C")
    fen = Fenetre()
    # Ctrl-C : Qt garde la main pendant app.exec(), le minuteur rend régulièrement la main à Python
    signal.signal(signal.SIGINT, lambda *a: (print("Arrêt demandé…"), fen.close()))
    fen.show()
    if args.live:
        print("Mode --live : aucun fichier ne sera écrit. Fermer la fenêtre ou Ctrl-C pour arrêter.")
    fil.start()
    app.exec()

    args.arret.set()
    fil.join()
    if analyse is not None:
        analyse.fermer()
    if args.live:
        print("Session --live terminée, aucun fichier écrit.")
        return
    gc.sauvegarder(args, dec)


if __name__ == "__main__":
    main()
