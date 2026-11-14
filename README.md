# graves-radar

Réception passive et classification automatique des échos du radar **GRAVES** (143,050 MHz) avec un récepteur SDR : avions, satellites, météores, Lune.

*English version below.*

---

## Le principe

GRAVES est le radar français de surveillance spatiale. Son émetteur, situé près de Broye-Aubigney-Montseugny (Haute-Saône), émet une porteuse continue à 143,050 MHz en balayant le ciel du sud par secteurs (commutation toutes les 3,2 s). Tout objet éclairé (avion, satellite, traînée ionisée d'un météore, Lune) renvoie un peu de ce signal. Un récepteur situé à quelques centaines de kilomètres le capte, décalé par effet Doppler : c'est une configuration **bistatique**.

Le décalage observé dépend de la vitesse de l'objet par rapport aux deux trajets émetteur → objet et objet → récepteur :

```
f_D = −(1/λ) · v⃗ · (û_GRAVES→objet + û_récepteur→objet),    λ = c / 143,050 MHz ≈ 2,096 m
```

Ces scripts vont au-delà de l'affichage d'une cascade. Ils **identifient** les échos en confrontant chaque trace à ce que la physique prévoit :

- **Avions** : le Doppler bistatique de chaque vol est calculé à partir de ses positions ADS-B (récepteur local readsb et/ou API adsb.fi). On mesure ensuite si un écho suit cette courbe, en comparant avec des courbes « leurres » décalées en fréquence pour estimer la significativité (z).
- **Satellites** : propagation SGP4 de tout le catalogue orbital (Space-Track ou Celestrak), puis **filtre adapté** : l'énergie est sommée le long de la courbe Doppler prévue, sur le meilleur tronçon de 60 s (un satellite n'est éclairé que pendant sa traversée du volume couvert par GRAVES), avec recherche du meilleur recalage temporel (erreur des TLE). Les courbes leurres ignorent la trace du candidat lui-même, et chaque case de la cascade est plafonnée (+12 dB) pour qu'un écho d'avion très fort, croisé brièvement, ne crée pas de fausse identification. Une attribution exclusive évite de confondre deux objets voisins (trains de Starlink). Verdicts : écho identifié, ou « candidat » (détection probable, à vérifier sur l'image).
- **Météores** : événements brefs de la cascade rapide (écho de tête, traînée sous-dense ou surdense), après élimination des avions, des satellites, des parasites large bande et des « éclaboussures » spectrales liées à la commutation des faisceaux. Les morceaux d'écho d'avion qui reviennent à la même fréquence au rythme du balayage (multiples de 3,2 s) sont regroupés et rattachés au vol ADS-B correspondant.
- **Dérive du récepteur** : l'oscillateur du SDR dérive en chauffant. La porteuse directe de GRAVES sert de référence pour la corriger.
- **Référence GPS** (facultative) : un récepteur GPS logiciel, sur l'entrée A du RSPdx, mesure la position du récepteur, l'écart de son oscillateur (en ppm, donc la position attendue de la porteuse GRAVES) et l'écart de l'horloge du PC. Utile sur le terrain, sans réseau.

## Les scripts

| Script | Rôle |
|---|---|
| `graves_capture.py` | Étape 1 : acquisition IQ (SoapySDR), décimation, cascades PNG, cascade en direct (matplotlib), mode simulation sans matériel |
| `graves_v2.py` | Affichage en direct V2 (pyqtgraph, ~15 images/s), mêmes options que `graves_capture.py`, capture d'écran F12, boutons + / − pour répartir la largeur entre les deux cascades, échelle de couleurs réglable (`--dyn`) |
| `graves_gps.py` | Calage GPS sur l'entrée A (récepteur GPS logiciel L1 C/A) : position, écart de l'oscillateur du SDR, écart de l'horloge du PC (`--regler-horloge` pour la corriger sans réseau). La référence `graves_gps.json` est lue automatiquement par `graves_capture.py` et `graves_v2.py` |
| `graves_doppler.py` | Doppler attendu pour la Lune et un satellite donné (prévisions) |
| `graves_adsb.py` | Étape 3 : journal ADS-B continu, puis corrélation avion par avion sur une capture |
| `graves_echos.py` | Étape 4 : détection et classification de tous les échos d'une capture (tableau, image annotée, journal CSV) |
| `graves_satellites.py` | Étape 5 : catalogue orbital, prévision des passages, identification des satellites sur une capture |
| `graves_direct.py` | Module d'analyse en temps réel (avions, satellites, météores) utilisé par `--analyse` |

## Matériel

Configuration de développement, à adapter :

- Antenne : discone pour les essais à domicile, Yagi 2 m orientée vers le sud pour les sessions en extérieur
- Préamplificateur faible bruit (LNA), utile avec une antenne sous un toit. Sur le terrain, avec une vue dégagée sur GRAVES, un préampli large bande sature sous l'effet des émetteurs FM (placés *avant* le filtre FM du RSPdx) : s'en passer, ou le faire précéder d'un filtre passe-bande 143 MHz
- Pour `graves_gps.py` : antenne GPS active sur l'entrée A, alimentée par un bias-T externe (3 V pour une antenne 3 V ; le Bias-T interne du RSPdx, 4,7 V sur l'entrée B, ne convient pas)
- SDR : **SDRplay RSPdx**, piloté par SoapySDR. D'autres SDR compatibles SoapySDR devraient fonctionner moyennant quelques adaptations dans `capturer()`.
- Ordinateur : NVIDIA Jetson Orin Nano sous Ubuntu 24.04 (tout Linux récent convient)
- Optionnel : récepteur ADS-B local exposant un `aircraft.json` (readsb, dump1090)

## Installation

```bash
git clone https://github.com/cyberbab52/graves-radar.git
cd graves-radar
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pour l'acquisition réelle, il faut aussi installer l'API SDRplay, SoapySDR et le module [SoapySDRPlay3](https://github.com/pothosware/SoapySDRPlay3). Le mode `--simulation` fonctionne sans aucun matériel.

## Configuration

**Position du récepteur** : `~/.config/graves/station.ini`. Sans ce fichier, une position d'exemple (Paris) est utilisée avec un avertissement : les Doppler prévus seraient alors faux. On peut aussi passer `--lat --lon --alt` en ligne de commande.

```ini
[station]
nom = Maison
lat = 48.1000
lon = 5.1000
alt = 300          ; altitude de l'antenne (m), sol + hauteur du mât
```

**Identifiants Space-Track** (facultatifs, pour le catalogue complet) : variables d'environnement `SPACETRACK_USER` et `SPACETRACK_PASSWORD`, ou fichier `~/.config/graves/spacetrack.ini` en droits 600 :

```ini
[spacetrack]
identity = votre@adresse
password = votre_mot_de_passe
```

Sans compte Space-Track, `--source celestrak` télécharge les satellites actifs.

## Utilisation

```bash
# Essai sans matériel
python3 graves_capture.py --simulation --duree 130 --iq
python3 graves_echos.py graves_AAAAMMJJ_HHMMSS_iq_16666.npy --sans-adsb --sans-satellites

# Session réelle
python3 graves_satellites.py catalogue                           # une fois par jour (avec Internet)
python3 graves_gps.py --test-antenne                             # facultatif : antenne GPS sur l'entrée A
python3 graves_gps.py                                            # facultatif : calage (position, oscillateur, horloge)
python3 graves_adsb.py journal --source local --source adsbfi   # terminal 1 : journal ADS-B
python3 graves_adsb.py journal --terrain                         # … ou sur le terrain (connexion mobile : adsb.fi,
                                                                 #     gzip, 1 requête / 3 s, rayon 200 NM)
python3 graves_v2.py --duree 900 --iq --analyse                  # terminal 2 : capture + affichage en direct

# Analyse a posteriori
python3 graves_echos.py   graves_..._iq_16666.npy                # classification complète
python3 graves_adsb.py    superpose graves_..._iq_16666.npy      # détail des avions
python3 graves_satellites.py superpose graves_..._iq_16666.npy   # détail des satellites

# Prévisions
python3 graves_satellites.py passages --heures 6
python3 graves_doppler.py lune --heures 24
```

Chaque script affiche toutes ses options avec `-h`. Quelques options utiles :

- `graves_v2.py --dyn 35` : échelle de couleurs jusqu'à +35 dB quand les échos sont très forts ; `--ips 8` si le processeur peine (débordements USB) ; `--porteuse HZ` pour imposer la position de la porteuse directe.
- `graves_satellites.py superpose … --tous` : trace aussi les 30 meilleures courbes non retenues. `--max` (objets testés, 1500 par défaut), `--fenetre` (±7 kHz), `--troncon` (60 s), `--plafond` (12 dB).
- Si le flux du SDR s'interrompt (USB, alimentation), la durée manquante est remplacée par des zéros : la datation de la suite de la capture reste juste, et le trou est noté dans le `.json`.

## Limites et hypothèses

- Le volume couvert par GRAVES n'est pas publié. `graves_doppler.py` retient l'hypothèse d'un azimut de 90° à 270° et d'une élévation de 10° à 45°, tirée d'observations d'amateurs. Vos observations permettront de l'affiner.
- Les seuils de classification ont été réglés sur une seule station. Ils devront sans doute être ajustés à d'autres installations.
- L'analyse en direct est indicative (latence de quelques secondes). La classification de référence reste celle de `graves_echos.py` sur une capture enregistrée.
- **Orientation de l'antenne** : le signal direct de GRAVES arrive presque à l'horizontale. Une Yagi pointée vers le sud mais peu inclinée le reçoit dans son lobe principal (porteuse 35 à 45 dB au-dessus du bruit, éclaboussures à chaque commutation) et ne couvre que le bas du volume où GRAVES éclaire les satellites. Une élévation de 30 à 45° atténue le signal direct et vise ce volume ; les avions, vus bas sur l'horizon, deviennent plus faibles.
- **Satellites** : vérifier la vraisemblance physique d'une identification (indice de détectabilité, distance, recalage de l'ordre de la seconde pour des éléments orbitaux récents) et la trace sur l'image produite.
- **Météores** : les traînées sous-denses ont un Doppler proche de zéro, comme les avions en diffusion vers l'avant et les éclaboussures de la porteuse. Le comptage reste indicatif, surtout en soirée quand le trafic aérien est dense.
- **Terrain** : par temps froid, l'oscillateur du SDR dérive (plusieurs Hz à 143 MHz) et sa température en mode GPS diffère de celle en mode GRAVES : la porteuse est cherchée à ±40 Hz de la prévision GPS, ou à ±15 Hz d'une valeur donnée par `--porteuse`.
- Projet expérimental, en cours de développement.

## Sources de données et conditions d'utilisation

- **Space-Track.org** : compte gratuit requis. Leurs conditions interdisent de redistribuer les données. Les catalogues téléchargés restent donc en local (exclus par `.gitignore`).
- **adsb.fi** : API ouverte, usage personnel non commercial, 1 requête par seconde au maximum. Les images produites mentionnent la source.
- **Celestrak** : éléments orbitaux publics.

Ce projet fait uniquement de la **réception passive** d'une porteuse non modulée : aucun décodage, aucune émission.

Les fichiers de capture (`.json`) contiennent la position du récepteur. Pensez-y avant de partager des captures.

## Contributions

Retours d'observation, captures d'autres stations et améliorations sont les bienvenus : ouvrez une *issue* ou une *pull request*.

## Licence

MIT (voir [LICENSE](LICENSE)).

---

## English

**graves-radar** passively receives echoes from the French **GRAVES** space-surveillance radar (CW carrier at 143.050 MHz) with an SDR, and automatically classifies them as aircraft, satellites, meteors or the Moon.

The geometry is bistatic: the transmitter is near Broye-Aubigney-Montseugny (eastern France) and the receiver sits a few hundred kilometres away. The scripts go beyond displaying a waterfall and **identify** each echo by checking it against what physics predicts:

- **Aircraft**: the bistatic Doppler curve of every flight is computed from ADS-B positions (local readsb and/or the adsb.fi API) and matched against the waterfall. Significance is estimated against frequency-shifted decoy curves.
- **Satellites**: SGP4 propagation of the full orbital catalogue (Space-Track or Celestrak), then a **matched filter** along each predicted Doppler curve with a search over the time offset (TLE error). Exclusive assignment prevents neighbouring objects, such as Starlink trains, from being confused.
- **Meteors**: short events in the fast waterfall (head echoes, under- and overdense trails), after removing aircraft, satellites, broadband interference and spectral splatter caused by beam switching.
- **Receiver drift** is corrected using the direct GRAVES carrier as a reference.
- **GPS reference** (optional): a software GPS receiver on the RSPdx input A measures the receiver position, the SDR oscillator offset and the PC clock offset (`graves_gps.py`).

Quick start:

```bash
pip install -r requirements.txt
python3 graves_capture.py --simulation --duree 130 --iq        # no hardware needed
python3 graves_echos.py graves_*_iq_16666.npy --sans-adsb --sans-satellites
```

The code, comments and command-line options are in French. Each script documents its options with `-h`. Real acquisition was developed with an SDRplay RSPdx through SoapySDR; other SoapySDR devices should work with minor changes.

Receiver position goes in `~/.config/graves/station.ini` (otherwise an example position in Paris is used, with a warning). Space-Track credentials go in environment variables or `~/.config/graves/spacetrack.ini` and are never stored in the code. Space-Track data must not be redistributed, so downloaded catalogues are excluded from the repository. adsb.fi is for personal, non-commercial use only.

This is passive reception of an unmodulated carrier only: nothing is decoded and nothing is transmitted. Experimental project; feedback and observations from other stations are welcome.

License: MIT.
