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
- **Satellites** : propagation SGP4 de tout le catalogue orbital (Space-Track ou Celestrak), puis **filtre adapté** : l'énergie est sommée le long de la courbe Doppler prévue, avec recherche du meilleur recalage temporel (erreur des TLE). Une attribution exclusive évite de confondre deux objets voisins (trains de Starlink).
- **Météores** : événements brefs de la cascade rapide (écho de tête, traînée sous-dense ou surdense), après élimination des avions, des satellites, des parasites large bande et des « éclaboussures » spectrales liées à la commutation des faisceaux.
- **Dérive du récepteur** : l'oscillateur du SDR dérive en chauffant. La porteuse directe de GRAVES sert de référence pour la corriger.

## Les scripts

| Script | Rôle |
|---|---|
| `graves_capture.py` | Étape 1 : acquisition IQ (SoapySDR), décimation, cascades PNG, cascade en direct (matplotlib), mode simulation sans matériel |
| `graves_v2.py` | Affichage en direct V2 (pyqtgraph, ~15 images/s), mêmes options que `graves_capture.py`, capture d'écran F12 |
| `graves_doppler.py` | Doppler attendu pour la Lune et un satellite donné (prévisions) |
| `graves_adsb.py` | Étape 3 : journal ADS-B continu, puis corrélation avion par avion sur une capture |
| `graves_echos.py` | Étape 4 : détection et classification de tous les échos d'une capture (tableau, image annotée, journal CSV) |
| `graves_satellites.py` | Étape 5 : catalogue orbital, prévision des passages, identification des satellites sur une capture |
| `graves_direct.py` | Module d'analyse en temps réel (avions, satellites, météores) utilisé par `--analyse` |

## Matériel

Configuration de développement, à adapter :

- Antenne : discone pour les essais à domicile, Yagi 2 m orientée vers le sud pour les sessions en extérieur
- Préamplificateur faible bruit (LNA)
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
python3 graves_adsb.py journal --source local --source adsbfi   # terminal 1 : journal ADS-B
python3 graves_satellites.py catalogue                           # une fois par jour
python3 graves_v2.py --duree 600 --iq --analyse                  # terminal 2 : capture + affichage en direct

# Analyse a posteriori
python3 graves_echos.py   graves_..._iq_16666.npy                # classification complète
python3 graves_adsb.py    superpose graves_..._iq_16666.npy      # détail des avions
python3 graves_satellites.py superpose graves_..._iq_16666.npy   # détail des satellites

# Prévisions
python3 graves_satellites.py passages --heures 6
python3 graves_doppler.py lune --heures 24
```

Chaque script affiche toutes ses options avec `-h`.

## Limites et hypothèses

- Le volume couvert par GRAVES n'est pas publié. `graves_doppler.py` retient l'hypothèse d'un azimut de 90° à 270° et d'une élévation de 10° à 45°, tirée d'observations d'amateurs. Vos observations permettront de l'affiner.
- Les seuils de classification ont été réglés sur une seule station. Ils devront sans doute être ajustés à d'autres installations.
- L'analyse en direct est indicative (latence de quelques secondes). La classification de référence reste celle de `graves_echos.py` sur une capture enregistrée.
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
