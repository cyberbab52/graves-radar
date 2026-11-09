#!/usr/bin/env python3
"""
graves_doppler.py — Doppler bistatique attendu pour les échos du radar GRAVES
Émetteur : GRAVES, Broye-Aubigney-Montseugny (70)   47.3480 N, 5.5151 E
Récepteur : ~/.config/graves/station.ini, ou --lat/--lon/--alt

f_D = -(1/λ) · d/dt ( |E→C| + |C→R| ),   λ = c / 143,050 MHz ≈ 2,0957 m

Le script :
  * Lune  : Doppler bistatique + élévation/azimut vus de GRAVES et du récepteur,
            avec un drapeau "éclairée" quand la Lune est dans le volume couvert.
  * Satellites : à partir d'un fichier TLE (ou téléchargement Celestrak),
            liste les passages dans le volume éclairé par GRAVES et la courbe
            de Doppler attendue (à superposer à la cascade).

Usage :
  python3 graves_doppler.py lune   [--debut 2026-11-02T18:00] [--heures 24]
  python3 graves_doppler.py sat    --norad 25544 [--tle fichier.txt] [--heures 24]
  python3 graves_doppler.py sat    --groupe stations   (tous les objets du groupe Celestrak)
Options communes : --lat --lon --alt (récepteur), --audio 1000 (fréquence audio
de la porteuse en USB, défaut 1000 Hz, i.e. VFO sur 143,049 MHz), --png fichier.png

Hypothèses à ajuster (non publiées officiellement, issues d'observations
d'amateurs) : couverture GRAVES en azimut 90°–270° (demi-espace sud) et en
élévation ~10°–45°. Modifier AZ_MIN/AZ_MAX/EL_MIN/EL_MAX si les observations
montrent autre chose.

Dépendances : pip install skyfield skyfield-data matplotlib numpy
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from skyfield.api import load, wgs84, EarthSatellite

C = 299_792_458.0
F0 = 143.050e6
LAMBDA = C / F0

GRAVES = dict(lat=47.3480, lon=5.5151, alt=200.0)
RECEPTEUR = dict(lat=48.8566, lon=2.3522, alt=35.0)   # exemple (Paris)
try:                                              # position de la station (~/.config/graves/station.ini)
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import graves_adsb as _a
    RECEPTEUR = dict(lat=_a.STATION[0], lon=_a.STATION[1], alt=_a.STATION[2])
except Exception:
    pass

# Volume éclairé (hypothèse, à ajuster)
AZ_MIN, AZ_MAX = 90.0, 270.0
EL_MIN, EL_MAX = 10.0, 45.0

DT = 0.5  # pas de dérivation numérique (s)


def charger_ephemerides():
    ts = load.timescale()
    try:
        from skyfield_data import get_skyfield_data_path
        from skyfield.api import Loader
        eph = Loader(get_skyfield_data_path())("de421.bsp")
    except Exception:
        eph = load("de421.bsp")
    return ts, eph


def doppler_bistatique(tx, rx, cible, temps, ts):
    """Doppler (Hz) et chemin total (km) pour une cible Skyfield (Lune ou satellite).
    tx/rx : positions observateur (planète Terre + topos pour la Lune, topos seul pour un satellite)."""
    jd = temps.tt
    t1 = ts.tt_jd(jd - DT / 2 / 86400)
    t2 = ts.tt_jd(jd + DT / 2 / 86400)

    def chemin(t):
        d_tx = (cible - tx).at(t).distance().km
        d_rx = (cible - rx).at(t).distance().km
        return d_tx + d_rx

    L1, L2 = chemin(t1), chemin(t2)
    dLdt = (L2 - L1) * 1000.0 / DT  # m/s
    return -dLdt / LAMBDA, chemin(temps)


def eclaire(az, el):
    return (az >= AZ_MIN) & (az <= AZ_MAX) & (el >= EL_MIN) & (el <= EL_MAX)


def grille_temps(ts, debut, heures, pas_s):
    n = int(heures * 3600 / pas_s) + 1
    base = debut.replace(tzinfo=timezone.utc)
    return ts.utc(base.year, base.month, base.day, base.hour, base.minute,
                  base.second + np.arange(n) * pas_s)


def mode_lune(args):
    ts, eph = charger_ephemerides()
    terre, lune = eph["earth"], eph["moon"]
    tx_topo = wgs84.latlon(GRAVES["lat"], GRAVES["lon"], elevation_m=GRAVES["alt"])
    rx_topo = wgs84.latlon(args.lat, args.lon, elevation_m=args.alt)
    tx, rx = terre + tx_topo, terre + rx_topo

    t = grille_temps(ts, args.debut, args.heures, 60)
    alt_g, az_g, _ = tx.at(t).observe(lune).apparent().altaz()
    alt_r, az_r, _ = rx.at(t).observe(lune).apparent().altaz()
    fd, L = doppler_bistatique(tx, rx, lune, t, ts)
    on = eclaire(az_g.degrees, alt_g.degrees) & (alt_r.degrees > 0)

    print(f"Lune — récepteur {args.lat:.4f}N {args.lon:.4f}E — λ = {LAMBDA:.4f} m")
    print("Heure UTC          | Az/El GRAVES  | Az/El RX      | Doppler (Hz) | Audio USB (Hz) | Retard (s) | Éclairée")
    for i in range(0, len(t), args.pas_affichage):
        dt = t[i].utc_datetime()
        print(f"{dt:%Y-%m-%d %H:%M}   | {az_g.degrees[i]:5.1f}/{alt_g.degrees[i]:5.1f} | "
              f"{az_r.degrees[i]:5.1f}/{alt_r.degrees[i]:5.1f} | {fd[i]:+10.1f}   | "
              f"{args.audio + fd[i]:10.1f}     | {L[i] / C * 1000:6.3f}     | {'OUI' if on[i] else '-'}")

    if on.any():
        idx = np.where(on)[0]
        print(f"\nFenêtre(s) favorable(s) : {len(idx)} min au total. "
              f"Doppler dans la fenêtre : {fd[idx].min():+.0f} à {fd[idx].max():+.0f} Hz")
    else:
        print("\nAucune fenêtre favorable sur la période (Lune hors du volume GRAVES ou sous l'horizon du récepteur).")

    if args.png:
        tracer(args.png, [x.utc_datetime() for x in t], fd, on, "Lune — Doppler bistatique GRAVES",
               args.audio)


def lire_tle(args, ts):
    if args.tle:
        return load.tle_file(args.tle, ts=ts)
    if args.groupe:
        url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={args.groupe}&FORMAT=tle"
    else:
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={args.norad}&FORMAT=tle"
    return load.tle_file(url, filename=f"tle_{args.norad or args.groupe}.txt", reload=True, ts=ts)


def mode_sat(args):
    ts = load.timescale()
    sats = lire_tle(args, ts)
    if args.norad and not args.groupe:
        sats = [s for s in sats if s.model.satnum == int(args.norad)] or sats
    tx = wgs84.latlon(GRAVES["lat"], GRAVES["lon"], elevation_m=GRAVES["alt"])
    rx = wgs84.latlon(args.lat, args.lon, elevation_m=args.alt)

    t = grille_temps(ts, args.debut, args.heures, 2)
    toutes = []
    for sat in sats:
        alt_g, az_g, _ = (sat - tx).at(t).altaz()
        alt_r, _, _ = (sat - rx).at(t).altaz()
        on = eclaire(az_g.degrees, alt_g.degrees) & (alt_r.degrees > 0)
        if not on.any():
            continue
        fd, L = doppler_bistatique(tx, rx, sat, t, ts)
        # découpe en passages
        idx = np.where(on)[0]
        coupures = np.where(np.diff(idx) > 1)[0]
        debuts = np.r_[idx[0], idx[coupures + 1]]
        fins = np.r_[idx[coupures], idx[-1]]
        for a, b in zip(debuts, fins):
            toutes.append((t[a].utc_datetime(), sat.name, a, b, fd, L, alt_g, az_g))

    toutes.sort(key=lambda x: x[0])
    if not toutes:
        print("Aucun passage dans le volume éclairé par GRAVES sur la période.")
        return
    for d0, nom, a, b, fd, L, alt_g, az_g in toutes[: args.max]:
        duree = (b - a) * 2
        print(f"\n{nom} — {d0:%Y-%m-%d %H:%M:%S} UTC — {duree} s dans le volume")
        print("  Heure UTC  | Az/El GRAVES  | Doppler (Hz) | Audio USB (Hz) | Distance totale (km)")
        pas = max(1, (b - a) // 10)
        for i in range(a, b + 1, pas):
            print(f"  {t[i].utc_datetime():%H:%M:%S}   | {az_g.degrees[i]:5.1f}/{alt_g.degrees[i]:5.1f} | "
                  f"{fd[i]:+10.0f}   | {args.audio + fd[i]:10.0f}     | {L[i]:8.0f}")

    if args.png:
        d0, nom, a, b, fd, L, alt_g, az_g = toutes[0]
        a0, b0 = max(0, a - 30), min(len(t) - 1, b + 30)
        tt = [x.utc_datetime() for x in t[a0:b0 + 1]]
        on = np.zeros(b0 - a0 + 1, bool)
        on[a - a0:b - a0 + 1] = True
        tracer(args.png, tt, fd[a0:b0 + 1], on, f"{nom} — Doppler bistatique GRAVES", args.audio)


def tracer(png, temps, fd, on, titre, audio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(temps, fd, color="0.7", lw=1, label="hors volume éclairé")
    fd_on = np.where(on, fd, np.nan)
    ax.plot(temps, fd_on, color="C3", lw=2.5, label="dans le volume GRAVES (écho possible)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Doppler (Hz)")
    sec = ax.secondary_yaxis("right", functions=(lambda y: y + audio, lambda y: y - audio))
    sec.set_ylabel(f"Audio USB (Hz), porteuse à {audio} Hz")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Heure UTC")
    ax.set_title(titre)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"\nGraphique écrit : {png}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["lune", "sat"])
    p.add_argument("--lat", type=float, default=RECEPTEUR["lat"])
    p.add_argument("--lon", type=float, default=RECEPTEUR["lon"])
    p.add_argument("--alt", type=float, default=RECEPTEUR["alt"])
    p.add_argument("--debut", type=lambda s: datetime.fromisoformat(s),
                   default=datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0),
                   help="début en UTC, ex. 2026-11-02T18:00")
    p.add_argument("--heures", type=float, default=24)
    p.add_argument("--audio", type=float, default=1000.0)
    p.add_argument("--norad", type=str, default="25544", help="numéro NORAD (25544 = ISS)")
    p.add_argument("--groupe", type=str, default=None, help="groupe Celestrak (stations, visual, starlink…)")
    p.add_argument("--tle", type=str, default=None, help="fichier TLE local")
    p.add_argument("--max", type=int, default=20, help="nombre max de passages affichés")
    p.add_argument("--pas-affichage", dest="pas_affichage", type=int, default=30,
                   help="mode lune : une ligne toutes les N minutes")
    p.add_argument("--png", type=str, default=None)
    args = p.parse_args()
    (mode_lune if args.mode == "lune" else mode_sat)(args)


if __name__ == "__main__":
    main()
