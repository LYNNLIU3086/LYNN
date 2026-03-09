# Saturn E-Ring Composition

Interactive dashboard for exploring dust grain composition data from Cassini's Cosmic Dust Analyzer (CDA). Built for the CS 81 Cassini project at Caltech.

28,145 confirmed grains collected between 2004–2017, classified by mass spectrum into four types: Ice, Salt, Organic, and Other.

**Live:** https://edju03-cassini-visualization.hf.space/

## Running locally

```
pip install dash dash-bootstrap-components plotly pandas numpy scipy
python app.py
```

Opens at `localhost:8050`.

## Dashboard

- **Spatial map** — 2D/3D grain positions around Saturn with ring system and moon orbits (Enceladus, Tethys, Dione, Rhea)
- **Composition vs distance** — organic and salt fractions binned by radial distance, with point-biserial correlation
- **Mass spectrum** — click any grain to see its 200-channel spectrum with labeled peaks (H₂O, Na, Si, Fe, etc.)
- **Latent space** — autoencoder embedding from DustMAP showing spectral clustering by type
- **Radial strip** — drag-to-zoom radial distribution with live selection stats
- **Year slider** — cumulative filter through Cassini mission years (2004–2017)
- **Moon selector** — switch distance reference between Saturn and its moons

## Data

`CS81 Cassini CDA Composition-selected/ConfDataSpicy.csv` — 28,145 grains, 265 columns: spatial coordinates, timestamps, moon distances, M3 categories, mass spectra (AMU 1–200), and autoencoder latent coordinates.

## Other files

- `ering.py` — standalone Plotly HTML visualization with time slider
- `.github/workflows/keep-alive.yml` — pings the HF Space every 14 minutes to prevent it from sleeping
