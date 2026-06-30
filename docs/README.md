# CFO project page

Static site for **Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design** (ICML 2026).

Served via GitHub Pages from this `docs/` folder on `main`.
Live URL: **https://svenlg.github.io/constrained-flow-optimization/**

## Editing

- Content: `index.html` (single file, plain HTML + Bulma)
- Figures: `static/images/` (PNGs converted from the paper's PDFs at ~150 DPI)
- Styling: `static/css/index.css` (small custom overrides on top of Bulma)

To preview locally:

```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000
```

Template adapted from the [Nerfies project page](https://github.com/nerfies/nerfies.github.io) (CC BY-SA 4.0).
