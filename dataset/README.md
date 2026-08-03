# Trek Guardian Datasets

This folder holds **real** high-altitude hypoxia physiology data used to train
the early-warning model. Synthetic data is not used.

## Sources

| Dataset | Citation | DOI / Figshare |
|---------|----------|----------------|
| **Harespod** | Yang et al., *Scientific Data* (2024) | [10.6084/m9.figshare.c.6623344](https://doi.org/10.6084/m9.figshare.c.6623344) |
| **HAPP** | Jia, *Scientific Data* (2025) | [10.6084/m9.figshare.29947679](https://doi.org/10.6084/m9.figshare.29947679) |

Paper links:
- Harespod: https://www.nature.com/articles/s41597-024-03065-x
- HAPP: https://www.nature.com/articles/s41597-025-06508-1

## Download

From the repo root:

```bash
cd ml_model
pip install -r requirements.txt
python download_datasets.py
# optional (~77MB extra continuous Harespod archive):
python download_datasets.py --with-continuous
python preprocess.py
```

Raw downloads land in `dataset/raw/` (gitignored).  
Processed 1 Hz vitals land in `dataset/processed/` (gitignored).

## Honesty note

These are hypobaric-chamber recordings under peer-reviewed protocols.
They are **not** labeled Himalayan AMS field events. The ML task is
**early hypoxia warning** (predict future SpO2 severity from recent trends),
not diagnosis of acute mountain sickness.
