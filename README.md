# CATEM Telepresence Evaluation

This repository scaffolds the CATEM research project for telepresence evaluation. It includes synthetic sample data, a scoring prototype, validation utilities, and a Streamlit dashboard.

## Project Structure

- `README.md` — project overview and usage
- `requirements.txt` — Python dependencies
- `app.py` — prototype runner for data loading and scoring
- `data/` — raw, processed, and synthetic datasets
- `src/` — modules for loading, preprocessing, scoring, validation, and visualization
- `dashboards/` — Streamlit dashboard implementation
- `notebooks/` — exploratory and validation notebooks
- `outputs/` — generated figures, reports, and results
- `docs/` — CATEM framework and validation documentation

## Getting Started

1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the prototype app

```bash
python app.py
```

3. Start the dashboard

```bash
streamlit run dashboards/streamlit_dashboard.py
```

## Notes

This scaffold begins with a synthetic dataset in `data/synthetic/catem_sample_data.csv` to test scoring and dashboard logic before integrating Cornell data.
