# Base44 Setup Notes

## Project Overview
CodeAlpha Data Science Internship — two Python ML scripts (Iris Classification, Sales Prediction). No web server originally; a Flask dashboard (`app.py`) was added to serve results on port 3000.

## Stack
- Python 3.12 (via `python:3.12-slim` Docker image)
- Flask web dashboard (`app.py`) serving on port 3000
- ML deps: scikit-learn, pandas, numpy, matplotlib, seaborn

## Running
```
docker compose -f docker-compose.base44.yml up -d
```
- Installs pip dependencies on container startup, then runs `python app.py`
- Flask runs in debug mode with auto-reload (binds 0.0.0.0:3000)
- Healthcheck: `GET /` returns 200

## Key Details
- Flask is server-rendered (not SPA); navigating between `/` and `/task4` causes full page loads
- Pre-generated plots (PNGs) and result CSVs live in each task folder; the dashboard displays them as base64-embedded images
- "Re-run Analysis" buttons execute the original ML scripts (`iris_classification.py`, `sales_prediction.py`) and refresh results
- No external secrets or credentials needed
