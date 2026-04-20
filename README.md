# Customer Segmentation with K-Means, SQL, FastAPI, and Streamlit

A recruiter-facing data science project that segments retail customers using **RFM features** (Recency, Frequency, Monetary value), profiles the resulting clusters, and exposes the outcome through both a **FastAPI service** and a **Streamlit dashboard**.

## Why this project matters

Marketing teams do not need “all customers.” They need **distinct groups with distinct actions**:
- high-value loyal customers to retain,
- promising customers to grow,
- at-risk customers to re-engage,
- low-value or one-off customers to avoid over-spending on.

This project turns raw online retail transactions into customer-level intelligence that can support campaign targeting and CRM prioritization.

## Dataset

This project uses the **Online Retail II** dataset from the UCI Machine Learning Repository.
It contains transaction records for a UK-based non-store online retailer from **December 2009 to December 2011**.

### Raw variables used
- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

## Project objectives

- clean raw transactional data,
- engineer customer-level features,
- build RFM-based customer segments using K-Means,
- evaluate cluster quality with silhouette score,
- assign human-readable business labels,
- expose cluster predictions via API,
- let non-technical users explore segments in a Streamlit app.

## Tech stack

- **Python**
- **SQL** (SQLite-style analysis workflow)
- **scikit-learn**
- **FastAPI**
- **Streamlit**
- **Plotly / Matplotlib**
- **UCI data ingestion via `ucimlrepo`**

## Repository structure

```text
customer-segmentation-kmeans/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_segmentation_eda.ipynb
├── reports/
│   └── figures/
├── sql/
│   └── customer_analysis.sql
├── src/
│   ├── data_download.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── tests/
│   └── test_features.py
├── README.md
└── requirements.txt
```

## How to run

### 1) Create environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Download data and build features
```bash
python -m src.data_download
python -m src.features
```

### 3) Train the clustering model
```bash
python -m src.train
```

### 4) Launch the API
```bash
uvicorn api.main:app --reload
```

### 5) Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

## Main outputs

- customer-level feature table in `data/processed/customer_features.csv`
- fitted scaler and K-Means model in `models/`
- segment profiling summary in `data/processed/cluster_summary.csv`

## Business deliverables

This repo is designed to support CV-ready claims such as:
- Built an end-to-end customer segmentation workflow from transactional retail data.
- Engineered RFM features and customer-level metrics from over 1M raw transactions.
- Clustered customers with K-Means and translated clusters into actionable business personas.
- Delivered model access via FastAPI and a stakeholder-facing Streamlit dashboard.

## Example CV bullets

- Built a customer segmentation system on real-world retail transactions using Python, SQL, and K-Means clustering.
- Engineered RFM and behavioral features, then profiled customer groups for targeted marketing actions.
- Deployed the segmentation workflow with FastAPI and Streamlit for interactive exploration and scoring.

## Notes

- Raw data is intentionally not committed by default because of size and repository hygiene.
- The project scripts download the dataset directly from the public source.
- Cluster names are heuristic and should be reviewed with business stakeholders before live use.
