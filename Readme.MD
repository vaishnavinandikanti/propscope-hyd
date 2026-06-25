# PropScope HYD — Spatial Real Estate Valuation Engine

Geospatial ML platform predicting property buy price and monthly rental yield for any clicked location in Hyderabad — built on 7,487 real 99acres listings.

---

## Live Demo

**Launch PropScope HYD →**

https://propscope-hyd-njwfzgznqguutiqdmpptyg.streamlit.app

---

## What It Does

Click any location on the Hyderabad map. PropScope HYD instantly returns:

* Buy price prediction — Random Forest, R² = 0.7986
* Monthly rental estimate — Random Forest, R² = 0.8180
* Annual yield (cap rate) — 2.0% to 5.5% p.a. based on spatial demand
* Nearest Metro station — resolved from 33 real Hyderabad Metro coordinates
* Investment scenario modelling — EMI, 10-year equity projection, ROI
* Multi-pin portfolio comparison — save and compare locations side by side

---

## Model Performance

| Model             | Target         | R² (Test) | CV R² (5-fold) | RMSE         | MAE          |
| ----------------- | -------------- | --------- | -------------- | ------------ | ------------ |
| Random Forest     | Buy Price      | 0.7986    | 0.8226         | Rs. 210.1L   | Rs. 57.6L    |
| Random Forest     | Monthly Rental | 0.8180    | —              | Rs. 63.8K/mo | Rs. 17.1K/mo |
| Gradient Boosting | Buy Price      | 0.7739    | 0.8142         | Rs. 256.2L   | Rs. 65.4L    |
| SVR (RBF)         | Buy Price      | 0.5981    | 0.5981         | Rs. 278.9L   | Rs. 83.8L    |
| Ridge Regression  | Buy Price      | −4.60     | N/A            | Overflow     | —            |

Random Forest was selected after benchmarking all four algorithms with 5-fold cross-validation. The CV-to-test gap of 0.024 indicates low overfitting.

---

# Architecture

```text
spatial-valuation-engine/
│
├── pipeline.py          # Data cleaning → feature engineering → model training → .pkl
├── app.py               # 4-tab Streamlit dashboard
├── Hyderabad.csv        # 99acres dataset (9,487 raw → 7,487 after cleaning)
├── requirements.txt     # Python dependencies
├── packages.txt         # System dependency for GeoPandas on Streamlit Cloud
└── README.md
```

### Two-file architecture

* `pipeline.py` runs once, saves `spatial_valuation_model.pkl` containing both models, scaler, encoder classes, and benchmark results.
* `app.py` loads the `.pkl` at startup and serves all four tabs.

---

## Feature Engineering

### Structural Features (9)

Extracted and cleaned from raw 99acres columns — bedrooms, area (sq metres → sq ft), age, furnishing level, balconies, total floors, floor number, facing direction (Vastu-encoded), amenity count.

### Spatial Features (5) — the differentiator

Computed entirely from lat/lon coordinates using GeoPandas projected to EPSG:3857 (Web Mercator, metres). Raw GPS degrees cannot be used directly because at Hyderabad's latitude (~17°N), 1° longitude ≈ 106 km ≠ 1° latitude ≈ 111 km.

| Feature          | Landmark      | Why It Matters                                                          |
| ---------------- | ------------- | ----------------------------------------------------------------------- |
| dist_hitec_m     | HITEC City    | #1 by feature importance — IT employment hub driving 5km radius premium |
| dist_charminar_m | Charminar     | Captures North (premium) vs South (older) value gradient                |
| dist_jubilee_m   | Jubilee Hills | Separates luxury segment from mid-range                                 |
| dist_airport_m   | RGIA Airport  | Connectivity driver — business/expat demand                             |
| dist_hussain_m   | Hussain Sagar | City centre proxy — correlates with commute length                      |

---

## Dashboard — 4 Tabs

### Tab 1 — Map & Valuation

Click-to-predict Folium map with Street/Satellite tile switching. Geofenced to Hyderabad boundary. Property type selector (Premium flat / Villa / Penthouse / Standard Apt), 14 amenity checkboxes, facing direction with Vastu demand labels.

Valuation output:

* Buy price
* Price/sqft
* Monthly rental
* Yield cap rate

Nearest Metro station resolved dynamically from 33 real station coordinates.

### Tab 2 — Scenario Modeler

Three investment strategy presets:

* Safe Buy-and-Hold
* Modern Luxury Upgrade
* Leveraged Capital Play

Custom sliders for:

* Renovation budget
* Down payment
* Interest rate
* Appreciation rate

KPI cards:

* Total Initial Cash
* 10-year ROI
* Capital Break-Even Year

10-year Asset Value vs Cumulative Return chart.

### Tab 3 — Leverage & CapEx Analysis

EMI calculation using standard amortisation formula over 20-year term.

Three-line 10-year chart:

* Asset Value
* Equity Built
* Remaining Debt

KPIs:

* 10-year Equity Accumulation
* Cumulative Cash Flow
* Net ROI Multiplier

### Tab 4 — Portfolio Cases

Save multiple location pins and compare side by side.

Each pin shows:

* Area name (resolved from 30+ Hyderabad locality references)
* Coordinates
* Property type
* Price
* Rent
* Yield

Horizontal bar chart comparison when 2+ pins are saved.

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/vaishnavinandikanti/propscope-hyd
cd propscope-hyd

# 2. Install dependencies
pip install -r requirements.txt

# On Mac, if geopandas fails:
# brew install gdal
# pip install geopandas

# 3. Train the model (runs once, takes ~2 minutes)
python3 pipeline.py

# 4. Launch the dashboard
python3 -m streamlit run app.py
```

The terminal output from `pipeline.py` shows the benchmark table — Random Forest winning with R² = 0.7986 and CV R² = 0.8226.

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (with `spatial_valuation_model.pkl` in `.gitignore`)
2. Go to `share.streamlit.io` → New app
3. Select repo → branch: `main` → main file: `app.py`
4. Click Deploy

The `packages.txt` file tells Streamlit Cloud to install `libgdal-dev` (required for GeoPandas).

The app auto-runs `pipeline.py` on first deploy if the `.pkl` is missing.

---

## Dataset

**Source:** 99acres.com via Kaggle — arvanshul/gurgaon-real-estate-99acres-com

**File:** Hyderabad.csv extracted from multi-city archive

* Raw: 9,487 listings · 55 columns
* After cleaning: 7,487 listings
* Price range: Rs. 5 Lakhs – Rs. 50 Crore
* Rent range: Rs. 1.1K – Rs. 1,792.5K/month (cap rate derived)

### Data engineering steps

* Lat/lon parsed from `MAP_DETAILS` JSON string column
* Price parsed from strings like `'69.25 L'`, `'2.02-2.87 Cr'`, and `'Price on Request'` using regex
* Area converted from sq metres to sq ft (`× 10.764`) where values < 1000
* Facing direction decoded from 99acres numeric codes (0–8) to compass directions

---

## Tech Stack

| Library                   | Purpose                                                              |
| ------------------------- | -------------------------------------------------------------------- |
| pandas / numpy            | Data cleaning, feature matrix, log transforms                        |
| geopandas / shapely       | EPSG:4326 → EPSG:3857 projection, distance computation, geofencing   |
| scikit-learn              | RandomForest, GradientBoosting, Ridge, SVR, StandardScaler, KFold CV |
| streamlit                 | 4-tab dashboard, session state, widget rendering                     |
| folium + streamlit-folium | Interactive map, tile switching, click event capture                 |
| pickle                    | Model and artifact serialisation                                     |

---

## Known Limitations

* Rental target is synthetic — no raw rental column in 99acres data; rent derived from cap rate formula, not actual rental transactions.
* Geofence is a bounding box — rectangular boundary, not the actual GHMC polygon.
* Property type multipliers are post-hoc — applied after prediction as business logic, not learned from labelled type data.
* Ridge Regression R² = −4.6 — excluded due to near-collinear spatial features causing coefficient overflow after log transformation; documented and expected.

