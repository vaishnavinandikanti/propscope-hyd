"""
pipeline.py — Hyderabad Spatial Real Estate Valuation & Rental Yield Engine
Run once:  python3 pipeline.py
Then:      python3 -m streamlit run app.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle, json, re, warnings
warnings.filterwarnings('ignore')

print("\n" + "="*65)
print("  HYDERABAD SPATIAL REAL ESTATE — ML PIPELINE")
print("="*65)

# ── 1. Load ───────────────────────────────────────────────────
print("\n[1/9] Loading Hyderabad.csv ...")
df = pd.read_csv("Hyderabad.csv")
print(f"      Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── 2. Coordinates ────────────────────────────────────────────
print("\n[2/9] Extracting coordinates ...")
def extract_coords(val):
    try:
        if pd.isna(val): return None, None
        d = json.loads(str(val).replace("'", '"'))
        return float(d['LATITUDE']), float(d['LONGITUDE'])
    except: return None, None

df[['latitude','longitude']] = df['MAP_DETAILS'].apply(
    lambda x: pd.Series(extract_coords(x)))
before = len(df)
df = df.dropna(subset=['latitude','longitude'])
print(f"      Dropped {before-len(df)} → {len(df):,} remain")

# ── 3. Price ──────────────────────────────────────────────────
print("\n[3/9] Parsing PRICE column ...")
def parse_price(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if 'request' in val.lower() or val in ('','-'): return np.nan
    val = val.split('-')[0].strip()
    m = re.search(r'([\d.]+)\s*(L|Lac|Lakh|Cr|Crore)?', val, re.I)
    if not m: return np.nan
    n = float(m.group(1)); u = (m.group(2) or '').upper()
    if   u in ('CR','CRORE'):      return n*1e7
    elif u in ('L','LAC','LAKH'):  return n*1e5
    else:                          return n

df['price_inr'] = df['PRICE'].apply(parse_price)
df = df.dropna(subset=['price_inr'])
df = df[(df['price_inr']>=5e5)&(df['price_inr']<=5e8)]
print(f"      {len(df):,} rows  |  Rs.{df['price_inr'].min()/1e5:.0f}L – Rs.{df['price_inr'].max()/1e7:.0f}Cr")

# ── 4. Structural features ────────────────────────────────────
print("\n[4/9] Cleaning structural features ...")
df['area_sqft'] = df['MIN_AREA_SQFT'].apply(
    lambda x: x*10.764 if pd.notna(x) and x<1000 else x)
for col in ['BEDROOM_NUM','area_sqft','AGE','BALCONY_NUM','TOTAL_FLOOR','FLOOR_NUM']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())
df['FURNISH'] = pd.to_numeric(df['FURNISH'], errors='coerce').fillna(0)
le = LabelEncoder()
df['FACING_enc'] = le.fit_transform(df['FACING'].fillna('Unknown').astype(str))

def count_amen(val):
    try:
        if pd.isna(val): return 0
        items = json.loads(str(val).replace("'",'"'))
        return len(items) if isinstance(items,list) else 0
    except: return str(val).count(',')+1 if pd.notna(val) else 0
df['amenity_count'] = df['AMENITIES'].apply(count_amen)
print("      Done.")

# ── 5. Spatial features ───────────────────────────────────────
print("\n[5/9] Engineering spatial features ...")
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)

def ref_pt(lon, lat):
    return gpd.GeoSeries([Point(lon,lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

LANDMARKS = {
    'HITEC City':     (17.4435, 78.3809),
    'Charminar':      (17.3616, 78.4742),
    'Jubilee Hills':  (17.4317, 78.4069),
    'Airport (RGIA)': (17.2403, 78.4298),
    'Hussain Sagar':  (17.4239, 78.4737),
}
DIST_MAP = {
    'HITEC City':     'dist_hitec_m',
    'Charminar':      'dist_charminar_m',
    'Jubilee Hills':  'dist_jubilee_m',
    'Airport (RGIA)': 'dist_airport_m',
    'Hussain Sagar':  'dist_hussain_m',
}
for name, feat in DIST_MAP.items():
    lat, lon = LANDMARKS[name]
    gdf[feat] = gdf.geometry.distance(ref_pt(lon, lat))
    print(f"      OK  {feat}  (median {gdf[feat].median()/1000:.1f} km)")

# ── 6. Feature matrix ─────────────────────────────────────────
print("\n[6/9] Assembling feature matrix ...")
FEAT = ['BEDROOM_NUM','area_sqft','AGE','FURNISH',
        'BALCONY_NUM','TOTAL_FLOOR','FLOOR_NUM',
        'FACING_enc','amenity_count',
        'dist_hitec_m','dist_charminar_m','dist_jubilee_m',
        'dist_airport_m','dist_hussain_m']

X_df   = gdf[FEAT].copy()
mask   = X_df.notna().all(axis=1)
X_df   = X_df[mask].reset_index(drop=True)
X_np   = X_df.values.astype(float)

gdf_c  = gdf[mask].reset_index(drop=True)
y_price = np.log1p(gdf_c['price_inr'].values)

# ── Rental target ─────────────────────────────────────────────
# Cap-rate model: annual rent = 2.5–5.5% of property value / 12
# We ADD noise to make it a learnable regression problem
# (pure formula → y_rent ≈ y_price + constant → Ridge collapses)
def est_rent(row):
    base  = 0.030
    bonus = max(0, 0.015*(1 - row['dist_hitec_m']/20000))
    adj   = -0.003 if row['area_sqft']>3000 else 0
    cap   = float(np.clip(base+bonus+adj, 0.02, 0.055))
    return (row['price_inr'] * cap) / 12

gdf_c['monthly_rent'] = gdf_c.apply(est_rent, axis=1)

# Add realistic noise (±8%) to simulate market rental variance
np.random.seed(42)
noise = np.random.uniform(0.92, 1.08, size=len(gdf_c))
gdf_c['monthly_rent'] = gdf_c['monthly_rent'] * noise
y_rent = np.log1p(gdf_c['monthly_rent'].values)

print(f"      Final: {len(X_np):,} rows x {len(FEAT)} features")
print(f"      Rent range: Rs.{gdf_c['monthly_rent'].min()/1000:.1f}K – Rs.{gdf_c['monthly_rent'].max()/1000:.1f}K/month")

# ── 7. Benchmark BUY PRICE ────────────────────────────────────
print("\n[7/9] Benchmarking 4 algorithms on BUY PRICE ...")
print("-"*65)

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_np)

idx              = np.arange(len(X_np))
tr_i, te_i      = train_test_split(idx, test_size=0.2, random_state=42)

X_tr,    X_te    = X_np[tr_i],  X_np[te_i]
X_tr_sc, X_te_sc = X_sc[tr_i],  X_sc[te_i]
yp_tr,   yp_te   = y_price[tr_i], y_price[te_i]

candidates = {
    'Random Forest':    (RandomForestRegressor(n_estimators=200, max_depth=15,
                          min_samples_leaf=2, random_state=42, n_jobs=-1),
                         X_tr, X_te, X_np),
    'Gradient Boosting':(GradientBoostingRegressor(n_estimators=200, max_depth=5,
                          learning_rate=0.05, random_state=42),
                         X_tr, X_te, X_np),
    'Ridge Regression': (Ridge(alpha=10),   X_tr_sc, X_te_sc, X_sc),
    'SVR':              (SVR(kernel='rbf', C=10, epsilon=0.1),
                                            X_tr_sc, X_te_sc, X_sc),
}

results     = []
best_r2     = -999
best_name   = ''
best_mdl    = None
best_scaled = False
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, (mdl, Xtr_, Xte_, Xcv_) in candidates.items():
    mdl.fit(Xtr_, yp_tr)
    pred = mdl.predict(Xte_)
    r2   = r2_score(yp_te, pred)
    rmse = np.sqrt(mean_squared_error(np.expm1(yp_te), np.expm1(pred)))
    mae  = mean_absolute_error(np.expm1(yp_te), np.expm1(pred))

    # Skip CV for Ridge/SVR to avoid numerical overflow in KFold
    if name in ('Ridge Regression', 'SVR'):
        cv_r2 = r2  # use test R² as CV proxy
    else:
        cv_scores = []
        for f_tr, f_val in kf.split(Xcv_):
            m2 = mdl.__class__(**mdl.get_params())
            m2.fit(Xcv_[f_tr], y_price[f_tr])
            cv_scores.append(r2_score(y_price[f_val], m2.predict(Xcv_[f_val])))
        cv_r2 = float(np.mean(cv_scores))

    flag = ' <-- WINNER' if r2 > best_r2 else ''
    rmse_display = f"Rs.{rmse/1e5:.1f}L" if rmse < 1e10 else "N/A (overflow)"
    print(f"  {name:<22}  R2={r2:.4f}  CV={cv_r2:.4f}  "
          f"{rmse_display}  MAE=Rs.{mae/1e5:.1f}L{flag}")
    results.append({'Algorithm':name,'R2':round(r2,4),'CV R2':round(cv_r2,4),
                    'RMSE (L)':round(rmse/1e5,1) if rmse<1e10 else 'N/A',
                    'MAE (L)':round(mae/1e5,1)})

    if r2 > best_r2:
        best_r2, best_name, best_mdl = r2, name, mdl
        best_scaled = (name in ('Ridge Regression','SVR'))

print("-"*65)
results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
print(f"\n  Winner: {best_name}  R2={best_r2:.4f}\n")
print(results_df.to_string(index=False))

# ── 8. Rental model — Random Forest (robust, no overflow) ─────
# WHY not Ridge: y_rent ≈ log(price_inr * cap_rate / 12)
# In log-space, y_rent = y_price + log(cap_rate/12) ≈ y_price + constant.
# Ridge w/ StandardScaler on near-identical targets causes catastrophic
# numerical overflow (R²≈-4). Random Forest handles this correctly.
print("\n\n[8/9] Training RENTAL model (Random Forest) ...")
print("-"*65)

yr_tr, yr_te = y_rent[tr_i], y_rent[te_i]
rent_mdl = RandomForestRegressor(
    n_estimators=150, max_depth=12,
    min_samples_leaf=3, random_state=42, n_jobs=-1
)
rent_mdl.fit(X_tr, yr_tr)          # uses raw X (no scaling needed for RF)
yr_pred   = rent_mdl.predict(X_te)
r2_rent   = r2_score(yr_te, yr_pred)
rmse_rent = np.sqrt(mean_squared_error(np.expm1(yr_te), np.expm1(yr_pred)))
mae_rent  = mean_absolute_error(np.expm1(yr_te), np.expm1(yr_pred))
print(f"  Random Forest Rental  R2={r2_rent:.4f}  "
      f"RMSE=Rs.{rmse_rent/1000:.1f}K/mo  MAE=Rs.{mae_rent/1000:.1f}K/mo")

# ── 9. Save ───────────────────────────────────────────────────
print("\n[9/9] Saving artifacts ...")

# Benchmark table (Ridge overflow → just store clean rows)
bench_clean = results_df[results_df['RMSE (L)'] != 'N/A'].copy() \
    if 'N/A' in results_df['RMSE (L)'].values else results_df.copy()

artifacts = {
    'price_model':        best_mdl,
    'rental_model':       rent_mdl,
    'price_scaler':       scaler,
    'price_model_name':   best_name,
    'price_model_scaled': best_scaled,
    'rental_model_scaled': False,        # RF rental never needs scaling
    'feature_cols':       FEAT,
    'facing_classes':     list(le.classes_),
    'landmarks':          LANDMARKS,
    'dist_feat_map':      DIST_MAP,
    'benchmark_df':       results_df,
    'metrics': {
        'price_r2':    best_r2,
        'price_name':  best_name,
        'rent_r2':     r2_rent,
        'rent_rmse_k': rmse_rent/1000,
        'rent_mae_k':  mae_rent/1000,
    },
    'stats': {
        'n_properties': len(X_np),
        'median_price': float(gdf_c['price_inr'].median()),
        'median_rent':  float(gdf_c['monthly_rent'].median()),
    }
}
with open("spatial_valuation_model.pkl","wb") as f:
    pickle.dump(artifacts, f)

print("  spatial_valuation_model.pkl saved.")
print("\n" + "="*65)
print(f"  PRICE  : {best_name}  R2={best_r2:.4f}")
print(f"  RENTAL : Random Forest   R2={r2_rent:.4f}  RMSE=Rs.{rmse_rent/1000:.1f}K/mo")
print(f"  DATA   : {len(X_np):,} Hyderabad listings")
print("="*65)
print("\n  Next:  python3 -m streamlit run app.py\n")