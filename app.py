"""
app.py — Hyderabad GEO-VALUATION ENGINE
4-tab PropTech dashboard matching the reference UI.
Run: python3 -m streamlit run app.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Geo-Valuation Engine", page_icon="🏙️",
                   layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#fafafa;color:#1e293b;}
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"]{
  padding:0 28px !important;max-width:100% !important;
}
@media (max-width: 640px){
  .block-container,
  [data-testid="stMainBlockContainer"],
  [data-testid="stAppViewBlockContainer"],
  [data-testid="block-container"]{
    padding:0 14px !important;
  }
  .navbar, .eng-footer{ margin:0 -14px; padding-left:14px; padding-right:14px; }
}
section[data-testid="stSidebar"]{display:none;}
*{overflow-wrap:break-word;word-wrap:break-word;}

/* ── Nav bar ── */
.navbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 28px;height:64px;background:#ffffff;
  border-bottom:1px solid #e2e8f0;position:sticky;top:0;z-index:100;
  box-shadow:0 1px 3px rgba(0,0,0,0.04);flex-wrap:wrap;gap:8px;
  margin:0 -28px;
}
.nav-brand{display:flex;align-items:center;gap:12px;min-width:0;}
.nav-logo{
  width:38px;height:38px;border-radius:9px;flex-shrink:0;
  background:linear-gradient(135deg,#6366f1,#8b5cf6);
  display:flex;align-items:center;justify-content:center;
  font-size:17px;
}
.nav-title{font-family:'Space Grotesk',sans-serif;font-weight:800;font-size:16px;color:#1e293b;letter-spacing:0.5px;white-space:nowrap;}
.nav-sub{font-size:9px;letter-spacing:3px;color:#94a3b8;text-transform:uppercase;margin-top:1px;white-space:nowrap;}
.nav-tabs{display:flex;gap:4px;}
.nav-tab{
  padding:7px 16px;border-radius:7px;font-size:13px;font-weight:500;
  color:#64748b;cursor:pointer;border:none;background:transparent;
  transition:all 0.15s;white-space:nowrap;
}
.nav-tab:hover{color:#1e293b;background:#f1f5f9;}
.nav-tab.active{background:#eef2ff;color:#4338ca;border:1px solid #c7d2fe;}

/* ── Shared components ── */
.panel{background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:22px 24px;box-shadow:0 1px 2px rgba(0,0,0,0.03);}
.panel-title{font-family:'Space Grotesk',sans-serif;font-size:17px;font-weight:700;color:#1e293b;margin-bottom:4px;}
.panel-sub{font-size:12px;color:#94a3b8;margin-bottom:20px;}
.kpi-row{display:flex;gap:14px;flex-wrap:wrap;}
.kpi{
  flex:1;min-width:140px;background:#ffffff;border:1px solid #e2e8f0;
  border-radius:12px;padding:18px 20px;box-shadow:0 1px 2px rgba(0,0,0,0.03);
  min-width:0;
}
.kpi-label{font-size:9px;letter-spacing:2.5px;text-transform:uppercase;color:#94a3b8;font-family:'JetBrains Mono',monospace;margin-bottom:8px;}
.kpi-val{font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;word-break:break-word;}
.kpi-sub{font-size:11px;color:#94a3b8;margin-top:4px;font-family:'JetBrains Mono',monospace;}
.green{color:#059669;} .red{color:#dc2626;} .indigo{color:#4f46e5;}
.amber{color:#d97706;} .teal{color:#0d9488;} .white{color:#1e293b;}

.sec-label{
  font-size:9px;letter-spacing:3px;text-transform:uppercase;
  color:#94a3b8;font-family:'JetBrains Mono',monospace;
  margin:22px 0 10px 0;padding-bottom:8px;border-bottom:1px solid #e2e8f0;
}
.badge{
  display:inline-block;padding:3px 10px;border-radius:5px;
  font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;
  max-width:100%;overflow-wrap:break-word;
}
.badge-green{background:#ecfdf5;color:#059669;border:1px solid #a7f3d0;}
.badge-indigo{background:#eef2ff;color:#4f46e5;border:1px solid #c7d2fe;}
.badge-amber{background:#fffbeb;color:#d97706;border:1px solid #fde68a;}

/* ── Tab 1 — Map & Valuation ── */
.map-header{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:14px;
}
.map-active-badge{
  display:flex;align-items:center;gap:6px;font-size:11px;color:#059669;
  background:#ecfdf5;border:1px solid #a7f3d0;border-radius:6px;padding:4px 10px;
  flex-shrink:0;
}
.dot-pulse{
  width:7px;height:7px;border-radius:50%;background:#059669;
  animation:pulse 1.5s infinite;
}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.4;}}
.prop-type-row{display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap;}
.pt-btn{
  padding:7px 16px;border-radius:8px;font-size:12px;font-weight:600;
  border:1px solid #e2e8f0;color:#64748b;background:#ffffff;cursor:pointer;
}
.pt-btn.sel{background:#ecfdf5;color:#059669;border-color:#a7f3d0;}
.spec-row{display:flex;justify-content:space-between;align-items:flex-start;
  padding:10px 0;border-bottom:1px solid #f1f5f9;font-size:13px;gap:10px;
  flex-wrap:wrap;}
.spec-label{color:#64748b;min-width:0;flex:1 1 auto;word-break:break-word;}
.spec-val{color:#1e293b;font-family:'JetBrains Mono',monospace;font-weight:500;
  text-align:right;flex-shrink:0;white-space:nowrap;}
.spec-val.green{color:#059669;}
.val-big{font-family:'Space Grotesk',sans-serif;font-size:32px;font-weight:700;color:#1e293b;letter-spacing:-0.5px;word-break:break-word;}
.thesis-box{
  background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
  padding:16px 18px;font-size:12.5px;color:#64748b;line-height:1.8;
  font-style:italic;margin-top:12px;
}
.thesis-box b{color:#1e293b;font-style:normal;}
.coord-bar{
  display:flex;gap:16px;align-items:center;padding:10px 16px;
  background:#f8fafc;border-top:1px solid #e2e8f0;
  font-family:'JetBrains Mono',monospace;font-size:11px;
  border-radius:0 0 12px 12px;flex-wrap:wrap;
}
.coord-item{min-width:0;}
.coord-item span{color:#94a3b8;margin-right:6px;letter-spacing:1px;}
.coord-item b{color:#4f46e5;}
.coord-val-pill{
  background:#eef2ff;border:1px solid #c7d2fe;border-radius:6px;
  padding:5px 12px;color:#4338ca;font-size:12px;display:flex;align-items:center;gap:6px;
  white-space:nowrap;
}

/* ── Tab 2 — Scenario ── */
.scenario-card{
  background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
  padding:20px;cursor:pointer;transition:border-color 0.15s;
  box-shadow:0 1px 2px rgba(0,0,0,0.03);
}
.scenario-card.sel{border-color:#6366f1;background:#eef2ff;}
.scenario-card:hover{border-color:#c7d2fe;}
.sc-title{font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:700;color:#1e293b;margin-bottom:6px;}
.sc-desc{font-size:12px;color:#94a3b8;line-height:1.6;margin-bottom:14px;}
.sc-specs{display:grid;grid-template-columns:1fr 1fr;gap:6px;}
.sc-spec{font-size:11px;color:#64748b;}
.sc-spec b{color:#334155;}

/* ── Tab 3 — Leverage ── */
.lev-row{display:flex;justify-content:space-between;align-items:flex-start;
  padding:10px 0;border-bottom:1px solid #f1f5f9;font-size:13px;
  gap:10px;flex-wrap:wrap;}
.lev-label{color:#64748b;min-width:0;word-break:break-word;}
.lev-val{font-family:'JetBrains Mono',monospace;color:#1e293b;font-weight:500;
  text-align:right;flex-shrink:0;white-space:nowrap;}
.lev-val.red{color:#dc2626;}

/* ── Tab 4 — Portfolio ── */
.pin-card{
  background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
  padding:18px 20px;margin-bottom:12px;box-shadow:0 1px 2px rgba(0,0,0,0.03);
}
.pin-header{display:flex;justify-content:space-between;align-items:flex-start;}
.pin-coords{font-family:'JetBrains Mono',monospace;font-size:11px;color:#94a3b8;}
.pin-price{font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:#059669;}

/* ── Footer ── */
.eng-footer{
  border-top:1px solid #e2e8f0;padding:12px 28px;
  font-family:'JetBrains Mono',monospace;font-size:10px;
  color:#94a3b8;display:flex;justify-content:space-between;
  background:#ffffff;margin-top:auto;flex-wrap:wrap;gap:8px;
  margin-left:-28px;margin-right:-28px;
}

/* ── Streamlit native widget overrides for light theme ── */
.stSelectbox label, .stSlider label, .stCheckbox label,
.stSlider div[data-testid="stTickBarMin"],
.stSlider div[data-testid="stTickBarMax"]{
  color:#475569 !important;
}
.stCheckbox label p, .stCheckbox span{color:#334155 !important;}

/* Selectbox closed value + dropdown menu */
div[data-baseweb="select"] > div{
  background:#ffffff!important;border-color:#e2e8f0!important;color:#1e293b!important;
}
div[data-baseweb="select"] *{color:#1e293b!important;}
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"]{
  background:#ffffff!important;border:1px solid #e2e8f0!important;
}
li[role="option"]{color:#1e293b!important;background:#ffffff!important;}
li[role="option"]:hover, li[aria-selected="true"]{
  background:#eef2ff!important;color:#1e293b!important;
}

/* Slider track + value bubble */
.stSlider [data-baseweb="slider"] [role="slider"]{background:#6366f1!important;border-color:#6366f1!important;}
.stSlider [data-baseweb="slider"] > div > div{background:#e2e8f0!important;}
div[data-testid="stThumbValue"]{color:#4f46e5!important;background:#eef2ff!important;}

/* Buttons */
.stButton button{border-radius:8px!important;font-weight:600!important;}
.stButton button[kind="secondary"]{
  background:#ffffff!important;color:#334155!important;border:1px solid #e2e8f0!important;
}
.stButton button[kind="secondary"]:hover{
  background:#f8fafc!important;border-color:#c7d2fe!important;color:#1e293b!important;
}
.stButton button[kind="primary"]{
  background:#4f46e5!important;color:#ffffff!important;border:1px solid #4f46e5!important;
}
.stButton button[kind="primary"]:hover{background:#4338ca!important;}

/* hide streamlit chrome */
#MainMenu,footer,.stDeployButton,[data-testid="stHeader"]{display:none!important;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    with open("spatial_valuation_model.pkl", "rb") as f:
        return pickle.load(f)

art          = load_artifacts()
price_model  = art['price_model']
rental_model = art['rental_model']
scaler       = art['price_scaler']
need_scale   = art['price_model_scaled']
FACING_CLS   = art['facing_classes']
LANDMARKS    = art['landmarks']
metrics      = art['metrics']
stats        = art['stats']

HYD_FENCE = Polygon([(78.20,17.20),(78.20,17.65),(78.65,17.65),(78.65,17.20)])
def in_hyd(lat,lon): return HYD_FENCE.contains(Point(lon,lat))

def dist_m(lat1,lon1,lat2,lon2):
    p1=gpd.GeoSeries([Point(lon1,lat1)],crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    p2=gpd.GeoSeries([Point(lon2,lat2)],crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    return p1.distance(p2)

# ── Real Hyderabad Metro stations (Red/Blue/Green line, approx coords) ──
METRO_STATIONS = {
    'Miyapur':            (17.4946, 78.3822),
    'JNTU College':       (17.4933, 78.3915),
    'KPHB Colony':        (17.4858, 78.3920),
    'Kukatpally':         (17.4849, 78.4080),
    'Balanagar':          (17.4709, 78.4310),
    'Erragadda':          (17.4537, 78.4400),
    'ESI Hospital':       (17.4490, 78.4480),
    'SR Nagar':           (17.4430, 78.4480),
    'Ameerpet':           (17.4374, 78.4482),
    'Punjagutta':         (17.4253, 78.4525),
    'Irrum Manzil':       (17.4225, 78.4622),
    'Khairatabad':        (17.4080, 78.4660),
    'Lakdi-ka-pul':       (17.4013, 78.4690),
    'Assembly':           (17.3950, 78.4730),
    'Nampally':           (17.3920, 78.4760),
    'Narayanguda':        (17.4020, 78.4830),
    'Sultan Bazaar':      (17.3870, 78.4870),
    'MG Bus Station':     (17.3790, 78.4810),
    'Malakpet':           (17.3740, 78.4940),
    'New Market':         (17.3700, 78.5010),
    'Musarambagh':        (17.3650, 78.5080),
    'Dilsukhnagar':       (17.3680, 78.5290),
    'Chaitanyapuri':      (17.3650, 78.5370),
    'Victoria Memorial':  (17.3600, 78.5440),
    'LB Nagar':           (17.3460, 78.5500),
    'Madhapur':           (17.4480, 78.3920),
    'Durgam Cheruvu':     (17.4330, 78.3920),
    'HITEC City':         (17.4500, 78.3800),
    'Raidurg':            (17.4280, 78.3700),
    'Begumpet':           (17.4440, 78.4660),
    'Parade Ground':      (17.4480, 78.4990),
    'Secunderabad East':  (17.4350, 78.5040),
    'Paradise':           (17.4430, 78.4920),
    'RTC X Roads':        (17.4040, 78.4960),
    'Chikkadpally':       (17.3990, 78.4920),
}

# ── Famous Hyderabad landmarks (for "nearest landmark" feature) ──
# Replaces the old fixed KBR-only reference — now picks whichever
# well-known landmark is actually closest to the clicked point.
FAMOUS_LANDMARKS = {
    'Charminar':              (17.3616, 78.4747),
    'Golconda Fort':          (17.3833, 78.4011),
    'KBR National Park':      (17.4180, 78.4070),
    'Hussain Sagar Lake':     (17.4239, 78.4738),
    'NTR Gardens':            (17.4125, 78.4737),
    'Birla Mandir':           (17.4062, 78.4691),
    'Salar Jung Museum':      (17.3713, 78.4804),
    'Hyderabad Central Mall': (17.4257, 78.4524),
    'GVK One Mall':           (17.4239, 78.4475),
    'Inorbit Mall':           (17.4337, 78.3923),
    'Sudha Cars Museum':      (17.4123, 78.5210),
    'Ramoji Film City':       (17.2543, 78.6808),
    'Shilparamam':            (17.4530, 78.3720),
    'Necklace Road':          (17.4280, 78.4660),
    'Lumbini Park':           (17.4040, 78.4760),
    'Paradise Circle':        (17.4430, 78.4920),
    'Begumpet Airport':       (17.4530, 78.4680),
}

# ── Facing direction: 99acres FACING codes mapped to compass directions ──
# The raw FACING column from 99acres uses numeric codes 0-8. This is the
# standard mapping used across 99acres listing exports.
FACING_DIRECTION_MAP = {
    '0': 'Not Specified', '1': 'East',      '2': 'North',
    '3': 'North - East',  '4': 'West',      '5': 'South',
    '6': 'South - East',  '7': 'South - West', '8': 'North - West',
    'Unknown': 'Not Specified',
}

# ── Area name lookup for Portfolio Cases ──
# Hyderabad neighbourhood centroids — used to label saved pins with the
# nearest known locality name (simple nearest-centroid reverse geocoding,
# no external API / network call needed).
AREA_CENTROIDS = {
    'Miyapur':        (17.4946, 78.3822), 'Kukatpally':    (17.4849, 78.4080),
    'KPHB':           (17.4858, 78.3920), 'Madhapur':      (17.4480, 78.3920),
    'HITEC City':     (17.4500, 78.3800), 'Gachibowli':    (17.4400, 78.3489),
    'Jubilee Hills':  (17.4317, 78.4069), 'Banjara Hills': (17.4150, 78.4500),
    'Ameerpet':       (17.4374, 78.4482), 'Punjagutta':    (17.4253, 78.4525),
    'Begumpet':       (17.4440, 78.4660), 'Khairatabad':   (17.4080, 78.4660),
    'Narayanguda':    (17.4020, 78.4830), 'Nampally':      (17.3920, 78.4760),
    'Abids':          (17.3870, 78.4790), 'Charminar':     (17.3616, 78.4747),
    'Malakpet':       (17.3740, 78.4940), 'Dilsukhnagar':  (17.3680, 78.5290),
    'LB Nagar':       (17.3460, 78.5500), 'Secunderabad':  (17.4399, 78.4983),
    'Tarnaka':        (17.4280, 78.5210), 'Uppal':         (17.4019, 78.5590),
    'Shamshabad':     (17.2403, 78.4298), 'Manikonda':     (17.4015, 78.3850),
    'Kondapur':       (17.4615, 78.3640), 'Miyapur West':  (17.5000, 78.3700),
    'Attapur':        (17.3760, 78.4290), 'Mehdipatnam':   (17.3920, 78.4380),
    'Tolichowki':     (17.4000, 78.4180), 'Lingampally':   (17.4930, 78.3210),
    'Bowenpally':     (17.4720, 78.4630), 'Alwal':         (17.5040, 78.5050),
}

def predict(lat,lon,bedrooms,area_sqft,age,furnish,balconies,floors,floor_n,facing,amenities):
    LM_NAMES =['HITEC City','Charminar','Jubilee Hills','Airport (RGIA)','Hussain Sagar']
    FEAT_DIST=['dist_hitec_m','dist_charminar_m','dist_jubilee_m','dist_airport_m','dist_hussain_m']
    dists={}
    for lm,fd in zip(LM_NAMES,FEAT_DIST):
        lt,ln=LANDMARKS[lm]; dists[fd]=dist_m(lat,lon,lt,ln)
    raw=np.array([[bedrooms,area_sqft,age,furnish,balconies,floors,floor_n,facing,amenities,
                   dists['dist_hitec_m'],dists['dist_charminar_m'],dists['dist_jubilee_m'],
                   dists['dist_airport_m'],dists['dist_hussain_m']]])
    pv=scaler.transform(raw) if need_scale else raw
    rv=raw   # Rental model is Random Forest — never needs scaling
    price=float(np.expm1(price_model.predict(pv)[0]))
    rent =float(np.expm1(rental_model.predict(rv)[0]))
    return price,rent,dists

def nearest_metro(lat, lon):
    """Returns (station_name, distance_km) of the closest metro station."""
    best_name, best_d = None, float('inf')
    for name,(slat,slon) in METRO_STATIONS.items():
        d = dist_m(lat,lon,slat,slon)
        if d < best_d:
            best_d, best_name = d, name
    return best_name, best_d/1000

def nearest_landmark(lat, lon):
    """Returns (landmark_name, distance_km) of the closest famous Hyderabad landmark."""
    best_name, best_d = None, float('inf')
    for name,(plat,plon) in FAMOUS_LANDMARKS.items():
        d = dist_m(lat,lon,plat,plon)
        if d < best_d:
            best_d, best_name = d, name
    return best_name, best_d/1000

def nearest_area_name(lat, lon):
    """Returns the nearest known Hyderabad locality name for a coordinate."""
    best_name, best_d = None, float('inf')
    for name,(alat,alon) in AREA_CENTROIDS.items():
        d = dist_m(lat,lon,alat,alon)
        if d < best_d:
            best_d, best_name = d, name
    return best_name



# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
if 'tab'         not in st.session_state: st.session_state.tab='map'
if 'clicked_lat' not in st.session_state: st.session_state.clicked_lat=17.4435
if 'clicked_lon' not in st.session_state: st.session_state.clicked_lon=78.3809
if 'prop_type'   not in st.session_state: st.session_state.prop_type='Premium flat'
if 'bedrooms'    not in st.session_state: st.session_state.bedrooms=2
if 'area_sqft'   not in st.session_state: st.session_state.area_sqft=1250
if 'build_year'  not in st.session_state: st.session_state.build_year=2005
if 'scenario'    not in st.session_state: st.session_state.scenario=0
if 'saved_pins'  not in st.session_state: st.session_state.saved_pins=[]
if 'map_style'   not in st.session_state: st.session_state.map_style='Dark'

LM_COLORS={'HITEC City':'#f59e0b','Charminar':'#f87171','Jubilee Hills':'#a78bfa',
           'Airport (RGIA)':'#60a5fa','Hussain Sagar':'#2dd4bf'}
LM_DESC  ={'HITEC City':'IT Hub — Google, Microsoft, Infosys',
           'Charminar':'Historic Old City centre',
           'Jubilee Hills':'Premium residential zone',
           'Airport (RGIA)':'Rajiv Gandhi Intl Airport',
           'Hussain Sagar':'City centre / lake reference'}

# ═══════════════════════════════════════════════════════════════
# NAV BAR
# ═══════════════════════════════════════════════════════════════
t=st.session_state.tab
tabs_html=''
for tid,tlabel in [('map','🗺️ Map & Valuation'),('scenario','✦ Scenario Modeler'),
                   ('leverage','📈 Leverage Analysis'),('portfolio','🗂️ Portfolio Cases')]:
    cls='active' if t==tid else ''
    tabs_html+=f'<button class="nav-tab {cls}" onclick="">{tlabel}</button>'

st.markdown(f"""
<div class="navbar">
  <div class="nav-brand">
    <div class="nav-logo">🏙️</div>
    <div>
      <div class="nav-title">PROPSCOPE HYD</div>
      <div class="nav-sub">Spatial Real Estate Analytics Suite</div>
    </div>
  </div>
  <div style="font-size:10px;color:#94a3b8;letter-spacing:2px;font-family:'JetBrains Mono',monospace;white-space:nowrap;">
    HYDERABAD · 99ACRES DATASET
  </div>
</div>
""", unsafe_allow_html=True)

# Tab switcher using Streamlit buttons styled to look like the nav
st.markdown("""
<style>
/* Force tab buttons to never wrap text */
.stButton button { white-space: nowrap !important; }
/* Make tab row scroll horizontally on very small screens */
[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; overflow-x: auto; }
</style>
""", unsafe_allow_html=True)

tab_cols = st.columns([1.3, 1.2, 1.3, 1.2, 3])
with tab_cols[0]:
    if st.button("🗺️ Map & Valuation",   use_container_width=True,
                 type="primary" if t=='map'       else "secondary"):
        st.session_state.tab='map';       st.rerun()
with tab_cols[1]:
    if st.button("✦ Scenario Modeler",   use_container_width=True,
                 type="primary" if t=='scenario'  else "secondary"):
        st.session_state.tab='scenario';  st.rerun()
with tab_cols[2]:
    if st.button("📈 Leverage Analysis", use_container_width=True,
                 type="primary" if t=='leverage'  else "secondary"):
        st.session_state.tab='leverage';  st.rerun()
with tab_cols[3]:
    if st.button("🗂️ Portfolio Cases",   use_container_width=True,
                 type="primary" if t=='portfolio' else "secondary"):
        st.session_state.tab='portfolio'; st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ── TAB 1: MAP & VALUATION ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════
if st.session_state.tab == 'map':
    left, right = st.columns([1.1, 1], gap="medium")

    with left:
        # Map header
        st.markdown("""
        <div style='padding:0 4px;'>
          <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;'>
            <div>
              <div style='font-family:Space Grotesk,sans-serif;font-weight:700;font-size:15px;
                          letter-spacing:1px;text-transform:uppercase;color:#94a3b8;'>
                HYDERABAD PROPERTY ENGINE MAP
              </div>
            </div>
            <div class="map-active-badge"><div class="dot-pulse"></div>ACTIVE</div>
          </div>
          <div style='font-size:12px;color:#94a3b8;margin-bottom:12px;'>
            Select coordinates on the map grid to recalculate valuations
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Map style toggle
        ms_col1, ms_col2 = st.columns(2)
        with ms_col1:
            if st.button("🗺️ Street Map", use_container_width=True,
                         type="secondary" if st.session_state.map_style!='Street' else "primary"):
                st.session_state.map_style='Street'; st.rerun()
        with ms_col2:
            if st.button("🛰️ Satellite", use_container_width=True,
                         type="secondary" if st.session_state.map_style!='Satellite' else "primary"):
                st.session_state.map_style='Satellite'; st.rerun()

        # Build map
        tile_map = {
            'Dark':      ("CartoDB dark_matter",""),
            'Street':    ("OpenStreetMap",""),
            'Satellite': ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                          "Esri WorldImagery"),
        }
        tu,ta = tile_map.get(st.session_state.map_style, tile_map['Dark'])
        m = folium.Map(location=[17.43,78.42], zoom_start=12, tiles=tu, attr=ta, prefer_canvas=True)

        for name,(lat,lon) in LANDMARKS.items():
            c=LM_COLORS.get(name,'#fff')
            folium.CircleMarker([lat,lon],radius=9,color=c,fill=True,fill_color=c,
                fill_opacity=0.85,weight=2,
                tooltip=folium.Tooltip(
                    f"<b style='color:{c};font-family:monospace'>{name}</b><br>"
                    f"<span style='color:#94a3b8;font-size:11px'>{LM_DESC[name]}</span>",
                    style="background:#f8fafc;border:1px solid #cbd5e1;padding:8px;border-radius:8px;"
                )).add_to(m)

        # Show saved pins
        for pin in st.session_state.saved_pins:
            folium.CircleMarker([pin['lat'],pin['lon']],radius=7,
                color='#ef4444',fill=True,fill_color='#ef4444',fill_opacity=0.7,
                tooltip=f"Saved: ₹{pin['price']/1e5:.1f}L").add_to(m)

        # Active pin
        folium.CircleMarker(
            [st.session_state.clicked_lat, st.session_state.clicked_lon],
            radius=13, color='#10b981', fill=True, fill_color='#10b981',
            fill_opacity=0.3, weight=3,
            tooltip="Active location"
        ).add_to(m)
        folium.CircleMarker(
            [st.session_state.clicked_lat, st.session_state.clicked_lon],
            radius=5, color='#10b981', fill=True, fill_color='#10b981', fill_opacity=1
        ).add_to(m)

        folium.PolyLine([[17.20,78.20],[17.65,78.20],[17.65,78.65],[17.20,78.65],[17.20,78.20]],
            color="#6366f1",weight=1.5,opacity=0.2,dash_array="6 6").add_to(m)

        map_data = st_folium(m, width="100%", height=420, returned_objects=["last_clicked"])

        if map_data and map_data.get("last_clicked"):
            clat = map_data["last_clicked"]["lat"]
            clon = map_data["last_clicked"]["lng"]
            if in_hyd(clat,clon):
                st.session_state.clicked_lat = clat
                st.session_state.clicked_lon = clon
                st.rerun()

        # Coord bar
        clat=st.session_state.clicked_lat; clon=st.session_state.clicked_lon
        st.markdown(f"""
        <div class="coord-bar">
          <div class="coord-item"><span>LATITUDE:</span><b>{clat:.5f}</b></div>
          <div class="coord-item"><span>LONGITUDE:</span><b>{clon:.5f}</b></div>
          <div style="flex:1"></div>
          <div class="coord-val-pill">📍 Active Pin</div>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("<div style='padding:0 4px;'>", unsafe_allow_html=True)

        # — Investment Specs —
        st.markdown("""
        <div style='margin-bottom:16px;'>
          <div style='font-family:Space Grotesk,sans-serif;font-size:18px;font-weight:700;color:#1e293b;'>
            ⚙️ Investment Specs
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Property type
        prop_types = ['Premium flat','Villa','Penthouse','Standard Apt']
        # Multipliers applied on top of base ML prediction — the dataset has no
        # explicit property-type column, so this scales the base valuation
        # to reflect typical Hyderabad market premiums for each category.
        PROP_TYPE_MULT = {
            'Premium flat': 1.00,
            'Villa':        1.45,
            'Penthouse':    1.30,
            'Standard Apt': 0.85,
        }
        pt_cols = st.columns(4)
        for i,pt in enumerate(prop_types):
            with pt_cols[i]:
                sel = st.session_state.prop_type==pt
                if st.button(pt, key=f"pt_{i}", use_container_width=True,
                             type="primary" if sel else "secondary"):
                    st.session_state.prop_type=pt; st.rerun()
        st.markdown(
            f"<div style='font-size:11px;color:#94a3b8;margin-top:-6px;'>"
            f"{st.session_state.prop_type} applies a "
            f"<b style='color:#f59e0b;'>{PROP_TYPE_MULT[st.session_state.prop_type]:.2f}x</b> "
            f"market premium vs a standard flat.</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Sliders
        area  = st.slider("Useable Super Built-up Area (SqFt)", 400, 5000,
                          st.session_state.area_sqft, 50, key="area_sl")
        st.session_state.area_sqft=area

        bhk_col, bath_col = st.columns(2)
        with bhk_col:
            beds = st.selectbox("Bedrooms Count",
                [1,2,3,4,5,6], index=st.session_state.bedrooms-1, key="beds_sel")
            st.session_state.bedrooms=beds
        with bath_col:
            baths = st.selectbox("Bathrooms Count", [1,2,3,4,5], index=1, key="bath_sel")
        # Each bathroom beyond 1 adds a small premium (real signal: ensuite
        # bathrooms raise resale value, roughly 1.5% per extra bathroom)
        bath_mult = 1.0 + (baths - 1) * 0.015

        build_yr = st.slider("Construction Vintage", 1990, 2025,
                             st.session_state.build_year, 1, key="yr_sl")
        st.session_state.build_year=build_yr
        age = 2026 - build_yr

        furnish_map={"Unfurnished":0,"Semi-Furnished":1,"Fully Furnished":2}
        furnish_lbl=st.selectbox("Furnishing Status",list(furnish_map.keys()),index=1,key="furn_sel")
        furnish_val=furnish_map[furnish_lbl]

        # Facing direction — map raw codes (e.g. '0'..'8') to compass names.
        # FACING_CLS holds the raw label-encoder classes (strings like '0','1'...);
        # we display the human-readable direction but keep the underlying
        # encoded index for the model.
        facing_display = [FACING_DIRECTION_MAP.get(c, c) for c in FACING_CLS] \
                          if FACING_CLS else ["East","North","West","South","Not Specified"]
        # Deduplicate while preserving order + index mapping
        seen = {}
        display_options = []
        option_to_idx = []
        for idx, disp in enumerate(facing_display):
            if disp not in seen:
                seen[disp] = True
                display_options.append(disp)
                option_to_idx.append(idx)

        facing_choice = st.selectbox("Facing Direction", display_options, key="face_sel")
        facing_val = option_to_idx[display_options.index(facing_choice)]

        # Amenities checklist — replaces the old "count" slider.
        # The model only needs a count, but showing the actual checklist
        # makes it clear *which* amenities you're including.
        AMENITY_LIST = [
            "Swimming Pool","Gymnasium","Clubhouse","24/7 Security",
            "Power Backup","Lift / Elevator","Children's Play Area",
            "Jogging Track","Indoor Games","Visitor Parking",
            "CCTV Surveillance","Intercom","Rainwater Harvesting","Garden / Park",
        ]
        st.markdown("<div style='font-size:13px;color:#94a3b8;margin-top:4px;margin-bottom:4px;'>Amenities</div>",
                    unsafe_allow_html=True)
        DEFAULT_ON = {"24/7 Security","Power Backup","Lift / Elevator","Visitor Parking"}
        amen_cols = st.columns(2)
        selected_amenities = []
        for i,a in enumerate(AMENITY_LIST):
            with amen_cols[i % 2]:
                checked = st.checkbox(a, value=(a in DEFAULT_ON), key=f"amen_{i}")
                if checked: selected_amenities.append(a)
        amenities = len(selected_amenities)
        st.markdown(f"<div style='font-size:11px;color:#94a3b8;margin-top:-4px;'>"
                    f"{amenities} of {len(AMENITY_LIST)} amenities selected</div>",
                    unsafe_allow_html=True)

        # ─ Predict ─
        clat=st.session_state.clicked_lat; clon=st.session_state.clicked_lon
        price, rent, dists = predict(clat,clon,beds,area,age,furnish_val,1,12,4,facing_val,amenities)
        # Apply property-type and bathroom multipliers on top of the base ML prediction
        type_mult = PROP_TYPE_MULT[st.session_state.prop_type]
        price = price * type_mult * bath_mult
        rent  = rent  * type_mult * bath_mult
        annual_rent=rent*12
        cap_rate=annual_rent/price if price>0 else 0
        hitec_km=dists['dist_hitec_m']/1000

        st.markdown("<div class='sec-label'>VALUATION (PREDICTED PRICE) &nbsp;&nbsp; ANNUAL YIELD CAP RATE</div>",
                    unsafe_allow_html=True)
        v1,v2=st.columns(2)
        with v1:
            st.markdown(f"<div class='val-big'>₹{price:,.0f}</div>", unsafe_allow_html=True)
        with v2:
            st.markdown(f"<div style='font-family:Space Grotesk,sans-serif;font-size:28px;"
                        f"font-weight:700;color:#10b981;'>{cap_rate*100:.2f}% <span style='font-size:14px;color:#94a3b8;'>p.a.</span></div>",
                        unsafe_allow_html=True)

        # Spatial specs — real distances computed from the clicked coordinate
        metro_name, metro_dist = nearest_metro(clat, clon)
        landmark_name, landmark_dist = nearest_landmark(clat, clon)
        # Commercial density: higher near HITEC City / Gachibowli IT corridor,
        # decays with distance from the tech cluster (proxy score, 1-10)
        comm_score = min(10, max(1, int(10*(1-hitec_km/20))))

        for label,val,clr in [
            ("Nearest Metro Station:", f"{metro_name} Metro", "#1e293b"),
            ("Metro Transit Distance:", f"{metro_dist:.3f} km", "#1e293b"),
            (f"{landmark_name} distance:", f"{landmark_dist:.3f} km", "#1e293b"),
            ("Commercial Density score:", f"{comm_score}/10", "#10b981"),
        ]:
            st.markdown(f"""
            <div class="spec-row">
              <span class="spec-label">{label}</span>
              <span class="spec-val" style="color:{clr};">{val}</span>
            </div>""", unsafe_allow_html=True)

        # AI Thesis
        st.markdown("<div class='sec-label' style='margin-top:14px;'>🤖 HYDERABAD AI INVESTMENT THESIS</div>",
                    unsafe_allow_html=True)
        thesis = (
            f'The precise tech-corridor coordinates at (Lat {clat:.4f}, Lng {clon:.4f}) '
            f'command a strong valuation of ₹{price:,.0f} (approx. ₹{price/area:,.0f}/SqFt). '
            f'This is heavily leveraged by premium micro-spatial access, located just {metro_dist:.2f} km '
            f'from {metro_name} Metro station, which serves as a vital infrastructure backbone. '
            f'Nearby IT corridors like Cyber Towers HITEC City ({hitec_km:.3f} km) generate high '
            f'recurring commercial tenant demand. {landmark_name} ({landmark_dist:.2f} km) serves as a '
            f'notable nearby landmark, supporting neighbourhood desirability and capital appreciation. '
            f'This {beds} BHK {st.session_state.prop_type} (built {build_yr}) maintains a robust '
            f'risk-adjusted profile with a {cap_rate*100:.2f}% annual yield, representing a '
            f'high-liquidity real-estate holding in Hyderabad\'s booming tech belt. '
            f'[Analyst Sandbox Mode: High-Fidelity Local Model]'
        )
        st.markdown(f'<div class="thesis-box">"{thesis}"</div>', unsafe_allow_html=True)

        # Save pin + Forecaster buttons
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        sv_col, fc_col = st.columns([2,1])
        with sv_col:
            if st.button("＋  SAVE PIN COORDINATE", use_container_width=True, type="primary", key="save_pin"):
                st.session_state.saved_pins.append({
                    'lat':clat,'lon':clon,'price':price,'rent':rent,
                    'area':area,'beds':beds,'type':st.session_state.prop_type,
                    'area_name': nearest_area_name(clat, clon),
                })
                st.toast(f"📍 Pin saved — ₹{price/1e5:.1f}L")
        with fc_col:
            if st.button("Forecaster →", use_container_width=True, key="go_lev"):
                st.session_state.tab='leverage'; st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ── TAB 2: SCENARIO MODELER ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════
elif st.session_state.tab == 'scenario':
    st.markdown("""
    <div style='padding:20px 0 0;'>
      <div style='font-size:13px;color:#94a3b8;margin-bottom:20px;'>
        Build and compare asset leverage plans. Customize renovation impact, down payment structures, and macro rates side-by-side.
      </div>
    </div>
    """, unsafe_allow_html=True)

    pad = st.container()
    with pad:
        # Scenario presets
        SCENARIOS = [
            {"name":"Safe Buy-and-Hold",
             "desc":"Standard institutional approach. Low leverage, minor touch-up work, reliance on organic Hyderabad appreciation.",
             "furnishing":150_000,"down":0.25,"interest":0.085,"appreciation":0.085},
            {"name":"Modern Luxury Upgrade",
             "desc":"High value-add renovation strategy. Premium furnishing to attract IT executives, command lease premiums, and boost equity.",
             "furnishing":1_500_000,"down":0.20,"interest":0.09,"appreciation":0.12},
            {"name":"Leveraged Capital Play",
             "desc":"Aggressive low-down payment entry to maximize asset exposure with minimal upfront capital.",
             "furnishing":50_000,"down":0.15,"interest":0.095,"appreciation":0.075},
        ]

        sc_cols=st.columns(3, gap="medium")
        for i,(sc,col) in enumerate(zip(SCENARIOS,sc_cols)):
            with col:
                sel=st.session_state.scenario==i
                border='#6366f1' if sel else '#cbd5e1'
                bg='#eef2ff' if sel else '#ffffff'
                dot='🔵' if sel else ''
                st.markdown(f"""
                <div style='background:{bg};border:1px solid {border};border-radius:12px;padding:20px;'>
                  <div style='font-family:Space Grotesk,sans-serif;font-size:16px;font-weight:700;
                              color:#1e293b;margin-bottom:6px;'>{sc['name']} {dot}</div>
                  <div style='font-size:12px;color:#94a3b8;line-height:1.6;margin-bottom:14px;'>{sc['desc']}</div>
                  <div style='display:grid;grid-template-columns:1fr 1fr;gap:4px;'>
                    <div style='font-size:11px;color:#64748b;'>Furnishing: <b style='color:#94a3b8;'>₹{sc['furnishing']:,.0f}</b></div>
                    <div style='font-size:11px;color:#64748b;'>Down Pay: <b style='color:#94a3b8;'>{sc['down']*100:.0f}%</b></div>
                    <div style='font-size:11px;color:#64748b;'>Interest: <b style='color:#94a3b8;'>{sc['interest']*100:.1f}%</b></div>
                    <div style='font-size:11px;color:#64748b;'>Appreciation: <b style='color:#94a3b8;'>{sc['appreciation']*100:.1f}%</b></div>
                  </div>
                </div>""", unsafe_allow_html=True)
                if st.button(f"Select", key=f"sc_{i}", use_container_width=True):
                    st.session_state.scenario=i; st.rerun()

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        sc=SCENARIOS[st.session_state.scenario]

        # Get base price from current pin
        clat=st.session_state.clicked_lat; clon=st.session_state.clicked_lon
        beds=st.session_state.bedrooms; area=st.session_state.area_sqft
        age=2026-st.session_state.build_year
        base_price,base_rent,_=predict(clat,clon,beds,area,age,1,1,12,4,0,10)
        _pt_mult={"Premium flat":1.00,"Villa":1.45,"Penthouse":1.30,"Standard Apt":0.85}.get(st.session_state.prop_type,1.0)
        base_price*=_pt_mult; base_rent*=_pt_mult

        left2,right2=st.columns([1,1.1],gap="medium")

        with left2:
            st.markdown("""
            <div style='font-family:Space Grotesk,sans-serif;font-size:16px;font-weight:700;
                        color:#1e293b;margin-bottom:14px;display:flex;align-items:center;gap:8px;'>
              📋 Strategy Input Parameters
            </div>""", unsafe_allow_html=True)

            furnishing_budget=st.slider("Renovation & Furnishing Budget (₹)",
                0, 5_000_000, int(sc['furnishing']), 50_000,
                format="₹%d", key="furn_bud")
            down_pct=st.slider("Scenario Down Payment (%)",10,100,int(sc['down']*100),5,key="sc_down")
            interest_rate=st.slider("Active Interest Rate (%)",6.0,15.0,sc['interest']*100,0.1,
                                    format="%.1f%%",key="sc_int")/100
            appre_rate=st.slider("Expected Appreciation Rate (% / yr)",2.0,20.0,
                                 sc['appreciation']*100,0.5,format="%.1f%%",key="sc_app")/100

            monthly_rate = interest_rate / 12
            n = 240
            total_price  = base_price + furnishing_budget
            loan_amt     = total_price * (1 - down_pct / 100)
            emi = (loan_amt * monthly_rate * (1+monthly_rate)**n /
                   ((1+monthly_rate)**n - 1)) if loan_amt > 0 else 0
            # 10-year equity: property value minus remaining loan balance
            # Remaining balance after 10yr (120 payments) on a 20yr loan
            rem_bal = (loan_amt * ((1+monthly_rate)**n - (1+monthly_rate)**120) /
                       ((1+monthly_rate)**n - 1)) if loan_amt > 0 else 0
            equity_10 = base_price * (1 + appre_rate)**10 - rem_bal
            # ROI on initial cash outlay
            total_cash = total_price * (down_pct/100) + furnishing_budget
            gain_10 = equity_10 - base_price + base_rent * 12 * 10 - emi * 12 * 10
            roi_10 = gain_10 / total_cash if total_cash > 0 else 0
            roi_10 = max(-9.99, min(roi_10, 99.99))  # clamp to ±999%

            st.markdown(f"""
            <div style='background:#f8fafc;border:1px solid #cbd5e1;border-radius:10px;
                        padding:16px 18px;margin-top:16px;font-size:12px;color:#94a3b8;line-height:1.8;'>
              <span style='color:#6366f1;'>🛡</span>
              <b style='color:#94a3b8;'> Value-Add Forced Appreciation</b><br>
              Adding remodel work builds equity and commands rent premiums. Renovated properties
              are modeled with a <b style='color:#f59e0b;'>1.25x value multi-lift</b> and rental income scale.
            </div>""", unsafe_allow_html=True)

        with right2:
            # KPI cards (total_price, loan_amt, emi, roi_10 computed above)

            k1,k2,k3=st.columns(3)
            with k1:
                st.markdown(f"""<div class="kpi">
                  <div class="kpi-label">TOTAL INITIAL CASH</div>
                  <div class="kpi-val white" style='font-size:18px;'>₹{total_cash:,.1f}</div>
                  <div class="kpi-sub">Down payment + furnishing</div>
                </div>""",unsafe_allow_html=True)
            with k2:
                clr='green' if roi_10>0 else 'red'
                st.markdown(f"""<div class="kpi">
                  <div class="kpi-label">ADJUSTED YIELD (ROI)</div>
                  <div class="kpi-val {clr}">{'+' if roi_10>0 else ''}{roi_10*100:.1f}%</div>
                  <div class="kpi-sub">10-Year cumulative compounding</div>
                </div>""",unsafe_allow_html=True)
            with k3:
                breakeven=1 if roi_10>0 else int(total_cash/(base_rent*12+base_price*appre_rate))+1
                st.markdown(f"""<div class="kpi">
                  <div class="kpi-label">CAPITAL BREAK-EVEN</div>
                  <div class="kpi-val teal">Year {min(breakeven,10)}</div>
                  <div class="kpi-sub">Profit equals spent capital</div>
                </div>""",unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)

            # 10-year chart using SVG
            years=list(range(1,11))
            prop_vals=[base_price*(1+appre_rate)**y for y in years]
            cum_returns=[base_rent*12*y + base_price*((1+appre_rate)**y - 1) - emi*12*y for y in years]
            equity_built=[base_price*(1+appre_rate)**y - loan_amt*max(0,1-y/20) for y in years]

            # Normalise to chart space
            all_vals=prop_vals+[abs(v) for v in cum_returns]+equity_built
            max_v=max(all_vals); min_v=min(min(cum_returns),0)
            chart_h=220; chart_w=520; pad_l=60; pad_b=30; pad_t=20

            def to_x(yr): return pad_l+(yr-1)/(9)*(chart_w-pad_l-10)
            def to_y(v):  return pad_t+chart_h*(1-(v-min_v)/(max_v-min_v+1))

            def polyline(vals,color,width=2):
                pts=' '.join(f"{to_x(y+1):.1f},{to_y(v):.1f}" for y,v in enumerate(vals))
                return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'

            # Y-axis labels
            y_labels=''
            for v in [min_v,0,(max_v+min_v)/2,max_v]:
                y_label=f'₹{v/1e5:.0f}L'
                y_labels+=f'<text x="{pad_l-6}" y="{to_y(v)+4}" text-anchor="end" font-size="10" fill="#94a3b8" font-family="JetBrains Mono">{y_label}</text>'
                y_labels+=f'<line x1="{pad_l}" y1="{to_y(v)}" x2="{chart_w-10}" y2="{to_y(v)}" stroke="#e2e8f0" stroke-width="1"/>'

            x_labels=''.join(f'<text x="{to_x(y)}" y="{pad_t+chart_h+18}" text-anchor="middle" font-size="10" fill="#94a3b8" font-family="JetBrains Mono">Yr {y}</text>' for y in range(1,11,2))

            zero_y=to_y(0)
            svg=f"""
            <div style='background:#ffffff;border:1px solid #cbd5e1;border-radius:12px;padding:16px;'>
              <div style='font-family:Space Grotesk,sans-serif;font-size:14px;font-weight:600;
                          color:#1e293b;margin-bottom:4px;'>10-Year Scenario Performance Trend</div>
              <div style='font-size:11px;color:#94a3b8;margin-bottom:12px;'>
                Visualizing running profit accumulation and renovation budget payoff vectors.</div>
              <svg viewBox="0 0 {chart_w} {pad_t+chart_h+pad_b}" style="width:100%;overflow:visible;">
                {y_labels}{x_labels}
                <line x1="{pad_l}" y1="{zero_y}" x2="{chart_w-10}" y2="{zero_y}" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4 4"/>
                {polyline(prop_vals,'#14b8a6',2.5)}
                {polyline(cum_returns,'#6366f1',2)}
              </svg>
              <div style='display:flex;gap:16px;justify-content:center;margin-top:8px;'>
                <div style='font-size:11px;color:#14b8a6;display:flex;align-items:center;gap:5px;'>
                  <div style='width:20px;height:2px;background:#14b8a6;border-radius:1px;'></div>Asset Value
                </div>
                <div style='font-size:11px;color:#6366f1;display:flex;align-items:center;gap:5px;'>
                  <div style='width:20px;height:2px;background:#6366f1;border-radius:1px;'></div>Cumulative Return
                </div>
              </div>
              <div style='display:flex;justify-content:space-between;margin-top:14px;
                          padding-top:12px;border-top:1px solid #e2e8f0;'>
                <div style='font-size:12px;color:#94a3b8;'>Estimated Adjusted Rent:
                  <b style='color:#1e293b;'>₹{base_rent*1.25/1000:.0f}K/mo</b></div>
                <div style='font-size:12px;color:#94a3b8;'>Est. Monthly EMI Outlay:
                  <b style='color:#ef4444;'>-₹{emi/1000:.0f}K/mo</b></div>
              </div>
            </div>"""
            st.markdown(svg, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ── TAB 3: LEVERAGE ANALYSIS ────────────────────────────────
# ═══════════════════════════════════════════════════════════════
elif st.session_state.tab == 'leverage':
    st.markdown("""
    <div style='padding:20px 0 0;'>
      <div style='font-family:Space Grotesk,sans-serif;font-size:22px;font-weight:700;
                  color:#1e293b;margin-bottom:4px;'>📈 Leverage & CapEx Analysis</div>
      <div style='font-size:13px;color:#94a3b8;margin-bottom:20px;'>
        Model capital expenditure, EMI debt service, and long-term asset appreciation for the active property pin.
      </div>
    </div>""", unsafe_allow_html=True)

    clat=st.session_state.clicked_lat; clon=st.session_state.clicked_lon
    beds=st.session_state.bedrooms; area=st.session_state.area_sqft
    age=2026-st.session_state.build_year
    base_price,base_rent,_=predict(clat,clon,beds,area,age,1,1,12,4,0,10)
    _pt_mult={"Premium flat":1.00,"Villa":1.45,"Penthouse":1.30,"Standard Apt":0.85}.get(st.session_state.prop_type,1.0)
    base_price*=_pt_mult; base_rent*=_pt_mult

    lev_left,lev_right=st.columns([1,1.4],gap="medium")

    with lev_left:
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #cbd5e1;border-radius:14px;padding:22px 24px;'>
          <div style='font-family:Space Grotesk,sans-serif;font-size:16px;font-weight:700;
                      color:#1e293b;margin-bottom:4px;display:flex;align-items:center;gap:8px;'>
            📋 CapEx Assumptions
          </div>
          <div style='font-size:12px;color:#94a3b8;margin-bottom:18px;'>
            Adjust transactional financial parameters to model capital gain yield and operational leveraging options.
          </div>
        </div>""", unsafe_allow_html=True)

        down_pct2=st.slider("Down Payment Outlay (%)",10,100,20,5,
            help="10% Low Leverage → 100% Cash Purchase",key="lev_down")
        interest2=st.slider("Home Loan Interest Rate (% P&I)",6.0,15.0,6.2,0.1,
            format="%.1f%%",key="lev_int")/100
        appre2=st.slider("Projected Annual Capital Appreciation (%)",1.0,15.0,4.2,0.1,
            format="%.1f%%",key="lev_app")/100

        loan_amt2=base_price*(1-down_pct2/100)
        monthly_rate2=interest2/12; n2=240
        emi2=loan_amt2*(monthly_rate2*(1+monthly_rate2)**n2)/((1+monthly_rate2)**n2-1) if loan_amt2>0 else 0

        st.markdown(f"""
        <div style='background:#ffffff;border:1px solid #cbd5e1;border-radius:12px;padding:16px 18px;margin-top:8px;'>
          <div class="lev-row">
            <span class="lev-label">Total Purchase Value</span>
            <span class="lev-val">₹{base_price:,.1f}</span>
          </div>
          <div class="lev-row">
            <span class="lev-label">Home Loan Amount</span>
            <span class="lev-val">₹{loan_amt2:,.1f}</span>
          </div>
          <div class="lev-row" style='border:none;'>
            <span class="lev-label">Monthly Installment (EMI est.)</span>
            <span class="lev-val red">-₹{emi2:,.0f}/mo</span>
          </div>
        </div>""", unsafe_allow_html=True)

    with lev_right:
        # KPI row
        equity_10y=base_price*(1+appre2)**10 - loan_amt2*max(0,1-10/20)
        cum_cf=base_rent*12*10 - emi2*12*10
        roi_mult=(equity_10y/base_price-1)*100 if base_price>0 else 0

        k1,k2,k3=st.columns(3)
        with k1:
            st.markdown(f"""<div class="kpi">
              <div class="kpi-label">10-YR EQUITY ACCUMULATION</div>
              <div class="kpi-val white" style='font-size:17px;'>₹{equity_10y:,.0f}</div>
            </div>""",unsafe_allow_html=True)
        with k2:
            clr='green' if cum_cf>0 else 'red'
            st.markdown(f"""<div class="kpi">
              <div class="kpi-label">CUMULATIVE CASH FLOW</div>
              <div class="kpi-val {clr}" style='font-size:17px;'>{"+" if cum_cf>0 else ""}₹{cum_cf:,.0f}</div>
            </div>""",unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi">
              <div class="kpi-label">NET R.O.I. MULTIPLIER</div>
              <div class="kpi-val indigo" style='font-size:22px;'>{roi_mult:.0f}%</div>
            </div>""",unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)

        # 10-year chart: Asset Value / Equity Built / Remaining Debt
        years=list(range(1,11))
        asset_vals=[base_price*(1+appre2)**y for y in years]
        remaining_debt=[max(0,loan_amt2*(1-y/20)) for y in years]
        equity_built=[av-rd for av,rd in zip(asset_vals,remaining_debt)]

        all_v=asset_vals+equity_built+remaining_debt
        max_v2=max(all_v); min_v2=0
        chart_h2=220; chart_w2=560; pl=65; pb=30; pt=20

        def tx2(yr): return pl+(yr-1)/9*(chart_w2-pl-10)
        def ty2(v):  return pt+chart_h2*(1-v/(max_v2+1))

        def fill_area(vals,color,opacity=0.15):
            pts_top=' '.join(f"{tx2(y+1):.1f},{ty2(v):.1f}" for y,v in enumerate(vals))
            pts_bot=f"{tx2(10):.1f},{ty2(0):.1f} {tx2(1):.1f},{ty2(0):.1f}"
            return f'<polygon points="{pts_top} {pts_bot}" fill="{color}" opacity="{opacity}"/>'

        def line2(vals,color,w=2):
            pts=' '.join(f"{tx2(y+1):.1f},{ty2(v):.1f}" for y,v in enumerate(vals))
            return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round"/>'

        yl2=''
        for v in [0,max_v2/3,2*max_v2/3,max_v2]:
            yl2+=f'<text x="{pl-6}" y="{ty2(v)+4}" text-anchor="end" font-size="10" fill="#94a3b8" font-family="JetBrains Mono">₹{v/1e5:.0f}L</text>'
            yl2+=f'<line x1="{pl}" y1="{ty2(v)}" x2="{chart_w2-10}" y2="{ty2(v)}" stroke="#e2e8f0" stroke-width="1"/>'

        xl2=''.join(f'<text x="{tx2(y)}" y="{pt+chart_h2+18}" text-anchor="middle" font-size="10" fill="#94a3b8" font-family="JetBrains Mono">Yr {y}</text>' for y in range(1,11))

        svg2=f"""
        <div style='background:#ffffff;border:1px solid #cbd5e1;border-radius:12px;padding:18px;'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;'>
            <div>
              <div style='font-family:Space Grotesk,sans-serif;font-size:14px;font-weight:600;color:#1e293b;'>
                10-Year Asset Appreciation & Wealth Built</div>
              <div style='font-size:11px;color:#94a3b8;margin-top:3px;'>
                Equity value expands as Hyderabad property appreciates and outstanding loan decays.</div>
            </div>
            <div class="badge badge-indigo">Projected Amortization</div>
          </div>
          <svg viewBox="0 0 {chart_w2} {pt+chart_h2+pb}" style="width:100%;overflow:visible;">
            {yl2}{xl2}
            {fill_area(asset_vals,'#14b8a6',0.1)}
            {fill_area(equity_built,'#6366f1',0.12)}
            {line2(asset_vals,'#14b8a6',2.5)}
            {line2(equity_built,'#6366f1',2)}
            {line2(remaining_debt,'#f97316',1.5)}
            <line x1="{tx2(1)}" y1="{ty2(0)}" x2="{tx2(10)}" y2="{ty2(0)}" stroke="#ef4444" stroke-width="1" stroke-dasharray="4 4" opacity="0.4"/>
          </svg>
          <div style='display:flex;gap:16px;justify-content:center;margin-top:10px;'>
            <div style='font-size:11px;color:#14b8a6;display:flex;align-items:center;gap:5px;'>
              <div style='width:20px;height:2.5px;background:#14b8a6;'></div>Asset Value
            </div>
            <div style='font-size:11px;color:#6366f1;display:flex;align-items:center;gap:5px;'>
              <div style='width:20px;height:2px;background:#6366f1;'></div>Equity / Value Built
            </div>
            <div style='font-size:11px;color:#f97316;display:flex;align-items:center;gap:5px;'>
              <div style='width:20px;height:2px;background:#f97316;'></div>Remaining Debt
            </div>
          </div>
        </div>"""
        st.markdown(svg2, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ── TAB 4: PORTFOLIO CASES ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════
elif st.session_state.tab == 'portfolio':
    st.markdown("""
    <div style='padding:20px 0 0;'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>
        <div style='font-family:Space Grotesk,sans-serif;font-size:22px;font-weight:700;color:#1e293b;'>
          🗂️ Portfolio Cases
        </div>
      </div>
      <div style='font-size:13px;color:#94a3b8;margin-bottom:20px;'>
        Saved coordinate pins for side-by-side investment comparison.
      </div>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.saved_pins:
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                    padding:60px 24px;text-align:center;'>
          <div style='font-size:32px;margin-bottom:12px;'>🗂️</div>
          <div style='font-family:Space Grotesk,sans-serif;font-size:17px;font-weight:600;
                      color:#1e293b;margin-bottom:6px;'>No Location Pins Saved</div>
          <div style='font-size:13px;color:#94a3b8;max-width:420px;margin:0 auto;line-height:1.7;'>
            Use the <b style='color:#1e293b;'>"Save Coordinate Pin"</b> button on the
            Map & Valuation tab to pin multiple geographic grids and weigh returns side-by-side!
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Header
        cols_header=st.columns([1,1,1,1,1,0.5])
        headers=['Type','Location','Price','Rent/mo','Yield','']
        for col,h in zip(cols_header,headers):
            with col:
                st.markdown(f"<div style='font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#94a3b8;font-family:JetBrains Mono,monospace;padding:0 8px 8px;border-bottom:1px solid #e2e8f0;'>{h}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)

        pins_to_remove=[]
        for i,pin in enumerate(st.session_state.saved_pins):
            cap=pin['price']*0.035/12
            yield_pct=(cap*12/pin['price'])*100
            c1,c2,c3,c4,c5,c6=st.columns([1,1,1,1,1,0.5])
            with c1: st.markdown(f"<div style='padding:12px 8px;'><span class='badge badge-green'>{pin.get('type','Flat')}</span></div>",unsafe_allow_html=True)
            with c2:
                area_name = pin.get('area_name') or nearest_area_name(pin['lat'], pin['lon'])
                st.markdown(
                    f"<div style='padding:12px 8px;'>"
                    f"<div style='font-size:13px;color:#1e293b;font-weight:600;'>{area_name}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6366f1;margin-top:2px;'>"
                    f"{pin['lat']:.4f}°N, {pin['lon']:.4f}°E</div></div>",
                    unsafe_allow_html=True)
            with c3: st.markdown(f"<div style='padding:12px 8px;font-family:Space Grotesk,sans-serif;font-size:15px;font-weight:700;color:#1e293b;'>₹{pin['price']/1e5:.1f}L</div>",unsafe_allow_html=True)
            with c4: st.markdown(f"<div style='padding:12px 8px;font-family:JetBrains Mono,monospace;font-size:13px;color:#14b8a6;'>₹{pin['rent']/1000:.1f}K</div>",unsafe_allow_html=True)
            with c5: st.markdown(f"<div style='padding:12px 8px;font-family:Space Grotesk,sans-serif;font-size:15px;font-weight:600;color:#10b981;'>{yield_pct:.2f}%</div>",unsafe_allow_html=True)
            with c6:
                if st.button("✕",key=f"del_{i}",help="Remove pin"):
                    pins_to_remove.append(i)

        for i in sorted(pins_to_remove,reverse=True):
            st.session_state.saved_pins.pop(i)
        if pins_to_remove: st.rerun()

        if len(st.session_state.saved_pins)>=2:
            st.markdown("<div class='sec-label' style='margin:20px 0 12px;'>COMPARISON CHART</div>",
                        unsafe_allow_html=True)
            prices=[p['price']/1e5 for p in st.session_state.saved_pins]
            rents =[p['rent']/1000 for p in st.session_state.saved_pins]
            labels=[f"Pin {i+1}" for i in range(len(prices))]
            max_p=max(prices); chart_w3=500; bar_h=28; gap=10
            pt4=20  # chart top padding

            bars=''
            for i,(lbl,prc,rnt) in enumerate(zip(labels,prices,rents)):
                y_pos=pt4+i*(bar_h+gap)
                w=prc/max_p*(chart_w3-120)
                bars+=f'<rect x="80" y="{y_pos}" width="{w:.0f}" height="{bar_h}" fill="#6366f1" rx="4" opacity="0.8"/>'
                bars+=f'<text x="76" y="{y_pos+bar_h/2+4}" text-anchor="end" font-size="11" fill="#64748b" font-family="Inter">{lbl}</text>'
                bars+=f'<text x="{80+w+6}" y="{y_pos+bar_h/2+4}" font-size="11" fill="#94a3b8" font-family="JetBrains Mono">₹{prc:.1f}L</text>'

            total_h=len(prices)*(bar_h+gap)+40
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #cbd5e1;border-radius:12px;padding:16px;'>
              <div style='font-size:13px;font-weight:600;color:#1e293b;margin-bottom:12px;'>Price Comparison (₹ Lakhs)</div>
              <svg viewBox="0 0 {chart_w3} {total_h}" style="width:100%;">{bars}</svg>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="eng-footer">
  <div style='display:flex;gap:24px;'>
    <span>GRID: HYD_SPATIAL_CLUSTER_04</span>
    <span>METADATA: SEEDED_PROPERTIES_ACTIVE</span>
    <span>METRO_TRANSIT: COMPILATION_COMPLETE</span>
  </div>
  <div>v4.3.0-HYD &nbsp; © 2026 HYDERABAD SPATIAL RE SUITE</div>
</div>
""", unsafe_allow_html=True)