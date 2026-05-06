import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import pickle
import holidays
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Google Drive ühendus ──────────────────────────────────────────────────────
KAUSTA_ID = "1Mga1HaM6xaTb_R84tnNQUVvJ39fEzWSu"

@st.cache_resource
def drive_teenus():
    info = dict(st.secrets["gcp_service_account"])
    info["private_key"] = info["private_key"].replace("\\n", "\n")
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)

def laadi_drive_fail(teenus, failinimi):
    tulemused = teenus.files().list(
        q=f"name='{failinimi}' and '{KAUSTA_ID}' in parents",
        fields="files(id, name)"
    ).execute()
    failid = tulemused.get("files", [])
    if not failid:
        raise FileNotFoundError(f"Fail '{failinimi}' ei leitud Drive'ist")
    faili_id = failid[0]["id"]
    paring   = teenus.files().get_media(fileId=faili_id)
    puhver   = io.BytesIO()
    allalaadimine = MediaIoBaseDownload(puhver, paring)
    valmis = False
    while not valmis:
        _, valmis = allalaadimine.next_chunk()
    puhver.seek(0)
    return puhver

# ── Mudelite laadimine ────────────────────────────────────────────────────────
@st.cache_resource
def laadi_mudelid():
    teenus = drive_teenus()
    rf2          = joblib.load(laadi_drive_fail(teenus, "mudel_rf2.pkl"))
    xgb2         = joblib.load(laadi_drive_fail(teenus, "mudel_xgb2.pkl"))
    tunnused_uus = pickle.load(laadi_drive_fail(teenus, "tunnused_uus.pkl"))
    return rf2, xgb2, tunnused_uus

# ── Andmete laadimine Drive'ist ───────────────────────────────────────────────
@st.cache_data(ttl=86400)
def laadi_ajalugu():
    teenus = drive_teenus()
    puhver = laadi_drive_fail(teenus, "andmed_2020_2026.csv")
    df     = pd.read_csv(puhver)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

@st.cache_data(ttl=86400)
def laadi_testperiood():
    teenus = drive_teenus()
    puhver = laadi_drive_fail(teenus, "prognoos_testperiood_uus.csv")
    df     = pd.read_csv(puhver)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def laadi_päevane_prognoos(teenus, päev):
    try:
        puhver = laadi_drive_fail(teenus, f"prognoos_{päev}.csv")
        df     = pd.read_csv(puhver)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except:
        return None

# ── Elering API ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def tõmba_elering_viitajad():
    url      = "https://dashboard.elering.ee/api/system/with-plan"
    praegu   = datetime.now(timezone.utc)
    algus    = praegu - timedelta(hours=384)
    kõik     = []
    praegune = algus
    while praegune < praegu:
        järgmine = min(praegune + timedelta(days=30), praegu)
        params   = {
            "start": praegune.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   järgmine.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        vastus = requests.get(url, params=params)
        data   = vastus.json()["data"]
        df_r   = pd.DataFrame(data["real"])
        df_r["timestamp"] = pd.to_datetime(df_r["timestamp"], unit="s", utc=True)
        df_p   = pd.DataFrame(data["plan"])
        df_p["timestamp"] = pd.to_datetime(df_p["timestamp"], unit="s", utc=True)
        df     = pd.merge(df_r, df_p, on="timestamp", suffixes=("_real", "_plan"))
        kõik.append(df)
        praegune = järgmine
    df_kogu = pd.concat(kõik).drop_duplicates("timestamp").reset_index(drop=True)
    df_kogu = df_kogu.set_index("timestamp")
    df_kogu = df_kogu.apply(pd.to_numeric, errors="coerce")
    df_kogu = df_kogu.resample("h").mean().reset_index()
    return df_kogu

# ── Open-Meteo API ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def tõmba_ilmaprognoos():
    url    = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 58.5, "longitude": 25.0,
        "hourly": ["temperature_2m", "wind_speed_10m", "wind_direction_10m",
                   "precipitation", "relative_humidity_2m",
                   "surface_pressure", "sunshine_duration"],
        "timezone": "UTC", "wind_speed_unit": "ms", "forecast_days": 3,
    }
    vastus = requests.get(url, params=params)
    df     = pd.DataFrame(vastus.json()["hourly"])
    df     = df.rename(columns={
        "time":                 "timestamp",
        "temperature_2m":       "TA_kesk",
        "wind_speed_10m":       "WS10M_kesk",
        "wind_direction_10m":   "WD10M_kesk",
        "precipitation":        "PR1H_kesk",
        "relative_humidity_2m": "RH_kesk",
        "surface_pressure":     "PA0_kesk",
        "sunshine_duration":    "SDUR1H_kesk",
    })
    df["timestamp"]   = pd.to_datetime(df["timestamp"], utc=True)
    df["SDUR1H_kesk"] = df["SDUR1H_kesk"] / 60
    return df

# ── Prognoosi funktsioon ──────────────────────────────────────────────────────
def tee_prognoos(df_elering, df_ilm, prog_päev, rf2, xgb2, tunnused_uus):
    eesti_pyhad = holidays.Estonia()
    algus = pd.Timestamp(prog_päev, tz="UTC")
    lõpp  = algus + timedelta(hours=23)
    df    = df_ilm[(df_ilm["timestamp"] >= algus) & (df_ilm["timestamp"] <= lõpp)].copy().reset_index(drop=True)
    if len(df) == 0:
        return None
    eesti_aeg              = df["timestamp"].dt.tz_convert("Europe/Tallinn")
    df["kelleaeg"]         = eesti_aeg.dt.hour
    df["nädalapäev"]       = eesti_aeg.dt.dayofweek
    df["kuu"]              = eesti_aeg.dt.month
    df["aasta"]            = eesti_aeg.dt.year
    df["on_püha"]          = eesti_aeg.dt.date.apply(lambda x: 1 if x in eesti_pyhad else 0)
    df["on_nädalavahetus"] = (df["nädalapäev"] >= 5).astype(int)
    df["hooaeg"]           = df["kuu"].apply(
        lambda k: 1 if k in [12,1,2] else 2 if k in [3,4,5] else 3 if k in [6,7,8] else 4
    )
    def on_koolivaheaeg(kp):
        kuu, päev = kp.month, kp.day
        if kuu in [6,7,8]: return 1
        if kuu == 12 and päev >= 22: return 1
        if kuu == 1 and päev <= 7: return 1
        if kuu == 2 and 15 <= päev <= 21: return 1
        if kuu == 4 and 15 <= päev <= 21: return 1
        return 0
    df["on_koolivaheaeg"] = eesti_aeg.apply(lambda x: on_koolivaheaeg(x))
    df["tunde_ette"]      = (df["kelleaeg"] - 9) % 24 + 15
    df["TA_kesk2"]        = df["TA_kesk"] ** 2
    df["kelleaeg_TA"]     = df["kelleaeg"] * df["TA_kesk"]
    tarbimine = df_elering.set_index("timestamp")["consumption_real"]
    solar     = df_elering.set_index("timestamp")["solar_energy_production"]
    renewable = df_elering.set_index("timestamp")["production_renewable_real"]
    lag_dyn, lag168, lag336, s168, s336, r168, r336 = [], [], [], [], [], [], []
    for ts in df["timestamp"]:
        eesti_ts = ts.tz_convert("Europe/Tallinn")
        tunde    = int((eesti_ts.hour - 9) % 24 + 15)
        lag_dyn.append(tarbimine.get(ts - timedelta(hours=tunde), np.nan))
        lag168.append(tarbimine.get(ts - timedelta(hours=168), np.nan))
        lag336.append(tarbimine.get(ts - timedelta(hours=336), np.nan))
        s168.append(solar.get(ts - timedelta(hours=168), np.nan))
        s336.append(solar.get(ts - timedelta(hours=336), np.nan))
        r168.append(renewable.get(ts - timedelta(hours=168), np.nan))
        r336.append(renewable.get(ts - timedelta(hours=336), np.nan))
    df["tarbimine_lag_dyn"] = lag_dyn
    df["tarbimine_lag168"]  = lag168
    df["tarbimine_lag336"]  = lag336
    df["solar_lag168"]      = s168
    df["solar_lag336"]      = s336
    df["renewable_lag168"]  = r168
    df["renewable_lag336"]  = r336
    cols     = ["tarbimine_lag_dyn","tarbimine_lag168","tarbimine_lag336",
                "solar_lag168","solar_lag336","renewable_lag168","renewable_lag336"]
    df[cols] = df[cols].ffill()
    X                  = df[tunnused_uus]
    suvi               = df["kuu"].isin([5,6,7,8])
    df["prognoos_MWh"] = np.where(suvi, rf2.predict(X), xgb2.predict(X))
    return df[["timestamp", "prognoos_MWh"]]

# ── Lehe seadistus ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Eesti elektritarbimise prognoos", page_icon="⚡", layout="wide")
st.title("⚡ Eesti elektritarbimise prognoos")
st.caption("Andmed: Elering Live API | Ilm: Open-Meteo & EMHI")

# ── Andmete laadimine ─────────────────────────────────────────────────────────
with st.spinner("Laen mudeleid ja andmeid..."):
    rf2, xgb2, tunnused_uus = laadi_mudelid()
    df_ajalugu              = laadi_ajalugu()
    df_el_viitajad          = tõmba_elering_viitajad()
    df_ilm                  = tõmba_ilmaprognoos()
    try:
        df_test = laadi_testperiood()
    except:
        df_test = None

praegu   = datetime.now(timezone.utc)
tänane   = praegu.astimezone(ZoneInfo("Europe/Tallinn")).date()
järgmine = tänane + timedelta(days=1)

df_prog_homne = tee_prognoos(df_el_viitajad, df_ilm, järgmine, rf2, xgb2, tunnused_uus)

teenus       = drive_teenus()
df_prog_täna = laadi_päevane_prognoos(teenus, tänane)
if df_prog_täna is None:
    df_prog_täna = tee_prognoos(df_el_viitajad, df_ilm, tänane, rf2, xgb2, tunnused_uus)

# ── 1. Tänane ja homne prognoos ───────────────────────────────────────────────
st.subheader(f"📅 Tänane ({tänane}) ja homne ({järgmine}) prognoos")

if df_prog_homne is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Homne min",      f"{df_prog_homne['prognoos_MWh'].min():.0f} MWh")
    col2.metric("Homne max",      f"{df_prog_homne['prognoos_MWh'].max():.0f} MWh")
    col3.metric("Homne keskmine", f"{df_prog_homne['prognoos_MWh'].mean():.0f} MWh")

    tänane_algus   = pd.Timestamp(tänane, tz="UTC")
    tänane_lõpp    = tänane_algus + timedelta(hours=23)
    järgmine_algus = pd.Timestamp(järgmine, tz="UTC")
    järgmine_lõpp  = järgmine_algus + timedelta(hours=23)

    df_täna_el  = df_el_viitajad[
        (df_el_viitajad["timestamp"] >= tänane_algus) &
        (df_el_viitajad["timestamp"] <= tänane_lõpp)
    ]
    df_homne_el = df_el_viitajad[
        (df_el_viitajad["timestamp"] >= järgmine_algus) &
        (df_el_viitajad["timestamp"] <= järgmine_lõpp)
    ]

    fig1 = go.Figure()
    if df_täna_el["consumption_real"].notna().any():
        fig1.add_trace(go.Scatter(
            x=df_täna_el["timestamp"], y=df_täna_el["consumption_real"],
            name="Tegelik täna", line=dict(color="black", width=2)
        ))
    if df_prog_täna is not None:
        fig1.add_trace(go.Scatter(
            x=df_prog_täna["timestamp"], y=df_prog_täna["prognoos_MWh"],
            name="Meie prognoos täna", line=dict(color="orange", width=2, dash="dot")
        ))
    if df_täna_el["consumption_plan"].notna().any():
        fig1.add_trace(go.Scatter(
            x=df_täna_el["timestamp"], y=df_täna_el["consumption_plan"],
            name="Elering plaan täna", line=dict(color="steelblue", width=2, dash="dot")
        ))
    fig1.add_trace(go.Scatter(
        x=df_prog_homne["timestamp"], y=df_prog_homne["prognoos_MWh"],
        name="Meie prognoos homseks", line=dict(color="orange", width=2)
    ))
    if not df_homne_el.empty and df_homne_el["consumption_plan"].notna().any():
        fig1.add_trace(go.Scatter(
            x=df_homne_el["timestamp"], y=df_homne_el["consumption_plan"],
            name="Elering plaan homseks", line=dict(color="steelblue", width=2, dash="dash")
        ))
    fig1.add_vline(x=järgmine_algus, line_dash="dash", line_color="gray", opacity=0.5)
    fig1.update_layout(
        height=420, xaxis_title="Kellaaeg (UTC)", yaxis_title="MWh",
        legend=dict(orientation="h", y=1.18)
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── 2. Ajalooline vaade ───────────────────────────────────────────────────────
st.subheader("📊 Ajalooline vaade")

col1, col2 = st.columns(2)
algus_val  = col1.date_input("Algus", value=date(2025, 3, 15))
lõpp_val   = col2.date_input("Lõpp",  value=date.today())

mask_aj      = ((df_ajalugu["timestamp"].dt.date >= algus_val) &
                (df_ajalugu["timestamp"].dt.date <= lõpp_val))
df_aj_filter = df_ajalugu[mask_aj]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_aj_filter["timestamp"], y=df_aj_filter["consumption_real"],
    name="Tegelik", line=dict(color="black", width=1)
))
fig2.add_trace(go.Scatter(
    x=df_aj_filter["timestamp"], y=df_aj_filter["consumption_plan"],
    name="Elering (planeeritud)", line=dict(color="steelblue", width=1, dash="dash")
))
if df_test is not None:
    mask_test      = ((df_test["timestamp"].dt.date >= algus_val) &
                      (df_test["timestamp"].dt.date <= lõpp_val))
    df_test_filter = df_test[mask_test]
    if len(df_test_filter) > 0:
        fig2.add_trace(go.Scatter(
            x=df_test_filter["timestamp"], y=df_test_filter["kombi_uus"],
            name="Meie prognoos", line=dict(color="orange", width=1)
        ))
fig2.update_layout(
    height=450, xaxis_title="Kuupäev", yaxis_title="MWh",
    legend=dict(orientation="h", y=1.1)
)
st.plotly_chart(fig2, use_container_width=True)

# ── 3. Meetrikad ─────────────────────────────────────────────────────────────
if df_test is not None:
    st.subheader("📈 Mudeli täpsus (testperiood 15.03.2025 – 05.05.2026)")
    col1, col2, col3 = st.columns(3)
    for tulp, nimi in [("kombi_uus", "Meie prognoos"), ("consumption_plan", "Elering")]:
        m    = df_test[tulp].notna() & df_test["tegelik"].notna()
        mae  = mean_absolute_error(df_test["tegelik"][m], df_test[tulp][m])
        rmse = np.sqrt(mean_squared_error(df_test["tegelik"][m], df_test[tulp][m]))
        r2   = r2_score(df_test["tegelik"][m], df_test[tulp][m])
        col1.metric(f"MAE — {nimi}",  f"{mae:.1f} MWh")
        col2.metric(f"RMSE — {nimi}", f"{rmse:.1f} MWh")
        col3.metric(f"R² — {nimi}",   f"{r2:.3f}")
