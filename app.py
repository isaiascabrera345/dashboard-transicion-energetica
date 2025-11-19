# app.py
# ============================================================
# DASHBOARD: Transición Energética — Sudamérica (OWID)
# - Datos OWID energía + CO2 (CSVs públicos)
# - Mezcla robusta (TWh directo o shares×generación)
# - Tema visual minimal (CSS externo + Plotly theme)
# - Mapa grande, tabs limpias, keys por gráfico
# - Pestaña de pronósticos ARIMA + elasticidad CO₂/PIB
# ============================================================

from pathlib import Path
import os
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from statsmodels.tsa.arima.model import ARIMA

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ModuleNotFoundError:  # streamlit < 1.20 o cambios internos
    get_script_run_ctx = None

# ---------------------------#
# Configuración de la app
# ---------------------------#
st.set_page_config(
    page_title="Transición Energética — Sudamérica",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== CSS y tema Plotly =====
def inject_local_css(path: str = "assets/style.css"):
    # Asegura fuentes de íconos Material (evita textos como 'keyboard_double_arrow_right')
    st.markdown(
        """
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200">
        <style>
        .material-symbols-rounded, .material-icons, [class*="material-icons"]{
            font-family: 'Material Symbols Rounded','Material Icons',sans-serif !important;
            font-weight: normal; font-style: normal; font-size: 24px; line-height: 1;
            letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; direction: ltr;
            -webkit-font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    p = Path(path)
    if p.exists():
        st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def set_plotly_theme():
    pio.templates["icosystems_dark"] = pio.templates["plotly_dark"]
    t = pio.templates["icosystems_dark"].layout
    t.paper_bgcolor = "#0f1216"
    t.plot_bgcolor  = "#0f1216"
    t.font.color    = "#e7e9ee"

    # 🎨 Paleta con alto contraste y líneas claramente diferenciables
    pio.templates["icosystems_dark"].layout.colorway = [
        "#FF6B6B",  # rojo coral
        "#FFD93D",  # amarillo brillante
        "#6BCB77",  # verde esmeralda
        "#4D96FF",  # azul intenso
        "#C77DFF",  # violeta
        "#FF9F1C",  # naranja
        "#2EC4B6",  # turquesa
        "#F94144",  # rojo fuerte
        "#90BE6D",  # verde claro
        "#577590",  # azul grisáceo
    ]
    px.defaults.template = "icosystems_dark"
    px.defaults.color_continuous_scale = "Turbo"
    px.defaults.height = 540

inject_local_css()
set_plotly_theme()

# Production mode (Render/CI) for lighter defaults
PRODUCTION = bool(os.environ.get("RENDER") or os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("STREAMLIT_PROD"))

def _emit_streamlit_info(message: str, *, show_to_user: bool = False) -> None:
    """Registra mensajes informativos; opcionalmente los muestra en la UI."""
    logging.getLogger(__name__).info(message)
    if PRODUCTION and not show_to_user:
        return
    if get_script_run_ctx is None:
        return
    try:
        if get_script_run_ctx() is not None:
            st.info(message)
    except Exception:
        # Si Streamlit no tiene contexto (tests, scripts), ignoramos.
        pass

# ---------------------------#
# Parámetros / países
# ---------------------------#
SOUTH_AMERICA_ISO = ["ARG","BOL","BRA","CHL","COL","ECU","GUY","PRY","PER","SUR","URY","VEN"]
OWID_ENERGY_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
OWID_CO2_URL    = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
OWID_CHUNK_SIZE = 10_000

ENERGY_USECOLS = [
    "iso_code","country","year","electricity_generation",
    "electricity_from_coal","electricity_from_gas","electricity_from_oil",
    "electricity_from_hydro","electricity_from_nuclear","electricity_from_wind",
    "electricity_from_solar","electricity_from_biofuel","electricity_from_bioenergy",
    "electricity_from_other_renewables",
    "primary_energy_consumption",
    "coal_share_elec","gas_share_elec","oil_share_elec","hydro_share_elec",
    "nuclear_share_elec","wind_share_elec","solar_share_elec",
    "bioenergy_share_elec","other_renewables_share_elec",
]
ENERGY_FLOAT_COLS = [
    "electricity_generation",
    "electricity_from_coal","electricity_from_gas","electricity_from_oil",
    "electricity_from_hydro","electricity_from_nuclear","electricity_from_wind",
    "electricity_from_solar","electricity_from_biofuel","electricity_from_bioenergy",
    "electricity_from_other_renewables",
    "primary_energy_consumption",
    "coal_share_elec","gas_share_elec","oil_share_elec","hydro_share_elec",
    "nuclear_share_elec","wind_share_elec","solar_share_elec",
    "bioenergy_share_elec","other_renewables_share_elec",
]

CO2_USECOLS = ["iso_code","country","year","co2","co2_per_capita","gdp"]
CO2_FLOAT_COLS = ["co2","co2_per_capita","gdp"]
ENERGY_REQUIRED_COLS = ["iso_code","country","year","electricity_generation"]
CO2_REQUIRED_COLS = ["iso_code","country","year","co2"]

FORECAST_INDICATORS = {
    "co2_total": {
        "label": "CO₂ total (Mt)",
        "unit": "Mt",
        "description": "Emisiones totales anuales de CO₂ reportadas por OWID.",
    },
    "renewables_share": {
        "label": "Participación de renovables en generación (%)",
        "unit": "%",
        "description": "Share estimado de renovables dentro de la generación eléctrica.",
    },
    "primary_energy": {
        "label": "Consumo primario de energía (TWh)",
        "unit": "TWh",
        "description": "Consumo primario de energía reportado por OWID.",
    },
}
FORECAST_MIN_OBS = 8
FORECAST_MAX_PDQ = 2
ELASTICITY_MIN_OBS = 6

# ---------------------------#
# Carga y validación de datos
# ---------------------------#
def _read_owid_subset(
    url: str,
    usecols: list[str],
    float_cols: list[str],
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Lee el CSV remoto por chunks, filtra Sudamérica y minimiza memoria."""
    required = required_cols or []
    try:
        header_cols = pd.read_csv(url, nrows=0).columns
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer la cabecera del dataset OWID: {url}") from exc

    available = set(header_cols)
    missing_required = [col for col in required if col not in available]
    if missing_required:
        raise ValueError(
            "Faltan columnas esenciales en OWID: "
            + ", ".join(missing_required)
            + ". Actualiza la lista de columnas o revisa el dataset de origen."
        )

    selected_cols: list[str] = []
    for col in usecols:
        if col in available and col not in selected_cols:
            selected_cols.append(col)
    for col in required:
        if col in available and col not in selected_cols:
            selected_cols.append(col)

    optional_missing = [col for col in usecols if col not in available]
    if optional_missing:
        _emit_streamlit_info(
            "Columnas OWID no disponibles y omitidas: "
            + ", ".join(optional_missing)
        )

    try:
        iterator = pd.read_csv(
            url,
            usecols=selected_cols,
            chunksize=OWID_CHUNK_SIZE,
            dtype_backend="numpy_nullable",
        )
    except TypeError:
        iterator = pd.read_csv(url, usecols=selected_cols, chunksize=OWID_CHUNK_SIZE)

    frames = []
    for chunk in iterator:
        filtered = chunk[chunk["iso_code"].isin(SOUTH_AMERICA_ISO)]
        if not filtered.empty:
            frames.append(filtered)

    if not frames:
        return pd.DataFrame(columns=usecols)

    df = pd.concat(frames, ignore_index=True)

    for cat_col in ("iso_code", "country"):
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int16")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float32")

    return df

@st.cache_data(show_spinner=True, ttl=60*60)
def load_owid_energy() -> pd.DataFrame:
    if PRODUCTION:
        return _read_owid_subset(
            OWID_ENERGY_URL,
            ENERGY_USECOLS,
            ENERGY_FLOAT_COLS,
            required_cols=ENERGY_REQUIRED_COLS,
        )
    return pd.read_csv(OWID_ENERGY_URL)

@st.cache_data(show_spinner=True, ttl=60*60)
def load_owid_co2() -> pd.DataFrame:
    if PRODUCTION:
        return _read_owid_subset(
            OWID_CO2_URL,
            CO2_USECOLS,
            CO2_FLOAT_COLS,
            required_cols=CO2_REQUIRED_COLS,
        )
    return pd.read_csv(OWID_CO2_URL)

def validate_energy(df: pd.DataFrame) -> list:
    warns = []
    for col in ["iso_code","country","year","electricity_generation"]:
        if col not in df.columns:
            warns.append(f"Falta columna en energía: {col}")
    if "year" in df.columns and not pd.api.types.is_numeric_dtype(df["year"]):
        warns.append("Columna 'year' en energía no es numérica.")
    return warns

def validate_co2(df: pd.DataFrame) -> list:
    warns = []
    for col in ["iso_code","country","year","co2"]:
        if col not in df.columns:
            warns.append(f"Falta columna en CO₂: {col}")
    return warns

def list_sa_countries(df: pd.DataFrame) -> list:
    return sorted(df[df["iso_code"].isin(SOUTH_AMERICA_ISO)]["country"].unique())

def last_nonnull(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    return None if s.empty else s.iloc[-1]

def fmt(x, d=2, default="-"):
    try:
        return f"{float(x):,.{d}f}"
    except Exception:
        return default

@st.cache_data(show_spinner=False, ttl=60*60)
def filter_south_america(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["iso_code"].isin(SOUTH_AMERICA_ISO)].copy()
    if "year" in subset.columns:
        subset = subset.sort_values("year")
    return subset

@st.cache_data(show_spinner=False, ttl=60*10)
def slice_country_period(df: pd.DataFrame, country: str, start_year: int, end_year: int) -> pd.DataFrame:
    mask = (df["country"] == country) & (df["year"].between(start_year, end_year))
    subset = df.loc[mask].copy()
    if "year" in subset.columns:
        subset = subset.sort_values("year")
    return subset.reset_index(drop=True)

# ---------------------------#
# Logo loader robusto
# ---------------------------#
def load_logo_bytes() -> bytes | None:
    base = Path(__file__).parent / "assets"
    candidates = [
        "logo.png","logo.jpg","logo.jpeg","Logo.png",
        "Flat Vector Logo of ICO Systems.png","ICO Systems.png"
    ]
    for name in candidates:
        p = base / name
        if p.exists() and p.is_file():
            try:
                return p.read_bytes()
            except Exception:
                pass
    return None

# ---------------------------#
# Mezcla eléctrica robusta
# ---------------------------#
MIX_FROM_COLS = [
    "electricity_from_coal","electricity_from_gas","electricity_from_oil",
    "electricity_from_hydro","electricity_from_nuclear","electricity_from_wind",
    "electricity_from_solar","electricity_from_biofuel","electricity_from_bioenergy",
    "electricity_from_other_renewables"
]
MIX_SHARE_COLS = {
    "electricity_from_coal": "coal_share_elec",
    "electricity_from_gas": "gas_share_elec",
    "electricity_from_oil": "oil_share_elec",
    "electricity_from_hydro": "hydro_share_elec",
    "electricity_from_nuclear": "nuclear_share_elec",
    "electricity_from_wind": "wind_share_elec",
    "electricity_from_solar": "solar_share_elec",
    "electricity_from_biofuel": "bioenergy_share_elec",
    "electricity_from_bioenergy": "bioenergy_share_elec",
    "electricity_from_other_renewables": "other_renewables_share_elec",
}

def compute_renewable_share_row(row: pd.Series) -> float | None:
    cols = [
        "electricity_from_hydro","electricity_from_wind","electricity_from_solar",
        "electricity_from_biofuel","electricity_from_bioenergy",
        "electricity_from_other_renewables"
    ]
    ren = sum([row[c] for c in cols if c in row and pd.notna(row[c])])
    gen = row.get("electricity_generation", None)
    if gen is None or pd.isna(gen) or gen == 0:
        return None
    return 100.0 * ren / gen

@st.cache_data(show_spinner=False, ttl=60*60)
def build_mix_columns(df_country: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    work = df_country.copy()
    if work.empty: return pd.DataFrame(), []
    from_cols_present = [c for c in MIX_FROM_COLS if c in work.columns]
    if from_cols_present:
        df_mix = work[["year"] + from_cols_present].set_index("year")
        if not df_mix.dropna(how="all").empty:
            return df_mix.sort_index(), from_cols_present
    if "electricity_generation" not in work.columns: return pd.DataFrame(), []
    gen = work[["year","electricity_generation"]].set_index("year")
    pieces, used_cols = [], []
    for out_col, share_col in MIX_SHARE_COLS.items():
        if share_col in work.columns:
            tmp = work[["year", share_col]].set_index("year").join(gen, how="left")
            pieces.append(((tmp[share_col]/100.0) * tmp["electricity_generation"]).rename(out_col))
            used_cols.append(out_col)
    if not pieces: return pd.DataFrame(), []
    df_mix = pd.concat(pieces, axis=1); df_mix.index.name = "year"
    return df_mix.sort_index(), used_cols

@st.cache_data(show_spinner=False, ttl=60*60)
def build_renewable_ranking(energy_sa: pd.DataFrame) -> pd.DataFrame:
    tmp = energy_sa.dropna(subset=["iso_code"])
    if tmp.empty: return pd.DataFrame()
    idx = tmp.groupby("iso_code")["year"].idxmax()
    last = tmp.loc[idx].copy()
    shares = []
    for _, r in last.iterrows():
        row_df = pd.DataFrame([r]).copy()
        mix, _ = build_mix_columns(row_df)
        if mix.empty:
            shares.append(None); continue
        mix_row = mix.tail(1).iloc[0]
        ren_share = compute_renewable_share_row(pd.concat([mix_row, row_df.iloc[0]]))
        shares.append(ren_share)
    last["ren_share"] = shares
    return last[["country","iso_code","year","ren_share"]].dropna().sort_values("ren_share", ascending=False)

@st.cache_data(show_spinner=False, ttl=60*60)
def compute_renewables_share_timeseries(e_country: pd.DataFrame) -> pd.Series:
    if e_country.empty:
        return pd.Series(dtype=float)
    mix, _ = build_mix_columns(e_country)
    if mix.empty:
        return pd.Series(dtype=float)
    gen = e_country.set_index("year")["electricity_generation"]
    ren_cols = [c for c in mix.columns if any(kw in c for kw in [
        "hydro","wind","solar","biofuel","bioenergy","other_renewables"
    ])]
    ren_twh = mix[ren_cols].sum(axis=1)
    share = 100 * (ren_twh / gen.reindex(mix.index))
    return share.dropna()

# ---------------------------#
# Pronósticos ARIMA (helpers)
# ---------------------------#
def _clean_year_series(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype="float64")
    idx = pd.Series(s.index, dtype="object")
    years = pd.to_numeric(idx, errors="coerce")
    mask = years.notna()
    s = s.iloc[mask.values]
    if s.empty:
        return pd.Series(dtype="float64")
    years = years.loc[mask].astype("int64").to_numpy()
    s.index = pd.Index(years, name="year")
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index().astype("float64")

@st.cache_data(show_spinner=False, ttl=60*10)
def get_indicator_series(
    country: str,
    indicator_key: str,
    energy_sa: pd.DataFrame,
    co2_sa: pd.DataFrame,
) -> pd.Series:
    if indicator_key == "co2_total":
        df = co2_sa[co2_sa["country"] == country]
        if "co2" not in df.columns:
            return pd.Series(dtype="float64")
        series = df.set_index("year")["co2"]
    elif indicator_key == "renewables_share":
        df = energy_sa[energy_sa["country"] == country]
        series = compute_renewables_share_timeseries(df)
    elif indicator_key == "primary_energy":
        df = energy_sa[energy_sa["country"] == country]
        if "primary_energy_consumption" not in df.columns:
            return pd.Series(dtype="float64")
        series = df.set_index("year")["primary_energy_consumption"]
    else:
        return pd.Series(dtype="float64")
    return _clean_year_series(series)

def auto_arima_forecast(series: pd.Series, horizon: int) -> dict:
    y = _clean_year_series(series)
    if y.empty or len(y) < FORECAST_MIN_OBS:
        raise ValueError("Serie insuficiente para pronosticar.")
    dt_index = pd.to_datetime(y.index.astype(str), format="%Y", errors="coerce")
    mask = ~dt_index.isna()
    y = y.loc[mask]
    dt_index = dt_index[mask]
    if y.empty:
        raise ValueError("Serie sin años válidos.")
    y.index = pd.PeriodIndex(dt_index, freq="Y").to_timestamp()
    y = y[~y.index.duplicated(keep="last")].sort_index()
    best_res = None
    best_order = None
    best_aic = np.inf
    for p in range(FORECAST_MAX_PDQ + 1):
        for d in range(FORECAST_MAX_PDQ + 1):
            for q in range(FORECAST_MAX_PDQ + 1):
                try:
                    model = ARIMA(
                        y,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = model.fit()
                except Exception:
                    continue
                if res.aic < best_aic:
                    best_res = res
                    best_order = (p, d, q)
                    best_aic = res.aic
    if best_res is None or best_order is None:
        raise RuntimeError("No se pudo ajustar un ARIMA.")
    fc = best_res.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]
    history_df = pd.DataFrame({
        "year": y.index.year.astype(int),
        "valor": y.values.astype(float),
        "tipo": "Observado",
    })
    forecast_df = pd.DataFrame({
        "year": mean.index.year.astype(int),
        "valor": mean.values.astype(float),
        "tipo": "Pronóstico",
        "lower": lower.values.astype(float),
        "upper": upper.values.astype(float),
    })
    return {
        "history": history_df,
        "forecast": forecast_df,
        "order": best_order,
        "aic": float(best_aic) if np.isfinite(best_aic) else np.nan,
        "train_obs": int(len(y)),
        "last_value": float(history_df["valor"].iloc[-1]),
        "last_year": int(history_df["year"].iloc[-1]),
    }

# ---------------------------#
# Modelo alternativo: elasticidad CO₂-PIB
# ---------------------------#
def fit_log_elasticity(df: pd.DataFrame, min_obs: int = ELASTICITY_MIN_OBS) -> dict:
    work = df.dropna(subset=["co2", "gdp"]).copy()
    if work.empty:
        raise ValueError("No hay datos de CO₂ y PIB.")
    work = work[(work["co2"] > 0) & (work["gdp"] > 0)]
    if len(work) < min_obs:
        raise ValueError(f"Se requieren ≥{min_obs} observaciones positivas.")
    work = work.sort_values("year")
    log_gdp = np.log(work["gdp"].astype(float))
    log_co2 = np.log(work["co2"].astype(float))
    slope, intercept = np.polyfit(log_gdp, log_co2, 1)
    fitted = slope * log_gdp + intercept
    ss_res = float(np.sum((log_co2 - fitted) ** 2))
    ss_tot = float(np.sum((log_co2 - log_co2.mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)
    work = work.assign(
        log_gdp=log_gdp,
        log_co2=log_co2,
        fitted=np.exp(fitted),
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "data": work,
        "observations": int(len(work)),
    }

@st.cache_data(show_spinner=False, ttl=60*60)
def elasticity_league_table(co2_sa: pd.DataFrame, min_obs: int = ELASTICITY_MIN_OBS) -> pd.DataFrame:
    rows = []
    for country, df_country in co2_sa.groupby("country"):
        try:
            result = fit_log_elasticity(df_country, min_obs=min_obs)
        except Exception:
            continue
        rows.append({
            "country": country,
            "elasticity": result["slope"],
            "r2": result["r2"],
            "observations": result["observations"],
            "year_start": int(df_country["year"].min()),
            "year_end": int(df_country["year"].max()),
        })
    if not rows:
        return pd.DataFrame()
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values("elasticity", ascending=False)
    return ranking

# ---------------------------#
# Sidebar (controles)
# ---------------------------#
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.caption("Datos: Our World in Data (energía + CO₂). Refresca con ↻ si lo necesitas.")

# Cargar datos
try:
    energy = load_owid_energy()
    co2    = load_owid_co2()
except Exception as e:
    st.error("No se pudieron cargar los datos desde OWID. Verifica tu conexión.")
    st.exception(e)
    st.stop()

# Validación (informativa)
for w in (validate_energy(energy) + validate_co2(co2)):
    st.warning("• " + w)

# Filtrar Sudamérica con caché liviano
energy_sa = filter_south_america(energy)
co2_sa    = filter_south_america(co2)

# Controles
country_list = list_sa_countries(energy)
default_country = "Colombia" if "Colombia" in country_list else (country_list[0] if country_list else "Colombia")
with st.sidebar:
    country = st.selectbox("País (Sudamérica)", country_list, index=country_list.index(default_country) if default_country in country_list else 0)
    y_min = int(max(1960, energy_sa["year"].min() if not energy_sa.empty else 1960))
    y_max = int(energy_sa["year"].max() if not energy_sa.empty else datetime.now().year)
    yr = st.slider("Rango de años", y_min, y_max, (max(1990,y_min), y_max))
    show_pct = st.checkbox("Mezcla eléctrica en %", value=True)
    show_map = st.checkbox("Ver mapa de renovables (último año)", value=True)

# Filtrado por país y años (cacheado para evitar copiar dataframes completos)
e_c = slice_country_period(energy_sa, country, yr[0], yr[1])
c_c = slice_country_period(co2_sa, country, yr[0], yr[1])

# ---------------------------#
# Header con logo y título
# ---------------------------#
c_logo, c_title = st.columns([1,9], vertical_alignment="center")
with c_logo:
    _logo = load_logo_bytes()
    if _logo is not None:
        # Mostrar logo compacto para no tapar el contenido
        try:
            st.image(_logo, width=64)
        except TypeError:
            st.image(_logo, width=64)
    else:
        st.warning("No se encontró el logo en /assets. El dashboard continúa sin logo.")
with c_title:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand">
            <h1>Transición Energética — Sudamérica</h1>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------#
# KPIs
# ---------------------------#
k1,k2,k3,k4 = st.columns(4)

last_gen = last_nonnull(e_c.sort_values("year"), "electricity_generation")
k1.metric("Generación eléctrica (TWh, último año)", fmt(last_gen,2))

if not e_c.empty:
    mix_c, _used = build_mix_columns(e_c.sort_values("year"))
    if not mix_c.empty:
        last_year = mix_c.index.max()
        gen_last = e_c.loc[e_c["year"]==last_year, "electricity_generation"].dropna()
        gen_last = None if gen_last.empty else float(gen_last.iloc[0])
        if gen_last is not None:
            row = pd.concat([mix_c.loc[last_year], pd.Series({"electricity_generation": gen_last})])
            k2.metric("Renovables (% generación, último año)", fmt(compute_renewable_share_row(row),1))
        else:
            k2.metric("Renovables (%)", "—")
    else:
        k2.metric("Renovables (%)", "—")
else:
    k2.metric("Renovables (%)", "—")

last_co2_mt = last_nonnull(c_c.sort_values("year"), "co2")  # Mt
k3.metric("CO₂ total (Mt, último año)", fmt(last_co2_mt, 2))

if not c_c.empty and "gdp" in c_c.columns and "co2" in c_c.columns:
    c_c = c_c.copy()
    c_c["co2_intensity_kg_per_usd"] = (c_c["co2"] * 1e9) / c_c["gdp"]  # Mt->kg
    k4.metric("Intensidad CO₂/PIB (kg/USD)", fmt(last_nonnull(c_c.sort_values("year"), "co2_intensity_kg_per_usd"), 4))
else:
    k4.metric("Intensidad CO₂/PIB", "—")

# ---------------------------#
# Tabs
# ---------------------------#
tab_mix, tab_co2, tab_rank, tab_map, tab_forecast, tab_elasticity, tab_data = st.tabs([
    "Mezcla eléctrica",
    "Emisiones CO₂",
    "Ranking renovables",
    "Mapa regional",
    "Pronósticos",
    "Elasticidad CO₂/PIB",
    "Datos / Descargas"
])

# ---- Mezcla eléctrica ----
with tab_mix:
    st.subheader(f"Mezcla de generación eléctrica — {country}")
    mix_container = st.container()
    with mix_container:
        if e_c.empty:
            st.info("No hay registros para el rango seleccionado. Amplía los años o cambia de país.")
        else:
            df_mix, used_cols = build_mix_columns(e_c)
            if df_mix.empty or df_mix.dropna(how="all").empty:
                st.info("No se pudo construir la mezcla (ni TWh ni shares disponibles). Amplía el rango de años.")
            else:
                df_mix = df_mix.sort_index()
                if show_pct:
                    df_plot = (df_mix.div(df_mix.sum(axis=1), axis=0)*100).reset_index().melt("year", var_name="fuente", value_name="porcentaje")
                    fig_mix = px.area(df_plot, x="year", y="porcentaje", color="fuente", title="Participación por fuente (%)")
                else:
                    df_plot = df_mix.reset_index().melt("year", var_name="fuente", value_name="TWh")
                    fig_mix = px.area(df_plot, x="year", y="TWh", color="fuente", title="Generación por fuente (TWh)")
                fig_mix.update_traces(line=dict(width=2))
                fig_mix.update_layout(height=720, margin=dict(l=10, r=10, t=60, b=20), legend_title_text="Fuente")
                st.plotly_chart(fig_mix, use_container_width=True, key=f"mix_chart_{country}_{yr[0]}_{yr[1]}")
                st.caption("Columnas utilizadas para la mezcla: " + (", ".join(used_cols) if used_cols else "shares reconstruidos"))

# ---- Emisiones CO2 ----
with tab_co2:
    st.subheader(f"Emisiones de CO₂ — {country}")
    if c_c.empty or "co2" not in c_c.columns:
        st.info("No hay datos suficientes para este país/rango.")
    else:
        fig1 = px.line(c_c, x="year", y="co2", title="CO₂ total (Mt)")
        fig1.update_traces(line=dict(width=3))
        st.plotly_chart(fig1, use_container_width=True, key=f"co2_total_{country}_{yr[0]}_{yr[1]}")
        if "co2_per_capita" in c_c.columns:
            fig2 = px.line(c_c, x="year", y="co2_per_capita", title="CO₂ per cápita (ton/persona)")
            fig2.update_traces(line=dict(width=3))
            st.plotly_chart(fig2, use_container_width=True, key=f"co2_pc_{country}_{yr[0]}_{yr[1]}")
        if "gdp" in c_c.columns:
            c_c_plot = c_c.copy()
            c_c_plot["co2_intensity_kg_per_usd"] = (c_c_plot["co2"] * 1e9) / c_c_plot["gdp"]
            fig3 = px.line(c_c_plot, x="year", y="co2_intensity_kg_per_usd", title="Intensidad CO₂/PIB (kg/USD)")
            fig3.update_traces(line=dict(width=3))
            st.plotly_chart(fig3, use_container_width=True, key=f"co2_intensity_{country}_{yr[0]}_{yr[1]}")

# ---- Ranking renovables ----
with tab_rank:
    st.subheader("Ranking Sudamérica — % renovables en generación (último año por país)")
    rank = build_renewable_ranking(energy_sa)
    if rank.empty:
        st.info("No se pudo calcular el ranking.")
    else:
        fig_rank = px.bar(rank, x="country", y="ren_share", hover_data=["year"], title="Renovables (% del total)")
        fig_rank.update_yaxes(title="%")
        st.plotly_chart(fig_rank, use_container_width=True, key=f"rank_{yr[0]}_{yr[1]}")
        st.dataframe(rank.reset_index(drop=True))

# ---- Mapa regional ----
with tab_map:
    st.subheader("Mapa — renovables (% de la generación, último año)")
    if show_map:
        rank = build_renewable_ranking(energy_sa)
        if rank.empty:
            st.info("Sin datos para el mapa.")
        else:
            fig_map = px.choropleth(
                rank, locations="iso_code", color="ren_share",
                color_continuous_scale="Turbo", locationmode="ISO-3",
                scope="south america", title="Renovables (%) — último año disponible",
                hover_name="country", labels={"ren_share":"% renovables"}
            )
            fig_map.update_layout(height=820, margin=dict(l=10, r=10, t=60, b=10),
                                  coloraxis_colorbar=dict(title="% renovables", ticksuffix=" %"))
            fig_map.update_geos(fitbounds="locations", showcountries=True, showcoastlines=False,
                                showland=True, landcolor="rgba(255,255,255,0.02)", projection_type="mercator")
            st.plotly_chart(fig_map, use_container_width=True, key=f"map_{yr[0]}_{yr[1]}")
    else:
        st.info("Activa el mapa en la barra lateral.")

# ---- Pronósticos ----
with tab_forecast:
    st.subheader(f"Pronósticos ARIMA - {country}")
    st.caption("Modelos univariantes (p,d,q <= 2) con intervalos de confianza del 95%.")
    indicator_keys = list(FORECAST_INDICATORS.keys())
    indicator_key = st.selectbox(
        "Indicador a pronosticar",
        indicator_keys,
        format_func=lambda k: FORECAST_INDICATORS[k]["label"],
        key=f"forecast_indicator_{country}",
    )
    indicator_meta = FORECAST_INDICATORS[indicator_key]
    st.caption(indicator_meta["description"])
    horizon = st.slider("Horizonte de pronóstico (años)", 1, 10, 5)
    series = get_indicator_series(country, indicator_key, energy_sa, co2_sa)
    if series.empty:
        st.info("No hay datos históricos suficientes para este indicador/país.")
    else:
        min_year = int(series.index.min())
        max_year = int(series.index.max())
        if min_year == max_year:
            st.info("Se necesitan >=2 observaciones para entrenar el ARIMA.")
        else:
            default_start = max(min_year, max_year - 30)
            if default_start > max_year:
                default_start = min_year
            hist_range = st.slider(
                "Rango histórico utilizado",
                min_year,
                max_year,
                (default_start, max_year),
                key=f"forecast_range_{indicator_key}_{country}",
            )
            mask = (series.index >= hist_range[0]) & (series.index <= hist_range[1])
            subset = series.loc[mask].dropna()
            if len(subset) < FORECAST_MIN_OBS:
                st.warning(
                    f"Se requieren >={FORECAST_MIN_OBS} observaciones para estimar el modelo. "
                    "Amplía el rango histórico."
                )
            else:
                try:
                    forecast_res = auto_arima_forecast(subset, horizon)
                except Exception as exc:
                    st.error("No se pudo ajustar el ARIMA para este indicador.")
                    st.exception(exc)
                else:
                    fc_df = forecast_res["forecast"]
                    last_fc = float(fc_df["valor"].iloc[-1])
                    delta = last_fc - forecast_res["last_value"]
                    cols = st.columns(3)
                    cols[0].metric(
                        "Último dato observado",
                        fmt(forecast_res["last_value"], 2),
                        f"Año {forecast_res['last_year']}",
                    )
                    cols[1].metric(
                        "Orden ARIMA (p,d,q)",
                        str(forecast_res["order"]),
                        f"AIC {fmt(forecast_res['aic'], 1)}",
                    )
                    cols[2].metric(
                        f"Pronóstico {int(fc_df['year'].iloc[-1])}",
                        fmt(last_fc, 2),
                        fmt(delta, 2),
                    )

                    hist_df = forecast_res["history"]
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=hist_df["year"],
                            y=hist_df["valor"],
                            mode="lines+markers",
                            name="Observado",
                            line=dict(width=3),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=fc_df["year"],
                            y=fc_df["valor"],
                            mode="lines+markers",
                            name="Pronóstico",
                            line=dict(width=3, dash="dash"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=fc_df["year"],
                            y=fc_df["upper"],
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=fc_df["year"],
                            y=fc_df["lower"],
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor="rgba(77,150,255,0.20)",
                            name="IC 95%",
                            hovertemplate="Año %{x}<br>IC inferior: %{y:.2f}<extra></extra>",
                        )
                    )
                    fig.update_layout(
                        height=600,
                        margin=dict(l=10, r=10, t=60, b=20),
                        xaxis_title="Año",
                        yaxis_title=indicator_meta["label"],
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"forecast_chart_{indicator_key}_{country}",
                    )

                    fc_table = fc_df[["year", "valor", "lower", "upper"]].rename(
                        columns={
                            "year": "Año",
                            "valor": "Pronóstico",
                            "lower": "IC inf. 95%",
                            "upper": "IC sup. 95%",
                        }
                    )
                    st.dataframe(fc_table.round(2), use_container_width=True)

# ---- Elasticidad CO₂/PIB ----
with tab_elasticity:
    st.subheader(f"Elasticidad CO₂ - PIB · {country}")
    st.caption("Regresión log-log: una elasticidad de 1 implica que un crecimiento del 1% en el PIB se asocia con +1% en CO₂.")
    if c_c.empty or "gdp" not in c_c.columns:
        st.info("No hay datos suficientes de PIB y CO₂ para este país/rango.")
    else:
        try:
            elasticity_res = fit_log_elasticity(c_c, min_obs=ELASTICITY_MIN_OBS)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            slope = elasticity_res["slope"]
            r2 = elasticity_res["r2"]
            obs = elasticity_res["observations"]
            cols = st.columns(3)
            cols[0].metric("Elasticidad CO₂/PIB", f"{slope:.2f}")
            cols[1].metric("R² (ajuste log-log)", f"{r2:.2f}" if np.isfinite(r2) else "n/a")
            cols[2].metric("Observaciones", str(obs))

            scatter_df = elasticity_res["data"]
            fig_elast = px.scatter(
                scatter_df,
                x="gdp",
                y="co2",
                color="year",
                labels={"gdp": "PIB (USD corrientes)", "co2": "Emisiones CO₂ (Mt)", "year": "Año"},
                title="Relación CO₂ vs PIB (escala logarítmica)",
            )
            fig_elast.update_traces(marker=dict(size=9, line=dict(width=0)))
            fig_elast.update_xaxes(type="log")
            fig_elast.update_yaxes(type="log")
            line_df = scatter_df.sort_values("gdp")
            fig_elast.add_trace(
                go.Scatter(
                    x=line_df["gdp"],
                    y=line_df["fitted"],
                    mode="lines",
                    name="Ajuste log-log",
                    line=dict(color="#FFD93D", width=3),
                )
            )
            fig_elast.update_layout(
                height=620,
                margin=dict(l=10, r=10, t=60, b=20),
            )
            st.plotly_chart(
                fig_elast,
                use_container_width=True,
                key=f"elasticity_chart_{country}_{yr[0]}_{yr[1]}",
            )

    league = elasticity_league_table(co2_sa)
    if league.empty:
        st.info("No fue posible construir el ranking regional de elasticidades.")
    else:
        st.markdown("**Ranking regional de elasticidades (CO₂ vs PIB, log-log)**")
        st.caption("Calculado sobre toda la historia disponible de OWID para cada país (solo si hay ≥6 observaciones positivas).")
        st.dataframe(
            league.rename(columns={
                "country": "País",
                "elasticity": "Elasticidad",
                "r2": "R²",
                "observations": "Obs.",
                "year_start": "Año inicial",
                "year_end": "Año final",
            }).reset_index(drop=True).round({"Elasticidad": 2, "R²": 2}),
            use_container_width=True,
        )

# ---- Datos y descargas ----
with tab_data:
    st.subheader("Datos filtrados")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Energía (país/periodo)**")
        st.dataframe(e_c.reset_index(drop=True))
        st.download_button("Descargar energía (CSV)", data=e_c.to_csv(index=False),
                           file_name=f"energy_{country}_{yr[0]}_{yr[1]}.csv", mime="text/csv")
    with c2:
        st.markdown("**CO₂ (país/periodo)**")
        st.dataframe(c_c.reset_index(drop=True))
        st.download_button("Descargar CO₂ (CSV)", data=c_c.to_csv(index=False),
                           file_name=f"co2_{country}_{yr[0]}_{yr[1]}.csv", mime="text/csv")

# ---------------------------#
# Fuente + Footer
# ---------------------------#
st.markdown("---")
st.markdown(
    '<div class="source-note">Fuente de datos: Our World in Data — '
    '<a href="https://github.com/owid/energy-data" target="_blank">energía</a> y '
    '<a href="https://github.com/owid/co2-data" target="_blank">CO₂</a> (CSVs públicos y auditables).</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="footer">© <b>ICO Systems</b> 2025 — Todos los derechos reservados.</div>',
    unsafe_allow_html=True
)
