# app.py
# ============================================================
# DASHBOARD: Transición Energética — Sudamérica (OWID)
# - Datos OWID energía + CO2 (CSVs públicos)
# - Mezcla robusta (TWh directo o shares×generación)
# - Tema visual minimal (CSS externo + Plotly theme)
# - Mapa grande, tabs limpias, keys por gráfico
# - Pestaña VAR con AIC, ADF, Granger e IRFs (colores diferenciables)
# ============================================================

from pathlib import Path
from datetime import datetime
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

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

inject_local_css()
set_plotly_theme()

# ---------------------------#
# Parámetros / países
# ---------------------------#
SOUTH_AMERICA_ISO = ["ARG","BOL","BRA","CHL","COL","ECU","GUY","PRY","PER","SUR","URY","VEN"]
OWID_ENERGY_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
OWID_CO2_URL    = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

# ---------------------------#
# Carga y validación de datos
# ---------------------------#
@st.cache_data(show_spinner=True, ttl=60*60)
def load_owid_energy() -> pd.DataFrame:
    return pd.read_csv(OWID_ENERGY_URL)

@st.cache_data(show_spinner=True, ttl=60*60)
def load_owid_co2() -> pd.DataFrame:
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

def fmt(x, d=2, default="—"):
    try:
        return f"{float(x):,.{d}f}"
    except Exception:
        return default

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

# ---------------------------#
# Helpers VAR
# ---------------------------#
def compute_renewables_share_timeseries(e_country: pd.DataFrame) -> pd.Series:
    if e_country.empty: return pd.Series(dtype=float)
    mix, _ = build_mix_columns(e_country)
    if mix.empty: return pd.Series(dtype=float)
    gen = e_country.setindex("year")["electricity_generation"] if "setindex" in dir(e_country) else e_country.set_index("year")["electricity_generation"]
    ren_cols = [c for c in mix.columns if any(kw in c for kw in ["hydro","wind","solar","biofuel","bioenergy","other_renewables"])]
    ren_twh = mix[ren_cols].sum(axis=1)
    share = 100 * (ren_twh / gen.reindex(mix.index))
    return share.dropna()

def adf_result(x: pd.Series):
    x = x.dropna()
    if len(x) < 8: return {"stat": np.nan, "p": np.nan}
    try:
        stat, p, *_ = adfuller(x, autolag="AIC")
        return {"stat": stat, "p": p}
    except Exception:
        return {"stat": np.nan, "p": np.nan}

def make_var_frame(country: str, yr: tuple, energy_sa: pd.DataFrame, co2_sa: pd.DataFrame) -> pd.DataFrame:
    e_c = energy_sa[(energy_sa["country"]==country) & (energy_sa["year"].between(yr[0], yr[1]))].copy()
    c_c = co2_sa[(co2_sa["country"]==country) & (co2_sa["year"].between(yr[0], yr[1]))].copy()
    if e_c.empty or c_c.empty: return pd.DataFrame()
    ren_share = compute_renewables_share_timeseries(e_c)  # %
    elec_gen  = e_c.set_index("year")["electricity_generation"]
    co2_mt    = c_c.set_index("year")["co2"]   # Mt
    gdp_usd   = c_c.set_index("year")["gdp"] if "gdp" in c_c.columns else None
    df = pd.DataFrame(index=sorted(set(elec_gen.index) & set(co2_mt.index)))
    df.index.name = "year"
    df["co2"] = co2_mt.reindex(df.index)
    if gdp_usd is not None: df["gdp"] = gdp_usd.reindex(df.index)
    df["ren_share"] = ren_share.reindex(df.index)
    df["elec_gen"]  = elec_gen.reindex(df.index)
    return df.dropna()

def transform_for_var(df: pd.DataFrame, vars_sel: list, transform: str) -> pd.DataFrame:
    X = df[vars_sel].copy()
    if transform == "levels":
        return X.dropna()
    elif transform == "logdiff":
        for c in X.columns:
            X[c] = np.log(X[c]).replace([-np.inf, np.inf], np.nan)
        return X.diff().dropna()
    elif transform == "pctchg":
        return X.pct_change().dropna()
    else:
        return X.dropna()

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

# Filtrar Sudamérica
energy_sa = energy[energy["iso_code"].isin(SOUTH_AMERICA_ISO)].copy()
co2_sa    = co2[co2["iso_code"].isin(SOUTH_AMERICA_ISO)].copy()

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

# Filtrado por país y años
e_c = energy_sa[(energy_sa["country"]==country) & (energy_sa["year"].between(yr[0], yr[1]))].copy()
c_c = co2_sa[(co2_sa["country"]==country) & (co2_sa["year"].between(yr[0], yr[1]))].copy()

# ---------------------------#
# Header con logo y título
# ---------------------------#
c_logo, c_title = st.columns([1,9], vertical_alignment="center")
with c_logo:
    _logo = load_logo_bytes()
    if _logo is not None:
        st.image(_logo, use_container_width=False)
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
        gen_last = e_c.loc[e_c["year"]==last_year, "electricity_generation"]
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
tab_mix, tab_co2, tab_rank, tab_map, tab_var, tab_data = st.tabs([
    "Mezcla eléctrica",
    "Emisiones CO₂",
    "Ranking renovables",
    "Mapa regional",
    "Modelo VAR",
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

# ---- Modelo VAR ----
with tab_var:
    st.subheader(f"Modelo VAR — {country}")

    with st.expander("Configuración del modelo", expanded=True):
        vars_catalog = {
            "Emisiones CO₂ (Mt)": "co2",
            "PIB (USD corrientes)": "gdp",
            "% Renovables en generación": "ren_share",
            "Generación eléctrica (TWh)": "elec_gen",
        }
        var_labels = list(vars_catalog.keys())
        sel = st.multiselect("Variables (elige 2 a 4):", var_labels,
                             default=["Emisiones CO₂ (Mt)", "% Renovables en generación", "PIB (USD corrientes)"])
        vars_sel = [vars_catalog[v] for v in sel]
        transform = st.selectbox("Transformación", ["logdiff","pctchg","levels"], index=0,
                                 help="logdiff: Δlog(x); pctchg: % cambio; levels: sin cambios (si ya son estacionarias).")
        maxlags = st.slider("Máximo rezago para selección por AIC", 1, 6, 4)
        irf_h   = st.slider("Horizonte IRF (años)", 1, 12, 8)
        run = st.button("🧮 Estimar VAR", type="primary")

    df_var = make_var_frame(country, yr, energy_sa, co2_sa)
    if df_var.empty:
        st.info("No hay suficientes datos combinados para este país y rango.")
    elif len(vars_sel) < 2:
        st.info("Selecciona al menos 2 variables para el VAR.")
    else:
        X = transform_for_var(df_var, vars_sel, transform)
        st.markdown("**Ventana del modelo (tras transformaciones):**")
        st.write(f"{X.index.min()} — {X.index.max()}  ·  Observaciones: {len(X)}")
        st.dataframe(X.tail())

        cols_adf = st.columns(len(vars_sel))
        for i, c in enumerate(vars_sel):
            res = adf_result(X[c])
            ptxt = f"p={res['p']:.3f}" if not np.isnan(res['p']) else "—"
            cols_adf[i].metric(f"ADF: {c}", ptxt, help="H0: raíz unitaria (no estacionaria)")

        if run:
            if len(X) < (maxlags + 8):
                st.warning("Pocas observaciones para ese máximo de rezagos. Reduce 'máximo rezago' o amplía el rango.")
            else:
                try:
                    sel_order = VAR(X).select_order(maxlags=maxlags)
                    p_opt = int(sel_order.aic)
                    st.success(f"Rezago óptimo por AIC: p = {p_opt}")

                    model = VAR(X).fit(p_opt)
                    st.markdown("**Resumen VAR:**")
                    st.text(model.summary())

                    st.markdown("**Causalidad de Granger (p-valores):**")
                    rows = []
                    for caused, causing in itertools.permutations(X.columns, 2):
                        try:
                            test = model.test_causality(caused, causing, kind='f')
                            rows.append({"causado": caused, "causal": causing, "p_value": float(test.pvalue)})
                        except Exception:
                            rows.append({"causado": caused, "causal": causing, "p_value": np.nan})
                    mat = pd.DataFrame(rows).pivot(index="causado", columns="causal", values="p_value").round(3)
                    st.dataframe(mat)

                    st.markdown("**Funciones Impulso–Respuesta (IRF):**")
                    irf = model.irf(irf_h)
                    for resp in X.columns:
                        curves = []
                        for shock in X.columns:
                            vals = irf.irfs[:, X.columns.get_loc(resp), X.columns.get_loc(shock)]
                            curves.append(pd.DataFrame({
                                "h": np.arange(irf_h+1),
                                "respuesta": resp, "shock": shock, "valor": vals
                            }))
                        df_plot = pd.concat(curves, ignore_index=True)

                        # 🎨 IRF con colores intensos y leyenda horizontal
                        fig_irf = px.line(
                            df_plot, x="h", y="valor", color="shock",
                            title=f"IRF de {resp} (shock por variable)",
                            labels={"h": "horizonte (años)", "valor": "respuesta acumulada"},
                            color_discrete_sequence=[
                                "#FF6B6B", "#FFD93D", "#6BCB77",
                                "#4D96FF", "#C77DFF", "#FF9F1C"
                            ]
                        )
                        fig_irf.update_traces(line=dict(width=3))
                        fig_irf.update_layout(
                            height=420,
                            margin=dict(l=10, r=10, t=50, b=20),
                            legend_title_text="Shock",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom", y=-0.35,
                                xanchor="center", x=0.5,
                                font=dict(size=12)
                            ),
                            title=dict(font=dict(size=18))
                        )
                        st.plotly_chart(fig_irf, use_container_width=True,
                                        key=f"irf_{resp}_{country}_{yr[0]}_{yr[1]}")

                except Exception as e:
                    st.error("Falló la estimación del VAR.")
                    st.exception(e)
                    st.info("Revisa que las variables no sean colineales o que haya observaciones suficientes tras la transformación.")

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
