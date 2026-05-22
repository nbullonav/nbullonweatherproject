import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURACIÓN DEL DASHBOARD
# -----------------------------
st.set_page_config(
    page_title="Forecast de Estaciones",
    layout="wide"
)

st.title("📈 Forecast de 3 Estaciones (Historic + SARIMAX + XGB + Hybrid)")

tab1, tab2 = st.tabs(["📊 Forecast General", "🌊 Escenarios ENSO"])

with tab1:    
    st.header("📊 Forecast General")  
    # -----------------------------
    # CARGA DE DATOS
    # -----------------------------
    @st.cache_data
    def load_station(station_id):
        df = pd.read_csv(f"data/pred_{station_id}.csv")
        df["time"] = pd.to_datetime(df["time"])
        return df

    stations = {
        "Estación 84501": "84501",
        "Estación 84401": "84401",
        "Estación 84370": "84370"
    }

    # -----------------------------
    # SIDEBAR
    # -----------------------------
    st.sidebar.header("Opciones (sólo tab#1 Forecast General)")

    station_name = st.sidebar.selectbox("Selecciona estación", list(stations.keys()))

    station_id = stations[station_name]

    df = load_station(station_id)

    models = st.sidebar.multiselect(
        "Modelos a mostrar",
        ["historic", "sarimax", "xgb", "hybrid"],
        default=["historic", "sarimax", "xgb", "hybrid"]
    )

    df["time"] = pd.to_datetime(df["time"])

    # Fechas mínimas y máximas como datetime puro
    min_date = pd.to_datetime(df["time"].min()).to_pydatetime()
    max_date = pd.to_datetime(df["time"].max()).to_pydatetime()

    date_range = st.sidebar.slider(
        "Rango de fechas",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        key='slider_tab1'
    )

    start, end = date_range
    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)

    # Filtrar
    df_filtered = df[(df["time"] >= start) & (df["time"] <= end)]


    test_metrics = pd.read_csv(f"data/testsets_{station_id}.csv")
    test_metrics["time"] = pd.to_datetime(test_metrics["time"])
    test_metrics.set_index("time", inplace=True)

    # metricas streamlit

    def compute_metrics_test(df):
        metrics = {}
        for model in ["sarimax", "xgb", "hybrid"]:
            y_true = df["real"]
            y_pred = df[model]

            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))
            nse_val = nse(y_true, y_pred)
            kge_val = kge(y_true, y_pred)

            metrics[model] = {
                "rmse": rmse,
                "mae": mae,
                "nse": nse_val,
                "kge": kge_val
            }
        return metrics

    def nse(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

    def kge(y_true, y_pred):
        r = np.corrcoef(y_true, y_pred)[0, 1]
        alpha = np.std(y_pred) / np.std(y_true)
        beta = np.mean(y_pred) / np.mean(y_true)
        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)



    # -----------------------------
    # GRÁFICO
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    if "historic" in models:
        ax.plot(df_filtered["time"], df_filtered["historic"], label="Historic", color="black")

    if "sarimax" in models:
        ax.plot(df_filtered["time"], df_filtered["sarimax"], label="SARIMAX", linestyle="--", color="blue")

    if "xgb" in models:
        ax.plot(df_filtered["time"], df_filtered["xgb"], label="XGB", linestyle=":", color="orange")

    if "hybrid" in models:
        ax.plot(df_filtered["time"], df_filtered["hybrid"], label="Hybrid", color="red")

    ax.set_title(f"Forecast - Estación {station_id}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor (log o real)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # TABLA DE DATOS
    # -----------------------------
    st.subheader("Datos filtrados")
    st.dataframe(df_filtered)




    # -----------------------------
    # GRÁFICO DEL TEST SET
    # -----------------------------
    st.subheader("📉 Comparación en el Test Set (2016–2026)")

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(test_metrics.index, test_metrics["real"], label="Real", color="black")
    ax2.plot(test_metrics.index, test_metrics["sarimax"], label="SARIMAX", linestyle="--", color="blue")
    ax2.plot(test_metrics.index, test_metrics["xgb"], label="XGB", linestyle=":", color="orange")
    ax2.plot(test_metrics.index, test_metrics["hybrid"], label="Hybrid", color="red")

    ax2.set_title(f"Comparación de Modelos - Test Set {station_id}")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Valor real")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)


    st.subheader("📊 Métricas del Test Set (2016–2026)")

    metrics = compute_metrics_test(test_metrics)
    col1, col2, col3 = st.columns(3)
    for model, col in zip(["sarimax", "xgb", "hybrid"], [col1, col2, col3]):
        m = metrics[model]
        col.metric(f"{model.upper()} RMSE", f"{m['rmse']:.4f}")
        col.metric(f"{model.upper()} MAE", f"{m['mae']:.4f}")

    st.subheader("📈 Métricas Hidrológicas (NSE / KGE)")
    col1, col2, col3 = st.columns(3)
    for model, col in zip(["sarimax", "xgb", "hybrid"], [col1, col2, col3]):
        m = metrics[model]
        col.metric(f"{model.upper()} NSE", f"{m['nse']:.4f}")
        col.metric(f"{model.upper()} KGE", f"{m['kge']:.4f}")


with tab2:
    st.header("🌊 Escenarios ENSO (Hybrid)")

    # -----------------------------
    # Cargar escenarios ENSO
    # -----------------------------
    import os

    scenario_files = {
        "neutral": f"data/pred_{station_id}_neutral.csv",
        "nino_moderado": f"data/pred_{station_id}_nino_moderado.csv",
        "nino_fuerte": f"data/pred_{station_id}_nino_fuerte.csv",
        "nina_moderada": f"data/pred_{station_id}_nina_moderada.csv"
    }

    scenario_dfs = {}

    for scen, path in scenario_files.items():
        if os.path.exists(path):
            df_s = pd.read_csv(path)
            df_s["time"] = pd.to_datetime(df_s["time"])
            scenario_dfs[scen] = df_s

    # -----------------------------
    # Selector de escenarios
    # -----------------------------
    scen_selected = st.multiselect(
        "Selecciona escenarios ENSO a mostrar",
        list(scenario_dfs.keys()),
        default=list(scenario_dfs.keys())
    )

    # -----------------------------
    # Construir DataFrame combinado
    # -----------------------------
    historic = df[["time", "historic"]].copy()
    historic.set_index("time", inplace=True)

    plot_df = historic.copy()

    for scen in scen_selected:
        df_s = scenario_dfs[scen].copy()
        df_s = df_s.set_index("time")
        plot_df = plot_df.join(df_s["hybrid"].rename(scen), how="outer")
        

    # -----------------------------
    # Selector de rango de años para ENSO
    # -----------------------------
    st.subheader("Filtrar rango de años para comparar escenarios")

    min_enso_date = plot_df.index.min().to_pydatetime()
    max_enso_date = plot_df.index.max().to_pydatetime()

    enso_range = st.slider(
        "Rango de fechas (solo afecta escenarios ENSO)",
        min_value=min_enso_date,
        max_value=max_enso_date,
        value=(min_enso_date, max_enso_date),
        key="enso_range_slider"
    )

    enso_start, enso_end = enso_range
    enso_start = pd.to_datetime(enso_start).tz_localize(None)
    enso_end = pd.to_datetime(enso_end).tz_localize(None)

    plot_df_filtered = plot_df[(plot_df.index >= enso_start) & (plot_df.index <= enso_end)]


    # -----------------------------
    # Gráfico ENSO
    # -----------------------------

    # -----------------------------
    # Convertir de log(prcp) a precipitación real
    # -----------------------------
    plot_df_filtered_real = plot_df_filtered.copy()

    # Convertir histórico
    #plot_df_filtered_real["historic"] = np.expm1(plot_df_filtered_real["historic"])
    
    # Convertir escenarios seleccionados
    for scen in scen_selected:
        plot_df_filtered_real[scen] = plot_df_filtered_real[scen]


    fig3, ax3 = plt.subplots(figsize=(12, 5))

    ax3.plot(plot_df_filtered_real.index, plot_df_filtered_real["historic"], label="Histórico", color="black", linewidth=2)

    colors_scen = {
        "neutral": "green",
        "nino_moderado": "orange",
        "nino_fuerte": "red",
        "nina_moderada": "blue"
    }

    for scen in scen_selected:
        ax3.plot(
            plot_df_filtered_real.index,
            plot_df_filtered_real[scen],
            label=scen.replace("_", " ").title(),
            linestyle="--",
            color=colors_scen[scen]
        )

    ax3.axvline(historic.index[-1], color="gray", linestyle="--", linewidth=1)
    ax3.set_title(f"Escenarios ENSO - Estación {station_id}")
    ax3.set_xlabel("Fecha")
    ax3.set_ylabel("Precipitación (real)")
    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig3)

    # -----------------------------
    # Tabla ENSO
    # -----------------------------
    st.subheader("Datos de escenarios ENSO")
    st.dataframe(plot_df.tail(24))
