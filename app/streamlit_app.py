"""
F1 Race Prediction — Streamlit App
Pages:
  1. Driver & Team Form — rolling performance trends
  2. Race Predictor     — predicted top-3 probabilities for any race in the dataset
  3. Model Evaluation   — accuracy, recall, F1, confusion matrices
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
METRICS = ROOT / "metrics"

st.set_page_config(
    page_title="F1 Race Prediction",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading race data...")
def load_model_table() -> pd.DataFrame | None:
    p = PROCESSED / "model_table.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_resource(show_spinner="Loading models...")
def load_model(model_type: str, target: str):
    p = ARTIFACTS / f"{model_type}_{target}.pkl"
    return joblib.load(p) if p.exists() else None


@st.cache_data(show_spinner="Loading evaluation results...")
def load_eval_results() -> list[dict] | None:
    p = METRICS / "evaluation_results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_feature_names() -> list[str]:
    p = ARTIFACTS / "feature_names.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("🏎️ F1 Race Prediction")
st.sidebar.caption("Multi-source F1 Analytics Platform")

page = st.sidebar.radio(
    "Navigate",
    ["Driver & Team Form", "Race Predictor", "Model Evaluation"],
    index=0,
)

df = load_model_table()

if df is None:
    st.warning(
        "No model table found. Run the full pipeline first:\n\n"
        "```\n"
        "python src/ingestion/ingest_ergast.py\n"
        "python src/ingestion/ingest_openf1.py\n"
        "python src/processing/clean_data.py\n"
        "python src/features/build_features.py\n"
        "python src/models/train.py\n"
        "```"
    )
    st.stop()

# ── Page 1: Driver & Team Form ─────────────────────────────────────────────────
if page == "Driver & Team Form":
    st.title("Driver & Team Form")
    st.caption("Rolling performance across the 2021–2025 seasons")

    col1, col2 = st.columns(2)
    with col1:
        all_drivers = sorted(df["driver_id"].dropna().unique())
        selected_drivers = st.multiselect(
            "Select drivers",
            all_drivers,
            default=all_drivers[:5] if len(all_drivers) >= 5 else all_drivers,
        )
    with col2:
        seasons = sorted(df["season"].unique())
        season_range = st.select_slider(
            "Season range",
            options=seasons,
            value=(seasons[0], seasons[-1]),
        )

    mask = (
        df["driver_id"].isin(selected_drivers)
        & df["season"].between(season_range[0], season_range[1])
    )
    filtered = df[mask].copy()

    if filtered.empty:
        st.info("No data for selected filters.")
        st.stop()

    # Race-by-race finish positions
    race_label = filtered["season"].astype(str) + " R" + filtered["round"].astype(str)
    filtered = filtered.copy()
    filtered["race_label"] = race_label

    st.subheader("Finish Position per Race")
    fig1 = px.line(
        filtered.sort_values(["season", "round"]),
        x="race_label",
        y="finish_position",
        color="driver_id",
        markers=True,
        title="Race Finish Position (lower = better)",
        labels={"finish_position": "Finish Position", "race_label": "Race", "driver_id": "Driver"},
    )
    fig1.update_yaxes(autorange="reversed")
    fig1.update_layout(xaxis_tickangle=-45, height=420)
    st.plotly_chart(fig1, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Rolling Avg Finish (last 5 races)")
        if "avg_finish_last5" in filtered.columns:
            fig2 = px.line(
                filtered.sort_values(["season", "round"]),
                x="race_label",
                y="avg_finish_last5",
                color="driver_id",
                title="5-Race Rolling Average Finish",
                labels={"avg_finish_last5": "Avg Finish", "race_label": "Race"},
            )
            fig2.update_yaxes(autorange="reversed")
            fig2.update_layout(xaxis_tickangle=-45, height=380)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Rolling average feature not available.")

    with col4:
        st.subheader("Points per Race")
        if "points" in filtered.columns:
            fig3 = px.bar(
                filtered.sort_values(["season", "round"]),
                x="race_label",
                y="points",
                color="driver_id",
                barmode="group",
                title="Points Scored per Race",
                labels={"points": "Points", "race_label": "Race"},
            )
            fig3.update_layout(xaxis_tickangle=-45, height=380)
            st.plotly_chart(fig3, use_container_width=True)

    # Constructor comparison
    st.subheader("Constructor Average Points per Season")
    if "constructor_id" in df.columns and "points" in df.columns:
        constructor_season = (
            df[df["season"].between(season_range[0], season_range[1])]
            .groupby(["season", "constructor_id"])["points"]
            .mean()
            .reset_index()
            .rename(columns={"points": "avg_points"})
        )
        fig4 = px.bar(
            constructor_season,
            x="season",
            y="avg_points",
            color="constructor_id",
            barmode="group",
            title="Constructor Avg Points per Race by Season",
        )
        st.plotly_chart(fig4, use_container_width=True)


# ── Page 2: Race Predictor ─────────────────────────────────────────────────────
elif page == "Race Predictor":
    st.title("Race Predictor")
    st.caption("Predicted podium probabilities using the trained XGBoost model")

    feature_cols = load_feature_names()
    xgb_top3 = load_model("xgb", "is_top3")
    xgb_winner = load_model("xgb", "is_winner")

    if xgb_top3 is None or xgb_winner is None:
        st.warning("Models not found. Run `python src/models/train.py` first.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        selected_season = st.selectbox("Season", sorted(df["season"].unique(), reverse=True))
    with col2:
        rounds_in_season = sorted(df[df["season"] == selected_season]["round"].unique())
        selected_round = st.selectbox("Round", rounds_in_season)

    race_df = df[(df["season"] == selected_season) & (df["round"] == selected_round)].copy()

    if race_df.empty:
        st.warning("No data for this race.")
        st.stop()

    race_name = race_df["circuit_id"].iloc[0] if "circuit_id" in race_df.columns else f"Round {selected_round}"
    st.subheader(f"Season {selected_season} — Round {selected_round} ({race_name})")

    X = race_df[[c for c in feature_cols if c in race_df.columns]].fillna(0)
    if X.empty or set(feature_cols) - set(race_df.columns):
        missing = set(feature_cols) - set(race_df.columns)
        st.warning(f"Missing feature columns: {missing}")
        st.stop()

    race_df = race_df.copy()
    race_df["p_top3"] = xgb_top3.predict_proba(X)[:, 1]
    race_df["p_winner"] = xgb_winner.predict_proba(X)[:, 1]
    race_df["predicted_top3"] = (race_df["p_top3"] >= 0.5).astype(int)

    display = race_df[["driver_id", "p_top3", "p_winner"]].copy()
    display = display.rename(columns={
        "driver_id": "Driver",
        "p_top3": "P(Top 3)",
        "p_winner": "P(Winner)",
    })
    display = display.sort_values("P(Top 3)", ascending=False).reset_index(drop=True)
    display.index += 1
    display["P(Top 3)"] = display["P(Top 3)"].map("{:.1%}".format)
    display["P(Winner)"] = display["P(Winner)"].map("{:.1%}".format)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Predicted Probabilities")
        st.dataframe(display, use_container_width=True)

    with col_b:
        st.subheader("Top 3 Probability — Bar Chart")
        plot_df = race_df.sort_values("p_top3", ascending=False).head(10)
        fig = px.bar(
            plot_df,
            x="driver_id",
            y="p_top3",
            color="p_top3",
            color_continuous_scale="RdYlGn",
            title="Top-10 Drivers by Podium Probability",
            labels={"driver_id": "Driver", "p_top3": "P(Top 3)"},
            range_y=[0, 1],
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% threshold")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("Key Prediction Drivers (XGBoost Feature Importance)")
    importances = xgb_top3.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(12)
    fig_imp = px.bar(
        feat_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importances (is_top3 model)",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    if "finish_position" in race_df.columns:
        st.subheader("Actual vs Predicted (if race has completed)")
        actual = race_df[["driver_id", "finish_position", "p_top3", "predicted_top3"]].copy()
        actual["actual_top3"] = (actual["finish_position"] <= 3).astype(int)
        actual["correct"] = (actual["predicted_top3"] == actual["actual_top3"])
        actual = actual.sort_values("finish_position")
        actual = actual.rename(columns={
            "driver_id": "Driver", "finish_position": "Finish",
            "p_top3": "P(Top 3)", "predicted_top3": "Pred Top 3",
            "actual_top3": "Actual Top 3", "correct": "Correct",
        })
        actual["P(Top 3)"] = actual["P(Top 3)"].map("{:.1%}".format)
        st.dataframe(actual.reset_index(drop=True), use_container_width=True)


# ── Page 3: Model Evaluation ───────────────────────────────────────────────────
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.caption("Test set metrics — seasons held out from training")

    eval_results = load_eval_results()
    if eval_results is None:
        st.warning("No evaluation results found. Run `python src/models/evaluate.py` first.")
        st.stop()

    # Summary table
    summary_rows = []
    for r in eval_results:
        summary_rows.append({
            "Model": r["model"],
            "Target": r["target"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "ROC-AUC": r.get("roc_auc", "N/A"),
            "Test Rows": r["n_test"],
            "Positives": r["n_positive"],
        })
    summary_df = pd.DataFrame(summary_rows)
    st.subheader("Summary Metrics")
    st.dataframe(
        summary_df.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
        }),
        use_container_width=True,
    )

    # Radar chart comparison
    st.subheader("Model Comparison — Radar Chart")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1"]
    fig_radar = go.Figure()
    for _, row in summary_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics_to_plot] + [row[metrics_to_plot[0]]],
            theta=metrics_to_plot + [metrics_to_plot[0]],
            fill="toself",
            name=f"{row['Model']} ({row['Target']})",
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=420)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    n_cols = min(len(eval_results), 2)
    cols = st.columns(n_cols)
    for i, r in enumerate(eval_results):
        cm = r["confusion_matrix"]
        labels = r.get("confusion_matrix_labels", ["Neg", "Pos"])
        cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels],
                             columns=[f"Pred {l}" for l in labels])
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"{r['model']} — {r['target']}",
            aspect="auto",
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        with cols[i % n_cols]:
            st.plotly_chart(fig_cm, use_container_width=True)

    # Interpretation notes
    st.subheader("Interpretation Guide")
    st.markdown(
        """
        - **Recall** measures how many true podium finishers/winners were captured.
          For sports prediction, recall matters more than precision — missing a real podium finisher is costly.
        - **Precision** measures false-positive control: of all predicted positives, how many were right.
        - **ROC-AUC** measures ranked separation ability across all thresholds —
          closer to 1.0 means the model clearly separates podium from non-podium drivers.
        - Class imbalance (3 podiums per 20 drivers = 15% positive rate) is handled via
          `class_weight='balanced'` in Logistic Regression and `scale_pos_weight` in XGBoost.
        """
    )
