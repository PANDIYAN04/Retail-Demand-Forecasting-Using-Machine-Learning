"""Streamlit dashboard for retail demand analysis and prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modeling import DATA_PATH, INPUT_FEATURES, load_model_bundle, load_dataset
from modeling import validate_prediction_inputs

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
PREDICTION_COLUMN = "Predicted Demand (units)"
COLORS = {
    "primary": "#3157D5",
    "secondary": "#14B8A6",
    "accent": "#F59E0B",
    "background": "#F5F7FB",
    "text": "#172033",
}

st.markdown(
    f"""
    <style>
    .stApp,
    [data-testid="stAppViewContainer"] {{
        background: {COLORS["background"]};
        color: {COLORS["text"]};
    }}
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp p,
    .stApp label,
    .stApp [data-testid="stMarkdownContainer"] {{
        color: {COLORS["text"]};
    }}
    [data-testid="stSidebar"] {{
        background: #20283D;
    }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {{
        color: #F8FAFC;
    }}
    [data-testid="stMetric"] {{
        background: #FFFFFF; border: 1px solid #E6EAF2;
        padding: 16px 18px; border-radius: 12px;
    }}
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {{
        color: #526078 !important;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {{
        color: #172033 !important;
        font-size: 1.3rem !important;
        line-height: 1.2;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere;
    }}
    [data-testid="stMetricDelta"] {{
        color: #526078 !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.85rem;
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
    }}
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        background: #FFFFFF;
    }}
    .stPlotlyChart {{
        background: #FFFFFF;
        border: 1px solid #E6EAF2;
        border-radius: 12px;
        padding: 8px;
    }}
    .hero {{
        background: linear-gradient(120deg, #1D3FA8, #3157D5 58%, #14B8A6);
        color: white; padding: 28px 32px; border-radius: 16px;
        margin-bottom: 20px;
    }}
    .hero h1 {{ color: white !important; margin-bottom: 8px; }}
    .hero p {{ color: #E8EEFF !important; margin-bottom: 0; font-size: 1.05rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_resource
def get_model_bundle() -> dict:
    return load_model_bundle()


def show_prediction_error(error: Exception) -> None:
    st.error(str(error))


def make_prediction(data: pd.DataFrame, bundle: dict) -> np.ndarray:
    checked = validate_prediction_inputs(data, bundle["categories"])
    return np.asarray(bundle["pipeline"].predict(checked), dtype=float)


def render_home(data: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Retail Demand Forecasting</h1>
          <p>Explore historical retail patterns and estimate product demand
          from pricing, inventory, promotions, and category.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Dataset at a glance")
    total_demand = int(data["Demand"].sum())
    first_date = data["Date"].min().strftime("%b %d, %Y")
    last_date = data["Date"].max().strftime("%b %d, %Y")
    columns = st.columns(4)
    columns[0].metric("Records", f"{len(data):,}")
    columns[1].metric("Avg demand", f"{data['Demand'].mean():,.1f} units")
    columns[2].metric(
        "Total demand",
        f"{total_demand / 1_000_000:.2f}M units",
        help=f"Exact historical demand: {total_demand:,} units",
    )
    columns[3].metric("Categories", f"{data['Category'].nunique()}")
    st.caption(
        f"Historical records from {first_date} to {last_date} · "
        f"{data['Date'].nunique():,} distinct dates · "
        f"{data['Category'].nunique()} categories."
    )

    st.subheader("Monthly demand trend")
    monthly = (
        data.set_index("Date")["Demand"]
        .resample("MS")
        .mean()
        .rename("Average demand")
        .reset_index()
    )
    figure = px.line(
        monthly,
        x="Date",
        y="Average demand",
        markers=True,
        color_discrete_sequence=[COLORS["primary"]],
        labels={"Date": "Month", "Average demand": "Average demand (units)"},
    )
    figure.update_layout(margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
    st.plotly_chart(figure, width="stretch")
    st.info(
        "This model estimates demand from the fields available in the prediction "
        "form. It is a historical-data estimate, not a guarantee of future sales."
    )


def render_analysis(data: pd.DataFrame) -> None:
    st.title("Data Analysis")
    available_categories = sorted(data["Category"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Filter categories", available_categories, default=available_categories
    )
    filtered = data[data["Category"].isin(selected_categories)]
    if filtered.empty:
        st.warning("Select at least one category to display analysis.")
        return

    left, right = st.columns(2)
    with left:
        distribution = px.histogram(
            filtered,
            x="Demand",
            nbins=35,
            color_discrete_sequence=[COLORS["primary"]],
            title="Demand distribution",
            labels={"Demand": "Demand (units)"},
        )
        st.plotly_chart(distribution, width="stretch")
    with right:
        category_demand = (
            filtered.groupby("Category", as_index=False)["Demand"]
            .mean()
            .sort_values("Demand", ascending=False)
        )
        category_chart = px.bar(
            category_demand,
            x="Category",
            y="Demand",
            color="Category",
            title="Average demand by category",
            labels={"Demand": "Average demand (units)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(category_chart, width="stretch")

    left, right = st.columns(2)
    with left:
        sample = filtered.sample(min(5000, len(filtered)), random_state=42)
        price_chart = px.scatter(
            sample,
            x="Price",
            y="Demand",
            color="Category",
            opacity=0.55,
            title="Price versus demand",
            labels={"Price": "Price", "Demand": "Demand (units)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(price_chart, width="stretch")
    with right:
        promotion_demand = (
            filtered.groupby("Promotion", as_index=False)["Demand"]
            .mean()
            .sort_values("Promotion")
        )
        promotion_demand["Promotion"] = promotion_demand["Promotion"].map(
            {0: "No promotion", 1: "Promotion"}
        )
        promotion_chart = px.bar(
            promotion_demand,
            x="Promotion",
            y="Demand",
            color="Promotion",
            title="Average demand by promotion",
            labels={"Demand": "Average demand (units)"},
            color_discrete_sequence=[COLORS["secondary"], COLORS["accent"]],
        )
        st.plotly_chart(promotion_chart, width="stretch")

    discount_summary = (
        filtered.assign(Discount=filtered["Discount"].round().astype(int))
        .groupby("Discount", as_index=False)["Demand"]
        .mean()
    )
    discount_chart = px.line(
        discount_summary,
        x="Discount",
        y="Demand",
        markers=True,
        title="Discount versus average demand",
        labels={"Discount": "Discount (%)", "Demand": "Average demand (units)"},
        color_discrete_sequence=[COLORS["primary"]],
    )
    st.plotly_chart(discount_chart, width="stretch")
    st.caption(
        "These charts describe associations in the historical dataset; "
        "they do not establish that price, discounts, or promotions caused a change."
    )


def render_prediction(bundle: dict) -> None:
    st.title("Demand Prediction")
    st.write("Enter the product and pricing details to estimate demand in units.")
    categories = bundle["categories"]
    with st.form("single_prediction"):
        price, discount, inventory = st.columns(3)
        with price:
            price_value = st.number_input("Price", min_value=0.0, value=50.0, step=1.0)
        with discount:
            discount_value = st.number_input(
                "Discount (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0
            )
        with inventory:
            inventory_value = st.number_input(
                "Inventory level", min_value=0, value=100, step=1
            )

        promotion, competitor, category = st.columns(3)
        with promotion:
            promotion_value = st.selectbox("Promotion", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with competitor:
            competitor_value = st.number_input(
                "Competitor price", min_value=0.0, value=50.0, step=1.0
            )
        with category:
            category_value = st.selectbox("Product category", categories)
        submitted = st.form_submit_button("Predict demand", type="primary")

    if submitted:
        row = pd.DataFrame(
            {
                "Price": [price_value],
                "Discount": [discount_value],
                "Inventory Level": [inventory_value],
                "Promotion": [promotion_value],
                "Competitor Pricing": [competitor_value],
                "Category": [category_value],
            }
        )
        try:
            prediction = float(make_prediction(row, bundle)[0])
            if not np.isfinite(prediction):
                raise ValueError("The model returned a non-finite prediction.")
        except (ValueError, TypeError) as error:
            show_prediction_error(error)
            return
        st.success(f"Estimated demand: **{max(0, round(prediction)):,} units**")
        st.caption("Estimate generated by the saved preprocessing and prediction pipeline.")


def render_model_performance(bundle: dict) -> None:
    st.title("Model Performance")
    st.caption(
        f"Selected using the chronological validation period. "
        f"Final test period: {bundle['test_dates']['start']} to "
        f"{bundle['test_dates']['end']}."
    )
    st.subheader(f"Selected model: {bundle['selected_model']}")

    comparison = pd.DataFrame(bundle["validation_metrics"])
    st.subheader("Validation model comparison")
    st.dataframe(
        comparison.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "The model with the lowest validation MAE is selected. The final test "
        "period was kept separate from model selection."
    )

    metrics = bundle["test_metrics"]
    st.subheader("Selected model on the untouched test period")
    columns = st.columns(3)
    columns[0].metric("MAE", f"{metrics['MAE']:.2f} units")
    columns[1].metric("RMSE", f"{metrics['RMSE']:.2f} units")
    columns[2].metric("R²", f"{metrics['R2']:.3f}")

    prediction_path = PROJECT_DIR / "models" / "test_predictions.csv"
    if prediction_path.is_file():
        predictions = pd.read_csv(prediction_path, parse_dates=["Date"])
        actual_vs_predicted = px.scatter(
            predictions.sample(min(4000, len(predictions)), random_state=42),
            x="Actual Demand",
            y="Predicted Demand",
            opacity=0.55,
            title="Actual versus predicted demand (test period)",
            labels={"Actual Demand": "Actual demand (units)", "Predicted Demand": "Predicted demand (units)"},
            color_discrete_sequence=[COLORS["primary"]],
        )
        lower = min(predictions["Actual Demand"].min(), predictions["Predicted Demand"].min())
        upper = max(predictions["Actual Demand"].max(), predictions["Predicted Demand"].max())
        actual_vs_predicted.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                name="Ideal prediction",
                line=dict(color=COLORS["accent"], dash="dash"),
            )
        )
        st.plotly_chart(actual_vs_predicted, width="stretch")

    importance_path = PROJECT_DIR / "models" / "feature_importance.csv"
    if importance_path.is_file():
        importance = pd.read_csv(importance_path).head(15).sort_values("Importance")
        importance_chart = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top feature importance",
            color_discrete_sequence=[COLORS["secondary"]],
        )
        st.plotly_chart(importance_chart, width="stretch")
        st.caption(
            "Feature importance shows model reliance, not causal impact. "
            "For linear regression, absolute coefficient magnitude is shown."
        )


def render_bulk_prediction(bundle: dict) -> None:
    st.title("Bulk Prediction")
    st.write("Upload a CSV with one row per scenario and these required columns:")
    st.code(", ".join(INPUT_FEATURES), language="text")
    template = pd.DataFrame(
        [
            {
                "Price": 50.0,
                "Discount": 10,
                "Inventory Level": 100,
                "Promotion": 1,
                "Competitor Pricing": 52.0,
                "Category": bundle["categories"][0],
            }
        ]
    )
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="demand_prediction_template.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Upload prediction CSV", type=["csv"])
    if upload is None:
        return

    try:
        uploaded_data = pd.read_csv(upload)
        checked = validate_prediction_inputs(uploaded_data, bundle["categories"])
        if checked.empty:
            raise ValueError("The uploaded CSV contains no prediction rows.")
        predictions = np.asarray(bundle["pipeline"].predict(checked), dtype=float)
        if not np.isfinite(predictions).all():
            raise ValueError("The model returned a non-finite prediction.")
        results = uploaded_data.copy()
        results[PREDICTION_COLUMN] = np.maximum(0, np.rint(predictions)).astype(int)
    except (
        ValueError,
        TypeError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        show_prediction_error(error)
        return

    st.success(f"Generated predictions for {len(results):,} rows.")
    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button(
        "Download predictions",
        results.to_csv(index=False).encode("utf-8"),
        file_name="demand_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:
    st.sidebar.title("Retail Demand")
    section = st.sidebar.radio(
        "Navigate",
        [
            "Home Dashboard",
            "Data Analysis",
            "Demand Prediction",
            "Model Performance",
            "Bulk Prediction",
        ],
    )
    try:
        data = get_dataset()
        bundle = get_model_bundle()
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(str(error))
        st.info("Check that the dataset and trained model bundle are present.")
        st.stop()

    if section == "Home Dashboard":
        render_home(data)
    elif section == "Data Analysis":
        render_analysis(data)
    elif section == "Demand Prediction":
        render_prediction(bundle)
    elif section == "Model Performance":
        render_model_performance(bundle)
    else:
        render_bulk_prediction(bundle)


if __name__ == "__main__":
    main()
