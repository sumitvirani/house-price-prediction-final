"""
House Price Prediction – Streamlit Frontend
============================================
Entrypoint: streamlit run streamlit_app.py  (or  streamlit run frontend/app.py)
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_json(endpoint: str):
    """GET an API endpoint and return parsed JSON, or None on error."""
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_dataset():
    """Load the house_data.csv used for the Data Explorer section."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "house_data.csv")
    # fallback for root-level entrypoint
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "house_data.csv")
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/000000/home--v2.png",
    width=80,
)
st.sidebar.title("🏠 House Price Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict Price", "📊 Data Explorer", "🤖 Model Insights", "ℹ️ About"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"API: `{API_BASE}`")

# ── Health check banner ────────────────────────────────────────────────────────
health = fetch_json("/health")
if health is None:
    st.warning(
        f"⚠️  Cannot reach the backend at **{API_BASE}**. "
        "Start the API server or set the `API_BASE` environment variable."
    )
    backend_ok = False
else:
    backend_ok = health.get("model_loaded", False)
    if not backend_ok:
        st.warning("⚠️  Backend reached but model is not loaded. Run `python model/train.py`.")

# ── Feature info ───────────────────────────────────────────────────────────────
feature_info = fetch_json("/feature-info") or {}

# ── Default feature values (fallback when API is unreachable) ──────────────────
_DEFAULTS = {
    "MedInc":     {"display_name":"Median Income (10k USD)","min":0.5, "max":15.0, "default":3.87, "step":0.1},
    "HouseAge":   {"display_name":"House Age (years)",       "min":1.0, "max":52.0, "default":28.0, "step":1.0},
    "AveRooms":   {"display_name":"Average Rooms",           "min":1.0, "max":20.0, "default":5.43, "step":0.1},
    "AveBedrms":  {"display_name":"Average Bedrooms",        "min":1.0, "max":5.0,  "default":1.10, "step":0.1},
    "Population": {"display_name":"Block Population",        "min":3.0, "max":5000.0,"default":1425.0,"step":10.0},
    "AveOccup":   {"display_name":"Average Occupancy",       "min":1.0, "max":10.0, "default":3.07, "step":0.1},
    "Latitude":   {"display_name":"Latitude",                "min":32.5,"max":42.0, "default":35.63,"step":0.01},
    "Longitude":  {"display_name":"Longitude",               "min":-124.4,"max":-114.3,"default":-119.57,"step":0.01},
}
if not feature_info:
    feature_info = _DEFAULTS

FEATURE_NAMES = list(feature_info.keys())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Predict Price
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Price":
    st.title("🏠 House Price Prediction")
    st.markdown(
        "Enter the block-group characteristics below and click **Predict Price** "
        "to get an estimated median house value."
    )
    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🏡 Property Features")

        input_vals = {}
        row1_cols = st.columns(2)
        row2_cols = st.columns(2)
        row3_cols = st.columns(2)
        row4_cols = st.columns(2)
        col_pairs = [row1_cols, row2_cols, row3_cols, row4_cols]

        for idx, feat in enumerate(FEATURE_NAMES):
            info  = feature_info[feat]
            col   = col_pairs[idx // 2][idx % 2]
            with col:
                input_vals[feat] = st.number_input(
                    label   = info.get("display_name", feat),
                    min_value = float(info["min"]),
                    max_value = float(info["max"]),
                    value     = float(info["default"]),
                    step      = float(info["step"]),
                    help      = info.get("description", ""),
                    key       = f"input_{feat}",
                )

        predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

    with col_right:
        st.subheader("📌 Feature Guide")
        for feat, info in feature_info.items():
            st.markdown(
                f"**{info.get('display_name', feat)}** – "
                f"<small>{info.get('description','')}</small>",
                unsafe_allow_html=True,
            )

    # ── Prediction ────────────────────────────────────────────────────────────
    if predict_btn:
        if not backend_ok:
            st.error("❌ Backend is not available. Cannot make predictions.")
        else:
            with st.spinner("Calling prediction API …"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/predict",
                        json=input_vals,
                        timeout=15,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        price      = data["predicted_price"]
                        price_low  = data["price_range_low"]
                        price_high = data["price_range_high"]

                        st.markdown("---")
                        st.markdown("### 💰 Prediction Result")

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Predicted Price",  f"${price:,.0f}")
                        m2.metric("Low Estimate",     f"${price_low:,.0f}")
                        m3.metric("High Estimate",    f"${price_high:,.0f}")

                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode  = "gauge+number+delta",
                            value = price,
                            title = {"text": "Predicted House Price (USD)"},
                            delta = {"reference": 200_000},
                            gauge = {
                                "axis": {"range": [0, 800_000], "tickformat": "$,.0f"},
                                "bar":  {"color": "#3b82d4"},
                                "steps": [
                                    {"range": [0,       200_000], "color": "#d1fae5"},
                                    {"range": [200_000, 400_000], "color": "#fef9c3"},
                                    {"range": [400_000, 800_000], "color": "#fee2e2"},
                                ],
                                "threshold": {
                                    "line":  {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": price_high,
                                },
                            },
                        ))
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)

                        # Confidence bar
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(
                            x=[price_high - price_low],
                            base=[price_low],
                            orientation="h",
                            marker_color="#93c5fd",
                            name="Confidence Range",
                        ))
                        fig2.add_trace(go.Scatter(
                            x=[price], y=[0],
                            mode="markers",
                            marker=dict(size=16, color="#1d4ed8", symbol="diamond"),
                            name="Predicted Price",
                        ))
                        fig2.update_layout(
                            title="Prediction Confidence Range",
                            xaxis_title="Price (USD)",
                            xaxis_tickformat="$,.0f",
                            height=180,
                            showlegend=True,
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                    elif resp.status_code == 422:
                        errors = resp.json().get("detail", [])
                        st.error("❌ Validation error:")
                        for e in errors:
                            st.write(f"• {e.get('loc', '')} – {e.get('msg', '')}")
                    else:
                        st.error(f"❌ API error {resp.status_code}: {resp.text[:300]}")

                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to backend at {API_BASE}.")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("Explore the California Housing dataset used to train the model.")

    df = load_dataset()
    if df is None:
        st.error(
            "Dataset not found. Please run `python model/train.py` first to generate `data/house_data.csv`."
        )
    else:
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Records",  f"{len(df):,}")
        k2.metric("Features",       len(df.columns) - 1)
        k3.metric("Avg Price",      f"${df['Price'].mean():,.0f}")
        k4.metric("Median Price",   f"${df['Price'].median():,.0f}")

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Distributions", "🗺️ Geographic", "🔗 Correlations", "📋 Raw Data"]
        )

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df, x="Price", nbins=60,
                    title="House Price Distribution",
                    labels={"Price": "Price (USD)"},
                    color_discrete_sequence=["#3b82d4"],
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                feat_sel = st.selectbox("Feature to plot", FEATURE_NAMES, key="dist_feat")
                fig2 = px.histogram(
                    df, x=feat_sel, nbins=50,
                    title=f"{feat_sel} Distribution",
                    color_discrete_sequence=["#7c5cd8"],
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Scatter
            x_feat = st.selectbox("X axis",    FEATURE_NAMES, index=0, key="sc_x")
            fig3 = px.scatter(
                df.sample(min(3000, len(df)), random_state=42),
                x=x_feat, y="Price",
                title=f"{x_feat} vs Price",
                labels={"Price": "Price (USD)"},
                opacity=0.5,
                color_discrete_sequence=["#3b82d4"],
                trendline="ols",
            )
            st.plotly_chart(fig3, use_container_width=True)

        with tab2:
            sample = df.sample(min(5000, len(df)), random_state=42)
            fig_map = px.scatter_mapbox(
                sample,
                lat="Latitude", lon="Longitude",
                color="Price",
                color_continuous_scale="Viridis",
                size_max=8,
                zoom=5,
                mapbox_style="open-street-map",
                title="California House Prices – Geographic Distribution",
                labels={"Price": "Price (USD)"},
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with tab3:
            numeric_cols = FEATURE_NAMES + ["Price"]
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Feature Correlation Matrix",
                aspect="auto",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with tab4:
            st.dataframe(df.head(500), use_container_width=True)
            st.caption(f"Showing first 500 of {len(df):,} rows.")

            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download full dataset",
                data=csv,
                file_name="house_data.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Model Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Insights":
    st.title("🤖 Model Insights")

    # Metrics
    metrics_resp = fetch_json("/metrics")
    model_info   = fetch_json("/model-info")

    if metrics_resp:
        st.subheader("📐 Model Performance Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test R²",    f"{metrics_resp['test_r2']:.4f}")
        c2.metric("Test MAE",   f"${metrics_resp['test_mae']:,.0f}")
        c3.metric("Test RMSE",  f"${metrics_resp['test_rmse']:,.0f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Train R²",   f"{metrics_resp['train_r2']:.4f}")
        c5.metric("Train MAE",  f"${metrics_resp['train_mae']:,.0f}")
        c6.metric("Train RMSE", f"${metrics_resp['train_rmse']:,.0f}")

        # Train vs Test bar chart
        fig_metrics = go.Figure(data=[
            go.Bar(name="Train", x=["R²","MAE","RMSE"],
                   y=[metrics_resp["train_r2"], metrics_resp["train_mae"]/100_000, metrics_resp["train_rmse"]/100_000],
                   marker_color="#3b82d4"),
            go.Bar(name="Test",  x=["R²","MAE","RMSE"],
                   y=[metrics_resp["test_r2"],  metrics_resp["test_mae"]/100_000,  metrics_resp["test_rmse"]/100_000],
                   marker_color="#7c5cd8"),
        ])
        fig_metrics.update_layout(
            barmode="group", title="Train vs Test Metrics (MAE & RMSE scaled to 100k USD)",
            yaxis_title="Value", height=350,
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    else:
        st.info("Metrics unavailable. Start the backend to load metrics.")

    # Feature importance
    st.markdown("---")
    st.subheader("🔍 Feature Importance")

    fi_path = os.path.join(os.path.dirname(__file__), "model", "feature_importance.csv")
    if not os.path.exists(fi_path):
        fi_path = os.path.join(os.path.dirname(__file__), "..", "model", "feature_importance.csv")

    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path).sort_values("importance", ascending=True)
        fig_fi = px.bar(
            fi_df, x="importance", y="feature",
            orientation="h",
            title="Feature Importance (GradientBoostingRegressor)",
            labels={"importance": "Importance", "feature": "Feature"},
            color="importance",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance file not found. Run `python model/train.py` first.")

    # Model info
    if model_info:
        st.markdown("---")
        st.subheader("ℹ️ Model Information")
        st.json(model_info)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – About
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Application")
    st.markdown("""
## 🏠 House Price Prediction

This application predicts **California median house prices** using a
**GradientBoostingRegressor** trained on the
[California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

---

### Technology Stack
| Layer       | Technology                          |
|-------------|-------------------------------------|
| ML Model    | scikit-learn – GradientBoostingRegressor |
| Backend API | FastAPI + Uvicorn                   |
| Frontend    | Streamlit + Plotly                  |
| Data        | California Housing (sklearn built-in) |
| Deployment  | Render (backend) + Streamlit Cloud  |

---

### Features Used
| Feature    | Description |
|------------|-------------|
| MedInc     | Median income (tens of thousands USD) |
| HouseAge   | Median house age (years) |
| AveRooms   | Average rooms per household |
| AveBedrms  | Average bedrooms per household |
| Population | Block group population |
| AveOccup   | Average household occupancy |
| Latitude   | Block group latitude |
| Longitude  | Block group longitude |

---

### API Endpoints
| Method | Endpoint       | Description            |
|--------|---------------|------------------------|
| GET    | /             | API root               |
| GET    | /health       | Health check           |
| GET    | /feature-info | Feature metadata       |
| GET    | /metrics      | Model performance      |
| GET    | /model-info   | Model details          |
| POST   | /predict      | Predict house price    |

---

### Local Development
```bash
pip install -r requirements.txt
python model/train.py
uvicorn backend.main:app --reload --port 8000
streamlit run streamlit_app.py
```
    """)
