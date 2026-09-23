
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Nassau_Candy_Cleaned.csv"
MODEL_PATH = ROOT / "models" / "best_model_random_forest.pkl"
IMG_DIR = ROOT / "dashboard_screenshots"

st.set_page_config(
    page_title="Nassau Candy | Factory Optimization",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
:root {
  --navy: #0b2f59;
  --blue: #0f56a8;
  --purple: #3421df;
  --ink: #152238;
  --muted: #6b7280;
  --card: #ffffff;
  --bg: #f4f7fb;
}
.stApp { background: var(--bg); }
.block-container { padding-top: 1.2rem; max-width: 1450px; }
.hero {
  background: linear-gradient(120deg,#092d55 0%,#0d4e92 55%,#3421df 100%);
  padding: 2.3rem 2.5rem; border-radius: 24px; color: white;
  box-shadow: 0 16px 40px rgba(11,47,89,.18);
}
.hero h1 { font-size: 2.5rem; margin-bottom: .35rem; }
.hero p { font-size: 1.08rem; opacity: .92; max-width: 900px; }
.badge {
  display:inline-block; padding:.32rem .72rem; border-radius:999px;
  background:rgba(255,255,255,.15); margin-right:.4rem; font-size:.82rem;
}
.card {
  background:white; border-radius:18px; padding:1.1rem 1.25rem;
  box-shadow:0 8px 25px rgba(15,40,70,.07); border:1px solid #e7edf5;
}
.kpi {
  background:white; border-radius:18px; padding:1rem 1.1rem;
  border-left:5px solid #3421df; box-shadow:0 7px 22px rgba(15,40,70,.06);
}
.kpi .label { color:#6b7280; font-size:.85rem; }
.kpi .value { color:#13294b; font-size:1.65rem; font-weight:800; margin-top:.2rem; }
.section-title { color:#0b2f59; font-weight:800; margin-top:.4rem; }
.small { color:#667085; font-size:.9rem; }
div[data-testid="stMetric"] {
  background:white; border:1px solid #e6ebf2; border-radius:15px;
  padding:12px 15px; box-shadow:0 5px 18px rgba(15,40,70,.05);
}
</style>
""", unsafe_allow_html=True)

# ---------- Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    for c in ["Order Date", "Ship Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    if "Order Date" in df.columns:
        df["Order Year"] = df["Order Date"].dt.year
        df["Order Month"] = df["Order Date"].dt.month
        df["Order Quarter"] = df["Order Date"].dt.quarter
        df["Order Day of Week"] = df["Order Date"].dt.dayofweek
    return df

df = load_data()

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, "Model file not found."
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

FACTORIES = ["Lot's O' Nuts", "Wicked Choccy's", "Sugar Shack", "Secret Factory", "The Other Factory"]
MODEL_FEATURES = [
    "Product ID","Product Name","Division","Region","State/Province",
    "Ship Mode","Factory","Sales","Units","Cost","Gross Profit",
    "Profit Margin","Cost Percentage","Unit Price","Order Year",
    "Order Month","Order Quarter","Order Day of Week"
]

def simulate_factories(row):
    if model is None:
        return pd.DataFrame()
    records = []
    for factory in FACTORIES:
        x = pd.DataFrame([{
            "Product ID": row["Product ID"],
            "Product Name": row["Product Name"],
            "Division": row["Division"],
            "Region": row["Region"],
            "State/Province": row["State/Province"],
            "Ship Mode": row["Ship Mode"],
            "Factory": factory,
            "Sales": row["Sales"],
            "Units": row["Units"],
            "Cost": row["Cost"],
            "Gross Profit": row["Gross Profit"],
            "Profit Margin": row["Profit Margin"],
            "Cost Percentage": row["Cost Percentage"],
            "Unit Price": row["Unit Price"],
            "Order Year": row["Order Year"],
            "Order Month": row["Order Month"],
            "Order Quarter": row["Order Quarter"],
            "Order Day of Week": row["Order Day of Week"],
        }])[MODEL_FEATURES]
        pred = float(model.predict(x)[0])
        records.append({"Factory": factory, "Predicted Lead Time (days)": pred})
    out = pd.DataFrame(records).sort_values("Predicted Lead Time (days)")
    return out

# ---------- Sidebar ----------
st.sidebar.markdown("## 🍫 Nassau Candy")
st.sidebar.caption("Factory Reallocation & Shipping Optimization")
page = st.sidebar.radio(
    "Portfolio",
    [
        "🏠 Overview",
        "📊 Executive Dashboard",
        "🧪 EDA & Business Insights",
        "🤖 ML Model",
        "🎯 Live Factory Simulator",
        "💡 Recommendations",
        "🖼️ Dashboard Gallery",
        "📦 Project Files",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Built from the supplied Nassau Candy dataset, notebooks, model, EDA report and Power BI dashboard.")

# ---------- Overview ----------
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero">
      <span class="badge">Decision Intelligence</span>
      <span class="badge">Machine Learning</span>
      <span class="badge">Power BI</span>
      <span class="badge">Optimization</span>
      <h1>Nassau Candy Factory Optimization</h1>
      <p>Factory Reallocation & Shipping Optimization Recommendation System — a portfolio-ready analytics application that combines EDA, predictive lead-time modeling, five-factory what-if simulation, recommendations and Power BI reporting.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Unique Orders", f"{df['Order ID'].nunique():,}")
    c3.metric("Products", f"{df['Product Name'].nunique():,}")
    c4.metric("Factories", f"{df['Factory'].nunique():,}")

    st.markdown("### 🎯 Business question")
    st.markdown("""
    **What should be changed to improve shipping performance while protecting profitability?**

    The project moves beyond descriptive reporting by predicting shipping lead time under alternative
    factory assignments and identifying reassignment opportunities.
    """)

    a,b,c = st.columns(3)
    with a:
        st.markdown('<div class="card"><h3>1. Understand</h3><p>Clean the transaction data, engineer lead-time and profitability features, and identify regional, product and factory patterns.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><h3>2. Predict</h3><p>Use the trained Random Forest pipeline to estimate lead time for alternative factory scenarios.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="card"><h3>3. Recommend</h3><p>Compare current vs. best predicted factory and apply the project recommendation rule to prioritize opportunities.</p></div>', unsafe_allow_html=True)

    st.markdown("### 📌 Project results from the supplied EDA report")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Orders evaluated", "8,549")
    k2.metric("Orders marked for reassignment", "4,163")
    k3.metric("Product coverage", "53.33%")
    k4.metric("Predicted average reduction", "7.11 days")

    st.info("Important: the supplied EDA report states that the overall predicted reduction is 0.48%, so recommendation volume should not be interpreted as the same thing as service improvement.")

# ---------- Executive Dashboard ----------
elif page == "📊 Executive Dashboard":
    st.markdown("## Executive Dashboard")
    st.caption("Interactive portfolio version of the Executive Summary Power BI page.")

    total_sales = df["Sales"].sum()
    total_profit = df["Gross Profit"].sum()
    avg_lead = df["Lead_Time"].mean()
    orders = df["Order ID"].nunique()
    margin = total_profit / total_sales if total_sales else np.nan

    a,b,c,d,e = st.columns(5)
    a.metric("Total Sales", f"${total_sales:,.2f}")
    b.metric("Gross Profit", f"${total_profit:,.2f}")
    c.metric("Avg Lead Time", f"{avg_lead:,.1f} days")
    d.metric("Total Orders", f"{orders:,}")
    e.metric("Avg Profit Margin", f"{margin:.2%}")

    left,right = st.columns(2)
    with left:
        trend = df.groupby("Order Year", dropna=True).agg(Sales=("Sales","sum"), Gross_Profit=("Gross Profit","sum")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=trend["Order Year"], y=trend["Sales"], name="Sales")
        fig.add_scatter(x=trend["Order Year"], y=trend["Gross_Profit"], name="Gross Profit", yaxis="y2", mode="lines+markers")
        fig.update_layout(title="Sales & Profit Trend", yaxis_title="Sales", yaxis2=dict(title="Gross Profit", overlaying="y", side="right"), height=390)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        reg = df.groupby("Region").agg(Avg_Lead_Time=("Lead_Time","mean")).reset_index().sort_values("Avg_Lead_Time")
        fig = px.bar(reg, x="Avg_Lead_Time", y="Region", orientation="h", title="Lead Time by Region")
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

    left,right = st.columns(2)
    with left:
        fac = df.groupby("Factory").agg(Avg_Lead_Time=("Lead_Time","mean")).reset_index().sort_values("Avg_Lead_Time")
        fig = px.bar(fac, x="Avg_Lead_Time", y="Factory", orientation="h", title="Factory Performance")
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        sm = df.groupby("Ship Mode").agg(Avg_Lead_Time=("Lead_Time","mean")).reset_index()
        fig = px.bar(sm, x="Ship Mode", y="Avg_Lead_Time", title="Shipping Mode Performance")
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

# ---------- EDA ----------
elif page == "🧪 EDA & Business Insights":
    st.markdown("## EDA & Business Insights")
    st.caption("Charts are regenerated from the supplied cleaned dataset; headline findings follow the supplied EDA report.")

    a,b,c = st.columns(3)
    a.metric("Observed Lead Time", f"{df['Lead_Time'].min():.0f}–{df['Lead_Time'].max():.0f} days")
    b.metric("Median Lead Time", f"{df['Lead_Time'].median():.0f} days")
    c.metric("Mean Profit Margin", f"{df['Profit Margin'].mean():.1%}")

    left,right = st.columns(2)
    with left:
        fig = px.histogram(df, x="Lead_Time", nbins=45, title="Shipping Lead Time Distribution")
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        reg = df.groupby("Region").agg(Sales=("Sales","sum"), Profit=("Gross Profit","sum"), Avg_Lead=("Lead_Time","mean")).reset_index()
        fig = px.bar(reg, x="Region", y=["Sales","Profit"], barmode="group", title="Regional Sales & Profit")
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

    prod = df.groupby("Product Name").agg(Orders=("Order ID","nunique"), Sales=("Sales","sum"), Profit=("Gross Profit","sum"), Avg_Lead=("Lead_Time","mean")).reset_index().sort_values("Sales", ascending=False)
    st.plotly_chart(px.bar(prod.head(15), x="Sales", y="Product Name", orientation="h", title="Top Products by Sales"), use_container_width=True)

    st.markdown("### Business observations")
    observations = [
        "Pacific has the highest sales and profit in the supplied EDA aggregation, while Gulf is lowest on both.",
        "The five leading Wonka Bar products account for most of the product-level sales shown in the EDA.",
        "Current factory assignments are highly concentrated in Lot's O' Nuts and Wicked Choccy's.",
        "The EDA report flags date parsing and future shipment dates as material data-quality issues that should be resolved before production use.",
    ]
    for x in observations:
        st.markdown(f"- {x}")

# ---------- Model ----------
elif page == "🤖 ML Model":
    st.markdown("## Machine Learning Model")
    st.caption("Model details are based on the supplied Model Training and Recommendation Engine notebooks.")

    a,b,c = st.columns(3)
    a.metric("Model", "Random Forest")
    b.metric("MAE", "138.59 days")
    c.metric("R²", "0.687")

    comparison = pd.DataFrame({
        "Model":["Linear Regression","Gradient Boosting","Random Forest"],
        "MAE":[183.32,168.66,138.59],
        "RMSE":[211.91,195.80,172.15],
        "R²":[0.53,0.60,0.687]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("### Features used by the recommendation workflow")
    st.code(", ".join(MODEL_FEATURES), language="text")

    if model_error:
        st.warning("The included model could not be loaded in this environment. The deployment requirements pin the scikit-learn version used by the supplied model.")
        st.caption(model_error)
    else:
        st.success("The supplied model loaded successfully.")

    st.markdown("### Production note")
    st.info("The supplied EDA report recommends re-validating the model after date-quality correction because the parsed lead-time target has known data-quality issues.")

# ---------- Live Simulator ----------
elif page == "🎯 Live Factory Simulator":
    st.markdown("## Live Factory Optimization Simulator")
    st.caption("Select a product and operating context, then compare predicted lead time across all five candidate factories.")

    if model is None:
        st.error("The supplied model could not be loaded. Deploy this project with the pinned dependencies in requirements.txt.")
        st.stop()

    p = st.selectbox("Product", sorted(df["Product Name"].dropna().unique()))
    region = st.selectbox("Region", sorted(df["Region"].dropna().unique()))
    ship = st.selectbox("Ship Mode", sorted(df["Ship Mode"].dropna().unique()))

    candidates = df[(df["Product Name"]==p) & (df["Region"]==region) & (df["Ship Mode"]==ship)]
    if candidates.empty:
        candidates = df[df["Product Name"]==p]
    row = candidates.iloc[0].copy()

    result = simulate_factories(row)
    current_factory = row["Factory"]
    current_pred = float(result.loc[result["Factory"]==current_factory, "Predicted Lead Time (days)"].iloc[0])
    best = result.iloc[0]
    reduction = current_pred - float(best["Predicted Lead Time (days)"])
    reduction_pct = reduction/current_pred*100 if current_pred else 0

    a,b,c,d = st.columns(4)
    a.metric("Current Factory", current_factory)
    b.metric("Current Predicted Lead", f"{current_pred:,.1f} days")
    c.metric("Suggested Factory", best["Factory"])
    d.metric("Predicted Reduction", f"{reduction:,.1f} days ({reduction_pct:.2f}%)")

    threshold = st.slider("Minimum improvement threshold for portfolio recommendation", 0.10, 5.00, 0.10, 0.10, format="%.2f%%")
    qualifies = best["Factory"] != current_factory and reduction_pct >= threshold

    if qualifies:
        st.success(f"Recommendation: consider reassignment from {current_factory} to {best['Factory']}.")
    else:
        st.info("Recommendation rule: keep the current factory unless the best alternative is different and clears the selected minimum improvement threshold.")

    fig = px.bar(result, x="Factory", y="Predicted Lead Time (days)", title="Factory Scenario Comparison")
    fig.add_hline(y=current_pred, line_dash="dash", annotation_text="Current factory prediction")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(result.style.format({"Predicted Lead Time (days)":"{:.2f}"}), use_container_width=True, hide_index=True)

# ---------- Recommendations ----------
elif page == "💡 Recommendations":
    st.markdown("## Recommendation Dashboard")
    st.caption("Portfolio summary of the recommendation-engine outputs reported in the supplied EDA document.")

    a,b,c,d = st.columns(4)
    a.metric("Orders Evaluated", "8,549")
    b.metric("Orders to Reassign", "4,163")
    c.metric("Orders to Keep", "4,386")
    d.metric("Reassignment Rate", "48.70%")

    a,b,c = st.columns(3)
    a.metric("Products Evaluated", "15")
    b.metric("Products Recommended", "8")
    c.metric("Recommendation Coverage", "53.33%")

    rec = pd.DataFrame({
        "Recommended Factory":["Wicked Choccy's","Lot's O' Nuts","The Other Factory","Secret Factory","Sugar Shack"],
        "Reassigned Orders":[1602,1213,503,465,380]
    })
    st.plotly_chart(px.bar(rec, x="Reassigned Orders", y="Recommended Factory", orientation="h", title="Recommended Factory Distribution"), use_container_width=True)

    st.markdown("### Recommendation interpretation")
    st.markdown("""
    The recommendation engine changes factory assignment only when the best factory differs from the
    current factory and the predicted lead-time improvement reaches the notebook's minimum threshold.
    The supplied workflow uses **0.10%** as that threshold.
    """)
    st.warning("The EDA report explicitly notes that 48.70% reassignment rate is recommendation volume, not a 48.70% service improvement. The overall predicted reduction is 0.48%.")

    top_products = df.groupby("Product Name").agg(Sales=("Sales","sum"), Profit=("Gross Profit","sum"), Avg_Lead=("Lead_Time","mean")).reset_index().sort_values("Sales", ascending=False).head(10)
    st.dataframe(top_products, use_container_width=True, hide_index=True)

# ---------- Gallery ----------
elif page == "🖼️ Dashboard Gallery":
    st.markdown("## Power BI Dashboard Gallery")
    st.caption("The original dashboard pages supplied with the project are shown below as portfolio visuals.")

    gallery = [
        ("01_Executive_Summary.png","Executive Summary"),
        ("02_Factory_Optimization_Simulator.png","Factory Optimization Simulator"),
        ("03_Recommendation_Dashboard.png","Recommendation Dashboard"),
        ("04_Risk_Profit_Impact.png","Risk & Profit Impact"),
        ("05_Shipping_Route_Analysis.png","Shipping & Route Analysis"),
        ("00_Project_Cover.png","Project Cover"),
    ]
    for filename, title in gallery:
        st.markdown(f"### {title}")
        st.image(str(IMG_DIR / filename), use_container_width=True)

# ---------- Files ----------
elif page == "📦 Project Files":
    st.markdown("## Project Files")
    st.write("This portfolio package includes the supplied analytical assets so the project can be reviewed, reproduced and submitted.")

    files = [
        ("Cleaned dataset","data/Nassau_Candy_Cleaned.csv"),
        ("Random Forest model","models/best_model_random_forest.pkl"),
        ("EDA report","reports/EDA_Report.docx"),
        ("Power BI dashboard","PowerBI_Nassau_Candy.pbix"),
        ("Data preprocessing notebook","notebooks/Data_Preprocessing_Feature_Engineering.ipynb"),
        ("EDA notebook","notebooks/EDA.ipynb"),
        ("Factory optimization notebook","notebooks/Factory_Optimization.ipynb"),
        ("Model training notebook","notebooks/Model_Training.ipynb"),
        ("Recommendation engine notebook","notebooks/Recommendation_Engine.ipynb"),
    ]
    for label, rel in files:
        pth = ROOT / rel
        if pth.exists():
            with open(pth, "rb") as f:
                st.download_button(
                    f"Download {label}",
                    f,
                    file_name=pth.name,
                    key=rel,
                )

    st.markdown("### Deployment")
    st.code("streamlit run app.py", language="bash")
    st.markdown("For cloud deployment, upload this folder to GitHub and create a Streamlit Community Cloud app using `app.py` as the entry point.")
