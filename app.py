"""
Streamlit app for the house price Linear Regression model.

Run locally:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from house_price_model import NEW_HOUSE, predict_price, train_model

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def get_model() -> dict:
    """Train once per server process and reuse across sessions."""
    return train_model(verbose=False)


result = get_model()
model, scaler, df = result["model"], result["scaler"], result["df"]
feature_columns, metrics = result["feature_columns"], result["metrics"]

st.title("🏠 House Price Predictor")
st.caption(
    "Linear Regression trained on historical sales. "
    f"{result['n_train']:,} train / {result['n_test']:,} test houses after outlier removal."
)

# ---------------------------------------------------------------- inputs
st.sidebar.header("House features")
house = {
    "bedrooms": st.sidebar.slider("Bedrooms", 1, 8, NEW_HOUSE["bedrooms"]),
    "bathrooms": st.sidebar.slider("Bathrooms", 1, 6, NEW_HOUSE["bathrooms"]),
    "floors": st.sidebar.slider("Floors", 1, 3, NEW_HOUSE["floors"]),
    "sqft_above": st.sidebar.number_input(
        "Living space above ground (sqft)", 300, 8000, NEW_HOUSE["sqft_above"], step=50),
    "sqft_basement": st.sidebar.number_input(
        "Basement (sqft)", 0, 3000, NEW_HOUSE["sqft_basement"], step=50),
    "sqft_lot": st.sidebar.number_input(
        "Lot size (sqft)", 500, 200000, NEW_HOUSE["sqft_lot"], step=500),
    "view": st.sidebar.slider("View (0 = worst, 4 = best)", 0, 4, NEW_HOUSE["view"]),
    "condition": st.sidebar.slider("Condition (1 = worst, 5 = best)", 1, 5, NEW_HOUSE["condition"]),
    "yr_built": st.sidebar.slider("Year built", 1900, 2015, NEW_HOUSE["yr_built"]),
    "waterfront": int(st.sidebar.checkbox("Waterfront", value=bool(NEW_HOUSE["waterfront"]))),
}
# Total living space is defined as above-ground + basement.
house["sqft_living"] = house["sqft_above"] + house["sqft_basement"]
st.sidebar.markdown(f"**Total living space:** {house['sqft_living']:,} sqft")

# ---------------------------------------------------------------- prediction
predicted = predict_price(model, scaler, feature_columns, house)
mae = metrics["Test"]["MAE ($)"]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Predicted price")
    st.metric("Estimate", f"${predicted:,.0f}")
    st.write(
        f"Typical error on unseen houses is about **${mae:,.0f}**, so treat this as a "
        f"range of roughly **${max(predicted - mae, 0):,.0f} – ${predicted + mae:,.0f}**."
    )
    if house["waterfront"]:
        st.info("Waterfront houses were all removed as outliers during training, "
                "so the model cannot price the waterfront premium.")
    if predicted <= 0:
        st.warning("The model predicts a non-positive price: these features are far "
                   "outside the range of the training data.")

with col2:
    st.subheader("Similar houses in the data")
    similar = df[
        (df["bedrooms"] == house["bedrooms"])
        & df["sqft_living"].between(house["sqft_living"] - 200, house["sqft_living"] + 200)
        & df["yr_built"].between(house["yr_built"] - 15, house["yr_built"] + 15)
    ]["price"]
    if similar.empty:
        st.write("No similar houses found (same bedrooms, ±200 sqft, ±15 years).")
    else:
        q1, median, q3 = similar.quantile([0.25, 0.5, 0.75])
        st.metric(f"Median of {len(similar)} similar houses", f"${median:,.0f}",
                  delta=f"{(predicted - median) / median:+.1%} prediction vs. median",
                  delta_color="off")
        st.write(f"Middle 50%: **${q1:,.0f} – ${q3:,.0f}**")
        if q1 <= predicted <= q3:
            st.success("The prediction falls within the typical range → looks reasonable.")
        else:
            st.warning("The prediction is outside the typical range of similar houses.")

# ---------------------------------------------------------------- evaluation
st.divider()
st.subheader("Model performance")
st.write("Model quality is judged on the **test set**; train is shown to check for overfitting.")

rows = []
for name in metrics["Test"]:
    tr, te = metrics["Train"][name], metrics["Test"][name]
    fmt = "{:.4f}" if "R^2" in name else ("{:.2f}" if "%" in name else "{:,.0f}")
    rows.append({"Metric": name, "Train": fmt.format(tr), "Test": fmt.format(te)})
st.table(pd.DataFrame(rows).set_index("Metric"))

gap = metrics["Train"]["R^2"] - metrics["Test"]["R^2"]
st.write(
    f"- The model explains **{metrics['Test']['R^2']:.1%}** of price variance on unseen houses.\n"
    f"- Train/test R² gap is **{gap:.3f}**: "
    f"{'no significant overfitting' if gap < 0.05 else 'possible overfitting'}."
)

with st.expander("Model coefficients (effect of +1 standard deviation)"):
    coefs = pd.Series(model.coef_, index=feature_columns).sort_values(key=abs, ascending=False)
    st.table(coefs.map(lambda v: f"${v:+,.0f}").rename("Effect on price"))
