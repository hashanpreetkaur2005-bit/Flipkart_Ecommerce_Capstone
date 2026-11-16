import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ------------------------------------------------------------
# Flipkart E-Commerce Decision Intelligence
# Based on the models and feature sets in Flipkart_Capstone.ipynb
# ------------------------------------------------------------

st.set_page_config(
    page_title="Flipkart E-Commerce Analytics",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Flipkart E-Commerce Decision Intelligence")
st.caption(
    "Customer behaviour analysis | Average Order Value estimation | "
    "Churn prediction | Customer segmentation"
)


def find_dataset():
    """Find the project dataset in the common folder locations."""
    root = Path(__file__).resolve().parent
    candidates = [
        root / "Dataset" / "flipkart_customers.csv",
        root / "Dataset" / "flipkart_customers.xlsx",
        root / "Dataset" / "flipkart_customers.xls",
        root / "Models" / "flipkart_final_dataset.csv",
        root / "Models" / "flipkart_final_dataset.xlsx",
        root / "Models" / "flipkart_final_dataset.xls",
        root / "data" / "flipkart_customers.csv",
        root / "models" / "flipkart_final_dataset.csv",
        root / "flipkart_customers.csv",
        root / "flipkart_customers.xlsx",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Search one level down if the folder/file capitalization differs.
    for folder in [root / "Dataset", root / "dataset", root / "Models",
                   root / "models", root / "data"]:
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file() and path.stem.lower() in {
                    "flipkart_customers", "flipkart_final_dataset"
                } and path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
                    return path
    return None


@st.cache_data
def load_data(file_path):
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    else:
        data = pd.read_excel(path)

    data.columns = data.columns.astype(str).str.strip()
    return data


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def prepare_aov_features(data):
    feature_names = [
        "Age", "Gender", "City_Tier", "Membership",
        "Tenure_Months", "Monthly_Visits",
        "Orders_Last_6_Months", "Discounts_Used"
    ]
    X = pd.get_dummies(data[feature_names], drop_first=True, dtype=float)
    return X, feature_names


def train_aov_model(data):
    X, feature_names = prepare_aov_features(data)
    y = data["Average_Order_Value"].astype(float).to_numpy()

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = X.iloc[train_idx].to_numpy(dtype=float)
    X_test = X.iloc[test_idx].to_numpy(dtype=float)
    y_train, y_test = y[train_idx], y[test_idx]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1

    X_train_scaled = (X_train - x_mean) / x_std
    X_test_scaled = (X_test - x_mean) / x_std

    y_mean = y_train.mean()
    y_std = y_train.std() or 1
    y_train_scaled = (y_train - y_mean) / y_std

    X_train_b = np.c_[np.ones(len(X_train_scaled)), X_train_scaled]
    X_test_b = np.c_[np.ones(len(X_test_scaled)), X_test_scaled]

    weights = np.zeros(X_train_b.shape[1])
    learning_rate = 0.01
    for _ in range(3000):
        errors = X_train_b @ weights - y_train_scaled
        gradients = (2 / len(y_train_scaled)) * (X_train_b.T @ errors)
        weights -= learning_rate * gradients

    predictions = (X_test_b @ weights) * y_std + y_mean
    mae = float(np.mean(np.abs(y_test - predictions)))
    rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))
    ss_total = np.sum((y_test - y_test.mean()) ** 2)
    r2 = float(1 - np.sum((y_test - predictions) ** 2) / ss_total) if ss_total else 0.0

    return {
        "weights": weights,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "columns": list(X.columns),
        "feature_names": feature_names,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "actual": y_test,
        "predicted": predictions,
    }


def predict_aov(model, customer):
    row = pd.DataFrame([customer])
    encoded = pd.get_dummies(row, drop_first=True, dtype=float)
    encoded = encoded.reindex(columns=model["columns"], fill_value=0)
    values = encoded.to_numpy(dtype=float)
    scaled = (values - model["x_mean"]) / model["x_std"]
    with_intercept = np.c_[np.ones(len(scaled)), scaled]
    return float((with_intercept @ model["weights"])[0] * model["y_std"] + model["y_mean"])


CHURN_FEATURES = [
    "Age", "Tenure_Months", "Monthly_Visits",
    "Orders_Last_6_Months", "Support_Tickets",
    "Late_Deliveries", "Returns", "Discounts_Used"
]


def train_churn_model(data):
    X = data[CHURN_FEATURES].to_numpy(dtype=float)
    y = data["Churn"].to_numpy(dtype=int)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train_b = np.c_[np.ones(len(X_train)), X_train]
    X_test_b = np.c_[np.ones(len(X_test)), X_test]

    weights = np.zeros(X_train_b.shape[1])
    for _ in range(3000):
        probabilities = sigmoid(X_train_b @ weights)
        gradient = (X_train_b.T @ (probabilities - y_train)) / len(y_train)
        weights -= 0.1 * gradient

    probabilities = sigmoid(X_test_b @ weights)
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = float(np.mean(predictions == y_test))
    tp = int(np.sum((y_test == 1) & (predictions == 1)))
    fp = int(np.sum((y_test == 0) & (predictions == 1)))
    fn = int(np.sum((y_test == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "weights": weights, "mean": mean, "std": std,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "actual": y_test, "predicted": predictions
    }


def predict_churn(model, customer):
    values = np.array([[customer[name] for name in CHURN_FEATURES]], dtype=float)
    scaled = (values - model["mean"]) / model["std"]
    values_b = np.c_[np.ones(len(scaled)), scaled]
    probability = float(sigmoid(values_b @ model["weights"])[0])
    return probability, int(probability >= 0.5)


def run_kmeans(values, k=3, max_iters=100, seed=42):
    rng = np.random.default_rng(seed)
    centers = values[rng.choice(len(values), k, replace=False)].copy()

    for _ in range(max_iters):
        distances = np.sqrt(((values[:, None, :] - centers) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([
            values[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(k)
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return labels


# --------------------------- Load data ---------------------------

dataset_path = find_dataset()

if dataset_path is None:
    st.error("Dataset file not found.")
    st.markdown(
        "Please keep your dataset inside the project folder using one of these paths:\n"
        "- `Dataset/flipkart_customers.csv`\n"
        "- `Dataset/flipkart_customers.xlsx`\n"
        "- `Models/flipkart_final_dataset.csv`\n\n"
        "The filename should be `flipkart_customers` or `flipkart_final_dataset`."
    )
    uploaded = st.file_uploader("Or upload your customer dataset here", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    else:
        st.stop()
else:
    try:
        df = load_data(str(dataset_path))
        st.success(f"Dataset loaded: {dataset_path.relative_to(Path(__file__).resolve().parent)}")
    except Exception as error:
        st.error(f"Could not read dataset: {error}")
        st.stop()


required_columns = set(CHURN_FEATURES + [
    "Age", "Gender", "City_Tier", "Membership", "Tenure_Months",
    "Monthly_Visits", "Orders_Last_6_Months", "Discounts_Used",
    "Average_Order_Value", "Total_Spending", "Churn"
])
missing = sorted(required_columns - set(df.columns))
if missing:
    st.error("The dataset is missing these required columns: " + ", ".join(missing))
    st.stop()

df = df.drop_duplicates().copy()

try:
    aov_model = train_aov_model(df)
    churn_model = train_churn_model(df)
except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()


# --------------------------- Dashboard ---------------------------

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "AOV Estimation", "Churn Prediction", "Customer Segmentation", "Dataset"]
)

if page == "Dashboard":
    st.subheader("Business Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Average Order Value", f"₹{df['Average_Order_Value'].mean():,.0f}")
    c3.metric("Total Customer Spending", f"₹{df['Total_Spending'].sum():,.0f}")
    c4.metric("Churn Rate", f"{df['Churn'].mean() * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Customer Membership")
        membership_counts = df["Membership"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(membership_counts.index.astype(str), membership_counts.values)
        ax.set_ylabel("Customers")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### Churn Distribution")
        churn_counts = df["Churn"].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.bar(["No Churn", "Churn"], [churn_counts.get(0, 0), churn_counts.get(1, 0)])
        ax.set_ylabel("Customers")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Key Model Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("AOV Model R²", f"{aov_model['r2']:.3f}")
    m2.metric("Churn Accuracy", f"{churn_model['accuracy'] * 100:.1f}%")
    m3.metric("Churn F1 Score", f"{churn_model['f1']:.3f}")

elif page == "AOV Estimation":
    st.subheader("Average Order Value (AOV) Estimation")
    st.write("This page uses the Linear Regression approach implemented in the project notebook.")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"₹{aov_model['mae']:,.2f}")
    m2.metric("RMSE", f"₹{aov_model['rmse']:,.2f}")
    m3.metric("R² Score", f"{aov_model['r2']:.4f}")

    st.markdown("#### Estimate AOV for a customer")
    with st.form("aov_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 64, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
            membership = st.selectbox("Membership", ["Regular", "Premium", "VIP"])
        with col2:
            tenure = st.slider("Tenure (months)", 1, 60, 24)
            visits = st.slider("Monthly Visits", 1, 24, 10)
            orders = st.slider("Orders in last 6 months", 1, 19, 6)
            discounts = st.slider("Discounts Used", 0, 14, 3)
        submitted = st.form_submit_button("Estimate Average Order Value")

    if submitted:
        customer = {
            "Age": age, "Gender": gender, "City_Tier": city_tier,
            "Membership": membership, "Tenure_Months": tenure,
            "Monthly_Visits": visits, "Orders_Last_6_Months": orders,
            "Discounts_Used": discounts
        }
        estimate = predict_aov(aov_model, customer)
        st.success(f"Estimated Average Order Value: ₹{max(0, estimate):,.2f}")

    st.markdown("#### Actual vs Predicted AOV (test data)")
    fig, ax = plt.subplots()
    ax.scatter(aov_model["actual"], aov_model["predicted"], alpha=0.65)
    low = min(aov_model["actual"].min(), aov_model["predicted"].min())
    high = max(aov_model["actual"].max(), aov_model["predicted"].max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual AOV (₹)")
    ax.set_ylabel("Predicted AOV (₹)")
    st.pyplot(fig)
    plt.close(fig)

elif page == "Churn Prediction":
    st.subheader("Customer Churn Prediction")
    st.write("The Logistic Regression model is trained using the churn features from your notebook.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{churn_model['accuracy'] * 100:.1f}%")
    c2.metric("Precision", f"{churn_model['precision']:.3f}")
    c3.metric("Recall", f"{churn_model['recall']:.3f}")
    c4.metric("F1 Score", f"{churn_model['f1']:.3f}")

    st.markdown("#### Predict churn for a customer")
    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Customer Age", 18, 64, 30, key="churn_age")
            tenure = st.slider("Tenure (months)", 1, 60, 24, key="churn_tenure")
            visits = st.slider("Monthly Visits", 1, 24, 10, key="churn_visits")
            orders = st.slider("Orders in last 6 months", 1, 19, 6, key="churn_orders")
        with col2:
            tickets = st.slider("Support Tickets", 0, 7, 2)
            late = st.slider("Late Deliveries", 0, 5, 1)
            returns = st.slider("Returns", 0, 5, 1)
            discounts = st.slider("Discounts Used", 0, 14, 3, key="churn_discounts")
        submitted = st.form_submit_button("Predict Customer Churn")

    if submitted:
        customer = {
            "Age": age, "Tenure_Months": tenure, "Monthly_Visits": visits,
            "Orders_Last_6_Months": orders, "Support_Tickets": tickets,
            "Late_Deliveries": late, "Returns": returns, "Discounts_Used": discounts
        }
        probability, prediction = predict_churn(churn_model, customer)
        st.metric("Estimated churn probability", f"{probability * 100:.1f}%")
        if prediction == 1:
            st.warning("Model result: Higher churn risk (Churn = 1)")
        else:
            st.success("Model result: Lower churn risk (Churn = 0)")

elif page == "Customer Segmentation":
    st.subheader("Customer Segmentation using K-Means")
    cluster_features = [
        "Age", "Tenure_Months", "Monthly_Visits",
        "Orders_Last_6_Months", "Average_Order_Value",
        "Total_Spending", "Discounts_Used", "Returns"
    ]
    values = df[cluster_features].to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    scaled = (values - means) / stds
    df_segmented = df.copy()
    df_segmented["Customer_Segment"] = run_kmeans(scaled, k=3)

    segment_summary = df_segmented.groupby("Customer_Segment").agg(
        Customers=("Customer_ID", "count") if "Customer_ID" in df_segmented.columns else ("Churn", "size"),
        Avg_Age=("Age", "mean"),
        Avg_Monthly_Visits=("Monthly_Visits", "mean"),
        Avg_AOV=("Average_Order_Value", "mean"),
        Avg_Total_Spending=("Total_Spending", "mean"),
        Avg_Orders=("Orders_Last_6_Months", "mean")
    ).round(2)

    st.dataframe(segment_summary, use_container_width=True)
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df_segmented["Average_Order_Value"],
        df_segmented["Total_Spending"],
        c=df_segmented["Customer_Segment"],
        alpha=0.7
    )
    ax.set_xlabel("Average Order Value (₹)")
    ax.set_ylabel("Total Spending (₹)")
    ax.set_title("Flipkart Customer Segments")
    st.pyplot(fig)
    plt.close(fig)

elif page == "Dataset":
    st.subheader("Customer Dataset")
    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button(
        "Download dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="flipkart_customer_data.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Academic capstone demonstration. The notebook describes a synthetically generated "
    "dataset for educational purposes; it is not actual Flipkart customer data."
)
