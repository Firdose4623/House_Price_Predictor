import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Boston House Price Predictor", layout="wide")

st.title("🏡 Boston House Price Prediction App")
st.markdown("Upload your dataset, explore the data, and train regression models to predict house prices.")

# Upload Data 
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with features and 'MEDV' as the target", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

if 'MEDV' not in df.columns:
    st.error("The uploaded file must contain a 'MEDV' column as the target.")
    st.stop()

# Navigation
section = st.radio("**Select a section:**", ["📊 Exploratory Data Analysis (EDA)", "📦 Train & Predict"], horizontal=True)

# SECTION 1: EDA
if section == "📊 Exploratory Data Analysis (EDA)":
    st.subheader(" Exploratory Data Analysis")

    with st.expander("📋 Dataset Overview"):
        st.dataframe(df.head())
        st.markdown(f"**Shape**: {df.shape[0]} rows, {df.shape[1]} columns")
        st.markdown(f"**Columns**: {', '.join(df.columns)}")

    with st.expander("📊 Summary Statistics"):
        st.dataframe(df.describe().T)

    with st.expander("📈 Correlation Heatmap"):
        corr = df.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax_corr.set_xticks(np.arange(len(corr.columns)))
        ax_corr.set_yticks(np.arange(len(corr.columns)))
        ax_corr.set_xticklabels(corr.columns, rotation=90)
        ax_corr.set_yticklabels(corr.columns)
        fig_corr.colorbar(im, ax=ax_corr)
        st.pyplot(fig_corr)

    with st.expander("🎯 Target Variable Distribution (MEDV)"):
        fig_medv, ax_medv = plt.subplots()
        ax_medv.hist(df['MEDV'], bins=30, color='skyblue', edgecolor='black')
        ax_medv.set_title("Distribution of MEDV")
        ax_medv.set_xlabel("MEDV")
        ax_medv.set_ylabel("Frequency")
        st.pyplot(fig_medv)

    with st.expander("📌 Feature vs MEDV Scatter Plots"):
        selected_features = st.multiselect(
            "Select features to visualize against MEDV:", 
            [col for col in df.columns if col != "MEDV"]
        )
        for feature in selected_features:
            fig_feat, ax_feat = plt.subplots()
            ax_feat.scatter(df[feature], df["MEDV"], alpha=0.5, color='green')
            ax_feat.set_xlabel(feature)
            ax_feat.set_ylabel("MEDV")
            ax_feat.set_title(f"{feature} vs MEDV")
            st.pyplot(fig_feat)

# SECTION 2: TRAIN & PREDICT 
elif section == "📦 Train & Predict":
    st.subheader(" Model Training and Prediction")
    with st.expander("⚠️ Missing Values Report"):
        st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])

    # Split X and y 
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]

    # Impute Missing Values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Train/Test Split 
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Feature Scaling 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Selection 
    st.sidebar.header("Model Settings")
    model_name = st.sidebar.selectbox("Choose a regression model:", ["Linear Regression", "Ridge Regression", "Lasso Regression"])

    if model_name == "Ridge Regression":
        alpha = st.sidebar.slider("Ridge Alpha", 0.01, 100.0, 10.0)
        model = Ridge(alpha=alpha)
    elif model_name == "Lasso Regression":
        alpha = st.sidebar.slider("Lasso Alpha", 0.01, 100.0, 1.0)
        model = Lasso(alpha=alpha, max_iter=10000)
    else:
        model = LinearRegression()

    # Train Model 
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Evaluation Metrics 
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * ((len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))

    with st.expander("📈 Evaluation Metrics"):
        st.markdown(f"**MAE**: {mae:.2f}")
        st.markdown(f"**RMSE**: {rmse:.2f}")
        st.markdown(f"**R² Score**: {r2:.3f}")
        st.markdown(f"**Adjusted R²**: {adj_r2:.3f}")

    # Regression Equation
    if model_name == "Linear Regression":
        st.markdown("### Regression Equation")
        terms = [f"{coef:.2f}×{name}" for coef, name in zip(model.coef_, X.columns)]
        equation = " + ".join(terms)
        st.markdown(f"**MEDV = {model.intercept_:.2f} + {equation}**")

    # Predict New Values 
    st.sidebar.header("Predict House Price")
    input_data = {}
    feature_names = X.columns
    for feature in feature_names:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())
        input_data[feature] = st.sidebar.slider(
            feature, min_value=min_val, max_value=max_val, value=mean_val,
            step=(max_val - min_val) / 100
        )

    input_df = pd.DataFrame([input_data])
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    predicted_price = model.predict(input_scaled)[0]

    st.sidebar.markdown(f"### 🏷️ Predicted Price: **${predicted_price * 1000:.2f}**")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Actual vs Predicted")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.scatter(y_test, y_pred, alpha=0.7, color='teal')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax1.set_xlabel("Actual MEDV")
        ax1.set_ylabel("Predicted MEDV")
        ax1.set_title("Actual vs Predicted")
        ax1.grid(True)
        st.pyplot(fig1)

    with col2:
        st.subheader("📉 Residuals Plot")
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(y_pred, residuals, alpha=0.7, color='orange')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel("Predicted MEDV")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs Predicted")
        ax2.grid(True)
        st.pyplot(fig2)

    # Residual Distribution
    with st.expander("Residual Distribution"):
        fig_res, ax_res = plt.subplots()
        ax_res.hist(residuals, bins=30, color='coral', edgecolor='black')
        ax_res.set_title("Histogram of Residuals")
        ax_res.set_xlabel("Residual")
        ax_res.set_ylabel("Frequency")
        st.pyplot(fig_res)

    # Feature Importance
    with st.expander("Feature Importance"):
        try:
            coefs = pd.Series(model.coef_, index=feature_names).sort_values()
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            coefs.plot(kind='barh', ax=ax3, color='purple')
            ax3.set_title("Feature Coefficients")
            ax3.axvline(x=0, color='black', linestyle='--')
            st.pyplot(fig3)
        except:
            st.warning("This model doesn't provide interpretable coefficients.")

    # Download Predictions
    results_df = pd.DataFrame({"Actual MEDV": y_test, "Predicted MEDV": y_pred})
    csv = results_df.to_csv(index=False)
    st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")

