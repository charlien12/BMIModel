# app.py

# -------------------------------
# Import libraries
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="Weight Prediction", page_icon="⚖️", layout="centered")
st.title("⚖️ Weight Prediction using Linear Regression")
st.write("Predict **Weight (kg)** from **Age** and **Height** using Linear Regression.")


data = pd.read_csv("age_height_weight_data.csv")

# Show data
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# Split data
# -------------------------------
X = data[["Age", "Height"]]
y = data["Weight"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train Linear Regression
# -------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_score = r2_score(y_test, lr_preds)

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔢 Enter Age and Height for Prediction")
age = st.slider("Select Age", 10, 80, 25)
height = st.slider("Select Height (cm)", 120, 200, 170)
input_df = pd.DataFrame([[age, height]], columns=["Age", "Height"])

# Prediction
lr_pred = lr_model.predict(input_df)[0]
st.success(f"Linear Regression Predicted Weight: **{lr_pred:.2f} kg**")

# -------------------------------
# Show model performance
# -------------------------------
st.subheader("📏 Model Accuracy (R² Score)")
st.write(f"**Linear Regression:** {lr_score:.2f}")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📈 Visualization")

fig, ax = plt.subplots()
ax.scatter(data["Height"], data["Weight"], c=data["Age"], cmap="viridis", alpha=0.8)
ax.scatter(height, lr_pred, color="red", s=100, label="Predicted Point")
plt.colorbar(ax.collections[0], label="Age")
ax.set_xlabel("Height (cm)")
ax.set_ylabel("Weight (kg)")
ax.legend()
st.pyplot(fig)
