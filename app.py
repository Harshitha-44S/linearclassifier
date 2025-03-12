import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('data.csv')

# Train a simple classification model
X = df[['feature1', 'feature2']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Simple Classification App")

# Sidebar for user input
st.sidebar.header("Input Features")
feature1 = st.sidebar.slider("Feature 1", float(df['feature1'].min()), float(df['feature1'].max()), float(df['feature1'].mean()))
feature2 = st.sidebar.slider("Feature 2", float(df['feature2'].min()), float(df['feature2'].max()), float(df['feature2'].mean()))

# Prediction
input_features = np.array([[feature1, feature2]])
prediction = model.predict(input_features)
probability = model.predict_proba(input_features)

st.write(f"### Prediction: {'Class 1 ✅' if prediction[0] == 1 else 'Class 0 ❌'}")
st.write(f"Probability of Class 0: {probability[0][0]:.2f}")
st.write(f"Probability of Class 1: {probability[0][1]:.2f}")

# Visualization
if st.checkbox("Show Data Visualization"):
    fig, ax = plt.subplots()
    sns.scatterplot(x='feature1', y='feature2', hue='target', data=df, ax=ax)
    st.pyplot(fig)
