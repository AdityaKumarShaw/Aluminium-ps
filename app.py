import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Aluminum Wire Rod Prediction Model")

# uploading data
st.header("Upload your training and testing data")
train_file = st.file_uploader("Upload training CSV", type="csv")
test_file = st.file_uploader("Upload testing CSV", type="csv")

if train_file and test_file:
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    #preview
    st.write("### Training Data")
    st.dataframe(train_data.head())

    features = train_data.drop(columns=['UTS', 'Elongation', 'Conductivity'])
    target = train_data[['UTS', 'Elongation', 'Conductivity']]
    
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Training
    st.header("Train the Random Forest Model")
    if st.button("Train Model"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        # Evaluation
        st.write("### Model Performance")
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")

        # Results
        st.write("### Prediction Results")
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=y_val['UTS'], y=y_pred[:, 0], label="UTS")
        sns.scatterplot(x=y_val['Elongation'], y=y_pred[:, 1], label="Elongation")
        sns.scatterplot(x=y_val['Conductivity'], y=y_pred[:, 2], label="Conductivity")
        plt.legend()
        st.pyplot(plt)
