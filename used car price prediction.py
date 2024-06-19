import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Load the dataset
# @st.cache  # Cache data for better performance
def load_data(file_path):
    data = pd.read_csv('CarPrice_Assignment.csv')
    return data


# Preprocess data
def preprocess_data(data):
    le = LabelEncoder()
    columns_to_encode = ['fueltype', 'enginetype']
    for column in columns_to_encode:
        data[column] = le.fit_transform(data[column])
    return data


# Train the model
def train_model(data, features, target):
    x = data[features]
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    alg = LinearRegression()
    alg.fit(x_train, y_train)
    return alg, x_test, y_test


# Predict using the trained model
def predict(model, x_test):
    return model.predict(x_test)


# Main function to run the Streamlit app
def main():
    st.title('Car Price Prediction App')

    # Load data
    data_file_path = '/content/CarPrice_Assignment.csv'
    data = load_data(data_file_path)

    # Preprocess data
    data = preprocess_data(data)

    # Display the dataset
#    st.subheader('Dataset')
 #   st.write(data)

    # Feature selection for prediction
    features = ['fueltype', 'horsepower', 'enginetype', 'boreratio']
    target = 'price'

    # Train the model
    model, x_test, y_test = train_model(data, features, target)

    # Display model performance
    accuracy_train = model.score(data[features], data[target])
    st.subheader('Model Performance')
    st.write(f"Training Accuracy (R^2 Score): {accuracy_train}")

    # User input for prediction
    st.subheader('Predict Car Price')

    # Input fields
    fueltype = st.selectbox('Fuel Type', data['fueltype'].unique())
    horsepower = st.number_input('Horsepower', min_value=data['horsepower'].min(), max_value=data['horsepower'].max())
    enginetype = st.selectbox('Engine Type', data['enginetype'].unique())
    boreratio = st.number_input('Bore Ratio', min_value=data['boreratio'].min(), max_value=data['boreratio'].max())

    # Predict using the trained model
    input_data = {'fueltype': fueltype, 'horsepower': horsepower, 'enginetype': enginetype, 'boreratio': boreratio}
    input_df = pd.DataFrame([input_data])

    if st.button('Predict'):
        prediction = predict(model, input_df[features])
        st.success(f'Predicted Car Price: ${prediction[0]:,.2f}')


if __name__ == '__main__':
    main()
