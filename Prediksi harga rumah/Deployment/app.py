import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Function to load the pickled model
def load_model():
    with open("..\Code\price_prediction.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def load_pca():
    with open("..\Code\pca_transform.pkl", "rb") as f:
        pca = pickle.load(f)
    return pca

# Function to scale the prediction back to dollars using StandardScaler
def scale_prediction(prediction):
    # Scaling parameters used during training
    mean_price = 476984.55943714274  # Mean house price
    std_price = 208371.26167027562
   # Standard deviation of house prices

    scaled_prediction = prediction * std_price + mean_price
    return scaled_prediction




saved_mean = np.array([3.329750329799189, 1.6881809742512337, 1975.5581668051009, 14610.408169248058, 1.4284946499242683, 1708.3309718082767, 1970.8110128499536, 1922.2551912835295, 12447.084526310646, 6.578541066106415, 2014.3226168954902, 0.17266819758635854, 3.4063614599110763, 7.530561391508281, 0.45219133238872333, 74.68114525822055, 0.00298040748522011, 47.55688812234329, -122.2132654028436, 98078.55577270729])
saved_std = np.array([0.9128847124369232, 0.6704659761079718, 774.8334603691661, 40109.55681337646, 0.5491614709044589, 727.2964609354087, 29.160540042301765, 614.9320098425608, 26538.59224971217, 3.1175266956275127, 0.4674889433983131, 0.6409504521819265, 0.6472280285027948, 1.0391917852731496, 0.6292633043508314, 378.76164705607835, 0.05451302458746089, 0.14103849190104215, 0.1424124032485596, 53.3369525145595])

def standardize_new_data(input_data):
    # Create a StandardScaler object with loaded scaling parameters
    scaler = StandardScaler()
    scaler.mean_ = saved_mean
    scaler.scale_ = saved_std

    # Transform the new input data using the same scaler
    standardized_data = scaler.transform(input_data)
    return standardized_data

# Main function for the Streamlit web app
def main():
    st.write(f"-1.224213 : ${scale_prediction(-1.224213)} harusnya 221900.0")
    st.write("""
    # Simple House Price Prediction App
    This app predicts the **House Price**!
    """)

    st.sidebar.header('User Input Features')

    # Define the features required for prediction
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'sqft_above', 'yr_built', 'sqft_living15', 'sqft_lot15', 'month',
       'year', 'view', 'condition', 'grade', 'sqft_basement', 'yr_renovated',
       'waterfront', 'lat', 'long', 'zipcode']
    # Create a form for user input
    st.header("Masukkan Data Properti")
    data = {}

    for feature in features:
        data[feature] = st.number_input(label=feature, value=0.0, step=1.0, format="%.2f")

    # Create a button to submit data
    submit_button = st.button("Proses Data")

    if submit_button:
        # Load the model
        model = load_model()

        # Convert input data to DataFrame
        df = pd.DataFrame(data, index=[0])
        standardized_input_data = standardize_new_data(df)
        print(standardized_input_data)
        pca = load_pca()
        pca_data = pca.transform(standardized_input_data)
        

        # Make prediction
        prediction = model.predict(pca_data)[0]


        # Scale the prediction back to dollars
        scaled_prediction = scale_prediction(prediction)


        # Display the prediction
        st.header("Prediksi Harga")
        st.write("Standarized value:",prediction)
        st.write(f"${scaled_prediction:.2f}")

if __name__ == "__main__":
    main()
