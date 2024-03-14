import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def load_model():
    with open("..\Code\model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def load_scaler():
    with open("..\Code\scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler



# Reverse scaling to get the price in USD
def scale_prediction(prediction):

    mean_price = 476984.55943714274  # Mean house price
    std_price = 208371.26167027562

    scaled_prediction = prediction * std_price + mean_price
    return scaled_prediction




saved_mean = np.array([1975.5581668051009, 7.530561391508281, 1922.2551912835295, 1708.3309718082767, 47.55688812234329, 1.6881809742512337, 3.329750329799189, 1.4284946499242683, 0.17266819758635854, 0.45219133238872333, 14610.408169248058, 12447.084526310646, 74.68114525822055, -122.2132654028436])
saved_std = np.array([774.8334603691661, 1.0391917852731496, 614.9320098425608, 727.2964609354087, 0.14103849190104215, 0.6704659761079718, 0.9128847124369232, 0.5491614709044589, 0.6409504521819265, 0.6292633043508314, 40109.55681337646, 26538.59224971217, 378.76164705607835, 0.1424124032485596])

def standardize_new_data(input_data, features):
    scaler = load_scaler()


    standardized_data = scaler.transform(input_data)
    print("ini standardized_data", standardized_data)
    standardized_df = pd.DataFrame(standardized_data, columns=features)
    return standardized_df

# Main function for the Streamlit web app
def main():

    st.write("""
    # Simple House Price Prediction App
    This app predicts the **House Price**!
    """)

    st.sidebar.header('User Input Features')


    features = ['sqft_living','grade','sqft_living15','sqft_above','lat','bathrooms','bedrooms','floors','view','sqft_basement','sqft_lot','sqft_lot15','yr_renovated','long']

    st.header("Masukkan Data Properti")
    data = {}

    for feature in features:
        data[feature] = st.number_input(label=feature, value=0.0, step=1.0, format="%.4f")

    submit_button = st.button("Proses Data")

    if submit_button:

        model = load_model()

        df = pd.DataFrame(data, index=[0], columns=features)
        print("ini df woi",df,end="\n\n\n\n")
        standardized_input_data = standardize_new_data(df, features)
        standardized_input_data = pd.DataFrame(standardized_input_data, columns=features)

        print("ni dah masuk")
        prediction = model.predict(standardized_input_data)[0]
        scaled_prediction = scale_prediction(prediction)

        st.header("Prediksi Harga")
        st.write("Standarized value:",prediction)
        st.write(f"${scaled_prediction:.2f}")

if __name__ == "__main__":
    main()
