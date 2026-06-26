import os
import pickle

import streamlit as st

# Path to the trained model produced by Model_Deployment.ipynb
MODEL_PATH = "classifier.pkl"


@st.cache_resource
def load_model(path):
    """Load the trained classifier once and cache it across reruns."""
    with open(path, "rb") as model_file:
        return pickle.load(model_file)


def predict(model, age, wc, htn, dm):
    """Return a human-readable CKD prediction for a single patient record."""
    # Pre-process the categorical user inputs into the encoding used at training time
    htn = 1 if htn == "yes" else 0
    dm = 1 if dm == "yes" else 0

    result = model.predict([[age, wc, htn, dm]])
    if result[0] == 0:
        return "Kidney Disease not detected"
    return "Kidney Disease found"


def main():
    """Define and render the Streamlit web page."""
    st.set_page_config(page_title="Kidney Disease Prediction", page_icon="🩺")

    # Front end header
    html_temp = """
    <div style="background-color:cyan;padding:13px">
    <h1 style="color:black;text-align:center;">Kidney Disease Prediction</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("")

    # Fail gracefully if the model has not been generated yet
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Trained model '{MODEL_PATH}' was not found. "
            "Please run Model_Deployment.ipynb first to generate it."
        )
        return

    model = load_model(MODEL_PATH)

    # Input widgets for the four features the model was trained on
    htn = st.selectbox("Hypertension (htn)", ("no", "yes"))
    dm = st.selectbox("Diabetes Mellitus (dm)", ("no", "yes"))
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    wc = st.number_input("White Blood Cell Count (wc)", min_value=0, step=100)

    # Run the prediction when the user clicks the button
    if st.button("Predict"):
        result = predict(model, age, wc, htn, dm)
        st.success(f"Report Result: {result}")


if __name__ == "__main__":
    main()
