import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Predictor Pro", page_icon="📈")

# Custom CSS for animations and styling
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main-header {
        animation: fadeIn 1.5s ease-out;
        color: #FF4B4B;
        text-align: center;
    }
    .stNumberInput {
        animation: fadeIn 2s ease-out;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        border-left: 5px solid #FF4B4B;
        animation: fadeIn 1s ease-in;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-header'>Salary Prediction Engine</h1>", unsafe_allow_html=True)
st.write("---")

# Load the model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.pkl' is in the same directory.")
    st.stop()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Details")
    # Using 'YearsExperience' as found in your model metadata 
    years_exp = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

with col2:
    st.subheader("Prediction")
    if st.button("Calculate Prediction"):
        # Reshape input for the model
        input_data = np.array([[years_exp]])
        prediction = model.predict(input_data)
        
        st.markdown(f"""
            <div class='prediction-card'>
                <h3>Estimated Value:</h3>
                <h2 style='color: #FF4B4B;'>${prediction[0]:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.info("Adjust the experience and click 'Calculate' to see the result.")

# Footer
st.markdown("---")
st.caption("Powered by Scikit-Learn 1.6.1 and Streamlit")
