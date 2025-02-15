import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler, LabelEncoder

# load the pickle file
def load_model():
    with open('student_lr_final_model.pkl', 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le


''' 
    user input needs to go through z-transformation which was done 
    during the model creation.
    input to the method
        data
        z-transformation and 
        label encoder
'''
def prepossing_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    processed_data = prepossing_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title('Student Performance Predicition')
    st.write('Enter your data to get a prediction for your performance')
    
    hr_studied = st.number_input('Hours Studied', min_value=1, max_value=10, value=5)
    prev_scores = st.number_input('Previous Scores', min_value= 20, max_value= 100, value = 40)
    extra_act = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
    sleep_hr = st.number_input('Sleep Hours', min_value= 4 , max_value= 9, value= 4)
    paper_solved = st.number_input('Number of question paper solved', min_value= 0, max_value= 10, value= 4)
    
    # define button for prediction
    if st.button('Score Prediction'):
        # do the mapping of the UI fileds to the actual column
        user_data = {
            'Hours Studied' : hr_studied,
            'Previous Scores' : prev_scores,
            'Extracurricular Activities' : extra_act,
            'Sleep Hours' : sleep_hr,
            'Sample Question Papers Practiced' : paper_solved
        }
        prediction = predict_data(user_data)
        
        st.success(f'Your prediction result is {prediction}')
        
if __name__ == '__main__':
    main()


