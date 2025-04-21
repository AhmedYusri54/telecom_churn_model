import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pickle as pk
import catboost
from wranglefuncation import wrangle

df = wrangle("telecom_churn.csv") # wrangle function to clean the data

st.title(f"Telecom Customer Churn Analysis Project.")
st.sidebar.header("Navigation")
st.sidebar.markdown("Created by [Ahmed Yusri](https://www.linkedin.com/in/ahmed-yusri-499a67313)")
st.sidebar.image("telecom.jpeg")

sidebar_option = st.sidebar.radio("Choose an Option:", ["Overview", "EDA", "Modeling", "Insights"])

if sidebar_option == "Overview":
    st.header("Data Overview")
    st.write(f"The Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    st.write(df.describe())
    st.write("The percentage of the Churn classes in the dataset.")
    st.write(df["churn"].value_counts())
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(df["churn"], ax=ax)
    plt.title("The percentage of the Churn classes in the dataset.")
    st.pyplot(fig)
    
    
elif sidebar_option == "EDA":
    st.header("Exploratory Data Analysis")
    st.markdown("### The Histogram plot for calls made feature in churn customers.")
    # See a histplot for the churn in call_made
    churn_data_calls_made=df[df['churn']==1]['calls_made']
    non_churn_calls_made=df[df['churn']==0]['calls_made']
    
    churn_value_calls_made=list(churn_data_calls_made.value_counts().sort_values())
    non_churn_value_calls_made=list(non_churn_calls_made.value_counts().sort_values())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(churn_value_calls_made, ax=ax)
    sns.histplot(non_churn_value_calls_made, ax=ax)
    plt.xlabel("call made")
    plt.legend(["churn", "non-churn"])
    plt.title("Histplot calls made churn customer")
    st.pyplot(fig)    
    st.write("The Histogram plot indecating the difference between the churn and non-churn customers in term of call made.")
    st.markdown("### The PieChart plot for the gender feature by churn target.")
    # See the churn percentage in age column
    churn_data_gender=df[df['churn']==1]['gender']
    gender_value=list(churn_data_gender.value_counts())
    # Create the pie plot 
    fig = px.pie(
        names=["Male", "Female"],
        values=gender_value,
        color_discrete_sequence=["orange", "blue"],
        title="Churn percentage of Gender"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("The PieChart plot indecating that there is an difference in Churn in about the gender.\nAs about of 60% of the churn costumers is Male so if there is any Advertising campaigns most study this difference.")
    st.markdown("### Boxplot for the estimated salary feature in telecom campanys by churn target.")
    fig1 = px.box(df, x="telecom_partner", y="estimated_salary",  color="churn", color_discrete_sequence=["orange", "blue"], title="Boxplot for the estimated salary feature in telecom campanys by churn target.")
    st.plotly_chart(fig1, use_container_width=True)
    st.write("The estimated salary feature in telecom campanys low effect in churn target.")

elif sidebar_option == "Modeling":
    st.markdown("## Customer Churn classification model.")
    user_age = st.slider("How old are you?", 18, 90, 28)
    
    user_telecom_partner = st.radio("Select your Telecom Partner",
    options=df["telecom_partner"].unique(),)
    
    user_gender = st.radio("Select your Gender", options=["Male", "Female"])
    if user_gender == "Male":
        user_gender = 1
    elif user_gender == "Female":
        user_gender = 0
         
    user_state = st.selectbox("Select your State",
    options=df["state"].unique(),)
    
    user_city = st.selectbox("Select your City",
    options=df["city"].unique(),)
    
    user_num_de = st.number_input('Enter your Number of Dependents', min_value=0, max_value=10, value=0, step=1)
    
    user_estimated_salary = st.number_input('Enter your Estimated Salary', min_value=0, max_value=10000000, value=0, step=1)
    
    user_calls_made = st.number_input('Enter your Calls Made', min_value=0, max_value=1000, value=0, step=1)
    
    user_data_used = st.number_input('Enter your Data Used', min_value=0, max_value=100000, value=0, step=1)
    
    user_sms_sent = st.number_input('Enter your SMS Sent', min_value=0, max_value=1000, value=0, step=1)
    
    
    user_data = {
        "telecom_partner": user_telecom_partner,
        "gender": user_gender,
        "age": user_age,
        "state": user_state,
        "city": user_city,
        "num_dependents": user_num_de,
        "estimated_salary": user_estimated_salary,
        "calls_made": user_calls_made,
        "sms_sent": user_sms_sent,
        "data_used": user_data_used,
    }
    
    # load the encoder and scalers
    with open("encoders.pkl", "rb") as f:
        encoders = pk.load(f)
    with open("scale.pkl", "rb") as f:
        scaler = pk.load(f)
        
    user_df = pd.DataFrame(user_data, index=[0])
    
    st.write(user_df)
    
    user_encoded = {}
    for col, value in user_data.items():
        if col in encoders:
            user_encoded[col] = encoders[col].transform([value])[0]
        else:
            user_encoded[col] = value    
        
    user_df = pd.DataFrame(user_encoded, index=[0])  
    user_scaled = scaler.transform(user_df) 
    
    # Load the model
    with open("best_model.pkl", "rb") as f:
        model = pk.load(f)
    
    if st.button("Predict"):
        # Make a prediction
        user_churn = model.predict(user_scaled)
        if user_churn[0] == 1:
            st.write("The customer is likely to churn.")
        else:
            st.write("The customer is not likely to churn.")   
            st.image("not_chunrn.jpg", caption="Happy Customer") 
    
elif sidebar_option == "Insights":
    st.header("My Insights")
    st.write("1. The dataset features not all of them are important to predict the churn target.")
    st.write("2. The dataset need a excessive features engineering to get the best model.")
    st.write("3. The inbalance classes in the dataset need to be handled by more techniques to see the alternative solutions to help in finding the best technique to go with.")
    
    
