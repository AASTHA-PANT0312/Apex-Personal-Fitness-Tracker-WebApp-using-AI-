#importing libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
import random
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.filterwarnings("ignore")




# Load Data
@st.cache_data      #tells streamlit to load the csv file once and then reuse it
def load_data():
    data = pd.read_csv('merged_fitness_data.csv')  # Loading the merged dataset
    return data

data = load_data()

# BMI Calculation
def calculate_bmi(weight, height):
    height_m = height / 100
    return weight / (height_m ** 2)

# BMI Category
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Pre-processing data
data['BMI'] = data.apply(lambda row: calculate_bmi(row['WeightKg'], row['AverageHeightCm']), axis=1)

data['BMI_Category'] = data['BMI'].apply(bmi_category)
data['BMR'] = 66 + (13.7 * data['WeightKg']) + (5 * data['AverageHeightCm']) - (6.8 * 30)  # Assuming avg age 30
data['TDEE'] = data['BMR'] * 1.55
data['CaloriesPerStep'] = data['Calories'] / (data['TotalSteps'] + 1)
data['SleepEfficiency'] = (data['TotalMinutesAsleep'] / (data['TotalTimeInBed'] + 1)) * 100
data['BodyFatPercentage'] = (1.20 * data['BMI']) + (0.23 * 30) - 16.2  # age 30 (assumption)

# Streamlit App Sidebar
st.sidebar.title("Apex")
app_mode = st.sidebar.radio("Go to:", ["🏠 Home",  "📊 Your Progress", "🤖 Calories Predictor", "🆚 Compare Yourself", "🩺 Health Tips","🎯 Fun Stats Cards"])


# Theme Switcher (Light/Dark Mode)

theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    dark_css = """
    <style>
        /* Main background */
        .stApp {
            background-color: #121212 !important;
            color: #E0E0E0 !important;
        }

        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E !important;
            color: #E0E0E0 !important;
        }

        /* Headers and text */
        h1, h2, h3, h4, h5, h6, p, label {
            color: #FAFAFA !important;
        }

        /* Input fields */
        input, textarea, select {
            background-color: #2A2A2A !important;
            color: #E0E0E0 !important;
            border: 1px solid #4A4A4A !important;
            border-radius: 5px !important;
        }

        /* Buttons */
        button {
            background-color: #30363D !important;
            color: #FAFAFA !important;
            border: 1px solid #555 !important;
            border-radius: 5px !important;
        }
        button:hover {
            background-color: #484F58 !important;
        }

        /* Radio buttons & Select boxes */
        div[data-baseweb="radio"], div[data-baseweb="select"] {
            background-color: #2A2A2A !important;
            color: #E0E0E0 !important;
        }

        /* Data tables */
        .stDataFrame, .stTable {
            background-color: #161B22 !important;
            color: #E0E0E0 !important;
            border: 1px solid #444 !important;
        }

        /* Metrics Boxes */
        .stMetric {
            background-color: #222 !important;
            color: #FAFAFA !important;
            border-radius: 5px !important;
            padding: 10px !important;
        }

        /* Charts (Plotly, Matplotlib, Seaborn) */
        .stPlotlyChart, .stPyplot {
            background-color: transparent !important;
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1E1E1E;
        }
        ::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #666;
        }

        /* Fix for upper white portion */
        header, [data-testid="stHeader"] {
            background-color: #121212 !important;
            color: #E0E0E0 !important;
        }

        /* Fix for 'Running' status bar in dark mode */
        [data-testid="stToolbar"] {
            background-color: #1E1E1E !important;
            color: #FAFAFA !important;
        }

    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)

    


# Home Page
if app_mode == "🏠 Home":
    st.title("🏋️‍♀️ Apex  Dashboard")

    # Get the current hour
    current_hour = datetime.now().hour

    # Determine the appropriate greeting
    if current_hour < 12:
        greeting = "🌅 Good Morning!"
    elif 12 <= current_hour < 18:
        greeting = "🌞 Good Afternoon!"
    else:
        greeting = "🌙 Good Evening!"

    # Display the greeting message
    st.markdown(f"## {greeting} Welcome to your interactive fitness analysis app!")

    # Quick overview of app features
    st.write("""
    - 📊 **Analyze** your activity, sleep, and calorie consumption data.
    - 🔥 **Predict** your daily calorie burn based on activity.
    - 🆚 **Compare** yourself with dataset averages.
    - 💡 **Get Tips** for a healthier lifestyle.
    - 📈 **Visualize** your fitness progress.
    """)

    # Generate a "Daily Fitness Number" for motivation
    daily_fitness_number = random.randint(1, 100)  # Random number for motivation
    fitness_messages = [
        "Keep pushing forward! Every step counts. 🚀",
        "Consistency is key! Stay on track. 🏃",
        "You’re stronger than you think! 💪",
        "Hydration is important! Drink water. 💧",
        "Small progress is still progress. Keep moving! 📈"
    ]
    daily_message = random.choice(fitness_messages)

    # Display the motivational number and message
    st.markdown(f"### 🎯 Your Daily Fitness Number: `{daily_fitness_number}`")
    st.info(daily_message)

    # Fun personalized message
    st.markdown("**Stay consistent and keep improving your health! 💪**")

elif app_mode == "📊 Your Progress":
    st.title("📊 Your Progress Tracker")

    # Motivational message based on progress
    progress_quotes = [
        "Every step counts! Keep pushing forward! 🚀",
        "Your only competition is YOU from yesterday. 💪",
        "Small progress is still progress! Keep going! 🌟",
        "Your body achieves what your mind believes! 🧠🏋️",
        "The best investment you can make is in your health! ❤️"
    ]
    st.markdown(f"### {random.choice(progress_quotes)}")

    # Step Progress
    st.subheader("🚶 Step Progress")
    steps_avg = data["TotalSteps"].mean()
    user_steps = st.number_input("Enter your steps today:", min_value=0, step=100, value=5000)

    col1, col2 = st.columns(2)
    with col1:
        fig_steps = px.pie(
            names=["Your Steps", "Target (Avg)"],
            values=[user_steps, max(steps_avg - user_steps, 0)],
            title="Daily Steps Progress",
            hole=0.4
        )
        st.plotly_chart(fig_steps)
    with col2:
        if user_steps < steps_avg:
            st.warning("You're below the average! Try taking a walk! 🚶")
        else:
            st.success("Great job! You're above average! 🎉")

    # Calorie Burn Progress
    st.subheader("🔥 Calories Burned")
    calories_avg = data["Calories"].mean()
    user_calories = st.number_input("Enter your estimated calories burned:", min_value=0, step=50, value=2000)

    col1, col2 = st.columns(2)
    with col1:
        fig_calories = px.bar(
            x=["Your Calories", "Dataset Avg"],
            y=[user_calories, calories_avg],
            title="Calories Burned Today",
            color=["Your Calories", "Dataset Avg"]
        )
        st.plotly_chart(fig_calories)
    with col2:
        if user_calories < calories_avg:
            st.warning("You're burning fewer calories than average. Try increasing activity! 💪")
        else:
            st.success("You're above average! Keep up the good work! 🎯")

    # Sleep Tracking - Auto-Calculated
    st.subheader("💤 Sleep Efficiency")
    sleep_avg = data["SleepEfficiency"].mean()

    # User inputs for sleep with validation
    user_sleep_time = st.slider("Hours of sleep:", min_value=0.0, max_value=24.0, step=0.1, value=7.0)
    user_time_in_bed = st.slider(
        "Total time spent in bed:", 
        min_value=user_sleep_time, max_value=24.0, step=0.1, value=max(user_sleep_time, 8.0)
    )

    # Auto-calculate sleep efficiency
    user_sleep_efficiency = (user_sleep_time / user_time_in_bed) * 100 if user_time_in_bed > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        fig_sleep = px.funnel(
            x=["Perfect (100%)", "Your Sleep Efficiency"],
            y=[100, user_sleep_efficiency],
            title="Sleep Efficiency",
        )
        st.plotly_chart(fig_sleep)
    with col2:
        if user_sleep_efficiency < sleep_avg:
            st.warning("Your sleep efficiency is below average. Try improving sleep hygiene! 🛌")
        else:
            st.success("You're well-rested! Keep it up! 😴")

    # Weekly Progress Heatmap
    st.subheader("📅 Weekly Activity Heatmap")
    weekly_steps = {
        "Monday": random.randint(4000, 10000),
        "Tuesday": random.randint(4000, 10000),
        "Wednesday": random.randint(4000, 10000),
        "Thursday": random.randint(4000, 10000),
        "Friday": random.randint(4000, 10000),
        "Saturday": random.randint(4000, 10000),
        "Sunday": random.randint(4000, 10000),
    }
    fig_weekly = px.bar(
        x=list(weekly_steps.keys()), y=list(weekly_steps.values()),
        title="Weekly Steps Activity",
        color=list(weekly_steps.values()),
        labels={"x": "Day", "y": "Steps"}
    )
    st.plotly_chart(fig_weekly)

    st.markdown("**Tracking your progress keeps you accountable! Keep up the good work! 🚀**")
  
elif app_mode == "🤖 Calories Predictor":
    st.title("🤖 Calorie Burn Prediction")
    st.markdown("Enter your daily activity details to estimate your calorie burn:")

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        total_steps = st.number_input("🚶‍♂️ Total Steps", min_value=0, step=100, value=5000)
        sedentary_minutes = st.number_input("🛋️ Sedentary Minutes", min_value=0, step=5, value=300)
        lightly_active_minutes = st.number_input("🚶 Lightly Active Minutes", min_value=0, step=5, value=60)
        fairly_active_minutes = st.number_input("🏃 Fairly Active Minutes", min_value=0, step=5, value=30)

    with col2:
        very_active_minutes = st.number_input("🏋️ Very Active Minutes", min_value=0, step=5, value=30)
        weight_kg = st.number_input("⚖️ Weight (in Kg)", min_value=30.0, step=0.5, value=70.0)
        user_height_cm = st.number_input("📏 Height (in cm)", min_value=120, step=1, value=170)
        sleep_hours = st.number_input("😴 Sleep (in hours)", min_value=0.0, step=0.1, value=7.0)

    # Calculate BMI
    bmi = weight_kg / ((user_height_cm / 100) ** 2)
    st.write(f"📊 **Your BMI:** `{bmi:.2f}`")

    # Encode BMI Category (as done in training)
    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_category_encoded = 0
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal weight"
        bmi_category_encoded = 1
    elif 25 <= bmi < 29.9:
        bmi_category = "Overweight"
        bmi_category_encoded = 2
    else:
        bmi_category = "Obese"
        bmi_category_encoded = 3

    st.info(f"🏷️ **BMI Category:** `{bmi_category}`")

    # **New: Auto-Calculate Body Fat Percentage**
    body_fat_percentage = (1.20 * bmi) + (0.23 * 30) - 16.2  # Using avg age 30
    st.write(f"📏 **Estimated Body Fat Percentage:** `{body_fat_percentage:.2f}%`")

    # **New: Sleep Efficiency Calculation**
    total_sleep_minutes = sleep_hours * 60
    total_bed_time = total_sleep_minutes + sedentary_minutes
    sleep_efficiency = (total_sleep_minutes / (total_bed_time + 0.01)) * 100
    st.write(f"💤 **Your Sleep Efficiency:** `{sleep_efficiency:.2f}%`")

    # Predict Calories Button
    if st.button("🔥 Predict Calories Burned"):
        try:
            # Additional calculated features
            age = 30  # Assume fixed age
            bmr = 66 + (13.7 * weight_kg) + (5 * user_height_cm) - (6.8 * age)
            tdee = bmr * 1.55  # Moderately active multiplier
            calories_per_step = tdee / 10000

            # Corrected input data (now includes all 14 features)
            input_data = np.array([[total_steps, sedentary_minutes, lightly_active_minutes,
                                    fairly_active_minutes, very_active_minutes, weight_kg, user_height_cm,
                                    bmi, bmr, tdee, calories_per_step, sleep_efficiency, 
                                    body_fat_percentage, bmi_category_encoded]], dtype=float)

            # Load the trained model
            with open("random_forest_model.pkl", "rb") as file:
                model = pickle.load(file)

            predicted_calories = model.predict(input_data)
            st.success(f"✅ Estimated Calories Burned: **{predicted_calories[0]:.2f} kcal**")

        except Exception as e:
            st.error(f"🚨 Something went wrong: {e}")

elif app_mode == "🆚 Compare Yourself":
    st.title("🆚 How Do You Compare?")
    
    # Subheading for user input
    st.markdown("### Enter Your Details to Compare with other Achievers:")

    # User inputs with improved UX
    user_weight = st.number_input("⚖️ Your Weight (kg)", min_value=30.0, max_value=200.0, step=0.5, format="%.1f")
    user_height = st.number_input("📏 Your Height (cm)", min_value=120.0, max_value=220.0, step=1.0, format="%.0f")

    # Improved Step Count Input: Allow up to 50,000 with validation
    user_steps = st.number_input("🚶 Your Daily Steps", min_value=0, max_value=50000, step=500, value=5000)
    if user_steps > 30000:
        st.warning("🏃‍♂️ That's a lot of steps! Make sure you're staying hydrated! 💧")

    if st.button("Compare"):
        # Handling column name variations
        weight_col = "WeightKg" if "WeightKg" in data.columns else st.error("⚠️ Weight column missing!")
        height_col = "HeightCm" if "HeightCm" in data.columns else (
            "AverageHeightCm" if "AverageHeightCm" in data.columns else st.error("⚠️ Height column missing!")
        )
        steps_col = "TotalSteps" if "TotalSteps" in data.columns else st.error("⚠️ Steps column missing!")

        # Compute dataset averages
        avg_weight = data[weight_col].mean()
        avg_height = data[height_col].mean()
        avg_steps = data[steps_col].mean()

        # Display dataset averages
        st.subheader("📊 ** Averages**")
        col1, col2, col3 = st.columns(3)
        col1.metric("⚖️ Avg Weight", f"{avg_weight:.2f} kg")
        col2.metric("📏 Avg Height", f"{avg_height:.2f} cm")
        col3.metric("🚶 Avg Steps", f"{avg_steps:.0f} steps")

        # Weight Comparison
        st.subheader("⚖️ Weight Comparison")
        fig_weight = px.bar(
            x=["Your Weight", "Dataset Avg"], y=[user_weight, avg_weight],
            color=["Your Weight", "Dataset Avg"],
            title="How Your Weight Compares",
            labels={"x": "Category", "y": "Weight (kg)"}
        )
        st.plotly_chart(fig_weight)

        # Weight insights
        if user_weight < avg_weight:
            st.success("You're below the average weight. Ensure you're maintaining a healthy BMI! 🍏")
        else:
            st.warning("You're above the average weight. A balanced diet and exercise can help! 🏋️")

        # Height Comparison
        st.subheader("📏 Height Comparison")
        fig_height = px.bar(
            x=["Your Height", "Dataset Avg"], y=[user_height, avg_height],
            color=["Your Height", "Dataset Avg"],
            title="How Your Height Compares",
            labels={"x": "Category", "y": "Height (cm)"}
        )
        st.plotly_chart(fig_height)

        # Height insights
        if user_height > avg_height:
            st.success("You're taller than the average! 🏀")
        elif user_height < avg_height:
            st.warning("You're shorter than the average. But height doesn't define fitness! 💪")

        # Steps Comparison
        st.subheader("🚶 Steps Comparison")
        fig_steps = px.bar(
            x=["Your Steps", "Dataset Avg"], y=[user_steps, avg_steps],
            color=["Your Steps", "Dataset Avg"],
            title="How Your Steps Compare",
            labels={"x": "Category", "y": "Steps"}
        )
        st.plotly_chart(fig_steps)

        # Steps insights
        if user_steps > avg_steps:
            st.success("You're walking more than average! Keep up the great work! 🚀")
        else:
            st.warning("You're below the average steps. Try adding a short walk to your daily routine! 🚶‍♂️")

        # Motivational Footer
        st.markdown("### 💡 Remember: Your progress matters more than comparison! Keep striving for better! 💪")


elif app_mode == "🩺 Health Tips":
    st.title("🩺 Daily / Weekly Health Tips")

    tips = [
        "💧 Drink at least 2 liters of water daily.",
        "🚶 Take short walks every hour if you're sitting for long periods.",
        "💤 Aim for at least 7 hours of sleep every night.",
        "🥗 Eat more fruits and vegetables with every meal.",
        "🧘 Stretch regularly to improve flexibility and posture.",
        "☕ Avoid sugary drinks and excessive caffeine intake.",
        "✅ Set small, achievable weekly health goals.",
        "🌿 Try deep breathing or meditation for stress relief.",
        "⏰ Maintain a consistent sleep schedule.",
        "📝 Track your fitness progress, but don't obsess over numbers."
    ]

    st.write("💡 **Here are some random health tips for you today:**")
    
    # Ensure unique tips are displayed
    selected_tips = random.sample(tips, 3)
    
    for tip in selected_tips:
        st.write(f"👉 {tip}")


elif app_mode == "🎯 Fun Stats Cards":
    st.title("🎯 Fun Apex Stats Cards")

    total_users = data.shape[0]
    avg_daily_steps = data["TotalSteps"].mean()
    avg_calories = data["Calories"].mean()
    avg_sleep_hours = data["TotalMinutesAsleep"].mean() / 60

    st.markdown("### 🗂️ Quick Stats")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Total Records", f"{total_users}")
    col2.metric("🚶 Avg Daily Steps", f"{avg_daily_steps:.0f} steps")
    col3.metric("🔥 Avg Calories", f"{avg_calories:.0f} kcal")
    col4.metric("😴 Avg Sleep", f"{avg_sleep_hours:.1f} hrs")

    st.write("#### ✅ You’re doing great! Keep tracking and improving! 💪")
st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ by [Aastha Pant]")




