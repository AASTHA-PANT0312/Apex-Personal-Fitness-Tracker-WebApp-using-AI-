## 📊 Apex: Personal Fitness Tracker Web App using Artificial Intelligence

### 🔎 Overview
Apex Personal Fitness Tracker is an end-to-end AI-powered solution built on real-world Fitbit data. This project enables users to:
- Visualize their daily activity, calories burned, heart rate, and sleep quality.
- Compare their fitness metrics with population averages.
- Predict daily calorie burn using trained machine learning models.
- Receive actionable health tips and motivational messages via an interactive Streamlit web app.

---

### 📁 Dataset
Due to file size constraints, all datasets are provided in a single zip (archive.zip) except merged_dataset


➡️  [Fitbit Dataset by Animesh Mahajan on Kaggle](https://www.kaggle.com/datasets/animeshmahajan/fitbit-dataset)  

👉 After downloading archive.zip, please extract it inside your project folder. It will create a datasets/ directory containing the following files:

dailyActivity_merged.csv

heartrate_seconds_merged.csv

hourlyCalories_merged.csv

hourlyIntensities_merged.csv

hourlySteps_merged.csv

sleepDay_merged.csv

weightLogInfo_merged.csv
The project uses the **Fitbit Fitness Tracker Dataset** from Kaggle, which includes:
- Daily Activity
- Daily Steps
- Daily Calories
- Daily Intensities
- Weight Logs
- Sleep Logs
- Heart Rate Logs

➡️ A **merged_fitness_data** was created for model training and deployment integration.

---

### 🔧 Feature Engineering
To enhance analysis and prediction, we engineered several features:
- **BMI (Body Mass Index)**
- **BMR (Basal Metabolic Rate)**
- **TDEE (Total Daily Energy Expenditure)**
- **Body Fat Percentage (BMI-based)**
- **Calories per Step**
- **Sleep Efficiency**
- **BMI Category (Underweight, Normal, Overweight, Obese)**

---

### 📊 Exploratory Data Analysis (EDA)
Performed with visual storytelling techniques, the EDA process included:
- Distributions of steps, calories, and sleep
- Correlation heatmaps
- Scatter plots, histograms, boxplots
- Comparative insights between active vs. sedentary users

---

### 🤖 Models Used
Three regression models were trained to predict daily calories burned:

| Model                      | Description                                                                            |
|----------------------------|----------------------------------------------------------------------------------------|
| **Linear Regression**      | Captures linear relationships between predictors and calories burned.                  |
| **Decision Tree Regressor**| Models non-linear relationships and allows easy interpretation.                        |
| **Random Forest Regressor**| An ensemble approach for more stable and accurate predictions.                         |

**Evaluation Metrics Used:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

---

### 🌟 Streamlit Web App Features
✅ Upload your own fitness data or explore provided samples  
📈 Dynamic visualizations for daily activity, calories, heart rate, and sleep  
📏 *Compare Yourself* tool against dataset averages  
⚡ Predict daily calories burned using trained models  
🎯 Display of *Health Tip of the Day* and motivational quotes  
📅 Progress tracker with fun statistic cards  
🌓 Light/Dark mode toggle  

---

### 🚀 Tech Stack
- **Python**
- **Pandas, NumPy** (Data processing)
- **Matplotlib, Seaborn, Plotly** (Visualizations)
- **Scikit-Learn** (ML models)
- **Joblib / Pickle** (Model serialization)
- **Streamlit** (Web app deployment)

---

### 📁 Project Structure

```bash
.
├── fitness.ipynb              # Jupyter notebook with full EDA, feature engineering & model training
├── app.py                     # Streamlit app for interactive visualization and prediction
├── data/                      #Download it either from the zip or the original data source
│   ├── dailyActivity.csv       # Fitbit daily activity data
│   ├── dailySteps.csv          # Daily step counts
│   ├── dailyCalories.csv       # Daily calories burned
│   ├── dailyIntensities.csv    # Daily exercise intensity data
│   ├── weightLogInfo.csv       # Weight logs from Fitbit users
│   ├── sleepDay.csv            # Sleep log information
│   └── heartrate_seconds.csv   # Heart rate logs
├── merged_fitness_data.csv         # Final combined dataset used for model training and app deployment
├── random_forest_model.pkl    # Trained Random Forest model used for calorie prediction
├── requirements.txt           # List of required Python libraries
└── README.md                  # Project documentation
```

---

### 💡 How to Run Locally

```bash
# Clone the repository
git clone <repository-link>

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

### 📚 References
- [Fitbit Fitness Tracker Data — Kaggle](https://www.kaggle.com/datasets/arashnic/fitbit)  
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.  
- Pedregosa et al., (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.  
- [Streamlit Documentation](https://docs.streamlit.io)  
- World Health Organization — [BMI Classification Guidelines](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)

---
🙏 Acknowledgments
Dataset Source: Fitbit Dataset by Animesh Mahajan — Kaggle

All credit for the dataset goes to the original creator.

