# Diabetes Progression Prediction Using Machine Learning

This project predicts diabetes progression using various machine learning models. It includes data preprocessing, exploratory data analysis (EDA), and the implementation of three models: Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor. The project compares model performance using Mean Squared Error (MSE) and R² score, and visualizes key findings with correlation heatmaps, distribution plots, and model comparison charts.

## Key Features
- **Data Analysis and Visualization:** Utilizes Pandas, Seaborn, and Matplotlib.
- **Machine Learning Models:** Implements Linear Regression, Random Forest, and Gradient Boosting.
- **Model Performance Evaluation:** Measures accuracy using MSE and R² score.
- **Visualizations:** Includes heatmaps, distribution plots, and model comparison charts.

## Installation

### Setting Up a Virtual Environment

1. **Create a Virtual Environment:**
   - For Mac/Linux:
     ```bash
     python3 -m venv env
     ```
   - For Windows:
     ```bash
     python -m venv env
     ```

2. **Activate the Virtual Environment:**
   - For Mac/Linux:
     ```bash
     source env/bin/activate
     ```
   - For Windows:
     ```bash
     .\env\Scripts\activate
     ```

3. **Install Dependencies:**
   Ensure you have the `requirements.txt` file in your project directory, then install the required packages:
   ```bash
   pip install -r requirements.txt
