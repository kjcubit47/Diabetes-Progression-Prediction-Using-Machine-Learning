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

4. **Deactivate the Virtual Environment:**
   When you’re done, you can deactivate the virtual environment by running:
   ```bash
    deactivate

## Usage
Run the Python script to execute the analysis and train the machine learning models. The script is named diabetes_prediction.py and can be run with:
    ```bash
    python diabetes_prediction.py
Results, including performance metrics and visualizations, will be displayed.


## Conclusions

- **Model Performance:** The Gradient Boosting Regressor generally outperforms the other models in predicting diabetes progression, providing the lowest MSE and highest R² score.
- **Feature Insights:** Key features influencing diabetes progression were identified through EDA and correlation analysis. These insights can inform future research and model improvements.
- **Visualization:** The visualizations effectively highlight the relationships between features and the target variable, and the performance of different models.
