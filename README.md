# Diabetes Progression Prediction Using Machine Learning

This project predicts diabetes progression using various machine learning models. It includes data preprocessing, exploratory data analysis (EDA), and the implementation of three models: Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor. The project compares model performance using Mean Squared Error (MSE) and R² score, and visualizes key findings with correlation heatmaps, distribution plots, and model comparison charts.

## Key Features
- **Data Analysis and Visualization:** Utilizes Pandas, Seaborn, and Matplotlib.
- **Machine Learning Models:** Implements Linear Regression, Random Forest, and Gradient Boosting.
- **Model Performance Evaluation:** Measures accuracy using MSE and R² score.
- **Visualizations:** Includes heatmaps, distribution plots, and model comparison charts.


## Model Performance Evaluation

### Mean Squared Error (MSE)

- **Definition:** Mean Squared Error (MSE) measures the average of the squared differences between the predicted values and the actual values. It quantifies how close the predicted values are to the actual values.
- **Interpretation:**
  - **Lower MSE:** Indicates better model performance, as the predictions are closer to the actual values.
  - **Higher MSE:** Indicates poorer model performance, with predictions being more spread out from the actual values.

### R-Squared
 represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It is a measure of how well the model explains the variability in the target variable.

- **Interpretation:**
  - **\( R^2 = 1 \):** Indicates a perfect fit; the model explains all the variability in the target variable.
  - **\( R^2 = 0 \):** Indicates the model does not explain any variability beyond the mean of the target variable.
  - **Negative \( R^2 \):** Indicates the model performs worse than a simple mean-based model.


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
