# Airline Passenger Satisfaction Prediction using Machine Learning

This project focuses on building machine learning models to predict whether an airline passenger is satisfied or dissatisfied based on survey responses. By leveraging several machine learning techniques, we aim to classify passenger satisfaction with high accuracy.

## Dataset Description:

The dataset used was sourced from [Kaggle](https://www.kaggle.com/datasets/johndddddd/customer-satisfaction). It contains over 130,000 survey entries detailing passenger feedback and flight information. There are 23 feature columns, including:
- **Survey-based ratings** on various aspects of the flight (scale of 1 to 5)
- **Demographic details** of passengers
- **Type of travel** (business/leisure)
- **Flight details**

The target variable is:
- **Satisfied (1)**
- **Neutral or Dissatisfied (0)**

## Preprocessing:

1. **Correlation Matrix**: We generated a heatmap to evaluate correlations. Features like 'Departure Delay in Minutes' were removed to avoid multicollinearity.
  
2. **Feature Encoding**:
   - **Label Encoding**: Applied to categorical columns like 'Customer Type' and 'Class'.
   - **One-Hot Encoding**: Applied to columns like 'Gender' and 'Type of Travel'.

3. **Train-Test Split**: The dataset was split into 70% training and 30% testing data to evaluate model performance.

4. **Feature Scaling**: Min-Max scaling was applied to ensure numeric features were within the range [0, 1], improving model accuracy.

## Machine Learning Models:

Three machine learning algorithms were implemented:
1. **Decision Tree Classifier**: Achieved an accuracy of **93.75%**.
2. **Logistic Regression**: Achieved an accuracy of **83.37%**.
3. **Random Forest Classifier**: The best-performing model with an accuracy of **95.74%**.

## Model Evaluation:

Models were evaluated based on several metrics:
- **Accuracy**: Correct predictions divided by total predictions.
- **Precision**: True positives among predicted positives.
- **Recall**: True positives among actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

Confusion matrices were also generated to visualize model performance.

**Random Forest Classifier** performed the best with:
- **Precision**: 96.71%
- **Recall**: 95.43%
- **F1-Score**: 96%

## Conclusion:

This project demonstrates how machine learning models can accurately predict airline passenger satisfaction. The **Random Forest Classifier** was the most reliable model, with the highest performance across all metrics.

## How to Run:

1. Install the required Python libraries:
   ```bash
   pip install sklearn pandas seaborn
2. Load the dataset and preprocess as outlined in the project.
3. Train and evaluate the models using the provided Jupyter Notebook.

## For more detailed information, refer to the [Project Report File](https://github.com/JOVi1506/AI-ML_Project/blob/main/Project%20Report.pdf)
