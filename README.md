# House Price Prediction

## Project Description

This project aims to predict house prices using machine learning techniques. It uses various features such as the number of bedrooms, bathrooms, square footage, and location to train models that can accurately estimate house values.

## Goals of the Project

- Analyze key factors influencing house prices.
- Preprocess and clean the dataset for modeling.
- Build and compare different regression models.
- Identify the best model for house price prediction.
- Make accurate predictions on new data.

## Dataset Details

The dataset used includes the following features:

- Number of bedrooms and bathrooms
- Square footage of the house and lot
- Year built and renovated
- Location details
- House condition and grade

The dataset may contain missing values and outliers, which were handled during preprocessing.

## Technologies Used

- Python
- Pandas, NumPy – for data manipulation
- Matplotlib, Seaborn – for visualization
- Scikit-learn – for building ML models
- Jupyter Notebook – for writing and running code

## Steps Followed

1. **Data Loading and Exploration**  
   Loaded the dataset and explored basic statistics and distributions.

2. **Data Preprocessing**  
   - Handled missing values  
   - Removed outliers  
   - Converted categorical variables into numeric form  
   - Applied feature scaling where needed

3. **Model Building**  
   Trained and tested multiple regression models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor

4. **Model Evaluation**  
   - Compared models using R-squared, MAE, RMSE
   - Selected the best model based on performance

5. **Prediction**  
   Used the final model to predict house prices on test data.

## Model Evaluation

| Model                     | R-squared | RMSE   |
|--------------------------|-----------|--------|
| Linear Regression        | 0.72      | 85,000 |
| Decision Tree Regressor  | 0.83      | 64,000 |
| Random Forest Regressor  | 0.89      | 52,000 |
| Gradient Boosting        | 0.91      | 48,000 |

*Note: These values are just examples. Replace them with your actual results.*

## Conclusion

The machine learning models built in this project effectively predict house prices. Among the models tested, the Gradient Boosting Regressor gave the best results. The project helps understand the most important features influencing house prices.

## Future Improvements

- Add more external data (like crime rate, school proximity)
- Implement a web application using Flask or Streamlit
- Use deep learning or advanced ensemble methods

## Author

R. Dharanidharan  
Data Scientist Trainee at DataMites  
Tirupattur District, Tamil Nadu, India
