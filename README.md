# GrowthLink_Task
## IMDb Movies India Analysis
This project analyzes the IMDb Movies India dataset to predict movie ratings based on various features such as genre, director, actors, duration, and votes. The analysis involves data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning model training and evaluation.

## Objectives
 Build a predictive model to estimate movie ratings based on different attributes.
 • Perform data preprocessing, including encoding categorical variables and handling missing values.
 • Engineer useful features like director success rate and average rating of similar movies.
 • Evaluate the model using appropriate techniques.
 • Expected outcome: A model that accurately predicts movie ratings based on given inputs.
 • Submit a structured GitHub repository with documentation on approach, preprocessing, and performance evaluation

## Data Understanding

The dataset has the following columns:

``Name``: Movie name   
``Year``: release year of the movies   
``Duration``: Movie duration   
``Genre``: Movie genre    
``Rating``: Movie rating    
``Votes``: Number of votes received    
``Director``: Movie director   
``Actor 1``: First main actor    
``Actor 2``: Second main actor    
``Actor 3``: Third main actor   

# Steps
# 1. Importing Libraries
The necessary libraries are imported, including pandas, numpy, matplotlib, seaborn, and scikit-learn for data manipulation, visualization, and machine learning.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# 2. Loading the Dataset
The dataset is loaded from a CSV file using pandas.

df = pd.read_csv("C:/Users/acer/Downloads/IMDb Movies India.csv/IMDb Movies India.csv", encoding='ISO-8859-1')
df.head()

# 3. Data Inspection
Basic information about the dataset is displayed, including the number of rows and columns, data types, and missing values.

df.info()
df.describe()
df.dtypes
df.shape
df.isna().sum()


# 4. Handling Missing Values
Missing values in the Rating column are dropped.

Missing values in the Genre column are filled with 'Unknown'.

Numeric values are extracted from the Duration column, and missing values are filled with the median.

Rows with missing values in the Actor 1, Actor 2, and Actor 3 columns are dropped.

df.dropna(subset=['Rating'], inplace=True)
df['Genre'].fillna('Unknown', inplace=True)
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)
df['Duration'].fillna(df['Duration'].median(), inplace=True)
df.dropna(subset=['Actor 1', 'Actor 2', 'Actor 3'], inplace=True)


# 5. Data Type Conversion
The Year and Votes columns are converted to numeric types.

df['Year'] = df['Year'].str.extract(r'(\d+)').astype(int)
df['Votes'] = df['Votes'].str.extract(r'(\d+)').astype(int)


# 6. Outlier Detection and Removal
Outliers in the Year and Votes columns are detected using z-scores and removed.

z_scores = pd.DataFrame()
for column in numerical_columns:
    z_scores[column] = (df[column] - df[column].mean()) / df[column].std()
z_score_threshold = 2
outliers = z_scores[(z_scores.abs() > z_score_threshold).any(axis=1)]
df = df[~((z_scores.abs() > z_score_threshold).any(axis=1))]


# 7. Exploratory Data Analysis (EDA)
The distribution of movie ratings and votes is visualized using histograms.

The top 10 directors with the most movies and the top 10 actors with the most movie appearances are identified and visualized.

The top 10 directors with the highest-rated movies and the top 10 highly rated movie genres are also visualized.

sns.histplot(data=df, x='Rating', bins=30, kde=True)
sns.histplot(data=df, x='Votes', bins=30, kde=True)
director_counts = df['Director'].value_counts()
top_10_directors = director_counts.head(10)
plt.bar(top_10_directors.index, top_10_directors.values)


# 8. Feature Engineering
Categorical columns (Name, Genre, Director, Actor 1, Actor 2, Actor 3) are one-hot encoded.

Numerical columns (Year, Duration, Votes) are scaled using MinMaxScaler.

ohe = OneHotEncoder(sparse=False)
X_categorical_encoded = ohe.fit_transform(X[categorical_columns])
scaler = MinMaxScaler()
X_numeric_scaled = scaler.fit_transform(X[numerical_columns])
X_final = pd.concat([X_numeric_scaled_df, X_categorical_encoded_df], axis=1)


# 9. Model Training and Evaluation
The dataset is split into training and testing sets.

A RandomForestRegressor and a GradientBoostingRegressor are trained and evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics.

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_test = rf_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)


# 10. Hyperparameter Tuning
Hyperparameter tuning is performed using GridSearchCV to find the best parameters for the RandomForestRegressor.

param_grid = {
    'n_estimators': [10, 20, 50],
    'max_depth': [None, 3, 10, 20],
    'min_samples_split': [1, 3, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_


# 11. Feature Importance
The importance of features is visualized using a bar plot to understand which features contribute the most to the model's predictions.

feature_importances = best_rf_model.feature_importances_
top_indices = np.argsort(feature_importances)[::-1][:10]
top_features = [feature_names[i] for i in top_indices]
plt.barh(range(10), top_importances, align='center')



