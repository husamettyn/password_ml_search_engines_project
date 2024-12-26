"""
This script is used to analyze the performance of different models on the passwords dataset.
It compares the performance of Decision Tree and Naive Bayes classifiers on the dataset.
It also plots the results of GridSearchCV for all parameters of the classifiers.

Functions:
    plot_all_params(cv_results, params, model_name):
        Plots the GridSearchCV results for all parameters of a given model.
"""


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from utils import import_dataset, password_features

# Function to plot GridSearchCV results for all parameters
def plot_all_params(cv_results, params, model_name):
    for param_name in params.keys():
        plt.figure(figsize=(10, 6))
        param_values = cv_results['param_' + param_name]
        
        # Convert None values to a string for plotting
        param_values = [str(val) if val is not None else 'None' for val in param_values]
        
        mean_test_scores = cv_results['mean_test_score']
        plt.plot(param_values, mean_test_scores, marker='o')
        plt.title(f'{model_name} - {param_name} vs Accuracy')
        plt.xlabel(param_name)
        plt.ylabel('Mean Test Accuracy')
        plt.grid(True)
        plt.show()

df = import_dataset()

print(df.info())

"""
Data columns (total 12 columns):
#   Column             Non-Null Count  Dtype
-   ------             --------------  -----
0   rank               500 non-null    float64
1   password           500 non-null    object
2   offline_crack_sec  500 non-null    float64
3   strength           500 non-null    float64
4   font_size          500 non-null    float64
5   length             500 non-null    float64
6   unique_chars       500 non-null    float64
7   uppercase_ratio    500 non-null    float64
8   lowercase_ratio    500 non-null    float64
9   digit_ratio        500 non-null    float64
10  special_ratio      500 non-null    float64
11  category_encoded   500 non-null    int32
"""

# Feature selection and preprocessing
X = df[['rank', 'offline_crack_sec', 'font_size', 'length', 'unique_chars', 'uppercase_ratio', 'lowercase_ratio', 'digit_ratio', 'special_ratio', 'category_encoded']]
y = df['strength']  # Target variable 'strength'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Decision Tree Classifier - Plot results for all parameters
print("\nDecision Tree Classifier")
dt_params = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}
dt = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring='accuracy', error_score='raise')
dt_grid.fit(X_train, y_train)

dt_best = dt_grid.best_estimator_
dt_predictions = dt_best.predict(X_test)
print(f"Best Parameters: {dt_grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, dt_predictions)}")

# Plot results for all Decision Tree parameters
plot_all_params(dt_grid.cv_results_, dt_params, 'Decision Tree')

# 2. Naive Bayes - Plot results for all parameters
print("\nNaive Bayes Classifier")
gnb_params = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
}
gnb = GaussianNB()
gnb_grid = GridSearchCV(gnb, gnb_params, cv=5, scoring='accuracy')
gnb_grid.fit(X_train, y_train)

gnb_best = gnb_grid.best_estimator_
gnb_predictions = gnb_best.predict(X_test)
print(f"Best Parameters: {gnb_grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, gnb_predictions)}")

# Plot results for all Naive Bayes parameters
plot_all_params(gnb_grid.cv_results_, gnb_params, 'Naive Bayes')

# # 3.  Logistic Regression - Plot results for all parameters

# print("\nLogistic Regression Classifier")
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Uyumlu parametre kombinasyonları için ayrı grid'ler oluşturalım
# param_grid = [
#     {
#         'penalty': ['l2', 'none'],
#         'C': [0.01, 0.1, 1, 10, 100, 1000],
#         'solver': ['lbfgs', 'newton-cg'],
#         'max_iter': [100, 200, 1000, 5000]
#     },
#     {
#         'penalty': ['l1', 'l2', 'elasticnet'],
#         'C': [0.01, 0.1, 1, 10, 100, 1000],
#         'solver': ['saga'],
#         'l1_ratio': [0.2, 0.5, 0.8],  # elasticnet için gerekli
#         'max_iter': [100, 200, 1000, 5000]
#     }
# ]

# # GridSearch'ü güncellenen param_grid ile çalıştırma
# grid_search = GridSearchCV(
#     LogisticRegression(random_state=42, class_weight='balanced'),
#     param_grid,
#     cv=3,
#     scoring='accuracy',
#     n_jobs=-1
# )

# grid_search.fit(X_train, y_train)

# # En iyi parametreler ve sonuçlar
# best_params = grid_search.best_params_
# print("En iyi parametreler:", best_params)

# # Sonuçların görselleştirilmesi
# results = grid_search.cv_results_

# def plot_parameter_effect(param_name):
#     plt.figure(figsize=(8, 6))
#     param_values = results['param_' + param_name]
#     mean_test_scores = results['mean_test_score']
#     plt.plot(param_values, mean_test_scores, marker='o')
#     plt.xlabel(param_name)
#     plt.ylabel('Mean Test Accuracy')
#     plt.title(f'Effect of {param_name} on Accuracy')
#     plt.grid(True)
#     plt.show()

# for param in ['penalty', 'C', 'solver', 'max_iter']:
#     plot_parameter_effect(param)
