from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import kagglehub
# Veriyi yükleme
import pandas as pd

path = kagglehub.dataset_download("utkarshx27/passwords")
df = pd.read_csv(f"{path}/passwords.csv").dropna()

# df.info()
"""
#   Column             Non-Null Count  Dtype
---  ------             --------------  -----
1   password           500 non-null    object
7   strength           500 non-null    float64
0   rank               500 non-null    float64 
5   offline_crack_sec  500 non-null    float64
8   font_size          500 non-null    float64
2   category           500 non-null    object
3   value              500 non-null    float64      DROP
4   time_unit          500 non-null    object       DROP
6   rank_alt           500 non-null    float64      DROP
"""

# Bağımsız ve bağımlı değişkenleri ayıralım
X = df[['rank', 'value', 'offline_crack_sec', 'font_size']]
y = df['strength']

# Veri setini train-test olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree modeli eğitimi
decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features= None, 
                                        min_samples_leaf=2, min_samples_split=2, random_state=42)
decision_tree.fit(X_train, y_train)
dt_predictions = decision_tree.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")

# Naive Bayes modeli eğitimi
naive_bayes = GaussianNB(var_smoothing=1e-5)
naive_bayes.fit(X_train, y_train)
nb_predictions = naive_bayes.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}")

# Logistic Regression modeli eğitimi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_regression = LogisticRegression(C=1000, max_iter=1000, penalty='l2', solver='lbfgs', random_state=42)
logistic_regression.fit(X_train_scaled, y_train)
lr_predictions = logistic_regression.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")
