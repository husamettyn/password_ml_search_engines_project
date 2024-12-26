import pandas as pd
import string
import kagglehub
from sklearn.preprocessing import LabelEncoder

def password_features(password):
    length = len(password)
    unique_chars = len(set(password))
    uppercase_count = sum(1 for char in password if char.isupper())
    lowercase_count = sum(1 for char in password if char.islower())
    digit_count = sum(1 for char in password if char.isdigit())
    special_count = sum(1 for char in password if char in string.punctuation)
    
    return pd.Series({
        'length': length,
        'unique_chars': unique_chars,
        'uppercase_ratio': uppercase_count / length,
        'lowercase_ratio': lowercase_count / length,
        'digit_ratio': digit_count / length,
        'special_ratio': special_count / length
    })
    
def import_dataset():
    # Load dataset
    path = kagglehub.dataset_download("utkarshx27/passwords")
    df = pd.read_csv(f"{path}/passwords.csv").dropna()

    # Drop rank_alt variable
    df = df.drop(columns=['rank_alt', 'time_unit', 'value'])


    # Apply feature extraction
    password_features_df = df['password'].apply(password_features)
    df = pd.concat([df, password_features_df], axis=1)

    # Category Encoding: Convert category to integer
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    df = df.drop(columns=['category'])
    return df