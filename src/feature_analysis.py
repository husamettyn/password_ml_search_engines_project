import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from utils import password_features, import_dataset

# Download latest version
df = import_dataset()

# print(df.info())
"""
#   Column             Non-Null Count  Dtype
---  ------             --------------  -----
1   password           500 non-null    object
7   strength           500 non-null    float64
0   rank               500 non-null    float64 
5   offline_crack_sec  500 non-null    float64
8   font_size          500 non-null    float64
2   category           500 non-null    object
3   value              500 non-null    float64      DROPPED
4   time_unit          500 non-null    object       DROPPED
6   rank_alt           500 non-null    float64      DROPPED
"""

# # Şifre kategorilerini görüntüle
# print("\nŞifre Kategorileri:")
# print(df['category'].unique())


# # Apply feature extraction
# password_features_df = df['password'].apply(password_features)
# # Concatenate new features with the original dataframe
# df = pd.concat([df, password_features_df], axis=1)

# # Category Encoding: Convert category to integer
# label_encoder = LabelEncoder()
# df['category_encoded'] = label_encoder.fit_transform(df['category'])

# # Drop original text columns
# df = df.drop(columns=['category'])

#print(df.info())
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

# Correlation Analysis
df = df.drop(columns=['font_size'])
numeric_columns = df.select_dtypes(include=['float64', 'int32'])
correlation_matrix = numeric_columns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix Heatmap")
plt.show()

# # Bar plot for category vs strength
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x='category', y='strength', ci=None, palette="viridis")
# plt.title("Category vs Strength")
# plt.xlabel("Category")
# plt.ylabel("Strength")
# plt.xticks(rotation=45)
# plt.show()

# # Password Analysis
# highest_strength_password = df[df['strength'] == df['strength'].max()]['password'].values[0]
# highest_strength_value = df['strength'].max()
# print(f"Highest Strength Password: {highest_strength_password} (Strength: {highest_strength_value})")

# highest_cracking_time_password = df[df['offline_crack_sec'] == df['offline_crack_sec'].max()]['password'].values[0]
# highest_cracking_time_value = df['offline_crack_sec'].max()
# print(f"Highest Cracking Time Password: {highest_cracking_time_password} (Cracking Time: {highest_cracking_time_value} seconds)")

# highest_strength_row = df[df['strength'] == df['strength'].max()]
# highest_strength_password = highest_strength_row['password'].values[0]
# offline_crack_sec = highest_strength_row['offline_crack_sec'].values[0]
# print(f"Offline Cracking Time for the Highest Strength Password: {offline_crack_sec} seconds")

# time_unit_mapping = {
#     'seconds': 1,
#     'minutes': 60,
#     'hours': 3600,
#     'days': 86400,
#     'weeks': 604800,        # 7 gün
#     'months': 2592000,      # 30 gün
#     'years': 31536000       # 365 gün
# }

# print("Unique Time Units:", df['time_unit'].unique())
# print("Value Column Sample:")
# print(df['value'].head())

# # Calculate online_crack_time in seconds
# df['online_crack_time'] = df['value'] * df['time_unit'].map(time_unit_mapping)

# # Normalize the 'online_crack_time' column using Min-Max Scaling
# scaler = MinMaxScaler()
# df['online_crack_time'] = scaler.fit_transform(df[['online_crack_time']])

# # Visualization: Online Crack Time vs Strength (Original)
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='strength', y='online_crack_time', data=df)
# plt.yscale('log')  # Use logarithmic scale for better visualization
# plt.title('Online Crack Time vs Strength (Original)')
# plt.xlabel('Strength')
# plt.ylabel('Online Crack Time (seconds)')
# plt.show()



# # Password Strength Analysis
# strength_counts = df['strength'].value_counts().sort_index()
# plt.bar(strength_counts.index, strength_counts.values)
# plt.xlabel('Password Strength')
# plt.ylabel('Number of Passwords')
# plt.title('Distribution of Password Strengths')
# plt.show()

# # Password Category Analysis
# category_counts = df['category'].value_counts()
# plt.figure(figsize=(10, 6))
# category_counts.plot(kind='bar')
# plt.xlabel('Password Category')
# plt.ylabel('Number of Passwords')
# plt.title('Distribution of Password Categories')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# # Cracking Time Analysis
# average_online_crack_time = df['value'].mean()
# average_offline_crack_time = df['offline_crack_sec'].mean()
# print('Average online crack time: ', average_online_crack_time)
# print('Average offline crack time: ', average_offline_crack_time)

# sns.barplot(x="password", y="offline_crack_sec", data=df)
# plt.xticks(rotation=45)
# plt.xlabel("Password")
# plt.ylabel("Offline Crack Time (seconds)")
# plt.title("Cracking Time Analysis")
# plt.tight_layout()
# plt.show()

# # Password Cracking Time Analysis by Category
# time_conversion = {
#     "seconds": 1,
#     "minutes": 60,
#     "hours": 3600
# }

# df["offline_crack_sec"] = df["offline_crack_sec"] * df["time_unit"].map(time_conversion)

# average_crack_time = df.groupby("category")["offline_crack_sec"].mean().reset_index()

# plt.figure(figsize=(10, 6))
# sns.barplot(x="category", y="offline_crack_sec", data=average_crack_time)
# plt.xlabel("Password Category")
# plt.ylabel("Average Cracking Time (seconds)")
# plt.title("Password Cracking Time Analysis by Category")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# # Password Length Analysis:
# df['password_length'] = df['password'].str.len()

# plt.figure(figsize=(10, 6))
# sns.boxplot(x='password_length', y='strength', data=df)
# plt.title("Password Length vs. Strength")
# plt.xlabel("Password Length")
# plt.ylabel("Strength")
# plt.xticks(rotation=90)
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.boxplot(x='password_length', y='offline_crack_sec', data=df)
# plt.title("Password Length vs. Cracking Time")
# plt.xlabel("Password Length")
# plt.ylabel("Offline Cracking Time (seconds)")
# plt.xticks(rotation=90)
# plt.show()

# # Font size ile diğer özellikler arasındaki korelasyonu inceleyelim
# font_correlations = df[['font_size', 'strength', 'length', 'unique_chars', 
#                        'uppercase_ratio', 'lowercase_ratio', 'digit_ratio', 
#                        'special_ratio']].corr()['font_size'].sort_values(ascending=False)

# print("\nFont Size Korelasyonları:")
# print(font_correlations)

# # Görselleştirme
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x='strength', y='font_size')
# plt.title("Şifre Gücü vs Font Boyutu")
# plt.xlabel("Şifre Gücü")
# plt.ylabel("Font Boyutu")
# plt.show()
