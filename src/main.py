import gradio as gr
import pandas as pd
import numpy as np
from train import df, decision_tree, naive_bayes, logistic_regression, scaler
from utils import password_features
import random
import string
from sklearn.preprocessing import LabelEncoder

# Model sözlüğü oluşturalım
models = {
    "Decision Tree": decision_tree,
    "Naive Bayes": naive_bayes,
    "Logistic Regression": logistic_regression
}

def predict_samples(model_name, sample_count):
    # Rastgele örnekleri seçelim
    random_indices = np.random.choice(len(df), size=sample_count, replace=False)
    X_samples = df.iloc[random_indices][['rank', 'offline_crack_sec', 'length', 'unique_chars', 
                                       'uppercase_ratio', 'lowercase_ratio', 'digit_ratio', 'special_ratio', 'category_encoded']]
    y_true = df.iloc[random_indices]['strength']
    
    # Seçilen modeli alalım
    model = models[model_name]
    
    # Logistic Regression için scaling gerekiyor
    if model_name == "Logistic Regression":
        X_samples = scaler.transform(X_samples)
    
    # Tahminleri alalım
    y_pred = model.predict(X_samples)
    
    # Sonuçları tablo için hazırlayalım
    results = []
    passwords = df.iloc[random_indices]['password']
    for i in range(sample_count):
        sample_features = X_samples.iloc[i] if model_name != "Logistic Regression" else X_samples[i]
        results.append([
            f"{passwords.iloc[i]}",
            f"{y_true.iloc[i]:.2f}",
            f"{y_pred[i]:.2f}"
        ])
    
    return results

# Kategorileri ve örnek şifreleri tanımlayalım
categories = [
    'password-related', 'simple-alphanumeric', 'animal', 'sport', 
    'cool-macho', 'name', 'fluffy', 'food', 'nerdy-pop', 'rebellious-rude'
]

# Kategorileri encode edelim
label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(categories)

# Kategori dropdown'u için örnek şifreler
category_examples = {
    'password-related': 'password123',
    'simple-alphanumeric': 'abc123',
    'animal': 'lionking',
    'sport': 'football',
    'cool-macho': 'rockstar',
    'name': 'johnsmith',
    'fluffy': 'bunny',
    'food': 'pizza',
    'nerdy-pop': 'starwars',
    'rebellious-rude': 'badboy'
}

# Gradio arayüzünü oluşturalım
with gr.Blocks() as demo:
    gr.Markdown("# Şifre Güçlülük Tahmini")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=list(models.keys()),
                value="Decision Tree",
                label="Model Seçiniz"
            )
            
            sample_count_dropdown = gr.Dropdown(
                choices=[10, 20, 30, 40, 50],  # Örnek sayısı seçenekleri
                value=20,
                label="Örnek Sayısı Seçiniz"
            )
            
            output_table = gr.Dataframe(
                headers=["Password", "Strength Real", "Strength Guess"],
                row_count=20,
                col_count=3,
                interactive=False
            )
            
            def update_table(model_name, sample_count):
                results = predict_samples(model_name, sample_count)
                return gr.Dataframe(
                    value=results,
                    headers=["Password", "Strength Real", "Strength Guess"],
                    row_count=sample_count,
                    col_count=3,
                    interactive=False
                )
            
            model_dropdown.change(
                fn=update_table,
                inputs=[model_dropdown, sample_count_dropdown],
                outputs=output_table
            )
            
            sample_count_dropdown.change(
                fn=update_table,
                inputs=[model_dropdown, sample_count_dropdown],
                outputs=output_table
            )
        
        with gr.Column():
            password_input = gr.Textbox(
                label="Şifre Giriniz",
                placeholder="Tahmin edilecek şifreyi buraya girin"
            )
            
            category_dropdown = gr.Dropdown(
                choices=[f"{cat} ({category_examples[cat]})" for cat in categories],
                label="Kategori Seçiniz"
            )
            
            def predict_password_strength(model_name, password, category):
                if not password or not category:
                    return "Lütfen şifre ve kategori seçiniz."
                
                # Kategori adını ayıklayalım
                category_name = category.split(' ')[0]
                category_encoded = label_encoder.transform([category_name])[0]
                
                # Dummy değerler
                rank = 5
                offline_crack_sec = 5
                font_size = 5
                
                # Şifre özelliklerini hesaplayalım
                features = password_features(password)
                features['category_encoded'] = int(category_encoded)
                features['rank'] = rank
                features['offline_crack_sec'] = features['unique_chars'] * 0.57
                
                model = models[model_name]
                
                # Series'i DataFrame'e çevirelim ve sütunları düzenleyelim
                features_df = pd.DataFrame([features])
                features_df = features_df[model.feature_names_in_]  # Modelin beklediği sırada sütunları düzenle
                
                print(features_df)
                
                # Logistic Regression için scaling
                if model_name == "Logistic Regression":
                    features_df = scaler.transform(features_df)
                
                # Tahmin alalım
                y_pred = model.predict(features_df)[0]
                
                return f"{y_pred:.2f}"

            password_output = gr.Textbox(
                label="Tahmin Edilen Güç",
                interactive=False
            )
            
            password_input.change(
                fn=predict_password_strength,
                inputs=[model_dropdown, password_input, category_dropdown],
                outputs=password_output
            )
            
            category_dropdown.change(
                fn=predict_password_strength,
                inputs=[model_dropdown, password_input, category_dropdown],
                outputs=password_output
            )

if __name__ == "__main__":
    demo.launch() 