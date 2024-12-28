import gradio as gr
import pandas as pd
import numpy as np
from train import df, decision_tree, naive_bayes, logistic_regression, scaler
from utils import password_features
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

def get_top_passwords_by_category(category_name):
    # Kategoriyi encode edelim
    category_encoded = label_encoder.transform([category_name])[0]
    
    # Kategoriyi filtreleyelim
    category_df = df[df['category_encoded'] == category_encoded]
    
    # Şifreleri güce göre sıralayalım ve en güçlü 5 tanesini seçelim
    top_passwords = category_df.nlargest(5, 'strength')[['password', 'strength']]
    
    # Şifreleri liste içinde liste formatında döndürelim
    return [[row['password'], f"{row['strength']:.2f}"] for _, row in top_passwords.iterrows()]

def get_weakest_passwords_by_category(category_name):
    # Kategoriyi encode edelim
    category_encoded = label_encoder.transform([category_name])[0]
    
    # Kategoriyi filtreleyelim
    category_df = df[df['category_encoded'] == category_encoded]
    
    # Şifreleri güce göre sıralayalım ve en güçsüz 5 tanesini seçelim
    weakest_passwords = category_df.nsmallest(5, 'strength')[['password', 'strength']]
    
    # Şifreleri liste içinde liste formatında döndürelim
    return [[row['password'], f"{row['strength']:.2f}"] for _, row in weakest_passwords.iterrows()]

# Başlangıçta tabloları doldurmak için fonksiyonları çağır
initial_model = "Decision Tree"
initial_sample_count = 20
initial_category = categories[0]

# Başlangıç değerlerini hesapla
initial_results = predict_samples(initial_model, initial_sample_count)
initial_top_passwords = get_top_passwords_by_category(initial_category)
initial_weakest_passwords = get_weakest_passwords_by_category(initial_category)

# Gradio arayüzünü oluşturalım
with gr.Blocks(theme='allenai/gradio-theme') as demo:
    gr.Markdown("Şifre Güçlülük Tahmini")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=list(models.keys()),
                value=initial_model,
                label="Model Seçiniz"
            )
            
            password_input = gr.Textbox(
                label="Şifre Giriniz",
                placeholder="Tahmin edilecek şifreyi buraya girin"
            )
            
            category_dropdown = gr.Dropdown(
                choices=[f"{cat} ({category_examples[cat]})" for cat in categories],
                value=f"{initial_category} ({category_examples[initial_category]})",
                label="Kategori Seçiniz"
            )

            password_output = gr.Textbox(
                label="Tahmin Edilen Güç",
                interactive=False
            )

        with gr.Column():
            sample_count_dropdown = gr.Dropdown(
                choices=[10, 20, 30, 40, 50],  # Örnek sayısı seçenekleri
                value=initial_sample_count,
                label="Örnek Sayısı Seçiniz"
            )

            output_table = gr.Dataframe(
                value=initial_results,  # Başlangıçta initial_results ile dolu
                headers=["Password", "Strength Real", "Strength Guess"],
                row_count=20,
                col_count=3,
                interactive=False
            )
            
            top_passwords_output = gr.Dataframe(
                value=initial_top_passwords,  # Başlangıçta initial_top_passwords ile dolu
                headers=["Top Passwords", "Strength"],
                row_count=5,
                col_count=2,
                interactive=False
            )

            weakest_passwords_output = gr.Dataframe(
                value=initial_weakest_passwords,  # Başlangıçta initial_weakest_passwords ile dolu
                headers=["Weakest Passwords", "Strength"],
                row_count=5,
                col_count=2,
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
            
            def predict_password_strength(model_name, password, category):
                if not password or not category:
                    return "Lütfen şifre ve kategori seçiniz."
                
                # Kategori adını ayıklayalım
                category_name = category.split(' ')[0]
                category_encoded = label_encoder.transform([category_name])[0]
                
                # Dummy değerler
                rank = 5
                
                # Şifre özelliklerini hesaplayalım
                features = password_features(password)
                features['category_encoded'] = int(category_encoded)
                features['rank'] = rank
                features['offline_crack_sec'] = features['unique_chars'] * 0.57
                
                model = models[model_name]
                
                # Series'i DataFrame'e çevirelim ve sütunları düzenleyelim
                features_df = pd.DataFrame([features])
                features_df = features_df[model.feature_names_in_]  # Modelin beklediği sırada sütunları düzenle
                
                # Logistic Regression için scaling
                if model_name == "Logistic Regression":
                    features_df = scaler.transform(features_df)
                
                # Tahmin alalım
                y_pred = model.predict(features_df)[0]
                
                return f"{y_pred:.2f}"

            category_dropdown.change(
                fn=lambda model_name, password, category: (
                    predict_password_strength(model_name, password, category),
                    get_top_passwords_by_category(category.split(' ')[0]),
                    get_weakest_passwords_by_category(category.split(' ')[0])
                ),
                inputs=[model_dropdown, password_input, category_dropdown],
                outputs=[password_output, top_passwords_output, weakest_passwords_output]
            )

    # Başlangıç değerlerini ayarlayalım
    initial_results = predict_samples(initial_model, initial_sample_count)
    initial_top_passwords = get_top_passwords_by_category(initial_category)
    initial_weakest_passwords = get_weakest_passwords_by_category(initial_category)
    password_output.value = predict_password_strength(initial_model, "", f"{initial_category} ({category_examples[initial_category]})")
    output_table.value = initial_results
    top_passwords_output.value = initial_top_passwords
    weakest_passwords_output.value = initial_weakest_passwords

if __name__ == "__main__":
    demo.launch() 