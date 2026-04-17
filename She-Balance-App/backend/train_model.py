import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import os

def clean_and_train_model():
    # Load your dataset
    df = pd.read_csv('data/anemiadata_cleaned.csv')
    
    # Clean the data
    df = df.dropna()
    df = df[df['Symptom'].notna()]
    
    # Create symptom text for training
    df['symptom_text'] = df['Symptom'].str.lower()
    
    # Prepare features and labels
    X = df['symptom_text']
    y = df['Remedy (Siddha Term)']
    
    # Create vectorizer
    vectorizer = CountVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=100,
        stop_words='english'
    )
    
    # Transform symptoms to numerical features
    X_vectorized = vectorizer.fit_transform(X)
    
    # Encode remedy labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train the model
    model = MultinomialNB(alpha=0.1)
    model.fit(X_vectorized, y_encoded)
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save everything
    joblib.dump(model, 'model/siddha_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
    joblib.dump(df, 'model/dataset.pkl')
    
    print("✅ AI Model trained successfully!")
    print(f"📊 Dataset size: {len(df)} remedies")
    
    # Test prediction
    test_text = "tiredness pale skin"
    test_vector = vectorizer.transform([test_text])
    prediction = model.predict(test_vector)
    predicted_remedy = label_encoder.inverse_transform(prediction)[0]
    print(f"🎯 Test prediction: '{test_text}' → {predicted_remedy}")

if __name__ == "__main__":
    clean_and_train_model()