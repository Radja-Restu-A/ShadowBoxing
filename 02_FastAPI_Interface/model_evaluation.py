import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path):
    """
    Load model dari file .pkl
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model berhasil dimuat dari {model_path}")
        return model
    except FileNotFoundError:
        print(f"File {model_path} tidak ditemukan!")
        return None
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return None

def evaluate_model(model, X_test, y_test, target_names=None):
    """
    Evaluasi model dengan menghitung akurasi, F1 score, dan confusion matrix
    """
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    
    # F1 score - sesuaikan average berdasarkan jenis klasifikasi
    if len(np.unique(y_test)) == 2:
        f1 = f1_score(y_test, y_pred, average='binary')
    else:
        f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print hasil
    print("=" * 50)
    print("HASIL EVALUASI MODEL")
    print("=" * 50)
    print(f"Akurasi: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return accuracy, f1, cm

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix'):
    """
    Visualisasi confusion matrix
    """
    plt.figure(figsize=(8, 6))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def main():
    # Contoh penggunaan dengan data dummy
    # Ganti dengan data dan model Anda sendiri
    
    # 1. Load model
    model_path = '01_Notebook_eksplorasi/shadow_classifier.pkl'  # Ganti dengan path model Anda
    model = load_model(model_path)
    
    if model is None:
        print("Membuat contoh data dummy untuk demonstrasi...")
        # Jika model tidak ada, buat contoh dengan data dummy
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        
        # Generate data dummy
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                                n_informative=15, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model dummy
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("Model dummy berhasil dibuat dan dilatih")
    else:
        # 2. Load data test Anda
        # Contoh: Ganti dengan cara load data Anda
        print("Silakan load data test Anda di sini...")
        print("Contoh:")
        print("X_test = pd.read_csv('test_features.csv')")
        print("y_test = pd.read_csv('test_labels.csv')")
        
        # Untuk demonstrasi, gunakan data dummy jika model berhasil dimuat
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=500, n_features=20, n_classes=3, 
                                n_informative=15, random_state=42)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
    
    # 3. Evaluasi model
    class_names = ['Class 0', 'Class 1', 'Class 2']  # Sesuaikan dengan kelas Anda
    accuracy, f1, cm = evaluate_model(model, X_test, y_test, target_names=class_names)
    
    # 4. Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    # 5. Simpan hasil ke file (opsional)
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
    
    # Simpan ke JSON
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nHasil evaluasi disimpan ke 'evaluation_results.json'")

if __name__ == "__main__":
    main()