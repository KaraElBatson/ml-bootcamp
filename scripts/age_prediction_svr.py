"""
Age prediction script using Support Vector Regression
Entegre edilmiş SVR pipeline - config.py ile uyumlu
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sys
import warnings
import joblib

# Suppress warnings
warnings.filterwarnings("ignore")

# Import configuration
try:
    from config import (
        SVR_PARAMS, 
        RAW_DATA_DIR, 
        MODELS_DIR, 
        MODEL_CONFIG
    )
except ImportError:
    # Fallback if config not found
    SVR_PARAMS = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"}
    RAW_DATA_DIR = os.path.join("data", "raw")
    MODELS_DIR = os.path.join("assets", "models")
    MODEL_CONFIG = {"test_size": 0.2, "random_state": 42}


def load_data(train_file="train.csv", test_file="test.csv"):
    """
    Load training and test datasets from configured data directory
    
    Parameters:
    -----------
    train_file : str
        Training data filename
    test_file : str
        Test data filename
    
    Returns:
    --------
    train_df, test_df : pandas DataFrames
    """
    train_path = os.path.join(RAW_DATA_DIR, train_file)
    test_path = os.path.join(RAW_DATA_DIR, test_file)
    
    print(f"Loading data from:\n  Train: {train_path}\n  Test: {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✓ Training data shape: {train_df.shape}")
    print(f"✓ Test data shape: {test_df.shape}")
    
    return train_df, test_df


def preprocess_data(train_df, test_df, target_col="Age", id_col="id"):
    """
    Preprocess the data for SVR model
    
    Parameters:
    -----------
    train_df : pandas DataFrame
        Training dataset
    test_df : pandas DataFrame
        Test dataset
    target_col : str
        Name of target column
    id_col : str
        Name of ID column
    
    Returns:
    --------
    X_train_processed, X_test_processed : numpy arrays
        Transformed features
    y_train : pandas Series
        Target variable
    preprocessor : ColumnTransformer
        Fitted preprocessor
    categorical_cols : list
        List of categorical column names
    numerical_cols : list
        List of numerical column names
    """
    print("\n" + "="*60)
    print("VERİ ÖN İŞLEME")
    print("="*60)
    
    # Hedef değişken ve özellikler
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[id_col, target_col])
    X_test = test_df.drop(columns=[id_col])
    
    # Ortak özellikleri belirle
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"✓ Ortak özellik sayısı: {len(common_features)}")
    
    # Kategorik ve sayısal sütunları ayır
    categorical_cols = [col for col in common_features if train_df[col].dtype == 'object']
    numerical_cols = [col for col in common_features if train_df[col].dtype != 'object']
    
    print(f"✓ Kategorik özellikler: {len(categorical_cols)}")
    print(f"✓ Sayısal özellikler: {len(numerical_cols)}")
    
    # Ön işleme pipeline'ı oluştur
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Fit ve transform işlemleri
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"✓ İşlenmiş eğitim verisi boyutu: {X_train_processed.shape}")
    print(f"✓ İşlenmiş test verisi boyutu: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, preprocessor, categorical_cols, numerical_cols


def train_and_evaluate_svr(X_train, y_train, svr_params=None, validation_size=None, random_state=None):
    """
    Train and evaluate SVR model with validation set
    
    Parameters:
    -----------
    X_train : numpy array
        Training features
    y_train : pandas Series or numpy array
        Training target
    svr_params : dict
        SVR hyperparameters (uses config if None)
    validation_size : float
        Validation set size (uses config if None)
    random_state : int
        Random seed (uses config if None)
    
    Returns:
    --------
    svr_model : SVR
        Trained model
    metrics : dict
        Validation metrics
    """
    print("\n" + "="*60)
    print("MODEL EĞİTİMİ VE DOĞRULAMA")
    print("="*60)
    
    # Parametreleri ayarla
    if svr_params is None:
        svr_params = SVR_PARAMS
    if validation_size is None:
        validation_size = MODEL_CONFIG.get("test_size", 0.2)
    if random_state is None:
        random_state = MODEL_CONFIG.get("random_state", 42)
    
    print(f"SVR Parametreleri: {svr_params}")
    print(f"Doğrulama seti boyutu: {validation_size}")
    
    # Eğitim ve doğrulama setlerine ayır
    X_train_svr, X_val_svr, y_train_svr, y_val_svr = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=random_state
    )
    
    # SVR modeli oluştur
    svr_model = SVR(**svr_params)
    
    # Modeli eğit
    print("\nModel eğitiliyor...")
    svr_model.fit(X_train_svr, y_train_svr)
    print("✓ Eğitim tamamlandı!")
    
    # Doğrulama seti üzerinde tahmin yap
    y_pred_val = svr_model.predict(X_val_svr)
    
    # Metrikleri hesapla - HATA DÜZELTİLDİ: y_val_svr, y_pred_val kullanılmalı
    mae_val = mean_absolute_error(y_val_svr, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val_svr, y_pred_val))
    r2_val = r2_score(y_val_svr, y_pred_val)
    
    metrics = {
        "mae": mae_val,
        "rmse": rmse_val,
        "r2": r2_val
    }
    
    print("\n" + "="*60)
    print("DOĞRULAMA SETİ SONUÇLARI")
    print("="*60)
    print(f"MAE (Mean Absolute Error)     : {mae_val:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse_val:.4f}")
    print(f"R² Score                      : {r2_val:.4f}")
    print("="*60)
    
    return svr_model, metrics


def train_final_svr(X_train, y_train, svr_params=None):
    """
    Train final SVR model on all training data
    
    Parameters:
    -----------
    X_train : numpy array
        All training features
    y_train : pandas Series or numpy array
        All training target
    svr_params : dict
        SVR hyperparameters (uses config if None)
    
    Returns:
    --------
    final_svr_model : SVR
        Final trained model
    """
    print("\n" + "="*60)
    print("TÜM VERİ İLE NİHAİ MODEL EĞİTİMİ")
    print("="*60)
    
    if svr_params is None:
        svr_params = SVR_PARAMS
    
    final_svr_model = SVR(**svr_params)
    final_svr_model.fit(X_train, y_train)
    
    print("✓ Nihai model başarıyla eğitildi!")
    
    return final_svr_model


def make_predictions(model, X_test, test_ids, output_file="submission_svr.csv"):
    """
    Make predictions on test data and save to file
    
    Parameters:
    -----------
    model : SVR
        Trained model
    X_test : numpy array
        Test features
    test_ids : pandas Series or list
        Test IDs
    output_file : str
        Output filename
    """
    print("\n" + "="*60)
    print("TEST VERİSİ ÜZERİNDE TAHMİN")
    print("="*60)
    
    y_pred_test = model.predict(X_test)
    
    # Tahminleri en yakın tam sayıya yuvarla
    y_pred_test_rounded = y_pred_test.round().astype(int)
    
    # Submission dosyası oluştur
    submission_df = pd.DataFrame({'id': test_ids, 'Age': y_pred_test_rounded})
    submission_df.to_csv(output_file, index=False)
    
    print(f"✓ Tahminler '{output_file}' dosyasına kaydedildi.")
    print(f"✓ Toplam tahmin sayısı: {len(submission_df)}")
    print(f"✓ Ortalama tahmin edilen yaş: {y_pred_test_rounded.mean():.2f}")
    print(f"✓ Min yaş: {y_pred_test_rounded.min()}, Max yaş: {y_pred_test_rounded.max()}")


def save_model_and_preprocessor(model, preprocessor, model_filename="svr_model.pkl", 
                                 preprocessor_filename="svr_preprocessor.pkl"):
    """
    Save trained model and preprocessor to disk
    
    Parameters:
    -----------
    model : SVR
        Trained model
    preprocessor : ColumnTransformer
        Fitted preprocessor
    model_filename : str
        Model filename
    preprocessor_filename : str
        Preprocessor filename
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    preprocessor_path = os.path.join(MODELS_DIR, preprocessor_filename)
    
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"\n✓ Model kaydedildi: {model_path}")
    print(f"✓ Preprocessor kaydedildi: {preprocessor_path}")

def main(train_file="train.csv", test_file="test.csv", output_file="submission_svr.csv",
         save_models=True):
    """
    Main function to run the SVR age prediction pipeline
    
    Parameters:
    -----------
    train_file : str
        Training data filename (in data/raw/)
    test_file : str
        Test data filename (in data/raw/)
    output_file : str
        Output submission filename
    save_models : bool
        Whether to save model and preprocessor
    
    Returns:
    --------
    results : dict
        Dictionary containing model, preprocessor, and metrics
    """
    print("\n" + "="*60)
    print("YAŞ TAHMİNİ SVR PİPELINE")
    print("="*60)
    
    # Random seed ayarla
    random_state = MODEL_CONFIG.get("random_state", 42)
    np.random.seed(random_state)
    
    # 1. Veri Yükleme
    train_df, test_df = load_data(train_file, test_file)
    
    # 2. Veri Ön İşleme
    X_train_processed, X_test_processed, y_train, preprocessor, cat_cols, num_cols = preprocess_data(
        train_df, test_df
    )
    
    # 3. Model Eğitimi ve Doğrulama
    svr_model, validation_metrics = train_and_evaluate_svr(X_train_processed, y_train)
    
    # 4. Tüm Veri ile Nihai Model Eğitimi
    final_svr_model = train_final_svr(X_train_processed, y_train)
    
    # 5. Test Verisi Üzerinde Tahmin
    make_predictions(final_svr_model, X_test_processed, test_df['id'], output_file)
    
    # 6. Model ve Preprocessor'ı Kaydet
    if save_models:
        save_model_and_preprocessor(final_svr_model, preprocessor)
    
    print("\n" + "="*60)
    print("PİPELINE BAŞARIYLA TAMAMLANDI!")
    print("="*60)
    
    results = {
        "model": final_svr_model,
        "preprocessor": preprocessor,
        "validation_metrics": validation_metrics,
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "output_file": output_file
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SVR Yaş Tahmin Pipeline")
    parser.add_argument("--train", default="train.csv", help="Eğitim veri dosyası")
    parser.add_argument("--test", default="test.csv", help="Test veri dosyası")
    parser.add_argument("--output", default="submission_svr.csv", help="Çıktı dosyası")
    parser.add_argument("--no-save", action="store_true", help="Model kaydetme")
    
    args = parser.parse_args()
    
    # Pipeline'ı çalıştır
    results = main(
        train_file=args.train,
        test_file=args.test,
        output_file=args.output,
        save_models=not args.no_save
    )
    
    print("\n✓ Tüm işlemler tamamlandı!")