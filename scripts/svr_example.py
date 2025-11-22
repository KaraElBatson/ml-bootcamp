"""
SVR Age Prediction - Kullanım Örneği
Bu script, SVR modelini çalıştırmak için basit bir örnek sağlar.
"""

from age_prediction_svr import main

if __name__ == "__main__":
    # Temel kullanım - train.csv ve test.csv dosyalarını data/raw/ klasöründen okur
    results = main(
        train_file="train.csv",
        test_file="test.csv",
        output_file="submission_svr.csv",
        save_models=True
    )
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("MODEL BİLGİLERİ")
    print("="*60)
    print(f"Doğrulama MAE: {results['validation_metrics']['mae']:.4f}")
    print(f"Doğrulama RMSE: {results['validation_metrics']['rmse']:.4f}")
    print(f"Doğrulama R²: {results['validation_metrics']['r2']:.4f}")
    print(f"\nKategorik özellik sayısı: {len(results['categorical_features'])}")
    print(f"Sayısal özellik sayısı: {len(results['numerical_features'])}")
    print(f"\nÇıktı dosyası: {results['output_file']}")
    print("="*60)

