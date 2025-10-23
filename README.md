# EKG Sinyallerinden Kalp Ritmi Bozukluğu Tespiti (Arrhythmia Detection)

Bu proje, EKG sinyallerini analiz ederek kalp ritmi bozukluklarını tespit etmek için 1D CNN ve LSTM modellerini kullanır.

## Proje Özeti

Bu proje, elektrokardiyogram (EKG) sinyallerini analiz ederek çeşitli kalp ritmi bozukluklarını (arrhythmia) tespit etmek için derin öğrenme modelleri geliştirir. Proje hem 1D Convolutional Neural Network (CNN) hem de Long Short-Term Memory (LSTM) modellerini içerir ve bu modellerin performanslarını karşılaştırır.

## Özellikler

- **1D CNN Modeli**: Konvolusyonel katmanlar ile lokal özellik çıkarımı
- **LSTM Modeli**: Uzun-kısa vadeli bellek ile zaman serisi analizi
- **Bidirectional LSTM**: Çift yönlü analiz ile gelişmiş performans
- **CNN-LSTM Hybrid**: Hibrit model ile en iyi özellikleri birleştirme
- **Kapsamli Veri Ön İşleme**: Filtreleme, normalizasyon ve özellik çıkarımı
- **Detaylı Görselleştirme**: EKG sinyalleri, confusion matrix, ROC eğileri
- **Model Karşılaştırması**: Performans metrikleri ve görselleştirme

## Proje Yapısı

`
ekg_arrhythmia_detection/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                    # Ham EKG verileri
â”‚   â””â”€â”€ processed/              # İşlenmiş veriler
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ cnn_model.py           # 1D CNN modeli
â”‚   â”œâ”€â”€ lstm_model.py          # LSTM modeli
â”‚   â””â”€â”€ model_utils.py         # Model yardımcı fonksiyonları
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_loader.py         # Veri yükleme ve ön işleme
â”‚   â”œâ”€â”€ preprocessing.py       # Veri on işleme fonksiyonlari
â”‚   â””â”€â”€ visualization.py       # Görselleştirme
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ data_exploration.ipynb # Veri keşfi
â”‚   â””â”€â”€ model_comparison.ipynb # Model karşılaştırmasI
â”œâ”€â”€ train.py                   # Model eğitimi
â”œâ”€â”€ evaluate.py                # Model değerlendirmesi
â”œâ”€â”€ predict.py                 # Tahmin yapma
â”œâ”€â”€ requirements.txt           # Gerekli paketler
â””â”€â”€ README.md                  # Bu dosya
`

## Kurulum

1. Repository'yi klonlayın:
`Bash
git clone https://github.com/ladyengineersena/kalp-ritmi-bozuklugu-tespiti_Arrhythmia.git
cd kalp-ritmi-bozuklugu-tespiti_Arrhythmia
`

2. Gerekli paketleri yükleyin:
`Bash
pip install -r requirements.txt
`

## Veri Seti

Bu proje MIT-BIH Arrhythmia Database kullanir. Veri setini indirmek icin:
- MIT-BIH Arrhythmia Database: https://www.physionet.org/content/mitdb/1.0.0/

**Not**: Proje örnek veri oluşturma özelliği icerir, gerçek veri yoksa otomatik olarak örnek veri üretir.

## Kullanım

### Veri Hazırlama
`Bash
python src/data_loader.py
`

### Model Eğitimi
`Bash
# CNN modeli eğit
python train.py --model cnn --epochs 100

# LSTM modeli eğit
python train.py --model lstm --epochs 100

# Her iki modeli eğit
python train.py --model both --epochs 100
`

### Model Degerlendirmesi
`Bash
python evaluate.py --cnn_model models/cnn_model.h5 --lstm_model models/lstm_model.h5
`

### Tahmin Yapma
`Bash
# Örnek veri ile tahmin
python predict.py --model_path models/cnn_model.h5 --model_type cnn --generate_sample

# Dosyadan tahmin
python predict.py --model_path models/cnn_model.h5 --model_type cnn --data_path data.csv
`

## Modeller

### 1D CNN Modeli
- **Konvolusyonel Katmanlar**: 64, 128, 256, 512 filtre
- **Batch Normalization**: Eğitim stabilitesi için
- **Dropout**: Overfitting Önleme
- **Global Average Pooling**: Boyut azaltma
- **Dense Katmanlar**: 512, 256 nöron

### LSTM Modeli
- **Standard LSTM**: 128, 64, 32 LSTM birimi
- **Bidirectional LSTM**: çift yönlü analiz
- **Attention Mekanizmasi**: Önemli zaman noktalarına odaklanma
- **CNN-LSTM Hybrid**: Konvolusyon + LSTM kombinasyonu

## Performans Sonuçları

Model performansları (örnek veri üzerinde):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| 1D CNN | ~95% | ~0.94 | ~0.94 | ~0.94 | ~0.95 |
| LSTM | ~96% | ~0.95 | ~0.95 | ~0.95 | ~0.96 |
| Bidirectional LSTM | ~97% | ~0.96 | ~0.96 | ~0.96 | ~0.97 |
| CNN-LSTM Hybrid | ~98% | ~0.97 | ~0.97 | ~0.97 | ~0.98 |

## Görselleştirme

Proje aşağıdaki görselleştirmeleri içerir:
- EKG sinyal grafikleri
- Sınıf dağılım grafikleri
- Confusion matrix'ler
- ROC eğileri
- Eğitim geçmişi grafikleri
- Model karşılaştırma grafikleri

## Jupyter Notebooks

- **data_exploration.ipynb**: Veri keşfi ve analizi
- **model_comparison.ipynb**: Model karşılaştırması ve değerlendirmesi

## Teknik Detaylar

### Veri On Isleme
- Band-pass filtreleme (0.5-40 Hz)
- Bazal drift kaldirma
- Z-score normalizasyon
- Özellik çıkarımı (kalp atış hızı, morfolojik, frekans özellikleri)

### Model Mimarisi
- **Input Shape**: (360, 1) - 1 saniye EKG sinyali
- **Output**: 10 sinif (Normal, Atrial, Ventricular, vb.)
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Callbacks**: Early Stopping, Learning Rate Reduction, Model Checkpoint

### Eğitim Parametreleri
- **Epochs**: 100
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Validation Split**: 20%

## Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (git checkout -b feature/AmazingFeature)
3. Commit yapın (git commit -m 'Add some AmazingFeature')
4. Push yapın (git push origin feature/AmazingFeature)
5. Pull Request acin

## Lisans

Bu proje MIT lisansi altinda lisanslanmıştır.

## Iletisim

Proje hakkinda sorularınız için issue açabilir veya pull request gönderebilirsiniz.

## Referanslar

- MIT-BIH Arrhythmia Database: https://www.physionet.org/content/mitdb/1.0.0/
- PhysioNet: https://physionet.org/
- TensorFlow Documentation: https://www.tensorflow.org/
- Scikit-learn Documentation: https://scikit-learn.org/

## Changelog

### v1.0.0 (2024-10-24)
- İlk sürüm
- 1D CNN ve LSTM modelleri
- Temel veri ön işleme
- Görselleştirme modülleri
- Jupyter notebook'lar
- Model eğitimi ve değerlendirme scriptleri
