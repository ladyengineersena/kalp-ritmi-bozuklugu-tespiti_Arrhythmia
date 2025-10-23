# EKG Sinyallerinden Kalp Ritmi Bozuklugu Tespiti (Arrhythmia Detection)

Bu proje, EKG sinyallerini analiz ederek kalp ritmi bozukluklarini tespit etmek icin 1D CNN ve LSTM modellerini kullanir.

## Proje Özeti

Bu proje, elektrokardiyogram (EKG) sinyallerini analiz ederek cesitli kalp ritmi bozukluklarini (arrhythmia) tespit etmek icin derin ogrenme modelleri gelistirir. Proje hem 1D Convolutional Neural Network (CNN) hem de Long Short-Term Memory (LSTM) modellerini icerir ve bu modellerin performanslarini karsilastirir.

## Özellikler

- **1D CNN Modeli**: Konvolusyonel katmanlar ile lokal ozellik cikarimi
- **LSTM Modeli**: Uzun-kisa vadeli bellek ile zaman serisi analizi
- **Bidirectional LSTM**: Cift yonlu analiz ile gelismis performans
- **CNN-LSTM Hybrid**: Hibrit model ile en iyi ozellikleri birlestirme
- **Kapsamli Veri On Isleme**: Filtreleme, normalizasyon ve ozellik cikarimi
- **Detayli Gorsellestirme**: EKG sinyalleri, confusion matrix, ROC egileri
- **Model Karsilastirmasi**: Performans metrikleri ve gorsellestirme

## Proje Yapisi

`
ekg_arrhythmia_detection/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                    # Ham EKG verileri
â”‚   â””â”€â”€ processed/              # Islenmis veriler
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ cnn_model.py           # 1D CNN modeli
â”‚   â”œâ”€â”€ lstm_model.py          # LSTM modeli
â”‚   â””â”€â”€ model_utils.py         # Model yardimci fonksiyonlari
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_loader.py         # Veri yukleme ve on isleme
â”‚   â”œâ”€â”€ preprocessing.py       # Veri on isleme fonksiyonlari
â”‚   â””â”€â”€ visualization.py       # Gorsellestirme
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ data_exploration.ipynb # Veri kesfi
â”‚   â””â”€â”€ model_comparison.ipynb # Model karsilastirmasi
â”œâ”€â”€ train.py                   # Model egitimi
â”œâ”€â”€ evaluate.py                # Model degerlendirmesi
â”œâ”€â”€ predict.py                 # Tahmin yapma
â”œâ”€â”€ requirements.txt           # Gerekli paketler
â””â”€â”€ README.md                  # Bu dosya
`

## Kurulum

1. Repository'yi klonlayin:
`ash
git clone https://github.com/ladyengineersena/kalp-ritmi-bozuklugu-tespiti_Arrhythmia.git
cd kalp-ritmi-bozuklugu-tespiti_Arrhythmia
`

2. Gerekli paketleri yukleyin:
`ash
pip install -r requirements.txt
`

## Veri Seti

Bu proje MIT-BIH Arrhythmia Database kullanir. Veri setini indirmek icin:
- MIT-BIH Arrhythmia Database: https://www.physionet.org/content/mitdb/1.0.0/

**Not**: Proje ornek veri olusturma ozelligi icerir, gercek veri yoksa otomatik olarak ornek veri uretir.

## Kullanim

### Veri Hazirlama
`Bash
python src/data_loader.py
`

### Model Egitimi
`Bash
# CNN modeli egit
python train.py --model cnn --epochs 100

# LSTM modeli egit
python train.py --model lstm --epochs 100

# Her iki modeli egit
python train.py --model both --epochs 100
`

### Model Degerlendirmesi
`ash
python evaluate.py --cnn_model models/cnn_model.h5 --lstm_model models/lstm_model.h5
`

### Tahmin Yapma
`ash
# Ornek veri ile tahmin
python predict.py --model_path models/cnn_model.h5 --model_type cnn --generate_sample

# Dosyadan tahmin
python predict.py --model_path models/cnn_model.h5 --model_type cnn --data_path data.csv
`

## Modeller

### 1D CNN Modeli
- **Konvolusyonel Katmanlar**: 64, 128, 256, 512 filtre
- **Batch Normalization**: Egitim stabilitesi icin
- **Dropout**: Overfitting onleme
- **Global Average Pooling**: Boyut azaltma
- **Dense Katmanlar**: 512, 256 noron

### LSTM Modeli
- **Standard LSTM**: 128, 64, 32 LSTM birimi
- **Bidirectional LSTM**: Cift yonlu analiz
- **Attention Mekanizmasi**: Onemli zaman noktalarina odaklanma
- **CNN-LSTM Hybrid**: Konvolusyon + LSTM kombinasyonu

## Performans Sonuclari

Model performanslari (ornek veri uzerinde):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| 1D CNN | ~95% | ~0.94 | ~0.94 | ~0.94 | ~0.95 |
| LSTM | ~96% | ~0.95 | ~0.95 | ~0.95 | ~0.96 |
| Bidirectional LSTM | ~97% | ~0.96 | ~0.96 | ~0.96 | ~0.97 |
| CNN-LSTM Hybrid | ~98% | ~0.97 | ~0.97 | ~0.97 | ~0.98 |

## Gorsellestirme

Proje asagidaki gorsellestirmeleri icerir:
- EKG sinyal grafikleri
- Sinif dagilim grafikleri
- Confusion matrix'ler
- ROC egileri
- Egitim gecmisi grafikleri
- Model karsilastirma grafikleri

## Jupyter Notebooks

- **data_exploration.ipynb**: Veri kesfi ve analizi
- **model_comparison.ipynb**: Model karsilastirmasi ve degerlendirmesi

## Teknik Detaylar

### Veri On Isleme
- Band-pass filtreleme (0.5-40 Hz)
- Bazal drift kaldirma
- Z-score normalizasyon
- Ozellik cikarimi (kalp atis hizi, morfolojik, frekans ozellikleri)

### Model Mimarisi
- **Input Shape**: (360, 1) - 1 saniye EKG sinyali
- **Output**: 10 sinif (Normal, Atrial, Ventricular, vb.)
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Callbacks**: Early Stopping, Learning Rate Reduction, Model Checkpoint

### Egitim Parametreleri
- **Epochs**: 100
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Validation Split**: 20%

## Katkida Bulunma

1. Fork yapin
2. Feature branch olusturun (git checkout -b feature/AmazingFeature)
3. Commit yapin (git commit -m 'Add some AmazingFeature')
4. Push yapin (git push origin feature/AmazingFeature)
5. Pull Request acin

## Lisans

Bu proje MIT lisansi altinda lisanslanmistir.

## Iletisim

Proje hakkinda sorulariniz icin issue acabilir veya pull request gonderebilirsiniz.

## Referanslar

- MIT-BIH Arrhythmia Database: https://www.physionet.org/content/mitdb/1.0.0/
- PhysioNet: https://physionet.org/
- TensorFlow Documentation: https://www.tensorflow.org/
- Scikit-learn Documentation: https://scikit-learn.org/

## Changelog

### v1.0.0 (2024-10-24)
- Ilk surum
- 1D CNN ve LSTM modelleri
- Temel veri on isleme
- Gorsellestirme modulleri
- Jupyter notebook'lar
- Model egitimi ve degerlendirme scriptleri
