# 🦁 Animals-10 Image Classification System

Bu proje, 10 farklı hayvan türünü (köpek, at, fil, kelebek, tavuk, kedi, inek, koyun, örümcek, sincap) yüksek doğrulukla sınıflandırmak için tasarlanmış, üretime hazır (production-ready) bir derin öğrenme sistemidir.

## 🚀 Proje Genel Bakışı

Sistem, **Transfer Learning** (Transferli Öğrenme) tekniğini kullanarak **MobileNetV2** mimarisi üzerine inşa edilmiştir. Google'ın MobileNetV2 modeli, özellikle mobil ve web tabanlı uygulamalar için optimize edilmiş, düşük gecikmeli ve yüksek performanslı bir modeldir.

### Ana Bileşenler:
1.  **Model Trainer (`model_trainer.py`):** Modeli eğiten ve hiperparametreleri yöneten modüler OOP yapısı.
2.  **Image Processor (`image_processor.py`):** Görüntüleri hem eğitim hem de tahmin (inference) süreci için hazırlayan yardımcı araçlar.
3.  **Streamlit UI (`app.py`):** Kullanıcı dostu, modern ve hızlı bir web arayüzü.

---

## 🛠️ Teknik Özellikler

-   **Model Mimarisi:** MobileNetV2 (include_top=False) + GAP + Dropout + Dense(Softmax).
-   **Giriş Boyutu:** 224x224x3 (RGB).
-   **Derleme:** Adam Optimizer + Categorical Crossentropy.
-   **Eğitim Stratejisi:** 
    -   *Faz 1:* Temel model dondurularak sadece yeni eklenen sınıflandırma katmanları eğitilir.
    -   *Faz 2 (Fine-tuning):* Son N katman çözülerek düşük öğrenme hızıyla (fine-tune) doğruluk artırılır.
-   **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
-   **Kütüphaneler:** TensorFlow 2.x, Streamlit, NumPy, Pillow.

---

## 💻 Kurulum

Öncelikle gerekli kütüphaneleri yükleyin:

```bash
pip install tensorflow streamlit numpy pillow
```

*Not: Eğer `google.protobuf` hatası alırsanız şu komutu kullanın:*
```bash
pip install --upgrade protobuf
```

---

## 📂 Dosya Yapısı

```text
.
├── data/raw-img/          # Hayvan sınıflarına göre ayrılmış klasörler
├── model_trainer.py       # Eğitim motoru (OOP)
├── image_processor.py     # Görüntü işleme yardımcıları
├── app.py                 # Streamlit arayüzü
├── animal_model.h5        # Kaydedilmiş model (Eğitim sonrası oluşur)
├── class_names.json       # Etiket eşleme dosyası (Eğitim sonrası oluşur)
└── README.md              # Dokümantasyon
```

---

## 📖 Kullanım Kılavuzu

### 1. Modelin Eğitilmesi

Dataset'iniz `data/raw-img` altında klasörler halinde hazırsa eğitimi şu komutla başlatabilirsiniz:

```bash
python model_trainer.py --data-dir "data/raw-img" --epochs 20 --fine-tune-epochs 10
```

**Argümanlar:**
- `--data-dir`: Veri setinin yolu.
- `--epochs`: İlk aşama eğitim tur sayısı.
- `--fine-tune-epochs`: İnce ayar tur sayısı.
- `--batch-size`: Paket boyutu (Varsayılan: 32).

### 2. Arayüzün Başlatılması

Eğitilen modeli test etmek veya kullanmak için web arayüzünü açın:

```bash
streamlit run app.py
```

**Arayüz Özellikleri:**
- Görsel yükleme (Drag & Drop).
- Tahminleme (Top-3 tahmin ve güven oranları).
- **Fallback Desteği:** Eğer kendi modeliniz henüz eğitilmemişse, sistem otomatik olarak genel ImageNet modelini yükleyerek sistemi çalışır halde tutar.

---

## 🧪 Model Performansı ve İzleme

Eğitim sırasında `model_trainer.py` otomatik olarak şunları yapar:
- Veriyi %20 oranında (veya belirtilen oranda) otomatik olarak eğitim/doğrulama diye böler.
- Data Augmentation (Döndürme, Yakınlaştırma, Kaydırma) uygulayarak modelin ezberlemesini (overfitting) engeller.
- En iyi modeli `animal_model.h5` olarak kaydeder.
- Sınıf isimlerini bir `.json` dosyasında saklar, böylece tahmin sürecinde hatalı etiketleme riskini ortadan kaldırır.

---

## 📝 Önemli Notlar

- **Hız:** MobileNetV2 hafif bir model olduğu için CPU üzerinde bile makul sürelerde tahmin yapabilir.
- **Doğruluk:** Daha iyi sonuçlar için veri setindeki her hayvan türü için en az 200+ kaliteli fotoğraf bulunması önerilir.
- **Genişletilebilirlik:** Yeni bir hayvan türü eklemek için `data/raw-img` içine yeni bir klasör açıp fotoğrafları eklemeniz ve eğitimi tekrar çalıştırmanız yeterlidir.

---
*Bu sistem Senior ML standartlarında temiz kod ve OOP prensipleriyle geliştirilmiştir.*
