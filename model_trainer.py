"""
Hayvan Sınıflandırma Eğitim Modülü.

Bu modül, MobileNetV2 kullanarak transfer öğrenme ile hayvan görüntüsü sınıflandırması için 
tam bir eğitim hattı sağlar. İki aşamalı eğitimi destekler:
dondurulmuş temel model ile başlangıç başlığı eğitimi ve ardından ince ayar (fine-tuning).

Örnek:
    trainer = AnimalClassifierTrainer(data_dir="data/raw-img")
    trainer.train(epochs=20, fine_tune_epochs=10)
"""

import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AnimalClassifierTrainer:
    """Hayvan görüntüsü sınıflandırması için üretim kalitesinde eğitici sınıf.

    Bu sınıf, MobileNetV2'yi temel model olarak kullanan transfer öğrenmeyi uygular.
    İki aşamalı eğitim desteği sunar: önce sınıflandırma başlığının eğitimi,
    ardından temel modelin üst katmanlarının ince ayarı (fine-tuning).

    Özellikler:
        data_dir: Veri seti kök dizin yolu.
        image_size: Hedef görüntü boyutları (yükseklik, genişlik).
        batch_size: Eğitim grubu başına örnek sayısı.
        validation_split: Doğrulama için ayrılacak veri oranı.
        model: Derlenmiş Keras modeli.
        class_names: İndeks sırasına göre sınıf adları listesi.
        train_generator: Eğitim veri oluşturucu.
        val_generator: Doğrulama veri oluşturucu.

    Örnek:
        >>> trainer = AnimalClassifierTrainer(
        ...     data_dir="data/raw-img",
        ...     batch_size=32,
        ...     validation_split=0.2
        ... )
        >>> history = trainer.train(epochs=20, fine_tune_epochs=10)
        >>> trainer.save_model("animal_model.h5")
    """

    def __init__(
        self,
        data_dir: str = "data/raw-img",
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2,
        seed: int = 42,
    ) -> None:
        """AnimalClassifierTrainer'ı başlatır.

        Argümanlar:
            data_dir: Sınıf klasörlerini içeren veri seti kök dizin yolu.
            image_size: (yükseklik, genişlik) olarak hedef görüntü boyutları. Varsayılan (224, 224).
            batch_size: Grup başına örnek sayısı. Varsayılan 32.
            validation_split: Doğrulama için ayrılan veri oranı. Varsayılan 0.2.
            seed: Tekrar üretilebilirlik için rastgelelik tohumu. Varsayılan 42.

        Hatalar:
            FileNotFoundError: data_dir dizini mevcut değilse.
            ValueError: validation_split 0 ile 1 arasında değilse.
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Veri dizini bulunamadı: {data_dir}")
        if not 0 < validation_split < 1:
            raise ValueError("validation_split 0 ile 1 arasında olmalıdır")

        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed

        self._set_seeds()
        self.model: Optional[Model] = None
        self.class_names: List[str] = []
        self.train_generator = None
        self.val_generator = None

        logger.info("AnimalClassifierTrainer başlatıldı. Veri dizini: %s", data_dir)

    def _set_seeds(self) -> None:
        """Tüm kütüphaneler genelinde tekrar üretilebilirlik için rastgelelik tohumlarını ayarlar."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        logger.info("Tekrar üretilebilirlik için rastgelelik tohumları %d olarak ayarlandı", self.seed)

    def _create_data_generators(self) -> None:
        """Artırılmış (augmentation) eğitim ve doğrulama veri oluşturucularını oluşturur.

        Eğitici oluşturucu; döndürme, yakınlaştırma, kaydırma ve yatay çevirme gibi 
        veri artırma tekniklerini uygular. Her iki oluşturucu da MobileNetV2 
        ön işlemesini uygular.
        """
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=self.validation_split,
        )

        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=self.validation_split,
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
            seed=self.seed,
            shuffle=True,
        )

        self.val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            seed=self.seed,
            shuffle=False,
        )

        # Sınıf adlarını doğru dizin sırasıyla çıkar
        class_indices = self.train_generator.class_indices
        self.class_names = [None] * len(class_indices)
        for class_name, index in class_indices.items():
            self.class_names[index] = class_name

        logger.info("Veri oluşturucular oluşturuldu. Sınıflar: %s", self.class_names)
        logger.info(
            "Eğitim örnekleri: %d, Doğrulama örnekleri: %d",
            self.train_generator.samples,
            self.val_generator.samples,
        )

    def _build_model(self, dropout_rate: float = 0.5) -> None:
        """MobileNetV2 tabanlı transfer öğrenme modelini oluşturur.

        Argümanlar:
            dropout_rate: Düzenlileştirme (regularization) için dropout oranı. Varsayılan 0.5.
        """
        num_classes = len(self.class_names)

        # Temel model (dondurulmuş)
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(self.image_size[0], self.image_size[1], 3),
        )
        base_model.trainable = False

        # Daha sağlam bir yapı kullanarak sınıflandırma başlığını oluştur
        inputs = tf.keras.Input(shape=(self.image_size[0], self.image_size[1], 3))
        # BatchNorm katmanlarının çıkarım (inference) modunda kalmasını sağlamak için training=False geçiyoruz
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = Dropout(dropout_rate, name="dropout")(x)
        outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

        self.model = Model(inputs, outputs, name="animal_classifier")

        logger.info(
            "%d sınıf için model oluşturuldu. Temel model donduruldu.", num_classes
        )

    def _compile_model(self, learning_rate: float = 1e-3) -> None:
        """Modeli Adam optimize edici ve metriklerle derler.

        Argümanlar:
            learning_rate: Adam optimize edici için başlangıç öğrenme oranı.
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        logger.info("Model learning_rate=%.6f ile derlendi", learning_rate)

    def _get_callbacks(
        self,
        checkpoint_path: str,
        patience_early_stop: int = 5,
        patience_reduce_lr: int = 3,
    ) -> List[tf.keras.callbacks.Callback]:
        """Erken durdurma, LR azaltma ve kontrol noktası kaydı için geri çağırmaları (callbacks) oluşturur.

        Argümanlar:
            checkpoint_path: En iyi model kontrol noktasının kaydedileceği yol.
            patience_early_stop: Erken durdurma öncesi beklenecek epoch sayısı. Varsayılan 5.
            patience_reduce_lr: LR düşürme öncesi beklenecek epoch sayısı. Varsayılan 3.

        Dönüş:
            Yapılandırılmış Keras geri çağırmaları listesi.
        """
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience_early_stop,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=patience_reduce_lr,
                min_lr=1e-7,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]
        return callbacks

    def _unfreeze_layers(self, num_layers: int) -> None:
        """İnce ayar için temel modelin son N katmanının dondurulmasını kaldırır.

        Argümanlar:
            num_layers: Temel modelin sonundan itibaren dondurulması kaldırılacak katman sayısı.
        """
        # Güncellenmiş _build_model fonksiyonumuzda temel model 1. indeksteki katmandır
        base_model = self.model.layers[1]
        base_model.trainable = True

        # Temel modelin son N katmanı hariç tüm katmanlarını dondur
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
        
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True

        trainable_count = sum(
            1 for layer in base_model.layers if layer.trainable
        )
        logger.info(
            "Temel modelin son %d katmanının dondurulması kaldırıldı. Temeldeki toplam eğitilebilir katman: %d",
            num_layers,
            trainable_count,
        )

    def train(
        self,
        epochs: int = 20,
        fine_tune_epochs: int = 10,
        fine_tune_layers: int = 30,
        initial_lr: float = 1e-3,
        fine_tune_lr: float = 1e-5,
        dropout_rate: float = 0.5,
        checkpoint_path: str = "animal_model.h5",
    ) -> Dict[str, List[float]]:
        """Tam iki aşamalı eğitim hattını yürütür.

        Aşama 1: Sadece dondurulmuş tabanlı sınıflandırma başlığını eğit.
        Aşama 2: Temel modelin son N katmanında ince ayar yap.

        Argümanlar:
            epochs: Başlangıç eğitimi için epoch sayısı. Varsayılan 20.
            fine_tune_epochs: İnce ayar için epoch sayısı. Varsayılan 10.
            fine_tune_layers: Dondurulması kaldırılacak temel model katman sayısı. Varsayılan 30.
            initial_lr: 1. aşama için öğrenme oranı. Varsayılan 1e-3.
            fine_tune_lr: 2. aşama için öğrenme oranı. Varsayılan 1e-5.
            dropout_rate: Sınıflandırma başlığındaki dropout oranı. Varsayılan 0.5.
            checkpoint_path: En iyi modelin kaydedileceği yol. Varsayılan "animal_model.h5".

        Dönüş:
            Her iki aşamadan gelen eğitim geçmişini içeren sözlük.

        Hatalar:
            RuntimeError: Veri oluşturucular başlatılamazsa.
        """
        logger.info("=" * 60)
        logger.info("Eğitim hattı başlatılıyor")
        logger.info("=" * 60)

        # Adım 1: Veriyi hazırla
        self._create_data_generators()
        if not self.train_generator or not self.val_generator:
            raise RuntimeError("Veri oluşturucular oluşturulamadı")

        # Adım 2: Modeli oluştur ve derle
        self._build_model(dropout_rate=dropout_rate)
        self._compile_model(learning_rate=initial_lr)
        self.model.summary(print_fn=logger.info)

        callbacks = self._get_callbacks(checkpoint_path)

        # Aşama 1: Sınıflandırma başlığını eğit
        logger.info("-" * 40)
        logger.info("Aşama 1: Sınıflandırma başlığı eğitiliyor")
        logger.info("-" * 40)

        history_phase1 = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1,
        )

        # Aşama 2: İnce ayar (Fine-tuning)
        logger.info("-" * 40)
        logger.info("Aşama 2: Son %d katmanda ince ayar yapılıyor", fine_tune_layers)
        logger.info("-" * 40)

        self._unfreeze_layers(fine_tune_layers)
        self._compile_model(learning_rate=fine_tune_lr)

        history_phase2 = self.model.fit(
            self.train_generator,
            epochs=fine_tune_epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1,
        )

        # Geçmişleri birleştir
        combined_history = {
            key: history_phase1.history[key] + history_phase2.history[key]
            for key in history_phase1.history.keys()
        }

        logger.info("=" * 60)
        logger.info("Eğitim tamamlandı!")
        logger.info("=" * 60)

        return combined_history

    def save_model(self, model_path: str = "animal_model.keras") -> None:
        """Eğitilmiş modeli ve sınıf adlarını diske kaydeder.

        Argümanlar:
            model_path: Model dosyasının kaydedileceği yol. Varsayılan "animal_model.keras".

        Hatalar:
            ValueError: Model henüz eğitilmemişse.
        """
        if self.model is None:
            raise ValueError("Kaydedilecek model yok. Önce modeli eğitin.")

        # Modeli kaydet (Önerilen .keras formatı)
        self.model.save(model_path)
        logger.info("Model şuraya kaydedildi: %s", model_path)

        # Sınıf adlarını modelin yanına JSON olarak kaydet
        model_dir = os.path.dirname(model_path) or "."
        class_names_path = os.path.join(model_dir, "class_names.json")

        with open(class_names_path, "w", encoding="utf-8") as f:
            json.dump({"class_names": self.class_names}, f, indent=2)

        logger.info("Sınıf adları şuraya kaydedildi: %s", class_names_path)

    def evaluate(self) -> Dict[str, float]:
        """Modeli doğrulama seti üzerinde değerlendirir.

        Dönüş:
            Metrik adlarını değerleriyle eşleştiren sözlük.

        Hatalar:
            ValueError: Model veya doğrulama oluşturucu mevcut değilse.
        """
        if self.model is None or self.val_generator is None:
            raise ValueError("Model ve doğrulama verileri mevcut olmalıdır")

        results = self.model.evaluate(self.val_generator, verbose=1)
        metrics = dict(zip(self.model.metrics_names, results))

        logger.info("Değerlendirme sonuçları: %s", metrics)
        return metrics


def main() -> None:
    """Hayvan sınıflandırıcıyı eğitmek için ana giriş noktası."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transfer öğrenme kullanarak bir hayvan görüntüsü sınıflandırıcı eğitin."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw-img",
        help="Veri seti kök dizin yolu",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Başlangıç eğitimi için epoch sayısı",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=10,
        help="İnce ayar için epoch sayısı",
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=30,
        help="Dondurulması kaldırılacak temel model katman sayısı",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Eğitim grubu boyutu",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="animal_model.keras",
        help="Kaydedilen model için yol",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Tekrar üretilebilirlik için rastgelelik tohumu",
    )

    args = parser.parse_args()

    try:
        trainer = AnimalClassifierTrainer(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        trainer.train(
            epochs=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune_layers=args.fine_tune_layers,
            checkpoint_path=args.output,
        )

        trainer.save_model(args.output)
        trainer.evaluate()

    except FileNotFoundError as e:
        logger.error("Veri seti bulunamadı: %s", e)
        raise
    except Exception as e:
        logger.exception("Eğitim başarısız oldu: %s", e)
        raise


if __name__ == "__main__":
    main()
