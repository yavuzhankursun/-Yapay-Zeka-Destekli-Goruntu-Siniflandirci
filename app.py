"""
Hayvan Görüntü Sınıflandırması İçin Streamlit Uygulaması.

Bu modül; eğitilmiş bir MobileNetV2 transfer öğrenme modeli kullanarak hayvan 
görüntülerini sınıflandırmak için modern ve kullanıcı dostu bir web arayüzü sağlar.

Çalıştırmak için: streamlit run app.py

Örnek:
    $ streamlit run app.py --server.port 8501
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

from image_processor import ImageUtils


class ClassificationUI:
    """Hayvan görüntüsü sınıflandırması için Streamlit kullanıcı arayüzü.

    Bu sınıf; model yükleme (yedekleme desteğiyle), görüntü yükleme yönetimi ve 
    tahmin görselleştirmesini sağlayarak tüm Streamlit uygulamasını kapsüller.

    Özellikler:
        image_utils: Ön işleme için ImageUtils örneği.
        model: Çıkarım için yüklenen Keras modeli.
        class_names: Sınıf adları listesi veya ImageNet yedeklemesi kullanılıyorsa None.
        is_imagenet_fallback: Yedek olarak ImageNet modelinin kullanılıp kullanılmadığı.
        model_path: Animals-10 model dosyasının yolu.
        class_names_path: Sınıf adları JSON dosyasının yolu.

    Örnek:
        >>> app = ClassificationUI()
        >>> app.run()
    """

    def __init__(
        self,
        model_path: str = "animal_model.keras",
        class_names_path: str = "class_names.json",
    ) -> None:
        """ClassificationUI uygulamasını başlatır.

        Argümanlar:
            model_path: Eğitilmiş model dosyasının yolu. Varsayılan "animal_model.h5".
            class_names_path: Sınıf adları JSON dosyasının yolu. Varsayılan "class_names.json".
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.image_utils = ImageUtils(target_size=(224, 224))
        self.model: Optional[tf.keras.Model] = None
        self.class_names: Optional[List[str]] = None
        self.is_imagenet_fallback: bool = False

    def _configure_page(self) -> None:
        """Streamlit sayfa ayarlarını ve stilini yapılandırır."""
        st.set_page_config(
            page_title="Hayvan Sınıflandırıcı",
            page_icon="🦁",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Modern stil için özel CSS
        st.markdown(
            """
            <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                padding: 1rem 0;
            }
            .sub-header {
                font-size: 1.2rem;
                color: #6c757d;
                text-align: center;
                margin-bottom: 2rem;
            }
            .prediction-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                padding: 1.5rem;
                color: white;
                text-align: center;
                margin: 1rem 0;
            }
            .confidence-bar {
                background: rgba(255,255,255,0.2);
                border-radius: 10px;
                height: 25px;
                margin: 0.5rem 0;
            }
            .stButton>button {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 0.75rem 2rem;
                font-weight: 600;
                width: 100%;
                transition: transform 0.2s;
            }
            .stButton>button:hover {
                transform: scale(1.02);
            }
            .info-box {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 1rem;
                border-radius: 0 10px 10px 0;
                margin: 1rem 0;
            }
            .warning-box {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 1rem;
                border-radius: 0 10px 10px 0;
                margin: 1rem 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def _render_sidebar(self) -> None:
        """Proje bilgilerini içeren yan paneli oluşturur."""
        with st.sidebar:
            st.image(
                "https://img.icons8.com/fluency/96/zoo.png",
                width=80,
            )
            st.title("🦁 Hayvan Sınıflandırıcı")
            st.markdown("---")

            st.subheader("📖 Hakkında")
            st.markdown(
                """
                Bu uygulama, MobileNetV2 ile transfer öğrenme tekniklerini kullanarak 
                hayvan görüntülerini 10 kategoriye ayırmak için **derin öğrenme** kullanır.
                """
            )

            st.subheader("🚀 Nasıl Kullanılır")
            st.markdown(
                """
                1. Bir görüntü yükleyin (JPG, PNG, JPEG)
                2. **Görüntüyü Sınıflandır** butonuna tıklayın
                3. Güven skorlarıyla birlikte tahminleri görüntüleyin
                """
            )

            st.subheader("🔧 Model Bilgisi")
            if self.is_imagenet_fallback:
                st.warning("⚠️ ImageNet demo modeli kullanılıyor")
                st.markdown("Kendi modelinizi `model_trainer.py` ile eğitin")
            else:
                st.success("✅ Animals-10 modeli yüklendi")
                if self.class_names:
                    with st.expander("Sınıflar"):
                        for i, name in enumerate(self.class_names):
                            st.text(f"{i}: {name}")

            st.markdown("---")
            st.subheader("💻 Eğitimi Çalıştır")
            st.code(
                "python model_trainer.py --data-dir data/raw-img",
                language="bash",
            )

            st.markdown("---")
            st.caption("TensorFlow ve Streamlit ile ❤️ kullanılarak yapıldı")

    @st.cache_resource
    def _load_model(_self) -> Tuple[tf.keras.Model, Optional[List[str]], bool]:
        """Sınıflandırma modelini yedekleme desteğiyle yükler.

        Eğitilmiş Animals-10 modelini yüklemeyi dener. Bulunamazsa,
        ImageNet ağırlıklarına sahip standart MobileNetV2'ye geri döner.

        Dönüş:
            Şunları içeren bir demet:
                - model: Yüklenen Keras modeli
                - class_names: Sınıf adları listesi veya ImageNet için None
                - is_imagenet: ImageNet yedeği kullanılıp kullanılmadığını belirten boolean
        """
        class_names = None
        is_imagenet = False

        # Animals-10 modelini yüklemeyi dene
        if os.path.exists(_self.model_path):
            try:
                model = tf.keras.models.load_model(_self.model_path)
                st.sidebar.success(f"✅ Yüklendi: {_self.model_path}")

                # Sınıf adlarını yüklemeyi dene
                if os.path.exists(_self.class_names_path):
                    with open(_self.class_names_path, "r") as f:
                        data = json.load(f)
                        class_names = data.get("class_names", None)
                    st.sidebar.info(f"📋 Sınıflar {_self.class_names_path} dosyasından yüklendi")
                else:
                    st.sidebar.warning(
                        f"⚠️ {_self.class_names_path} bulunamadı. Ham indeksler kullanılıyor."
                    )

                return model, class_names, False

            except Exception as e:
                st.sidebar.error(f"❌ Model yükleme hatası: {e}")

        # ImageNet MobileNetV2'ye geri dön
        st.sidebar.warning("⚠️ Animals-10 modeli bulunamadı. ImageNet demosu kullanılıyor.")
        model = MobileNetV2(weights="imagenet", include_top=True)
        return model, None, True

    def _initialize_model(self) -> None:
        """Modeli ve sınıf adlarını önbelleğe alınmış yükleyiciden başlatır."""
        self.model, self.class_names, self.is_imagenet_fallback = self._load_model()

    def _predict(self, model_input: np.ndarray) -> List[Tuple[str, float]]:
        """Çıkarımı çalıştırır ve sıralı tahminleri döndürür.

        Argümanlar:
            model_input: (1, 224, 224, 3) şeklinde önceden işlenmiş görüntü dizisi.

        Dönüş:
            Güven skoruna göre azalan sırada sıralanmış (sınıf_adı, güven) demetleri listesi.
        """
        predictions = self.model.predict(model_input, verbose=0)

        if self.is_imagenet_fallback:
            # ImageNet tahminlerinin kodunu çöz
            decoded = decode_predictions(predictions, top=5)[0]
            return [(name, float(conf)) for (_, name, conf) in decoded]
        else:
            # Animals-10 tahminlerini işle
            probs = predictions[0]
            results = []

            for idx, prob in enumerate(probs):
                if self.class_names and idx < len(self.class_names):
                    name = self.class_names[idx]
                else:
                    name = f"Sınıf {idx}"
                results.append((name, float(prob)))

            # Güven skoruna göre azalan şekilde sırala
            results.sort(key=lambda x: x[1], reverse=True)
            return results

    def _render_predictions(self, predictions: List[Tuple[str, float]], top_k: int = 3) -> None:
        """Tahmin sonuçlarını görsel güven çubuklarıyla oluşturur.

        Argümanlar:
            predictions: (sınıf_adı, güven) demetleri listesi.
            top_k: Görüntülenecek en iyi tahmin sayısı. Varsayılan 3.
        """
        top_predictions = predictions[:top_k]

        # Başlık
        model_type = "ImageNet Demosu" if self.is_imagenet_fallback else "Animals-10"
        st.markdown(f"### 🎯 Tahminler ({model_type})")

        if self.is_imagenet_fallback:
            st.info(
                "ℹ️ **Demo Modu**: ImageNet modeli kullanılıyor. "
                "Hayvana özgü tahminler için kendi Animals-10 modelinizi eğitin."
            )

        # En iyi tahmin vurgusu
        top_class, top_conf = top_predictions[0]
        st.markdown(
            f"""
            <div class="prediction-card">
                <h2 style="margin:0;">🏆 {top_class.replace('_', ' ').title()}</h2>
                <p style="font-size:1.5rem; margin:0.5rem 0;">%{top_conf*100:.1f} Güven Skoru</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tüm en iyi k tahminler
        st.markdown("#### En İyi Tahminler")

        for class_name, confidence in top_predictions:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(confidence, text=class_name.replace("_", " ").title())
            with col2:
                st.metric(label="", value=f"%{confidence*100:.1f}")

        # Detaylı tablo
        with st.expander("📊 Tüm Skorlar", expanded=False):
            st.dataframe(
                {
                    "Sınıf": [p[0].replace("_", " ").title() for p in predictions],
                    "Güven": [f"%{p[1]*100:.2f}" for p in predictions],
                    "Ham Skor": [f"{p[1]:.6f}" for p in predictions],
                },
                use_container_width=True,
            )

    def _render_main_content(self) -> None:
        """Ana uygulama içerik alanını oluşturur."""
        # Başlık
        st.markdown(
            '<h1 class="main-header">🦁 Hayvan Görüntü Sınıflandırıcı</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sub-header">Bir hayvan resmi yükleyin ve yapay zekanın onu tanımlamasına izin verin!</p>',
            unsafe_allow_html=True,
        )

        # İki sütunlu düzen
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("### 📤 Görüntü Yükle")

            uploaded_file = st.file_uploader(
                "Bir görüntü dosyası seçin",
                type=["jpg", "jpeg", "png"],
                help="Desteklenen formatlar: JPG, JPEG, PNG",
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                try:
                    # Görüntüyü işle
                    display_image, model_input = self.image_utils.process_uploaded_file(
                        uploaded_file
                    )

                    # Yüklenen görüntüyü göster
                    st.image(
                        display_image,
                        caption=f"📷 {uploaded_file.name}",
                        use_container_width=True,
                    )

                    # Tahmin için oturum durumuna (session state) kaydet
                    st.session_state["display_image"] = display_image
                    st.session_state["model_input"] = model_input
                    st.session_state["uploaded"] = True

                except Exception as e:
                    st.error(f"❌ Görüntü işleme hatası: {e}")
                    st.session_state["uploaded"] = False
            else:
                st.info("👆 Başlamak için bir görüntü yükleyin")
                st.session_state["uploaded"] = False

        with col2:
            st.markdown("### 🔮 Sınıflandırma Sonuçları")

            if st.session_state.get("uploaded", False):
                # Sınıflandır butonu
                if st.button("🚀 Görüntüyü Sınıflandır", type="primary"):
                    with st.spinner("Görüntü analiz ediliyor..."):
                        try:
                            model_input = st.session_state["model_input"]
                            predictions = self._predict(model_input)
                            st.session_state["predictions"] = predictions
                        except Exception as e:
                            st.error(f"❌ Tahmin başarısız: {e}")

                # Varsa tahminleri göster
                if "predictions" in st.session_state:
                    self._render_predictions(st.session_state["predictions"])
            else:
                st.markdown(
                    """
                    <div class="info-box">
                        <strong>👈 Bir görüntü yükleyin</strong><br>
                        Ardından tahminleri görmek için <strong>Görüntüyü Sınıflandır</strong> butonuna tıklayın.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def run(self) -> None:
        """Streamlit uygulamasını çalıştırır.

        Sayfayı yapılandıran, modeli yükleyen ve tüm kullanıcı 
        arayüzü bileşenlerini oluşturan ana giriş noktasıdır.

        Örnek:
            >>> app = ClassificationUI()
            >>> app.run()
        """
        # Sayfayı yapılandır
        self._configure_page()

        # Modeli başlat
        self._initialize_model()

        # Kullanıcı arayüzünü oluştur
        self._render_sidebar()
        self._render_main_content()

        # Alt bilgi (Footer)
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
                🧠 TensorFlow ve MobileNetV2 ile güçlendirildi | 
                🎨 Streamlit ile oluşturuldu |
                📦 Animals-10 Transfer Öğrenme
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Uygulama giriş noktası."""
    app = ClassificationUI(
        model_path="animal_model.h5",
        class_names_path="class_names.json",
    )
    app.run()


if __name__ == "__main__":
    main()
