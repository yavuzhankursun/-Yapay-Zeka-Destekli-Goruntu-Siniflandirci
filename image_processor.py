"""
Hayvan Sınıflandırması İçin Görüntü İşleme Yardımcı Araçları.

Bu modül; MobileNetV2 tabanlı sınıflandırma modelleri ile çıkarım (inference) yapmak üzere 
görüntülerin yüklenmesi, ön işlenmesi ve hazırlanması için yardımcı araçlar sağlar.

Örnek:
    from image_processor import ImageUtils
    
    processor = ImageUtils()
    pil_image, model_input = processor.process_image("yol/dosya.jpg")
    predictions = model.predict(model_input)
"""

import io
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class ImageUtils:
    """Görüntü yükleme ve ön işleme işlemleri için yardımcı sınıf.

    Bu sınıf, çeşitli kaynaklardan (dosya yolları, baytlar, dosya benzeri nesneler) 
    görüntüleri yüklemek ve bunları MobileNetV2 tabanlı modellerle çıkarım 
    yapmaya hazırlamak için yöntemler sunar.

    Özellikler:
        target_size: Yeniden boyutlandırılan görüntüler için hedef boyutlar (yükseklik, genişlik).

    Örnek:
        >>> utils = ImageUtils(target_size=(224, 224))
        >>> pil_img, model_input = utils.process_image("kedi.jpg")
        >>> print(model_input.shape)
        (1, 224, 224, 3)
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        """Hedef görüntü boyutları ile ImageUtils sınıfını başlatır.

        Argümanlar:
            target_size: (yükseklik, genişlik) olarak görüntünün yeniden boyutlandırılacağı hedef boyut.
                MobileNetV2 uyumluluğu için varsayılan değer (224, 224)'tür.
        """
        self.target_size = target_size

    def load_image(
        self, source: Union[str, bytes, Path, io.BytesIO]
    ) -> Image.Image:
        """Çeşitli kaynak türlerinden bir görüntü yükler.

        Dosya yollarından, ham baytlardan, Path nesnelerinden veya dosya benzeri 
        BytesIO nesnelerinden yüklemeyi destekler. Tutarlı işleme için görüntüleri 
        otomatik olarak RGB moduna dönüştürür.

        Argümanlar:
            source: Görüntü kaynağı. Aşağıdakilerden biri olabilir:
                - str: Görüntünün dosya yolu
                - bytes: Ham görüntü baytları
                - Path: Görüntüye işaret eden pathlib.Path nesnesi
                - BytesIO: Görüntü verilerini içeren dosya benzeri nesne

        Dönüş:
            RGB modunda PIL Görüntü nesnesi.

        Hatalar:
            FileNotFoundError: Dosya yolu mevcut değilse.
            ValueError: Kaynak türü desteklenmiyorsa.
            IOError: Görüntü açılamıyor veya kodu çözülemiyorsa.

        Örnek:
            >>> utils = ImageUtils()
            >>> img = utils.load_image("fotograf.jpg")
            >>> img = utils.load_image(görüntü_baytları)
        """
        try:
            if isinstance(source, bytes):
                image = Image.open(io.BytesIO(source))
            elif isinstance(source, io.BytesIO):
                source.seek(0)  # Başlangıçtan okuduğumuzdan emin olalım
                image = Image.open(source)
            elif isinstance(source, (str, Path)):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {path}")
                image = Image.open(path)
            else:
                raise ValueError(
                    f"Desteklenmeyen kaynak türü: {type(source)}. "
                    "Beklenen: str, bytes, Path, veya BytesIO."
                )

            # Gerekirse RGB'ye dönüştür (RGBA, gri tonlama vb. durumlar için)
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image

        except IOError as e:
            raise IOError(f"Görüntü yüklenemedi: {e}") from e

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Bir görüntüyü hedef boyutlara yeniden boyutlandırır.

        En iyi görsel sonuçlar için yüksek kaliteli Lanczos yeniden örneklemesini kullanır.

        Argümanlar:
            image: Yeniden boyutlandırılacak PIL Görüntü nesnesi.

        Dönüş:
            Yeniden boyutlandırılmış PIL Görüntü nesnesi.

        Örnek:
            >>> utils = ImageUtils(target_size=(224, 224))
            >>> resized = utils.resize_image(buyuk_goruntu)
            >>> print(resized.size)
            (224, 224)
        """
        return image.resize(self.target_size, Image.Resampling.LANCZOS)

    def image_to_array(self, image: Image.Image) -> np.ndarray:
        """Bir PIL Görüntüsünü numpy dizisine dönüştürür.

        Argümanlar:
            image: RGB modunda PIL Görüntü nesnesi.

        Dönüş:
            (yükseklik, genişlik, 3) şeklinde ve float32 veri tipinde numpy dizisi.

        Örnek:
            >>> utils = ImageUtils()
            >>> arr = utils.image_to_array(pil_goruntu)
            >>> print(arr.shape, arr.dtype)
            (224, 224, 3) float32
        """
        return np.array(image, dtype=np.float32)

    def add_batch_dimension(self, array: np.ndarray) -> np.ndarray:
        """Bir görüntü dizisine grup (batch) boyutu ekler.

        3D görüntü dizisini (Y, G, K), model çıkarımı için uygun olan 4D grup 
        dizisine (1, Y, G, K) genişletir.

        Argümanlar:
            array: (yükseklik, genişlik, kanallar) şeklinde numpy dizisi.

        Dönüş:
            (1, yükseklik, genişlik, kanallar) şeklinde numpy dizisi.

        Hatalar:
            ValueError: Giriş dizisi 3 boyutlu değilse.

        Örnek:
            >>> arr = np.zeros((224, 224, 3))
            >>> batched = utils.add_batch_dimension(arr)
            >>> print(batched.shape)
            (1, 224, 224, 3)
        """
        if array.ndim != 3:
            raise ValueError(
                f"3D dizi (Y, G, K) beklendi, {array.ndim}D dizi alındı"
            )
        return np.expand_dims(array, axis=0)

    def apply_preprocessing(self, array: np.ndarray) -> np.ndarray:
        """Bir görüntü dizisine MobileNetV2 ön işlemesini uygular.

        Piksel değerlerini [-1, 1] aralığına ölçeklendiren resmi MobileNetV2 
        ön işleme fonksiyonunu uygular.

        Argümanlar:
            array: (grup, yükseklik, genişlik, kanallar) veya 
                (yükseklik, genişlik, kanallar) şeklinde numpy dizisi.

        Dönüş:
            Model çıkarımı için hazır, önceden işlenmiş numpy dizisi.

        Örnek:
            >>> preprocessed = utils.apply_preprocessing(gruplanmis_dizi)
        """
        return preprocess_input(array)

    def process_image(
        self, source: Union[str, bytes, Path, io.BytesIO]
    ) -> Tuple[Image.Image, np.ndarray]:
        """Görüntüyü kaynaktan model için hazır formata işler.

        Tüm ön işleme adımlarını birleştiren ana yöntemdir:
        1. Görüntüyü kaynaktan yükle
        2. Hedef boyutlara yeniden boyutlandır
        3. Numpy dizisine dönüştür
        4. Grup boyutu ekle
        5. MobileNetV2 ön işlemesini uygular

        Argümanlar:
            source: Görüntü kaynağı (dosya yolu, bayt, Path veya BytesIO).

        Dönüş:
            Şunları içeren bir demet (tuple):
                - display_image: Görüntüleme için yeniden boyutlandırılmış PIL Görüntüsü (RGB)
                - model_input: (1, yükseklik, genişlik, 3) şeklinde, model.predict() için hazır dizi

        Hatalar:
            FileNotFoundError: Dosya yolu mevcut değilse.
            ValueError: Kaynak türü desteklenmiyorsa.
            IOError: Görüntü açılamıyor veya kodu çözülemiyorsa.

        Örnek:
            >>> utils = ImageUtils()
            >>> display_img, model_input = utils.process_image("kedi.jpg")
            >>> st.image(display_img, caption="Yüklenen Görüntü")
            >>> predictions = model.predict(model_input)
        """
        # Görüntüleme için yükle ve yeniden boyutlandır
        original_image = self.load_image(source)
        display_image = self.resize_image(original_image)

        # Model için hazırla
        array = self.image_to_array(display_image)
        batched = self.add_batch_dimension(array)
        model_input = self.apply_preprocessing(batched)

        return display_image, model_input

    def process_uploaded_file(
        self, uploaded_file
    ) -> Tuple[Image.Image, np.ndarray]:
        """Bir Streamlit UploadedFile nesnesini işler.

        Özellikle Streamlit dosya yüklemelerini işlemek için kolaylık sağlayan yöntem.

        Argümanlar:
            uploaded_file: st.file_uploader()'dan gelen Streamlit UploadedFile nesnesi.

        Dönüş:
            Şunları içeren bir demet:
                - display_image: Görüntüleme için yeniden boyutlandırılmış PIL Görüntüsü
                - model_input: Model çıkarımı için önceden işlenmiş numpy dizisi

        Hatalar:
            ValueError: uploaded_file None ise veya geçersizse.
            IOError: Görüntü kodu çözülemiyorsa.

        Örnek:
            >>> uploaded = st.file_uploader("Görüntü seçin", type=["jpg", "png"])
            >>> if uploaded:
            ...     display_img, model_input = utils.process_uploaded_file(uploaded)
        """
        if uploaded_file is None:
            raise ValueError("Dosya yüklenmedi")

        file_bytes = uploaded_file.read()
        return self.process_image(file_bytes)

    def validate_image(
        self, source: Union[str, bytes, Path, io.BytesIO]
    ) -> bool:
        """Bir kaynağın geçerli bir görüntü olarak yüklenip yüklenemeyeceğini doğrular.

        Tam işleme yapmadan görüntüyü yüklemeyi ve doğrulamayı dener. 
        Maliyetli işlemlerden önce ön doğrulama için kullanışlıdır.

        Argümanlar:
            source: Doğrulanacak görüntü kaynağı.

        Dönüş:
            Kaynak geçerli bir görüntüyse True, aksi takdirde False.

        Örnek:
            >>> if utils.validate_image(user_upload):
            ...     process_and_predict(user_upload)
            ... else:
            ...     st.error("Geçersiz görüntü dosyası")
        """
        try:
            image = self.load_image(source)
            image.verify()  # Görüntü bütünlüğünü doğrula
            return True
        except Exception:
            return False


def create_processor(target_size: Tuple[int, int] = (224, 224)) -> ImageUtils:
    """ImageUtils örneği oluşturmak için fabrika fonksiyonu.

    Yaygın yapılandırmalarla ImageUtils oluşturmak için kolaylık sağlayan fonksiyon.

    Argümanlar:
        target_size: Hedef görüntü boyutları. Varsayılan (224, 224).

    Dönüş:
        Yapılandırılmış ImageUtils örneği.

    Örnek:
        >>> processor = create_processor()
        >>> display, model_in = processor.process_image("fotograf.jpg")
    """
    return ImageUtils(target_size=target_size)
