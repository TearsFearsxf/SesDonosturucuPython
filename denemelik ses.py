import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS
import os

# PyTorch için güvenlik ayarlarını değiştir:
torch.serialization.default_load_weights_only = False
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

# Referans ses dosyasının tam yolunu belirt (Türkçe, temiz, yaklaşık 6-10 saniyelik bir örnek)
speaker_path = "C:/Users/muhte/OneDrive/Masaüstü/örnek_ses.wav"

if not os.path.exists(speaker_path):
    raise FileNotFoundError(f"Ses dosyası bulunamadı: {speaker_path}")

# XTTS-v2 modelini yükle (GPU kullanmıyorsan gpu=False; GPU varsa True yap)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Çıktı dosyasının yolunu belirle
output_path = os.path.join(os.getcwd(), "cikti_sesi.wav")

# Metni sese çevirirken ayarlamalar:
tts.tts_to_file(
    text="Son zamanlarda daha fazla sağlıklı beslenme benim için önemli hale gelmiştir, bu yüzden yemek tercihleri konusunda dikkatliyim.",
    speaker_wav=speaker_path,
    language="tr",
    file_path=output_path,
    temperature=0.65,   # Daha düşük sıcaklık, daha deterministik sonuç
    top_k=40,           # Daha dar seçim
    top_p=0.8,          # Daha dar seçim
    speed=0.95          # Hafif yavaşlatılmış hız, daha doğal algı için
)

print(f"Ses dosyası oluşturuldu: {output_path}")
