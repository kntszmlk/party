import torch
from sentence_transformers import SentenceTransformer
import pandas as pd

# GPU'nun mevcut olup olmadığını kontrol et
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'CUDA Kullanılabilir mi?: {torch.cuda.is_available()}')

# DataFrame'i yükleyin
df = pd.DataFrame({
    'Corpus': ["This is a test sentence.", "Another example sentence.", "How does this work?"]
})

# Modeli yükleyin ve GPU'ya taşıyın
model = SentenceTransformer('bert-base-nli-mean-tokens')
model = model.to(device)

# Vektör dönüştürme fonksiyonu
def get_vector(text):
    # Veriyi GPU'ya taşı
    inputs = model.encode(text, device=device)
    return inputs

# Paralel işlem için GPU kullanımını hızlandırmak
vectors = [get_vector(text) for text in df['Corpus']]

# Vektörleri DataFrame'e ekleyin
df['Vectors'] = vectors

print(df)
