# Pandas, verileri CSV gibi dosyalardan kolayca yüklemeye, veri manipülasyonu yapmaya, filtrelemeye ve analiz etmeye olanak sağlar.
import pandas as pd
# re (Regular Expressions): Düzenli ifadeler, metinleri işlemek ve desenlere göre arama yapmak için kullanılır.
import re  
# nltk.corpus: Natural Language Toolkit (NLTK), doğal dil işleme için kullanılan popüler bir Python kütüphanesidir. corpus modülü, çeşitli metin veri setlerini içerir.
from nltk.corpus import stopwords
# PorterStemmer: Kökleme (stemming), kelimelerin sonundaki ekleri kaldırarak onları kök haline getirir.
from nltk.stem import PorterStemmer
import nltk


data = pd.read_csv('training.1600000.processed.noemoticon.csv',header=None)

data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Tüm değerleri aynı çıktı. Bu sayede 'query' sütununun gereksiz olduğunu anlamış olduk.
for i in data['query']:
        if i!="NO_QUERY":
            print("Farklı değerler mevcut! : ",i)

# 'query' sütununu çıkarttım çünkü gereksiz.
data = data.drop(columns=['query'])

# PorterStemmer nesnesini oluşturuyoruz.
stemmer = PorterStemmer()

# NLTK stopwords veri setini indiriyoruz
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()  # Küçük harfe çevir
    text = re.sub(r'http\S+', '', text)  # URL'leri kaldır
    text = re.sub(r'@\w+', '', text)  # Kullanıcı adlarını kaldır
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Durdurma kelimelerini kaldır
    text = ' '.join(stemmer.stem(word) for word in text.split())  # Köklemeyi uygula
    return text

# Tüm metinleri temizle
data['cleaned_text'] = data['text'].apply(clean_text)

# 'sentiment' (duygu) sütunundaki farklı değerleri fonksiyon yardımıyla bulalım.
sentiment_degerleri= set()
for i in data["sentiment"]:
     sentiment_degerleri.add(i)
print(sentiment_degerleri)  # {0, 4} çıktısını aldık.

# Daha kısa bir şekilde benzersiz etiketleri ve sayılarını kontrol ettik.
print(data['sentiment'].value_counts())

# Kelime frekansını hesapladık.
word_freq = data['cleaned_text'].str.split(expand=True).stack().value_counts()

print("En sık geçen kelimeler:")
print(word_freq.head(10))