# Sahte Yorum Avcısı

**Türkçe yorumları TF-IDF vektörlerine dönüştürüp Multinomial Naive Bayes ile sınıflandıran FastAPI projesi.**

Bir yorum gönderildiğinde API, `gercek` / `sahte` etiketi ve modelin sahte sınıfına verdiği skoru döndürür. Proje; Polars ile veri hazırlama, metin vektörizasyonu, veri sızıntısını önleyen eğitim düzeni ve canlı tahmin akışını birlikte gösterir.

> Eğitim amaçlıdır. Veriler sentetik, etiketler varsayımsaldır. Model bir yorumun gerçekten bir insan veya bot tarafından yazıldığını doğrulamaz.

## Özellikler

- Polars ile 10.000 benzersiz sentetik yorum: 5.000 `gercek`, 5.000 `sahte`.
- Tek kelime ve iki kelimelik özelliklerle TF-IDF (`ngram_range=(1, 2)`).
- TF-IDF ve sınıflandırıcıyı birlikte tutan scikit-learn `Pipeline`.
- Eğitim ve test arasında ortak yorum şablonu bulunmayan değerlendirme.
- FastAPI, Swagger arayüzü, girdi doğrulama ve sözlük dışı metin kontrolü.
- Ayrıntılı metrik raporu, 10 otomatik test ve GitHub Actions iş akışı.

## Nasıl çalışır?

1. **Veri üretimi:** Her sınıfta 20 şablon × 25 ürün × 10 sayı = 5.000 yorum oluşturulur. `text`, `label`, `template_id` sütunları Polars DataFrame içinde tutulur. `0`: gerçek, `1`: sahte.
2. **Ayırma:** Şablonlar sınıf dağılımı korunarak %80 eğitim / %20 test olarak ayrılır. Aynı şablonun ürün veya sayı varyasyonları iki tarafa da düşmez. Sabit tohum: `42`.
3. **Vektörizasyon:** TF-IDF sözlüğü ve IDF ağırlıkları yalnız eğitim yorumlarından öğrenilir. Seyrek matris kullanıldığı için büyük ve çoğunlukla sıfır olan bir yoğun matris oluşturulmaz. Sık kelimelerin ayırt ediciliği azalır; kısa Türkçe stop-word listesi ayrıca çıkarılır.
4. **Öğrenme:** `MultinomialNB(alpha=1.0)` sınıflara göre özellik ağırlıklarını öğrenir. Naive Bayes doğrudan kelime sırasını modellemez; iki kelimelik özellikler bazı yan yana gelmeleri temsil eder. TF-IDF'nin negatif olmayan kesirli değerleri bu sınıflandırıcıyla kullanılabilir.
5. **Canlı tahmin:** Kaydedilen aynı sözlük ve model yüklenir. Gelen yorum yalnız `transform` işleminden geçirilir; API isteği sırasında yeniden eğitim yapılmaz. `predict_proba` içinden sınıf `1` skoru alınır. Eşik `0.5`tir.

Buradaki klasik model sayısal özellikler kullanır. Bu, bütün yapay zekâ sistemlerinin metin işleyemediği anlamına gelmez.

## Kurulum

**Python 3.12** ile geliştirilip test edilmiştir. Komutları `README.md` ve `requirements.txt` dosyalarının bulunduğu proje dizininde çalıştırın.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Komut İstemi (cmd):

```bat
.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Bağımlılıkları kurun ve modeli eğitin:

```bash
python -m pip install -r requirements.txt
python -m app.train
```

Eğitim; `data/reviews.csv`, `artifacts/model.joblib`, `artifacts/metrics.json` ve `reports/metrics.json` dosyalarını üretir. Model dosyası repoya dahil edilmez; kurulumdan sonra yerel olarak oluşturulur.

API'yi başlatın:

```bash
python -m uvicorn app.main:app --reload
```

Swagger arayüzü: **http://127.0.0.1:8000/docs**

`POST /predict` → **Try it out** → yorumunuzu yazın → **Execute**.

## API kullanımı

| Yöntem | Yol | İşlev |
| --- | --- | --- |
| GET | `/health` | Servisin ve modelin hazır olduğunu gösterir |
| POST | `/predict` | Tek bir yorum için tahmin üretir |
| GET | `/docs` | Etkileşimli API dokümantasyonu |

İstek gövdesi:

```json
{"text": "Harika harika harika kesinlikle al 10 numara!"}
```

Bu sürümle elde edilen örnek yanıt:

```json
{
  "label": "sahte",
  "fake_probability": 0.9966558380272409,
  "fake_percentage": 99.67,
  "threshold": 0.5,
  "warning": "Sentetik eğitim modelinin kalibre edilmemiş skoru; sahtecilik kanıtı değildir."
}
```

`fake_probability` 0–1 arasındaki model skorudur; `fake_percentage` bu skorun yüzdeye çevrilip iki basamakla yuvarlanmış halidir. Yuvarlama nedeniyle `100.0` görülmesi kesinlik anlamına gelmez. Skorlar kalibre edilmemiştir; gerçek dünyadaki sahtecilik olasılığı olarak yorumlanmamalıdır.

Geçerli giriş: baştaki/sondaki boşluklar temizlendikten sonra en az 3 karakter, en az bir harf ve en fazla 5.000 karakter. Eksik veya uygunsuz giriş `422` döndürür. Eğitim sözlüğünde hiçbir özelliğe eşleşmeyen metne de `422` döndürülür; sınıf öncüllerine dayanarak yanıltıcı tahmin verilmez.

## Ölçülen sonuçlar

Rapor: [reports/metrics.json](reports/metrics.json). Sonuçlar aynı eğitim komutuyla yeniden üretilebilir.

- Eğitim: **8.000 yorum / 32 şablon**.
- Test: **2.000 yorum / 8 ayrı şablon**; her sınıfta 1.000 yorum.
- Sözlük: **1.232 özellik**.
- Doğruluk: **%100**. Makro F1: **1,0000**.

| Sınıf | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Gerçek | 1,0000 | 1,0000 | 1,0000 |
| Sahte | 1,0000 | 1,0000 | 1,0000 |

Karışıklık matrisi:

| Gerçek etiket / Tahmin | Gerçek | Sahte |
| --- | ---: | ---: |
| Gerçek | 1.000 | 0 |
| Sahte | 0 | 1.000 |

Bu sabit sentetik test kümesinde bütün yorumlar doğru sınıflandırılmıştır. **%100 sonuç gerçek dünyada %100 başarı anlamına gelmez.** Üretilen yorumlar sınıfları ayıran belirgin örüntüler taşır ve test yalnız sekiz şablona dayanır; 2.000 varyasyon, 2.000 bağımsız gerçek kullanıcı deneyimi değildir. Şablon ayırma benzer metinlerden kaynaklanan sızıntıyı azaltır, gerçek dünya genellemesini kanıtlamaz. Önceki sürümle test şablonları da değiştiğinden iki sürümün doğrulukları doğrudan karşılaştırılamaz.

## Testler

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

10 test; veri sayısı ve benzersizliği, şablon ayrımı, servis sağlığı, iki örnek tahmin, skor aralığı/yüzde dönüşümü, uygunsuz girişler, sözlük dışı metin ve eksik model durumunu kontrol eder. `push` ve `pull_request` olaylarında GitHub Actions aynı test komutunu çalıştırır. Son yerel çalıştırmada **10 test geçti**; test istemcisinin `httpx` kullanımı için bağımlılık kaynaklı bir deprecation uyarısı görüldü.

## Dosya düzeni

| Dosya | Sorumluluk |
| --- | --- |
| `app/dataset.py` | Tekrar üretilebilir sentetik Polars veri kümesi |
| `app/train.py` | Şablon ayrımı, TF-IDF, Naive Bayes, kayıt ve metrikler |
| `app/main.py` | Model yükleme, girdi doğrulama ve FastAPI uçları |
| `tests/test_project.py` | Model ve API davranış testleri |
| `data/reviews.csv` | Üretilen 10.000 yorum ve etiketleri |
| `reports/metrics.json` | Ayrıntılı değerlendirme çıktısı |
| `.github/workflows/tests.yml` | Otomatik test iş akışı |

## Sınırlar ve geliştirme yönü

Sentetik veride tekrar, abartı ve satın almaya yönlendiren ifadeler sahte sınıfında yoğunlaşır. Yeni şablonlar, olumlu gerçek yorumların yanında olumsuz sahte yorum örneklerini de içerir. Gerçek kullanıcılar da böyle yazabilir; sahte yorumlar doğal ve olumsuz da olabilir. Dolayısıyla bu model, eğitim verisindeki yazım örüntülerini öğrenen bir başlangıç modelidir.

Gerçek bir uygulama için izinli ve güvenilir etiketlenmiş yorumlar, ürün/kullanıcı/zaman bazında bağımsız testler, olasılık kalibrasyonu ve yanlış suçlamayı azaltacak eşik seçimi gerekir. Bu sürüm, kullanıcı veya yorum kaldırma kararı vermek için uygun değildir.

`joblib` dosyaları Python nesneleri içerir. Yalnız kendi eğitim komutunuzla oluşturduğunuz veya kaynağına güvendiğiniz modeli yükleyin.

## Kaynaklar

- [Polars DataFrame](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html)
- [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [FastAPI](https://fastapi.tiangolo.com/)
