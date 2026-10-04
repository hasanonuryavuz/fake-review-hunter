"""Etiketleri varsayımsal olan, tekrar üretilebilir sentetik eğitim verisi."""
from pathlib import Path
import random
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ["kulaklık", "telefon", "çanta", "ayakkabı", "kitap", "lamba", "saat", "klavye", "fare", "monitör", "kazak", "mont", "şarj cihazı", "hoparlör", "suluk", "termos", "çaydanlık", "tencere", "yastık", "havlu", "süpürge", "masa", "sandalye", "kamera", "tablet"]
REAL = [
    "{item} gayet güzel, kargolama hızlıydı. {day} gündür kullanıyorum.",
    "{item} paketinde çizik vardı, iade talebi açtım. {day} gün bekledim.",
    "{item} beklediğimden küçük geldi ama işimi görüyor. {day} gündür bende.",
    "{item} malzemesi iyi, fiyatı biraz yüksek. Teslimat {day} gün sürdü.",
    "{item} fotoğraftakiyle aynı, kutu ezilmişti. {day} gün sonra ulaştı.",
    "{item} için satıcıya soru sordum, açıklayıcı cevap verdi. {day} gün kullandım.",
    "{item} ilk gün iyiydi, sonra arıza yaptı. {day} gün sonra servise verdim.",
    "{item} kalitesi orta, indirimde alınabilir. {day} günlük deneyimim bu.",
    "{item} rengi beklediğim gibi değil, değişim istedim. {day} gün oldu.",
    "{item} kullanışlı çıktı, ambalajı özenliydi. {day} gündür sorun yaşamadım.",
    '{item} çok hoşuma gitti; özellikle kullanım kolaylığı iyi. {day} gün denedim.',
    '{item} harika görünüyor ama bağlantı noktası gevşek. {day} gündür kullanıyorum.',
    '{item} beklentimi karşıladı, tavsiye ederim. Kargo {day} günde geldi.',
    '{item} fiyatına göre güzel; kullanım kılavuzu eksik çıktı. {day} gün oldu.',
    '{item} için iade süreci sorunsuz ilerledi. Ücret {day} günde yattı.',
    '{item} siparişim yanlış renk geldi, satıcı değişim yaptı. {day} gün bekledim.',
    '{item} uzun kullanımda rahat değil, kısa süre için uygun. {day} gündür bende.',
    '{item} tam istediğim gibi, tekrar alabilirim. {day} gün önce teslim edildi.',
    '{item} için olumsuz yorumları okudum ama bende sorun çıkmadı. {day} gün kullandım.',
    '{item} kötü paketlenmişti; ürün sağlam kaldı. {day} gün sonra teslim aldım.',
]
FAKE = [
    "{item} harika harika harika kesinlikle al 10 numara! {day} tane al!",
    "{item} mükemmel mükemmel herkese tavsiye hemen satın al! {day} yıldız!",
    "{item} efsane fırsat kaçırma en iyi ürün şimdi sipariş ver! {day} tane!",
    "{item} süper süper fiyat şahane düşünmeden al al al! {day} yıldız!",
    "{item} inanılmaz kusursuz muhteşem satıcı bir numara! {day} tane al!",
    "{item} harika fırsat herkese öneriyorum hemen al pişman olmazsın! {day} yıldız!",
    "{item} mükemmel kalite mükemmel fiyat mükemmel satıcı! {day} tane satın al!",
    "{item} piyasadaki en iyi en ucuz kaçırılmaz hemen sipariş! {day} yıldız!",
    "{item} şahane şahane kesinlikle almalısınız rakipsiz kalite! {day} tane!",
    "{item} muhteşem efsane süper ürün herkese tavsiye al! {day} yıldız!",
    '{item} berbat berbat berbat asla almayın en kötü ürün! {day} sıfır yıldız!',
    '{item} kesinlikle uzak dur herkes başka markayı alsın! {day} kere söylüyorum!',
    '{item} rezalet rezalet para tuzağı sakın satın alma! {day} sıfır puan!',
    '{item} için rakip yok bütün ürünlerden üstün hemen al! {day} tane sipariş!',
    '{item} hakkında olumsuz yazanlara inanmayın kusursuz süper! {day} yıldız!',
    '{item} on numara beş yıldız yüzde yüz garanti kaçırma! {day} tane al!',
    '{item} hayatınız değişecek inanılmaz sonuç hemen satın al! {day} yıldız!',
    '{item} dünyada bir numara kesin memnuniyet mükemmel! {day} tane sipariş ver!',
    '{item} beğenmeyen yok herkes alıyor hemen sen de al! {day} yıldız!',
    '{item} şimdi satın al bugün fırsat son fırsat süper süper! {day} tane al!',
]

def make_dataset(seed: int = 42) -> pl.DataFrame:
    rows = []
    for label, templates in ((0, REAL), (1, FAKE)):
        for number, template in enumerate(templates):
            for item in ITEMS:
                for day in range(1, 11):
                    rows.append({"text": template.format(item=item, day=day), "label": label,
                                 "template_id": f"{label}_{number}"})
    random.Random(seed).shuffle(rows)
    return pl.DataFrame(rows)

if __name__ == "__main__":
    destination = ROOT / "data" / "reviews.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    df = make_dataset()
    df.write_csv(destination)
    print(f"{df.height} sentetik yorum yazıldı: {destination}")
