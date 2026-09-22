# CLAUDE.md

Bu dosya, bu depoda çalışan her ajan (Claude Code oturumu) için talimat dosyasıdır. Yeni bir oturum başladığında önce bunu oku.

## Proje Nedir

Tek bir reyting bandına (2000-2200) odaklanan, arama yapmayan bir transformer politika ağı + Syzygy tablebase + hafif bir failsafe katmanından oluşan hibrit bir satranç motoru. Tam kapsam ve gerekçe: [docs/plans/00_Genel_Bakis.md](docs/plans/00_Genel_Bakis.md).

## Dosya Haritası

- `docs/plans/00_Genel_Bakis.md` — kapsam, bütçe, bilinen sınırlamalar, doğrulanmış referanslar.
- `docs/plans/01..06_*.md` — aşama aşama yapılacaklar listeleri (sırayla ilerler, her dosya bir öncekine/sonrakine link verir).
- `docs/reference/Teknoloji_Yigini_ve_Kaynaklar.md` — kullanılan kütüphaneler, dış kaynak adresleri (Lichess dump, Syzygy mirror, GPU kiralama), yerel geliştirme/notebook iş akışı.
- `docs/log/Ilerleme_Notlari.md` — iş günlüğü, aşağıdaki kurala göre güncellenir.

Yeni bir plan/karar dokümanı gerekiyorsa `docs/` altında konusuna uygun bir klasöre (plans/, reference/, log/ veya yeni bir klasöre) eklenir — kök dizine md dosyası atılmaz, CLAUDE.md hariç.

## İş Günlüğü Kuralı

Bir iş birimini (bir checklist maddesi, bir bug fix, bir aşamanın tamamı — ne olursa) bitirdikten sonra `docs/log/Ilerleme_Notlari.md` dosyasının **en üstüne** kısa bir not ekle: tarih, hangi aşama, ne yapıldı/hangi dosyalar değişti, sıradaki adım veya blocker varsa. Format dosyanın başında yazılı. Bu, farklı bir oturumda devam eden bir ajanın nerede kalındığını koda bakmadan anlaması için var — atlanmaz.

## Genel Kurallar

- **Kapsam sabit.** Tek reyting bandı, arama yok — bilinçli bir tasarım kararı. Modeli "daha güçlü" yapmak için arama/çok bantlı eğitim gibi kapsam genişletmeleri önerme; istenirse kullanıcı zaten söyler.
- **Teknoloji yığını sabit.** Python + PyTorch + `chess` (pip paketi `chess`, `python-chess` değil) + `zstandard` + `numpy` + `tensorboard`. Yeni bir bağımlılık eklemeden önce gerçekten gerekli mi diye sor (bkz. `docs/reference/Teknoloji_Yigini_ve_Kaynaklar.md`) — stdlib veya zaten kurulu bir paket işi görüyorsa onu kullan.
- **Kod `.py` dosyalarında yazılır, notebook değil.** Notebook (Kaggle/Colab) sadece ince bir başlatıcı: `git clone` + `pip install -r requirements.txt` + `python train.py ...` çalıştırır. Mantık notebook hücrelerine yazılmaz (debug edilemez, diff alınamaz).
- **Val/test ayrımı.** Test seti zaman-bazlı ayrılır (farklı ay/gün) ve eğitim boyunca hiç dokunulmaz. Val seti **oyun bazında** (pozisyon bazında değil) train verisinden ayrılır — pozisyon bazlı split sızıntıya yol açar.
- **Best-checkpoint.** Eğitimde son checkpoint değil, en iyi val top-1'e sahip checkpoint saklanır ve deployment/final test için kullanılır. Val, epoch sonunda değil belli adım aralıklarında (örn. her 5-10k step) ölçülür — epoch çok uzun sürebiliyor.
- **Yerel GPU (RTX 3050 4GB) sadece toy-scale test için.** Gerçek eğitim Kaggle (ücretsiz) veya kiralık GPU'da (Vast.ai/RunPod) yapılır. Lokalde büyük batch/tam veri setiyle eğitim denemesi başarısız olur, beklenen bir durum.
- **lichess-bot entegrasyonu `Homemade` motor sınıfı üzerinden.** UCI protokolü sıfırdan yazılmaz.
- **Bilinen sınırlamalar bug değildir.** Reyting bandı dışı rakiplere karşı öngörülemez orta oyun davranışı ve failsafe'in sadece bariz kayıpları yakalaması, kabul edilmiş tasarım kararları (bkz. `docs/plans/00_Genel_Bakis.md#bilinen-sınırlamalar`). Bunları "düzeltmeye" çalışma, sadece belgele.
- **Proje henüz git reposu değil.** İlk gerçek kodla birlikte `git init` önerilir; büyük/ikili dosyalar (veri shard'ları, checkpoint'ler, indirilen tablebase dosyaları) `.gitignore`'a eklenmeli.
