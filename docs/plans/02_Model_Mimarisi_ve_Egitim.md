# Aşama 2: Model Mimarisi ve Eğitim

[← Aşama 1 — Veri Toplama ve Hazırlama](01_Veri_Toplama_ve_Hazirlama.md) · Sonraki: [Aşama 3 — Tablebase Entegrasyonu](03_Tablebase_Entegrasyonu.md)

Mimari olarak ResNet yerine saf transformer tabanlı bir gövde tercih ediliyor. Tahtadaki 64 kareyi görsel bir piksel matrisi gibi değil, her biri taş tipi ve rengiyle kodlanmış 64 ayrı token olarak ele alan bir encoder-only transformer kullanılacak (artı sırası kimde, rok hakları gibi birkaç ek token). Bu seçim rastgele değil: satrançta bir filin veya kalenin etkisi bitişik karelerle sınırlı değil, tahtanın öbür ucuna kadar anında uzanıyor, bu da CNN'in yerel-komşuluk varsayımına göre self-attention'ın her kare her kareye tek katmanda bakabilmesine daha çok uyuyor. Bunu destekleyen somut sonuçlar da var: DeepMind'ın hiç CNN kullanmayan, saf decoder-only transformer'ı (270M parametre, Stockfish 16 action-value'larıyla 10M oyun / ~15 milyar veri noktası üzerinde eğitilmiş) hiçbir arama yapmadan [Lichess blitz'te 2895 Elo'ya (büyükusta seviyesi)](https://arxiv.org/html/2402.04494v1) ulaştı, ve tam olarak insan hamlesi tahminini hedefleyen ICLR 2026'da yayınlanan Chessformer çalışması, orijinal CNN tabanlı Maia'nın yüzde 46-52 aralığına ve önceki en iyi sonuç olan 355M parametreli aramalı modelin yüzde 55.9'una karşı, sadece 79M parametreyle (Maia-3) [yüzde 57.1 isabet oranına](https://arxiv.org/html/2605.19091v1) ulaştı — daha küçük model, daha iyi sonuç. 64 karelik bir dizi için tam self-attention maliyeti de zaten çok küçük, hesaplama açısından bir dezavantaj yok.

Chessformer ayrıca sabit öğrenilen pozisyon embedding'i yerine "Geometric Attention Bias" (GAB) adında, kare-arası geometrik mesafeyi (rank/file farkı) attention'a bias olarak ekleyen özel bir pozisyon kodlaması kullanıyor ve bunun isabet oranına katkısı var. Bu v1 için opsiyonel bir yükseltme olarak not ediliyor: düz öğrenilen kare embedding'i (aşağıdaki plan) daha az kod, daha hızlı kurulum sağlıyor; GAB sadece isabet oranı hedefin altında kalırsa denenecek.

## Yapılacaklar

- [ ] Tahtayı 64 token olarak kodla (her token: taş tipi + renk embedding'i, artı kare kimliği için öğrenilen pozisyon embedding'i), sırası kimde/rok/en passant için birkaç ek token ekle
- [ ] PyTorch'un nn.TransformerEncoder'ı ile birkaç katmanlı bir encoder kur, çıktı olarak kalkış/varış karesi softmax başlıklarını aynı şekilde koru
- [ ] Legal hamle maskeleme fonksiyonunu ekle (çıkarım anında)
- [ ] Adam optimizer, cross-entropy loss, batch size 256-512 ile eğitim döngüsünü kur
- [ ] İlk denemeleri lokalde toy veri setiyle (birkaç yüz pozisyon) doğrula, sonra ücretsiz Kaggle/Colab GPU'sunda küçük ölçekte dene
- [ ] Tam ölçekli eğitim için Vast.ai veya RunPod üzerinden bir RTX 4090 kirala (2026-09 itibarıyla RunPod Community Cloud ~0.34$/saat, Vast.ai spot fiyatları 0.11-0.50$/saat aralığında — plandaki 0.35-0.5$ tahmini hâlâ geçerli, spot ile daha da düşebilir)
- [ ] Val setini **epoch sonunda değil, belli adım aralıklarında** (örn. her 5-10k step, ya da duvar-saatine göre her X dakika) top-1/top-3 için değerlendir — veri seti milyonlarca pozisyon olduğu için bir epoch çok uzun sürer, epoch sonunu beklemek platoya girmeyi/overfit'i geç fark etmek ve GPU saatini boşa harcamak demek
- [ ] Son checkpoint'i değil, **en iyi val top-1'e sahip checkpoint'i** sakla (best-checkpoint / early stopping) — hem deployment'a hem Aşama 5'teki final test ölçümüne giden model bu olmalı
- [ ] İlk çalışan checkpoint hazır olur olmaz (tam eğitim bitmeden), [Aşama 6](06_Dagitim.md)'daki lichess-bot `Homemade` sargısını ve tablebase/failsafe entegrasyonunu bu checkpoint'le erken tak — legal hamle maskeleme, promosyon/rok/en passant kodlaması ve tetikleme eşikleri model kalitesinden bağımsız buglar, ne kadar geç yakalanırsa o kadar pahalıya patlar
- [ ] Hedef: Chessformer/Maia-3'ün insan hamlesi tahmini için bildirdiği yüzde 57.1 top-1 isabet oranına yakın bir sonuç (tek reyting bandına özelleşmiş daha küçük bir modelle bu seviyeye yaklaşmak gerçekçi bir hedef; bu sayı aspirasyonel, hedefin altında kalmak proje amacını (o bandda çalışan bir motor) geçersiz kılmaz)

## Tahmini Süre

1 hafta (mimari denemeleri dahil), toplam GPU maliyeti muhtemelen 20 doların altında.
