# ANN + Tablebase Satranç Motoru: Genel Bakış

2026-09-22 · @Someone

## Proje Kapsamı

Bu plan, tek bir reyting bandına (2000-2200) odaklanan, arama yapmayan bir ANN politika ağını iki tamamlayıcı katmanla güçlendiren hibrit bir satranç motoru inşa etmeyi hedefliyor. Ana bileşen, Lichess'teki 2000-2200 bandı oyunlarından öğrenen küçük bir ResNet politika ağı; bu ağ normal oyunun büyük kısmında (açılış ve orta oyun) o bandaki bir oyuncunun tipik hamlelerini taklit ediyor. Buna iki güvenlik katmanı ekleniyor: tahtada 5 veya daha az taş kaldığında devreye giren, kusursuz teknik oyun sonu sağlayan bir Syzygy tablebase sorgusu, ve ağın önerdiği hamlenin bariz bir taş kaptırma olup olmadığını kontrol eden hafif bir failsafe katmanı.

Amaç, tek bir reyting bandında insan gibi ve o bandda güvenilir şekilde çalışan bir motor üretmek — her rakibe karşı kazanan, arama motoru gücünde bir sistem değil. Reyting bandının dışındaki rakiplere karşı orta oyunda ortaya çıkabilecek öngörülemeyen davranış (dağılım dışı pozisyonlar yüzünden) baştan kabul edilen, çözülmemiş bir sınırlama olarak kalıyor; bu Bilinen Sınırlamalar bölümünde tekrar ele alınıyor.

## Aşamalar

1. [Veri Toplama ve Hazırlama](01_Veri_Toplama_ve_Hazirlama.md) — 3-5 gün
2. [Model Mimarisi ve Eğitim](02_Model_Mimarisi_ve_Egitim.md) — 1 hafta
3. [Tablebase Entegrasyonu](03_Tablebase_Entegrasyonu.md) — 1-2 gün
4. [Failsafe Katmanı](04_Failsafe_Katmani.md) — 1-2 gün
5. [Değerlendirme ve Test](05_Degerlendirme_ve_Test.md) — 2-3 gün
6. [Dağıtım (Opsiyonel)](06_Dagitim.md) — sargı Aşama 2 ile paralel, sonda 1 gün

Teknoloji yığını ve dış kaynak adresleri için [Teknoloji Yığını ve Kaynaklar](../reference/Teknoloji_Yigini_ve_Kaynaklar.md) dosyasına bak.

## Zaman ve Maliyet Bütçesi

Toplamda, yarı zamanlı bir hobi projesi olarak yaklaşık 3-4 haftalık bir efor ve büyük ölçüde ücretsiz ya da çok düşük maliyetli bir bütçe öngörülüyor.

| Aşama | Tahmini Süre | Tahmini Maliyet |
| --- | --- | --- |
| 1. Veri toplama ve hazırlama | 3-5 gün | Ücretsiz |
| 2. Model mimarisi ve eğitim | 1 hafta | 10-20 dolar (GPU kirası) |
| 3. Tablebase entegrasyonu | 1-2 gün | Ücretsiz |
| 4. Failsafe katmanı | 1-2 gün | Ücretsiz |
| 5. Değerlendirme ve test | 2-3 gün | Ücretsiz |
| 6. Dağıtım (opsiyonel, sargı Aşama 2 ile paralel) | 1 gün | Ücretsiz |
| Toplam | \~3-4 hafta | \~10-20 dolar |

## Bilinen Sınırlamalar

Bu tasarımın kapsamı bilinçli olarak dar tutuluyor, dolayısıyla aşağıdaki noktalar kabul edilen tasarım kararları, sonradan mutlaka çözülmesi gereken hatalar değil.

Model tek bir reyting bandına (2000-2200) sabitlendiği için çok farklı seviyedeki rakiplere karşı orta oyunda dağılım dışı pozisyonlarla karşılaşabilir, bu durumda davranışı öngörülemez hale gelebilir. Tablebase katmanı sadece 5 veya daha az taş kaldığında devreye girdiği için 6 taş ve üzeri karmaşık oyun sonlarında hâlâ ağın kendi (muhtemelen kusurlu) tahminine bağımlı kalınır. Failsafe katmanı sadece en bariz materyal kayıplarını yakalar, ince pozisyonel hatalar veya çok adımlı taktiksel gözden kaçırmalar bu katmanın kapsamı dışında kalır. Son olarak, sistem hiçbir zaman bir arama motoru gücünde olmayacak; amacı en güçlü hamleyi bulmak değil, insan gibi ama güvenilir şekilde kazanan bir stil üretmek.

## Doğrulanan Referanslar (2026-09-22 araştırması)

Plandaki teknik iddialar ve sayılar aşağıdaki kaynaklarla doğrulandı:

- [Grandmaster-Level Chess Without Search (DeepMind, arXiv 2402.04494)](https://arxiv.org/html/2402.04494v1) — 270M parametreli decoder-only transformer, arama yok, Lichess blitz'te 2895 Elo.
- [Chessformer: A Unified Architecture for Chess Modeling (ICLR 2026, arXiv 2605.19091)](https://arxiv.org/html/2605.19091v1) — Maia-3 (79M parametre) yüzde 57.1 isabet oranı, önceki en iyi (355M, aramalı) yüzde 55.9; ayrıca Geometric Attention Bias (GAB) mimarisi.
- [Maia-2: A Unified Model for Human-AI Alignment in Chess (arXiv 2409.20553)](https://arxiv.org/pdf/2409.20553) — 18 kanallı board encoding şemasının kaynağı.
- [python-chess Syzygy dokümantasyonu](https://python-chess.readthedocs.io/en/latest/syzygy.html) — `probe_wdl`/`probe_dtz` API'si.
- [Syzygy 3-4-5 tablebase indirme (tablebase.sesse.net)](http://tablebase.sesse.net/) — WDL ~378MB + DTZ ~561MB.
- [database.lichess.org](https://database.lichess.org/) — aylık PGN.zst dökümleri, `%clk`/`%eval` yorumları, `WhiteElo`/`BlackElo`/`Event` etiketleri.
- [lichess-bot (resmi, lichess-bot-devs)](https://github.com/lichess-bot-devs/lichess-bot) — Python 3.10+, `Homemade` motor sınıfı UCI yazmadan entegrasyon sağlıyor.
- GPU kiralama: RunPod Community Cloud ~0.34$/saat, Vast.ai spot 0.11-0.50$/saat (2026-09 itibarıyla).

Değişmeyen/doğrulanan varsayımlar: tablebase boyutu, GPU maliyet aralığı, DeepMind ve Chessformer sonuçları. Netleşen noktalar: Maia-2 kanal şeması artık tam tanımlı, filtreleme mantığı ham TimeControl parse etmek yerine hazır `Event`/`%clk` alanlarına dayanıyor, dağıtımda UCI yazmaya gerek yok.
