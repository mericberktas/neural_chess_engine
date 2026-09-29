# Eğitim Koşuları Karşılaştırması (run1-run6)

Şimdiye kadar tamamlanan gerçek GPU eğitim koşularının neyi nasıl farklı yaptığının özeti. Ayrıntılı olay anlatımı (bug'lar, operasyonel arızalar vb.) için [docs/log/Ilerleme_Notlari.md](../log/Ilerleme_Notlari.md)'ye bakın — bu dosya sadece koşular arası metodolojik farkları ve sonuçları özetler.

run1-run4'te mimari **sabit**: `ChessTransformer`, batch 256, d_model 256, nhead 8, num_layers 6, dim_feedforward 1024, dropout 0.1, ~4.76M parametre, 18 kanallı girdi. run5, `run5-rich-encoding` branch'inde girdiyi 21 kanala çıkardı (mimari boyutu aynı kaldı) — bkz. [docs/plans/02_Model_Mimarisi_ve_Egitim.md](../plans/02_Model_Mimarisi_ve_Egitim.md). run6, `run6-gab-see` branch'inde 22. kanalı (SEE riski) ve Geometric Attention Bias'ı (~349K ek parametre) ekledi.

## Özet tablo

| | run1 | run2 | run3 | run4 | run5 | run6 |
|---|---|---|---|---|---|---|
| Veri | 1 ay, 2.3M pozisyon | 4 ay, ~9M pozisyon | 8 ay, 18.0M train pozisyonu | 8 ay (run3 ile aynı) | 8 ay (run3/4 ile aynı) | **16 ay**, Drive harvest'ten |
| Girdi kanalları | 18 | 18 | 18 | 18 | 21 | **22** (+SEE riski) + GAB |
| LR programı | Sabit 3e-4 | Sabit 3e-4 | Sabit 3e-4 | `ReduceLROnPlateau` (patience 1, factor 0.5) | Aynı | Aynı — ama bkz. aşağıdaki not |
| Resume | — | — | — | run3'ten (sadece ağırlık) | — (mimari uyumsuz) | **2 kez zorla** (pod ölümleri) + 1 planlı (scheduler-fix testi için) |
| Sonuç (best) | step 8000, %32.9 | step 68000, %45.35 | step 64000, %45.08 | step 130000, %50.02 | step 108000, %50.18 | step 88000, **%50.61** ⚠️ |
| Nasıl bitti | Manuel `terminate_pod` | Gerçek early-stopping | Gerçek early-stopping | Gerçek early-stopping | Self-terminate | Gerçek early-stopping (step 96000) |
| Drive yolu | `run1/` | `run2/` | `run3/` | `run4/` | `run5/` | `run6/` (⚠️ sadece step 78000/%49.82 kurtarılabilir) |

**run6, scheduler resume-fix'i sonrası run4/5'i geçti** (%50.61 > %50.18 > %50.02) — GAB+SEE hipotezi doğrulandı, önceki %48.71 sonucu gerçekten bir bug'dan kaynaklanıyormuş (ayrıntı: [docs/log/Ilerleme_Notlari.md](../log/Ilerleme_Notlari.md)'nin 2026-09-28 notu).

⚠️ **Ama gerçek best'in (step 88000) ağırlıkları kayıp** — bu pod'un ağı yavaştı (~250KB/s), 61MB'lık checkpoint'in Drive sync'i o zamanki 180s timeout'a takıldı (4/7 yeni-best sync'i başarısız oldu), pod silinince o ağırlıklar sonsuza dek gitti. Drive'da fiilen duran son checkpoint step 78000 (%49.82) — yine de eski bug'lı sonucun (%48.71) üzerinde. Timeout 180s→600s'e çıkarıldı (2026-09-29 notu), aynı kayıp bir daha yaşanmamalı.

## Koşu koşu neden/ne değişti

**run1 — ilk temel çizgi.** Tek ay, tek epoch, hiçbir regülarizasyon/erken-durdurma/resume mekanizması henüz yoktu — pipeline'ın uçtan uca gerçekten çalıştığını doğrulamak içindi. %32.9 sonucu plandaki aspirasyonel hedefin (Chessformer/Maia-3 %57.1) altında ama beklenen bir ilk-koşu sonucuydu.

**run2 — veri 4 kat arttı, ilk gerçek early-stopping.** `--patience` flag'i bu koşu için eklendi (`train.py`), veri 4 aya çıktı. val_top1 45.35'e sıçradı ama step 68000 civarında plato yaptı; bu plato **regülarizasyonun eklenmesini tetikledi** (`train.py` docstring'inde de not edilir: "Added after run2 ... plateaued at val_top1 45.35%").

**run3 — 8 ay + regülarizasyon, ama hâlâ sabit LR.** Weight-decay (AdamW) ve label-smoothing eklendi, veri 8 aya çıktı (18M pozisyon). Beklenti fazlasıyla veri + regülarizasyonun overfitting'i geciktirip daha uzun/daha iyi bir eğrisi olmasıydı — ama sonuç run2'ye çok yakın (%45.08) çıktı, üstelik epoch 0'ın içinde (henüz 1 epoch bile bitmeden) platoladı. Bu, sorunun **veri miktarı değil optimizasyon** (sabit LR) olduğunu gösterdi.

**run4 — resume + LR decay, hedef aşıldı.** İki yeni mekanizma birlikte eklendi: `--resume-from` (bir checkpoint'ten ağırlık/optimizer/step/best-val_top1 geri yükleme) ve `torch.optim.lr_scheduler.ReduceLROnPlateau` (val_top1 platoya girince LR'yi otomatik yarılama). run3'ün checkpoint'inden (step 64000) devam edildi — ama run3'ün checkpoint'i optimizer state içermiyordu (o zaman henüz eklenmemişti), o yüzden optimizer sıfırdan ısındı, sadece model ağırlıkları ve step sayacı gerçekten "resume" oldu. LR 5 kez yarılandı (3e-4 → 9.37e-06) ve her yarılanmada küçük ama gerçek bir sıçrama geldi: %45.08 (resume noktası) → %49.34 → %49.69 → **%50.02**. Maia'nın (GitHub'daki gerçek `maia_config.yaml`, 6x64-SE LCZero ağı, muhtemelen bizim 4.76M parametremizden küçük) kademeli LR programı kullandığını görmek bu fikri doğrudan tetikledi.

**run5 — girdi zenginleştirme, beklenenden küçük kazanç.** `NUM_CHANNELS` 18→21: mobilite/legal-hamle maskesi (kare-bazlı, 64 token'ın her birine eklenen bir embedding) + son hamlenin kalkış/varış kareleri (en-passant token'ıyla aynı desende 2 yeni "extra token"). Motivasyon: Maia'nın (GitHub'daki gerçek config) bizden küçük bir ağla (6x64-SE, muhtemelen ~1M parametre altı) benzer/daha iyi sonuç alması, muhtemelen daha zengin girdisinden (hamle geçmişi vb.) geliyordu. Aynı 8 aylık veri + aynı regülarizasyon + aynı LR-decay tarifiyle sıfırdan eğitildi (resume mimari uyumsuzluk yüzünden mümkün değildi). Sonuç: **%50.02 → %50.18** (+0.16 puan) — pozitif ama run3→run4'teki (+4.94 puan) kazancın çok altında. Ayrıntılı olası nedenler için aşağıya bakın.

## run5'in küçük kazancı: olası nedenler

run4→run5 arası tek değişken girdi zenginliğiydi (veri, regülarizasyon, LR programı hepsi aynı), yine de kazanç run2→run3→run4 zincirindeki sıçramalardan çok daha küçük kaldı. Olası açıklamalar, en olası gördüğümüzden başlayarak:

1. **Görevin kendi gürültü tavanına yaklaşmış olabiliriz.** val_top1, gerçek bir insanın oynadığı hamleyi birebir tahmin etme oranı — bu metrik doğası gereği %100'e yaklaşamaz, çünkü çoğu pozisyonda birden fazla makul hamle vardır ve 2000-2200 aralığındaki farklı oyuncular farklı hamleler seçer (rating bandını tek bir "ortalama politika" olarak modelliyoruz). run4 ve run5'in ikisi de bağımsız olarak ~%50 civarında platoladı — bu, girdi ne kadar zenginleşirse zenginleşsin, bu ölçek/veri kombinasyonuyla ulaşılabilecek pratik tavana yakın olduğumuzu düşündürüyor. Yayınlanmış benzer çalışmalar (Maia) da benzer bantta ~%50-52'de tavan yapıyor.
2. **Yeni sinyaller modelin zaten örtük olarak çıkarabildiği bilgiyle örtüşüyor olabilir.** Mobilite (bir taşın legal hamlesi olup olmadığı), tahtanın deterministik bir fonksiyonu — yeterince kapasiteli bir transformer, dikkat mekanizmasıyla bunu zaten yaklaşık olarak öğrenebilir. Açıkça vermek modele hesaplama tasarrufu sağlar ama illa yeni bir tahmin gücü eklemez. Son hamle kareleri gerçekten yeni bilgi taşır (statik pozisyondan çıkarılamaz, transpozisyon belirsizliğini çözer) ama bunun bilgi içeriği sınırlı — sadece 1 yarı-hamlelik bağlam, çok atlı planları/tehditleri yakalamıyor.
3. **Optimizasyon/adım bütçesi etkileşimi.** Yeni embedding'ler (`mobility_embed`, 2 yeni extra-token tipi) sıfırdan başlıyor ve kendi "ısınma" sürelerine ihtiyaç duyuyor. run4'ün eğrisine göre ayarlanmış aynı `--lr-patience`/`--patience` tarifini run5'e olduğu gibi uygulamak, bu yeni parametrelerin tam faydaya dönüşmesinden önce LR'yi düşürüp erken durdurmuş olabilir — run5 (step 108000) run4'ten (step 130000) daha az adımda durdu.
4. **Füzyon mekanizması kaba kalmış olabilir.** Mobilite kare başına tek bir 0/1 embedding olarak ekleniyor — düşük bilgi yoğunluklu bir sinyal. Son hamle kareleri de 72 token'ın sadece 2 tanesi; dikkat mekanizması içinde "seyrelmiş" olabilirler. Daha güçlü bir füzyon (örn. sayısal özellik, tekrarlanan kanallar, veya modelin girdi işleme kısmına özel bir alt-katman) daha büyük bir fark yaratabilirdi.

Aksiyon önerisi: Bu analiz koda dönüştürülmedi, sadece belgelendi. Sıradaki en makul deney adımı (konuştuğumuz gibi) veri miktarını artırmak (16 ay) — eğer bu da benzer küçük bir kazanç getirirse, 1. madde (gürültü tavanı) güçlenir; belirgin bir kazanç getirirse, run3→run4→run5 zincirinin asıl darboğazının veri çeşitliliği olduğu doğrulanmış olur.

## Sıradaki adım

run6, son kurtarılabilir checkpoint'ten (step 78000, %49.82) 600s rclone timeout'uyla devam ettirilecek — hedef, step 88000'in %50.61'ini (veya daha iyisini) bu sefer gerçekten Drive'a güvenle kaydetmek.
