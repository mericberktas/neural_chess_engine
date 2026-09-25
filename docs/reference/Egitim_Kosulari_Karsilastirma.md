# Eğitim Koşuları Karşılaştırması (run1-run5)

Şimdiye kadar tamamlanan 5 gerçek GPU eğitim koşusunun neyi nasıl farklı yaptığının özeti. Ayrıntılı olay anlatımı (bug'lar, operasyonel arızalar vb.) için [docs/log/Ilerleme_Notlari.md](../log/Ilerleme_Notlari.md)'ye bakın — bu dosya sadece koşular arası metodolojik farkları ve sonuçları özetler.

run1-run4'te mimari **sabit**: `ChessTransformer`, batch 256, d_model 256, nhead 8, num_layers 6, dim_feedforward 1024, dropout 0.1, ~4.76M parametre, 18 kanallı girdi. run5, `run5-rich-encoding` branch'inde girdiyi 21 kanala çıkardı (mimari boyutu aynı kaldı) — bkz. [docs/plans/02_Model_Mimarisi_ve_Egitim.md](../plans/02_Model_Mimarisi_ve_Egitim.md).

## Özet tablo

| | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|
| Veri | 1 ay (2026-08), 40K oyun, 2.3M pozisyon | 4 ay (2026-08→05), ~9M pozisyon | 8 ay (2026-01→08), 18.0M train pozisyonu | 8 ay (run3 ile aynı, sıfırdan yeniden indirildi) | 8 ay (run3/4 ile aynı, Drive'daki harvest'ten) |
| Test ayı (zaman-ayrık) | 2026-04 | 2026-04 | 2025-12 | 2025-12 | 2025-12 (build sırasında donma nedeniyle atlandı — eğitim buna bağlı değil) |
| Girdi kanalları | 18 | 18 | 18 | 18 | **21** (+mobilite maskesi, +son hamle kalkış/varış kareleri) |
| Regülarizasyon (AdamW weight-decay, label smoothing) | Yok | Yok | Var (0.01 / 0.1) | Var (run3 ile aynı) | Var (aynı) |
| LR programı | Sabit 3e-4 | Sabit 3e-4 | Sabit 3e-4 | `ReduceLROnPlateau` (patience 1, factor 0.5) | Aynı (sıfırdan) |
| Resume | — (ilk koşu) | — (sıfırdan) | — (sıfırdan) | run3'ün checkpoint'inden (step 64000) — **sadece ağırlıklar** | — (mimari uyumsuz, sıfırdan zorunlu) |
| Early-stop patience | — (max_steps/1 epoch ile sınırlı) | 3 | 3 | 4 | 4 |
| Sonuç (best) | step 8000, val_top1 **%32.9**, val_top3 %47.1 | step 68000, val_top1 **%45.35**, val_top3 %59.82 | step 64000, val_top1 **%45.08**, val_top3 %59.53 | step 130000, val_top1 **%50.02**, val_top3 %63.28 | step 108000, val_top1 **%50.18**, val_top3 %63.54 |
| Nasıl bitti | Manuel doğrulama sonrası `terminate_pod` | Gerçek early-stopping | Gerçek early-stopping (kendi kendine self-terminate) | Gerçek early-stopping (kendi kendine self-terminate) | Kendi kendine self-terminate (early-stop/watchdog — pod log'u pod silindiği için sonradan doğrulanamadı) |
| Checkpoint boyutu | ~19MB (sadece ağırlık) | ~19MB (sadece ağırlık) | ~19MB (sadece ağırlık) | ~57MB (ağırlık + optimizer state) | ~57MB (aynı format) |
| Drive yolu | `gdrive:chess_bot/checkpoints/run1/` | `gdrive:chess_bot/checkpoints/run2/` | `gdrive:chess_bot/checkpoints/run3/` | `gdrive:chess_bot/checkpoints/run4/` | `gdrive:chess_bot/checkpoints/run5/` |

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

16-17 aylık veri havuzu (2025-04→2026-08) Drive'da harvest edildi (`gdrive:chess_bot/filtered_pgn/`) — `NUM_MONTHS=16` ile kullanılabilir.

`run6-gab-see` branch'inde (run5'ten türetildi) girdiye bir SEE-riski kanalı eklendi (21→22 kanal) ve Chessformer'ın Geometric Attention Bias'ı (dinamik, tahta-durumuna-bağlı attention bias'ı) implemente edildi — ayrıntı için [docs/log/Ilerleme_Notlari.md](../log/Ilerleme_Notlari.md)'nin 2026-09-25 notuna bakın. Tüm lokal testler yeşil; henüz bir pod'da eğitilmedi (mimari değişikliği yüzünden sıfırdan eğitim gerekiyor, `--resume-from` mümkün değil).
