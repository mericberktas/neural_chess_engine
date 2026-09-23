# Eğitim Koşuları Karşılaştırması (run1-run4)

Şimdiye kadar tamamlanan 4 gerçek GPU eğitim koşusunun neyi nasıl farklı yaptığının özeti. Ayrıntılı olay anlatımı (bug'lar, operasyonel arızalar vb.) için [docs/log/Ilerleme_Notlari.md](../log/Ilerleme_Notlari.md)'ye bakın — bu dosya sadece koşular arası metodolojik farkları ve sonuçları özetler.

Dört koşuda da mimari **sabit**: `ChessTransformer`, batch 256, d_model 256, nhead 8, num_layers 6, dim_feedforward 1024, dropout 0.1, ~4.76M parametre. Değişen tek şey veri miktarı, regülarizasyon/LR programı ve resume mekanizmasıdır — bkz. [docs/plans/02_Model_Mimarisi_ve_Egitim.md](../plans/02_Model_Mimarisi_ve_Egitim.md).

## Özet tablo

| | run1 | run2 | run3 | run4 |
|---|---|---|---|---|
| Veri | 1 ay (2026-08), 40K oyun, 2.3M pozisyon | 4 ay (2026-08→05), ~9M pozisyon | 8 ay (2026-01→08), 18.0M train pozisyonu | 8 ay (run3 ile aynı, sıfırdan yeniden indirildi) |
| Test ayı (zaman-ayrık) | 2026-04 | 2026-04 | 2025-12 | 2025-12 |
| Regülarizasyon (AdamW weight-decay, label smoothing) | Yok | Yok | Var (0.01 / 0.1) | Var (run3 ile aynı) |
| LR programı | Sabit 3e-4 | Sabit 3e-4 | Sabit 3e-4 | `ReduceLROnPlateau` (patience 1, factor 0.5) |
| Resume | — (ilk koşu) | — (sıfırdan) | — (sıfırdan) | run3'ün checkpoint'inden (step 64000) — **sadece ağırlıklar**, optimizer state run3'te henüz yoktu |
| Early-stop patience | — (max_steps/1 epoch ile sınırlı) | 3 | 3 | 4 (LR decay'e nefes payı için 3→4) |
| Sonuç (best) | step 8000, val_top1 **%32.9**, val_top3 %47.1 | step 68000, val_top1 **%45.35**, val_top3 %59.82 | step 64000, val_top1 **%45.08**, val_top3 %59.53 | step 130000, val_top1 **%50.02**, val_top3 %63.28 |
| Nasıl bitti | Manuel doğrulama sonrası `terminate_pod` | Gerçek early-stopping | Gerçek early-stopping (kendi kendine self-terminate) | Gerçek early-stopping (kendi kendine self-terminate) |
| Checkpoint boyutu | ~19MB (sadece ağırlık) | ~19MB (sadece ağırlık) | ~19MB (sadece ağırlık) | ~57MB (ağırlık + optimizer state) |
| Drive yolu | `gdrive:chess_bot/checkpoints/run1/` | `gdrive:chess_bot/checkpoints/run2/` | `gdrive:chess_bot/checkpoints/run3/` | `gdrive:chess_bot/checkpoints/run4/` |

## Koşu koşu neden/ne değişti

**run1 — ilk temel çizgi.** Tek ay, tek epoch, hiçbir regülarizasyon/erken-durdurma/resume mekanizması henüz yoktu — pipeline'ın uçtan uca gerçekten çalıştığını doğrulamak içindi. %32.9 sonucu plandaki aspirasyonel hedefin (Chessformer/Maia-3 %57.1) altında ama beklenen bir ilk-koşu sonucuydu.

**run2 — veri 4 kat arttı, ilk gerçek early-stopping.** `--patience` flag'i bu koşu için eklendi (`train.py`), veri 4 aya çıktı. val_top1 45.35'e sıçradı ama step 68000 civarında plato yaptı; bu plato **regülarizasyonun eklenmesini tetikledi** (`train.py` docstring'inde de not edilir: "Added after run2 ... plateaued at val_top1 45.35%").

**run3 — 8 ay + regülarizasyon, ama hâlâ sabit LR.** Weight-decay (AdamW) ve label-smoothing eklendi, veri 8 aya çıktı (18M pozisyon). Beklenti fazlasıyla veri + regülarizasyonun overfitting'i geciktirip daha uzun/daha iyi bir eğrisi olmasıydı — ama sonuç run2'ye çok yakın (%45.08) çıktı, üstelik epoch 0'ın içinde (henüz 1 epoch bile bitmeden) platoladı. Bu, sorunun **veri miktarı değil optimizasyon** (sabit LR) olduğunu gösterdi.

**run4 — resume + LR decay, hedef aşıldı.** İki yeni mekanizma birlikte eklendi: `--resume-from` (bir checkpoint'ten ağırlık/optimizer/step/best-val_top1 geri yükleme) ve `torch.optim.lr_scheduler.ReduceLROnPlateau` (val_top1 platoya girince LR'yi otomatik yarılama). run3'ün checkpoint'inden (step 64000) devam edildi — ama run3'ün checkpoint'i optimizer state içermiyordu (o zaman henüz eklenmemişti), o yüzden optimizer sıfırdan ısındı, sadece model ağırlıkları ve step sayacı gerçekten "resume" oldu. LR 5 kez yarılandı (3e-4 → 9.37e-06) ve her yarılanmada küçük ama gerçek bir sıçrama geldi: %45.08 (resume noktası) → %49.34 → %49.69 → **%50.02**. Maia'nın (GitHub'daki gerçek `maia_config.yaml`, 6x64-SE LCZero ağı, muhtemelen bizim 4.76M parametremizden küçük) kademeli LR programı kullandığını görmek bu fikri doğrudan tetikledi.

## Sıradaki adım

run5: girdi kodlamasını zenginleştirme (mobilite/legal-hamle maskesi + son hamle kareleri, `NUM_CHANNELS` 18→21) — mimari uyumsuz olduğu için resume mümkün değil, sıfırdan bir koşu olacak. Ayrı bir git branch'inde geliştiriliyor (bkz. commit geçmişi), henüz bir pod'da çalıştırılmadı.
