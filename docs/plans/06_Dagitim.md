# Aşama 6: Dağıtım (Opsiyonel)

[← Aşama 5 — Değerlendirme ve Test](05_Degerlendirme_ve_Test.md) · [Genel Bakışa dön](00_Genel_Bakis.md)

Bu aşama ikiye ayrılıyor. **Sargı/entegrasyon kısmı erken yapılır** (bkz. [Aşama 2](02_Model_Mimarisi_ve_Egitim.md)'nin son maddesi): tam eğitim bitmeden, ilk çalışan checkpoint'le lichess-bot bağlantısı kurulur — amaç model kalitesini değil, boru hattının (legal hamle maskeleme, promosyon/rok/en passant, tablebase/failsafe tetikleme eşikleri) doğru çalıştığını ucuza doğrulamak. Burada kalan iş, **tam eğitilmiş (best-checkpoint) modeli** bağlayıp gerçek/geniş çaplı canlı test yapmak — istersen bitişmiş sistemi bir Lichess hesabına bağlayıp gerçek insanlara ve diğer botlara karşı test etme imkanı bulursun. Bu genişletilmiş kısım tamamen opsiyonel, projenin teknik hedefini etkilemiyor ama somut bir gösterim ve daha geniş test alanı sağlıyor.

## Yapılacaklar

- [x] UCI protokolünü sıfırdan yazma: `src/engine.py`'deki `NeuralChessEngine` (tablebase → ANN top-k → failsafe sırasıyla) + `scripts/lichess_bot_homemade.py` (lichess-bot'un `Homemade`/`MinimalEngine` arayüzüne ince bir adaptör) — `tests/test_engine.py`'de tablebase önceliği, ANN yolu, failsafe kablolaması ayrı ayrı doğrulandı
- [x] İlk (gerçek, `run2/best.pt`) checkpoint'le sargı buglarını yakalama: Lichess hesabı gerekmeden 3 self-play parti oynatıldı (`selfplay_smoke.py`, tek seferlik doğrulama) — çökme yok, illegal hamle yok, failsafe partide ortalama 4 kez tetiklendi (aşırı sık değil)
- [x] lichess-bot bir Lichess bot hesabına bağlandı (`C:\Users\meric\lichess-bot`, ayrı bir checkout — `meric_bot`, en az bir gerçek parti oynandı, maia1'e karşı, 2026-09-23). 2026-09-30: `homemade.py`'daki `CHECKPOINT_PATH` varsayılanı `run6/best.pt`'ye (val_top1 %50.06, GAB+SEE mimarisi) güncellendi ve `NeuralChessEngine` üzerinden üç pozisyonla (açılış, orta oyun, asılı taş) smoke test edildi — hepsi legal hamle, failsafe asılı veziri doğru filtreledi. Not: `homemade.py`'ın `NEURAL_CHESS_SRC_DIR`'ı `chess_bot` reposunun **diskteki mevcut branch'ine** doğrudan bakıyor (git-branch-farkında değil) — repo şu an `run6-gab-see`'de kaldığı sürece doğru çalışır, başka bir branch'e geçilirse checkpoint/mimari uyuşmazlığı olur.
- [ ] Oynanan partileri kaydet, ileri iterasyon için kullan

## Tahmini Süre

Sargı kısmı Aşama 2 ile paralel gittiği için ek süre saymaz; sonda kalan geniş test 1 gün.
