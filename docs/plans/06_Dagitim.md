# Aşama 6: Dağıtım (Opsiyonel)

[← Aşama 5 — Değerlendirme ve Test](05_Degerlendirme_ve_Test.md) · [Genel Bakışa dön](00_Genel_Bakis.md)

Bu aşama ikiye ayrılıyor. **Sargı/entegrasyon kısmı erken yapılır** (bkz. [Aşama 2](02_Model_Mimarisi_ve_Egitim.md)'nin son maddesi): tam eğitim bitmeden, ilk çalışan checkpoint'le lichess-bot bağlantısı kurulur — amaç model kalitesini değil, boru hattının (legal hamle maskeleme, promosyon/rok/en passant, tablebase/failsafe tetikleme eşikleri) doğru çalıştığını ucuza doğrulamak. Burada kalan iş, **tam eğitilmiş (best-checkpoint) modeli** bağlayıp gerçek/geniş çaplı canlı test yapmak — istersen bitişmiş sistemi bir Lichess hesabına bağlayıp gerçek insanlara ve diğer botlara karşı test etme imkanı bulursun. Bu genişletilmiş kısım tamamen opsiyonel, projenin teknik hedefini etkilemiyor ama somut bir gösterim ve daha geniş test alanı sağlıyor.

## Yapılacaklar

- [x] UCI protokolünü sıfırdan yazma: `src/engine.py`'deki `NeuralChessEngine` (tablebase → ANN top-k → failsafe sırasıyla) + `scripts/lichess_bot_homemade.py` (lichess-bot'un `Homemade`/`MinimalEngine` arayüzüne ince bir adaptör) — `tests/test_engine.py`'de tablebase önceliği, ANN yolu, failsafe kablolaması ayrı ayrı doğrulandı
- [x] İlk (gerçek, `run2/best.pt`) checkpoint'le sargı buglarını yakalama: Lichess hesabı gerekmeden 3 self-play parti oynatıldı (`selfplay_smoke.py`, tek seferlik doğrulama) — çökme yok, illegal hamle yok, failsafe partide ortalama 4 kez tetiklendi (aşırı sık değil)
- [ ] (Sonda, best-checkpoint ile) lichess-bot'u tam eğitilmiş modelle bir Lichess hesabına bağla, geniş çaplı canlı test yap — **kullanıcının kendi Lichess bot hesabını/API token'ını açması gerekiyor**, bu adım bekliyor
- [ ] Oynanan partileri kaydet, ileri iterasyon için kullan

## Tahmini Süre

Sargı kısmı Aşama 2 ile paralel gittiği için ek süre saymaz; sonda kalan geniş test 1 gün.
