# Aşama 6: Dağıtım (Opsiyonel)

[← Aşama 5 — Değerlendirme ve Test](05_Degerlendirme_ve_Test.md) · [Genel Bakışa dön](00_Genel_Bakis.md)

Bu aşama ikiye ayrılıyor. **Sargı/entegrasyon kısmı erken yapılır** (bkz. [Aşama 2](02_Model_Mimarisi_ve_Egitim.md)'nin son maddesi): tam eğitim bitmeden, ilk çalışan checkpoint'le lichess-bot bağlantısı kurulur — amaç model kalitesini değil, boru hattının (legal hamle maskeleme, promosyon/rok/en passant, tablebase/failsafe tetikleme eşikleri) doğru çalıştığını ucuza doğrulamak. Burada kalan iş, **tam eğitilmiş (best-checkpoint) modeli** bağlayıp gerçek/geniş çaplı canlı test yapmak — istersen bitişmiş sistemi bir Lichess hesabına bağlayıp gerçek insanlara ve diğer botlara karşı test etme imkanı bulursun. Bu genişletilmiş kısım tamamen opsiyonel, projenin teknik hedefini etkilemiyor ama somut bir gösterim ve daha geniş test alanı sağlıyor.

## Yapılacaklar

- [ ] (Erken, Aşama 2 sırasında) UCI protokolünü sıfırdan yazma: resmi [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) (Python 3.10+, aktif proje) `Homemade` motor sınıfını destekliyor — modeli doğrudan bir Python sınıfı olarak bu arayüze eklemek, stdin/stdout üzerinden UCI mesajlaşması yazmaktan çok daha az kod
- [ ] (Erken) İlk checkpoint'le birkaç deneme/casual parti oynat, sargı buglarını (illegal hamle, crash, tablebase/failsafe'in hiç tetiklenmemesi) burada yakala
- [ ] (Sonda, best-checkpoint ile) lichess-bot'u tam eğitilmiş modelle bir Lichess hesabına bağla, geniş çaplı canlı test yap
- [ ] Oynanan partileri kaydet, ileri iterasyon için kullan

## Tahmini Süre

Sargı kısmı Aşama 2 ile paralel gittiği için ek süre saymaz; sonda kalan geniş test 1 gün.
