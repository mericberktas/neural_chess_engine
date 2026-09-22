# Aşama 3: Tablebase Entegrasyonu

[← Aşama 2 — Model Mimarisi ve Eğitim](02_Model_Mimarisi_ve_Egitim.md) · Sonraki: [Aşama 4 — Failsafe Katmanı](04_Failsafe_Katmani.md)

3-5 taşlı Syzygy tablebase'leri (doğrulandı: WDL dosyaları ~378MB + DTZ dosyaları ~561MB, toplam ~1GB'ın altında, ücretsiz) indirip python-chess'in `chess.syzygy.open_tablebase()` / `probe_wdl()` / `probe_dtz()` fonksiyonlarıyla entegre et. Tahtadaki toplam taş sayısı belirlenen eşiğin (örneğin 5) altına düştüğünde, ANN'ı tamamen devre dışı bırakıp tablebase'in önerdiği kusursuz hamleyi oyna. Not: DTZ sorgusu için hem WDL hem DTZ dosyaları birlikte gerekiyor, sadece WDL yetmiyor. İndirme adresleri için [Teknoloji Yığını ve Kaynaklar](../reference/Teknoloji_Yigini_ve_Kaynaklar.md) dosyasına bak.

## Yapılacaklar

- [ ] 3-5 taşlı Syzygy WDL ve DTZ dosyalarını indir
- [ ] python-chess ile tablebase sorgulama fonksiyonunu entegre et
- [ ] Karar mantığını ekle: taş sayısı eşiğin altındaysa ANN yerine tablebase hamlesini kullan
- [ ] Bilinen zor oyun sonlarında (fil-at matı, temel kale matı, basit piyon yarışları) doğru çalıştığını test et

## Tahmini Süre

1-2 gün, çoğunlukla entegrasyon ve test.
