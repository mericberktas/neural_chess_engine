# Aşama 4: Failsafe Katmanı

[← Aşama 3 — Tablebase Entegrasyonu](03_Tablebase_Entegrasyonu.md) · Sonraki: [Aşama 5 — Değerlendirme ve Test](05_Degerlendirme_ve_Test.md)

ANN'ın önerdiği en olası hamleyi oynamadan önce, o hamlenin bedavaya bir taş kaptırıp kaptırmadığını kontrol eden ucuz bir statik kontrol ekle (basit bir static exchange evaluation veya tek adımlık materyal kontrolü yeterli). Kırmızı bayrak kalkarsa, ağın bir sonraki en olası hamlesine geç; bu katman nadiren devreye girmeli, amacı genel oyunu iyileştirmek değil sadece en bariz kaptırmaları engellemek.

## Yapılacaklar

- [ ] Hamle sonrası tahtada basit bir materyal/güvenlik kontrolü yaz (taş bedavaya asılı kalıyor mu)
- [ ] Kontrolü ANN'ın çıktısına bir filtre olarak ekle, gerektiğinde bir sonraki en olası hamleye düş
- [ ] Devreye girme sıklığını izle, çok sık tetiklenirse eşiği gözden geçir

## Tahmini Süre

1-2 gün.
