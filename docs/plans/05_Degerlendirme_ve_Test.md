# Aşama 5: Değerlendirme ve Test

[← Aşama 4 — Failsafe Katmanı](04_Failsafe_Katmani.md) · Sonraki: [Aşama 6 — Dağıtım](06_Dagitim.md)

İki ayrı değerlendirme türü yapılmalı. Birincisi nicel: tutulan test setindeki gerçek 2000-2200 bandı oyunlarına karşı top-1/top-3 isabet oranı ölçümü. İkincisi nitel: modeli farklı güçteki rakiplere (500, 1200, 2000 civarı bot veya insanlar) karşı canlı oynatıp özellikle bariz hatalara ve oyun sonu pozisyonlarına verdiği tepkiyi gözlemlemek, tablebase ve failsafe katmanlarının doğru tetiklenip tetiklenmediğini doğrulamak.

## Yapılacaklar

- [ ] Tutulan test seti üzerinde isabet oranı ölçümü
- [ ] Farklı reytinglerde canlı test partileri oyna, sonuçları not al
- [ ] Tablebase ve failsafe katmanlarının doğru senaryolarda devreye girdiğini doğrula
- [ ] Reyting bandı dışı rakiplere karşı orta oyunda beklenmedik davranışları belgele (bu bir hata değil, kabul edilen bir sınırlama — bkz. [Genel Bakış: Bilinen Sınırlamalar](00_Genel_Bakis.md#bilinen-sınırlamalar))

## Tahmini Süre

2-3 gün.
