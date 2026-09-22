# İlerleme Notları

Her ajan/oturum, bir iş birimini bitirdikten sonra buraya kısa bir not düşer. Kural ve format için [CLAUDE.md](../../CLAUDE.md) dosyasına bak. Yeni notlar **en üste** eklenir (en yeni en üstte).

Format:

```
## YYYY-MM-DD — Kısa başlık

- Aşama: (ör. Aşama 1 — Veri Toplama)
- Yapıldı: ne tamamlandı, hangi dosyalar değişti
- Sıradaki adım / blocker: varsa
```

---

## 2026-09-22 — Aşama 3: Tablebase entegrasyonu tamamlandı

- Aşama: Aşama 3 — Tablebase Entegrasyonu
- Yapıldı: `src/tablebase.py` eklendi — `chess.syzygy.open_tablebase()` etrafında ince bir sarmalayıcı: `should_probe(board, piece_threshold=5)` taş sayısına bakıp tablebase'e mi danışılacağına karar veriyor, `best_move(board, tablebase)` her legal hamleyi deneyip `(bizim WDL, rakibin DTZ)` sözlük sırasına göre en iyisini seçiyor (WDL sınıfları arası ham DTZ karşılaştırması güvensiz olduğundan — "cursed win"/"blessed loss" DTZ değerleri ±100'ün dışında kodlanıyor — önce WDL sınıfı, sadece eşitlikte DTZ tie-break kullanılıyor). Eksik tablebase dosyası/rok hakkı/taş sayısı gibi durumlarda `KeyError` yakalanıp `None` dönülüyor, çökmüyor (çağıran ANN'a düşüyor). `tests/test_tablebase.py`: fil-at matı (KBNvK), temel kale matı (KRvK) ve bilerek "yanlış hamle kazananı elden kaçırır" tuzağı kurulmuş bir piyon yarışı (KPvKP, çoğu kral hamlesi kazancı berabereye/kayba çeviriyor, sadece doğru hamle kazancı koruyor) test ediliyor — hepsi genel bir "best_move, herhangi bir legal hamlenin ulaşabileceği en iyi WDL sonucuna ulaşır" assertion'ıyla doğrulanıyor, ayrıca eksik tablo durumunda çökmediği de test ediliyor. `.venv/Scripts/python.exe tests/test_tablebase.py` → `OK - all tablebase checks passed`. Test pozisyonları için gereken 3/4 taşlı Syzygy dosyaları (`KRvK`, `KBNvK`, `KPvK`, `KPvKP` — wdl+dtz, tablebase.lichess.ovh birincil mirror'dan) `data/tablebase/`'e indirildi, toplam ~789 KB (tam 3-4-5 setinin ~1GB'ına kıyasla önemsiz) — commit'e dahil edilmedi, `.gitignore`'a `*.rtbw`/`*.rtbz` eklendi (zaten `data/` de ignore'luydu, ekstra güvenlik).
- Sıradaki adım/not: Aşama 3 kapsamındaki üç checklist maddesi (indirme, entegrasyon, karar mantığı, zor oyun sonu testleri) tamam. Gerçek motora bağlanması (failsafe/inference döngüsünde `should_probe`+`best_move` çağrısı) Aşama 4/5'in işi. Tam 3-4-5 Syzygy setinin indirilmesi (gerçek deployment için gerekecek) hâlâ kullanıcı onayı bekliyor — bu oturum sadece test için gereken birkaç dosyayı çekti.

## 2026-09-22 — Aşama 1 filtreleri gerçek veride doğrulandı

- Aşama: Aşama 1 — Veri Toplama ve Hazırlama
- Yapıldı: `src/build_dataset.py`'nin filtre mantığı, 2026-08 dump'ından 3000 oyunluk canlı bir stream üzerinde ayrı bir tanılama script'iyle (proje deposuna dahil edilmedi, tek seferlik) doğrulandı. Sonuçlar: bullet/ultrabullet reddi %46.6, kalanın Elo bandı (2000-2200) reddi %50.3, toplamda oyunların %3.1'i kalifiye oldu (5 örnek: hepsi doğru şekilde "Rated Blitz game", Elo'lar bant içinde). Kalifiye oyunlardaki 6763 ply'da: `%clk` anotasyonu örneklenenlerin tamamında mevcuttu (0 eksik), açılış atlama (`skip-plies=10`) ply'ların %13.6'sını, zaman-baskısı filtresi (`<30s`) kalanların %10'unu elemiş — hiçbir filtre aşırı agresif/gevşek değil, tasarım varsayımlarıyla tutarlı.
- Sıradaki adım/not: %3.1 kalifikasyon oranıyla bir aylık dump muhtemelen hedef "birkaç milyon pozisyon"a tek başına ulaşır (doğrulanmadı, sadece oran ekstrapolasyonu). Aşama 3 (Tablebase Entegrasyonu) paralel bir ajana devredildi.

## 2026-09-22 — Plan dokümantasyonu kuruldu

- Aşama: Aşama 0 (planlama, henüz kod yok)
- Yapıldı: Proje planı araştırmayla doğrulanıp somutlaştırıldı, `docs/plans/` altında aşama başına ayrı dosyalara bölündü, `docs/reference/` altında teknoloji yığını ve kaynak adresleri dokümante edildi, kök dizine `requirements.txt` ve `CLAUDE.md` eklendi.
- Sıradaki adım: Aşama 1 — database.lichess.org'dan bir aylık PGN.zst dökümünü indirip streaming parse script'ine başlamak.
