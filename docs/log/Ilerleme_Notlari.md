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

## 2026-09-22 — Aşama 2: Model mimarisi ve eğitim döngüsü kuruldu

- Aşama: Aşama 2 — Model Mimarisi ve Eğitim
- Yapıldı: `src/model.py` eklendi — `ChessTransformer`: tahta 64 kare token'ı (taş tipi+renk embedding'i + öğrenilen kare pozisyon embedding'i, `encoding.py`'nin 12 taş kanalından `argmax` ile türetiliyor) artı sırası kimde/4 rok hakkı/en-passant için 6 ek token (paylaşılan bir "flag" (0/1) embedding'i + token-tipi embedding'i; en-passant token'ı ayrıca hedef karenin pozisyon embedding'ini de alıyor) — toplam 70 token, `nn.TransformerEncoder` (varsayılan 6 katman, d_model 256, 8 head) üzerinden geçip 64 kalkış-kare ve 64 varış-kare logit'i üreten iki ayrı linear head'e (her kare token'ının kendi çıktısından tek bir skaler) besleniyor. Legal hamle maskeleme üç fonksiyonla: `legal_move_mask(board)` (64x64 bool), `mask_move_logits(from_logits, to_logits, mask)` (birleşik skor matrisi, illegal hücreler -inf), `select_legal_move(...)` (en yüksek skorlu legal hamle, promosyonda vezir tercih ediliyor). `src/train.py` eklendi — `ShardDataset` (shard_*.npz dizinini tek bir flat dataset gibi indeksliyor, aktif shard'ı cache'liyor; `shuffle=True` ile global rastgele erişimde her örnekte shard değişebilir, toy-scale için sorun değil ama gerçek ölçekte yavaş — koda `ponytail:` notu düşüldü, shard-aware sampler upgrade yolu), Adam + iki başlık için ayrı cross-entropy toplamı, `--val-interval` adımda (epoch sonu değil) top-1/top-3 değerlendirme (birleşik from+to skor matrisinden gerçek hamlenin sırası), sadece val top-1 iyileştiğinde `torch.save` ile checkpoint (`model_state_dict` + `model_args` + step/top1/top3), TensorBoard (`SummaryWriter`), argparse CLI (batch size, lr, d_model/nhead/num_layers/dim_feedforward/dropout, val-interval, max-steps, vb.).
- Doğrulama: `tests/test_model.py` (forward shape kontrolü, legal-move mask'in startpos'ta bilinen legal/illegal kare çiftlerini doğru işaretlediği, `select_legal_move`'un her zaman legal bir hamle döndürdüğü, checkpoint save/load round-trip'inin ağırlıkları ve çıktıyı koruduğu) → `.venv/Scripts/python.exe tests/test_model.py` → `OK - all model checks passed`. Toy uçtan uca test: `python src/build_dataset.py --source .../lichess_db_standard_rated_2026-08.pgn.zst --out-dir data/toy --max-games 300` (300 oyun, 17698 pozisyon, train 17331/val 367 — commit'lenmedi, `data/` zaten gitignore'lu), ardından küçük hiperparametrelerle (`d_model 64, nhead 4, num_layers 2, dim_feedforward 128, batch 32`) lokal CPU'da 400 step `src/train.py` çalıştırıldı: loss 6.98→5.75, val top-1 %0.3→%4.1 (300 oyunluk toy set için anlamlı bir sayı değil, sadece akışın çalıştığının kanıtı), her 50 step'te val ölçüldü ve iyileşince checkpoint yazıldı (`checkpoints/toy/best.pt`, commit'lenmedi), checkpoint ayrı bir Python sürecinde `model_args`'tan yeniden kurulup başarıyla yüklendi. Hiçbir adımda çökme olmadı.
- Sıradaki adım/not: Bu sadece doğruluk/akış testi — gerçek eğitim (tam veri seti, kiralık RTX 4090, batch 256-512) planın sıradaki maddesi, kullanıcı onayı/bütçesi olmadan başlatılmadı. Mimari sabit öğrenilen kare embedding'i kullanıyor (Chessformer'ın GAB'ı değil) — plandaki not gereği bu v1 için kasıtlı, isabet oranı hedefin altında kalırsa denenecek bir yükseltme. `torch` bu ortamda CPU-only kuruldu (`torch 2.14.0+cpu`) — GPU kiralamasında CUDA wheel'i ayrıca kurulmalı. İlk çalışan checkpoint'in Aşama 6 (`Homemade` sargısı) ile erken entegrasyonu ve Aşama 4 (Failsafe) hâlâ başlanmadı.

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
