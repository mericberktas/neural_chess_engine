# Devir Notu (2026-09-22)

Geçici dosya — yeni bir worktree/oturum başlarken hızlı yön bulmak için. Kalıcı proje durumu her zaman `docs/log/Ilerleme_Notlari.md`'de; bu dosya sadece oraya bir giriş kapısı. İşin bir kısmını devraldıktan sonra silinebilir.

## Önce oku (sırayla)

1. **`CLAUDE.md`** (kök dizin) — kurallar, dosya haritası, iş günlüğü kuralı. Her oturumun ilk okuması gereken dosya.
2. **`docs/plans/00_Genel_Bakis.md`** — projenin tamamı: kapsam, 6 aşama, bütçe, bilinen sınırlamalar.
3. Üzerinde çalışacağın aşamanın dosyası: `docs/plans/0N_*.md`.
4. **`docs/log/Ilerleme_Notlari.md`** — en üstten aşağı oku, bugüne kadar tam olarak ne yapıldığının detaylı kaydı (bu dosyadaki özetten çok daha ayrıntılı).

## Şu ana kadar yapılanlar (özet)

- **Aşama 1 (Veri)** — `src/encoding.py` (Maia-2 18 kanal encoding) + `src/build_dataset.py` (stream/filtrele/shard'la) yazıldı. Gerçek canlı Lichess dump'ında hem uçtan uca duman testiyle hem ayrı bir filtre-oranı analiziyle doğrulandı (bullet %46.6, Elo bandı %50.3 eliyor, toplam kalifikasyon %3.1 — sağlıklı). **Henüz gerçek ölçekte (tam aylık dump) çalıştırılmadı**, sadece küçük örneklerle.
- **Aşama 3 (Tablebase)** — `src/tablebase.py` + `tests/test_tablebase.py` yazıldı, testler geçiyor. Sadece test için gereken birkaç küçük Syzygy dosyası (~789KB) indirildi; **tam 3-4-5 seti (~1GB) henüz indirilmedi**, gerçek dağıtımdan önce kullanıcı onayı gerekiyor.
- **Aşama 2 (Model), Aşama 4 (Failsafe), Aşama 5 (Değerlendirme)** — başlanmadı.
- Git: `main` branch, GitHub'a push edilmiş durumda, en son commit `b1cffce`.

## Bilmen gereken pratik detaylar

- Bu worktree'de `.venv` yok (gitignore'lu, sadece ana checkout'ta var) — `python -m venv .venv` + `pip install -r requirements.txt` ile kendi ortamını kur. `torch`/`tensorboard` henüz hiçbir yerde kurulu değil, Aşama 2'ye başlıyorsan onları da ekle.
- Bu ortamda Agent tool'unun `isolation: worktree` özelliği çalışmıyor ("WorktreeCreate hook yok") — paralel iş için manuel worktree (senin şu an yaptığın gibi) veya izolasyonsuz arka plan ajanı kullanılıyor.
- **Push etmeden önce kullanıcıya sor / onay al** — başka bir oturum aynı anda `main`'e push ediyor olabilir. Commit'i lokalde bırakmak güvenli, push paylaşılan bir eylem.
- Sadece kendi dosyalarını stage'le (`git add -A` değil), başka bir oturumun üzerinde çalıştığı dosyalara dokunma.

## Sırada ne var

Aşama 2 (Model Mimarisi ve Eğitim) veya Aşama 4 (Failsafe Katmanı) — ikisi de birbirinden bağımsız, hangisiyle başlanacağı henüz kullanıcı tarafından seçilmedi. `docs/plans/02_Model_Mimarisi_ve_Egitim.md` ve `docs/plans/04_Failsafe_Katmani.md`'ye bak.
