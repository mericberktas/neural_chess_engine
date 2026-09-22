# Aşama 1: Veri Toplama ve Hazırlama

[← Genel Bakış](00_Genel_Bakis.md) · Sonraki: [Aşama 2 — Model Mimarisi ve Eğitim](02_Model_Mimarisi_ve_Egitim.md)

Bu aşamanın çıktısı, model eğitimine hazır, pozisyon-hamle çiftlerinden oluşan bir veri seti ve bunu üreten tekrarlanabilir bir script. Kaynak, database.lichess.org üzerindeki aylık PGN dökümleri; zstandard ile akış halinde okuyarak tüm dosyayı diske açmadan işlemek mümkün.

## Yapılacaklar

- [ ] database.lichess.org'dan 1-2 aylık PGN.zst dökümünü indir (güncel aylık dosyalar sıkıştırılmış ~20-30GB civarında; `list.txt` üzerinden tam boyut kontrol edilebilir)
- [ ] python-chess ve zstandard ile oyunları akış halinde ayırıştıran bir script yaz
- [ ] Filtreleri uygula, doğrudan PGN etiketlerinden: `WhiteElo`/`BlackElo` her ikisi 2000-2200, `Event` alanında "Bullet"/"UltraBullet" geçmiyor (Lichess bu etiketi zaten "Rated Blitz game" gibi biçimlendiriyor, TimeControl'ü ayrıştırmaya gerek yok), hamle içi `[%clk ...]` yorumundan kalan süre 30 saniyenin altındaysa o hamleyi at, ilk 5-6 tam hamleyi (kitap/açılış aşaması) at
- [ ] Kalan oyunlardan (pozisyon, oynanan hamle) çiftlerini çıkar, Maia-2'nin 18 kanallı 8x8 tensor kodlamasını birebir kullan: kanal 1-12 = taş tipi×renk (pawn/knight/bishop/rook/queen/king × beyaz/siyah), kanal 13 = sırası kimde (tamamı 1 beyazsa), kanal 14-17 = 4 rok hakkı, kanal 18 = en passant karesi — bu şema Maia-2 makalesinde tanımlı, yeniden icat etmeye gerek yok
- [ ] Hedef büyüklük: birkaç milyon etiketli pozisyon; ayrı bir zaman diliminden (farklı bir ay veya son birkaç gün) bağımsız bir **test** seti ayır (eğitim boyunca hiç dokunulmaz, sadece Aşama 5'te bir kez kullanılır)
- [ ] Kalan (train) veriden ayrıca, **oyun bazında** (pozisyon bazında değil — aynı oyundaki pozisyonlar birbirine çok benzediği için pozisyon bazlı split sızıntıya yol açar) küçük bir **val** dilimi ayır; 30-50 bin pozisyon top-1/top-3 gibi bir doğruluk metriği için istatistiksel olarak yeterli, val'i büyütmek eğitim verisini boşa harcar
- [ ] Veri setini sıkıştırılmış shard'lar halinde (numpy veya parquet) diske yaz

## Tahmini Süre

3-5 gün, çoğunlukla script yazma ve filtreleme mantığını doğrulama. Not: 2000-2200 bandı toplam oyunların küçük bir yüzdesi (reyting dağılımı ~1500 civarında tepe yapıyor), tek bir aylık döküm muhtemelen yeterli olur; yetmezse ikinci ay eklenir.
