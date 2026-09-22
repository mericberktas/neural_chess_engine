# Teknoloji Yığını ve Kaynaklar

2026-09-22 · @Someone

Bu doküman [Genel Bakış](../plans/00_Genel_Bakis.md) planında kullanılacak kütüphaneleri, araçları ve dış kaynak adreslerini listeler. Seçim kriteri: önce zaten kullanılan bağımlılık, sonra stdlib/framework içi özellik, en son yeni paket — gereksiz bağımlılık eklenmiyor.

## Dil ve Çerçeve

- **Python 3.11+** — lichess-bot'un gerektirdiği 3.10+ ile uyumlu, güncel dil özellikleri için.
- **PyTorch** — model tanımı (`nn.TransformerEncoder`), eğitim döngüsü, checkpoint, mixed precision, TensorBoard entegrasyonu hepsi tek framework içinde.

## Satranç Mantığı

- **`chess`** (PyPI paket adı — proje adı "python-chess" ama kurulum `pip install chess`, `pip install python-chess` farklı/eski bir pakettir, karıştırılmamalı) — PGN parse, board temsili, legal hamle üretimi/maskeleme, Syzygy tablebase probing (`chess.syzygy`). Tüm satranç ihtiyacını tek paket karşılıyor, ayrı bir tablebase kütüphanesi gerekmiyor.

## Veri İşleme ve Depolama

- **`zstandard`** — `.pgn.zst` dosyalarını diske tam açmadan akış halinde okumak için.
- **`numpy`** — pozisyon/etiket tensörlerini `.npy`/`.npz` shard'lar halinde diske yazmak ve okumak için. Şema basit (tensor + label + game_id) olduğundan parquet/pyarrow gibi ek bir bağımlılığa gerek yok.

## Eğitim Yardımcıları (PyTorch içinde, ekstra paket değil)

- `torch.utils.data.Dataset` / `DataLoader` — veri yükleme.
- `torch.autocast` + `torch.cuda.amp.GradScaler` — mixed precision, kiralık RTX 4090'da hız/bellek kazancı.
- `torch.utils.tensorboard.SummaryWriter` (bunun için `tensorboard` paketi kurulu olmalı) — loss/val top-1-top-3 eğrisi takibi. wandb gibi harici bir servise hesap açmaya gerek yok.
- `torch.save` / `torch.load` — checkpoint kaydı (best-checkpoint mantığı için).

## Ortam ve Versiyon Kontrolü

- **`venv`** + `requirements.txt` — poetry/conda'ya gerek yok, tek kişilik hobi projesi için fazladan karmaşıklık.
- **`git`** — repo `git@github.com:mericberktas/neural_chess_engine.git`, `main` branch. Büyük/ikili dosyalar `.gitignore`'da (bkz. CLAUDE.md).

## Checkpoint Yedekleme (Google Drive)

Kiralık GPU instance'ı silinince (`vastai destroy instance`) diskteki her şey gidiyor — checkpoint'i instance'tan çekmeden silmemek gerekiyor. Bunun için `rclone` + kişisel Google Drive (Google One, 5TB) kullanılıyor; service account **değil**, normal OAuth ile kişisel hesap bağlandı (service account'un kendi ayrı 15GB kotası var, 5TB'a erişemiyor).

- Kurulum: `winget install Rclone.Rclone` (Windows). Linux/macOS/kiralık instance: `curl https://rclone.org/install.sh | sudo bash`.
- rclone'un paylaşılan client_id'si 2026'da emekliye ayrılıyor — kendi Google Cloud OAuth client_id'ini oluşturup (`rclone config` sırasında sorulur, adımlar [rclone.org/drive/#making-your-own-client-id](https://rclone.org/drive/#making-your-own-client-id)) kullanmak gerekiyor. OAuth consent screen "Testing" modda kalıyor (uygulama Google'a doğrulatılmadı) — kendi hesabını **Test users** listesine eklemek şart, yoksa 403 access_denied hatası alınır.
- Remote adı: `gdrive:` (kişisel Drive, Shared Drive/Team Drive değil).
- Klasör: `gdrive:chess_bot/checkpoints/`.
- Kullanım: `rclone copy checkpoints/toy/best.pt gdrive:chess_bot/checkpoints/` (kiralık instance'tan da aynı komut, rclone.conf'u oraya kopyalayarak).
- Kota: günlük 750GB upload limiti var, bizim kullanım (checkpoint'ler + gerekirse veri shard'ları) bunun çok altında.

## Dağıtım

- **[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)** — pip paketi değil, ayrı bir git clone; kendi `requirements.txt`'i var. `Homemade` motor sınıfı üzerinden UCI protokolü yazmadan entegre edilir.

## Yerel Geliştirme ve Notebook İş Akışı

Lokal GPU mobil RTX 3050 4GB — gerçek eğitim için yetersiz (transformer + batch 256-512 büyük ihtimalle OOM verir), ama kod geliştirme için değerli: küçük bir toy veri setiyle (birkaç yüz pozisyon, birkaç adım) uçtan uca akışı (veri yükleme, forward/backward, checkpoint, legal hamle maskeleme) Kaggle saatini veya kiralık GPU parasını harcamadan lokalde doğrulamaya yarar.

Notebook hücrelerine mantık kodu yazılmıyor — git diff alınamıyor, test edilemiyor, debug etmesi zor. Bunun yerine:

- Tüm gerçek mantık (`data.py`, `model.py`, `train.py`, `tablebase.py`, `failsafe.py`) normal `.py` dosyaları olarak lokalde geliştirilir ve test edilir.
- Kod GitHub'a push edilir (public repo yeterli, private gerekirse Kaggle/Colab'a personal access token ile clone edilebilir).
- Notebook (Kaggle/Colab/kiralık makine) ince bir başlatıcı olarak kullanılır: `!git clone ...`, `!pip install -r requirements.txt`, sonra doğrudan script çalıştırılır — `!python train.py --config configs/base.yaml`. Fonksiyonları notebook'a `import` etmek yerine script çalıştırmak, kernel state'iyle uğraşmamayı (kod değişince kernel restart + yeniden import gerekmiyor) sağlar.
- Uzun eğitim koşuları arka planda başlatılır: `!nohup python train.py > log.txt 2>&1 &` sonra `!tail -f log.txt` ile izlenir — tarayıcı/notebook bağlantısı kopsa da eğitim devam eder.

## Dış Kaynak Adresleri

| Kaynak | Adres | Ne için |
| --- | --- | --- |
| Lichess oyun verisi | [database.lichess.org/standard/](https://database.lichess.org/standard/) | Aylık `.pgn.zst` dökümleri |
| Syzygy tablebase (birincil) | [tablebase.lichess.ovh/tables/standard/{3,4,5}-{wdl,dtz}/](https://tablebase.lichess.ovh/) | 3-5 taşlı WDL/DTZ dosyaları, Lichess'in kendi mirror'ı — daha hızlı |
| Syzygy tablebase (yedek) | [tablebase.sesse.net/syzygy/3-4-5/](http://tablebase.sesse.net/) | Aynı dosyalar, birincil kaynak yavaş/erişilemezse |
| GPU kiralama (tam ölçek) | [vast.ai](https://vast.ai/), [runpod.io](https://runpod.io/) | RTX 4090, hazır PyTorch docker image'ıyla ortam kurmadan başla |
| GPU (ücretsiz prototipleme) | [kaggle.com/code](https://www.kaggle.com/code) | Haftalık ~30 saat P100/dual-T4, Colab'a göre daha öngörülebilir kota |
| lichess-bot referans kodu | [github.com/lichess-bot-devs/lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) | Dağıtım entegrasyonu |
| python-chess dokümantasyonu | [python-chess.readthedocs.io](https://python-chess.readthedocs.io/) | API referansı |
