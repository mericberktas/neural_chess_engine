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

## 2026-09-22 — Plan dokümantasyonu kuruldu

- Aşama: Aşama 0 (planlama, henüz kod yok)
- Yapıldı: Proje planı araştırmayla doğrulanıp somutlaştırıldı, `docs/plans/` altında aşama başına ayrı dosyalara bölündü, `docs/reference/` altında teknoloji yığını ve kaynak adresleri dokümante edildi, kök dizine `requirements.txt` ve `CLAUDE.md` eklendi.
- Sıradaki adım: Aşama 1 — database.lichess.org'dan bir aylık PGN.zst dökümünü indirip streaming parse script'ine başlamak.
