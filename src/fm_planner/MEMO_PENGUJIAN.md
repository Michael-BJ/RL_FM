# MEMO — Pengujian kualitas model FM TIDAK BOLEH memakai escape mode

**Dibuat 2026-07-27** setelah rangkaian uji SITL di world `poles` yang menunjukkan
escape mencemari hasil pengamatan secara sistematis.

---

## Aturan inti

> **Setiap pengujian yang bertujuan menilai KUALITAS MODEL (FM) wajib
> dijalankan dengan escape MATI.** Escape hanya boleh aktif untuk mode
> deployment/koleksi data, bukan untuk mengukur perilaku model.

### Perintah yang BENAR untuk pengujian model

```bash
ros2 launch fm_planner fm_planning_unknown.launch.py \
    model_path:=$HOME/saved_net/fm/run_20260724_190037/fm_planner_20260724_190037.pth \
    goal_x:=24.0 K:=8 n_steps:=2 publish_3d_map:=true \
    use_safety_guards:=false  use_lookahead_guard:=true  guard_clearance:=0.60
```

Cek banner startup — WAJIB terbaca:

```
Guard mode    : GUARDS 1+2, invalidate+hover, NO escape
```

Kalau terbaca `Guard mode : FULL (guards+escape)`, **hasilnya tidak sah untuk
menilai model** — ulangi dengan flag di atas.

> ⚠️ **KOREKSI 2026-07-27 (sore).** Versi pertama memo ini keliru: mode uji saat
> itu (banner lama `LOOK-AHEAD only`) ternyata **mematikan Guard 1 sepenuhnya** —
> Guard 1 (yang memeriksa POSISI DRONE SENDIRI) dulu hanya berjalan di bawah
> `self._use_guards`, sedangkan Guard 2 cuma memvalidasi TRAJEKTORI yang
> direncanakan. Akibatnya tidak ada yang memeriksa di mana drone benar-benar
> berada. Terukur di `gauntlet_train2`: clearance merosot
> **0.67 → 0.47 → 0.34 → 0.15 m** (pusat 0.45 m, jauh di bawah ambang 0.60)
> selama ~15 detik **tanpa satu pun guard menyala**, sampai bodi menyentuh
> dinding dan impuls Gazebo melempar drone ke **35 m/s**. World lapang
> (`poles`) tidak pernah mengekspos celah ini (esdf minimum 0.21 m sepanjang
> 8 leg). Sudah diperbaiki: Guard 1 kini ikut aktif di mode ini dengan aksi
> invalidate+hover yang sama (tetap TANPA escape).
>
> **Implikasi untuk hasil lama:** semua "tabrakan" di `gauntlet_train2` yang
> tercatat sebelum perbaikan ini **BUKAN bukti kualitas model** — itu artefak
> mode uji yang kehilangan separuh pengamannya. Jangan dikutip sebagai hasil.

---

## Kenapa escape mencemari hasil

Escape BUKAN keluaran model. Ia perintah kecepatan buta (`_escape_vel_xy`) yang
dipilih dari **hanya 8 arah diskrit** (kompas 45°) di
`_find_escape_direction()` (`fm_inference_base.py`). Saat escape aktif,
trajektori model dibuang (`_traj.invalidate()`) dan drone tidak lagi menerbangkan
apa pun yang dihasilkan FM.

Tiga cara ia merusak pengamatan:

### 1. Menghasilkan "berputar-putar di tempat" yang bukan salah model

Yaw drone diperbudak arah kecepatan (`msg.yaw = atan2(vel[1], vel[0])`), jadi
tiap escape berganti arah = drone **berputar fisik**. Karena arah escape dipilih
dari 8 arah diskrit dengan skor yang sering hampir seri, arah berturut-turut bisa
loncat 90–180°. Contoh nyata dari log run `poles` 2026-07-27:

```
vel=(0.00, 0.40)  →  vel=(-0.40, 0.00)  →  vel=(-0.00,-0.40)
     utara               barat                  selatan
```

Interval antar-escape banyak yang **+1.00s, +1.01s, +1.25s, +1.30s** — siklus
ping-pong ~1 detik, drone tidak sempat maju. Pengamat mudah menyimpulkan
"model bingung / mode collapse" padahal model tidak terlibat sama sekali.

### 2. Ping-pong escape ↔ Guard 2 tidak punya histeresis

Keduanya mengukur **titik berbeda di ruang**, jadi ambang histeresisnya tidak
pernah berlaku:

| | Yang diukur | Ambang (edt mentah) |
|---|---|---|
| Escape **keluar** | clearance di **posisi drone SEKARANG** | keluar saat `> 1.00` |
| Guard 2 **nyala** | clearance di **jalur 0.3–1.5 detik KE DEPAN** | nyala saat `< 0.40` |

Drone bisa aman di posisinya (keluar escape) tapi jalur di depannya masih
menembus obstacle yang sama → Guard 2 langsung nyala → escape lagi → berulang.

Diperparah kecepatan: horizon Guard 2 dalam **satuan waktu** (1.5 s), jadi drone
lebih cepat memindai **jarak fisik lebih jauh**. Terukur di dua run `poles`
(world, model, goal identik; hanya `planning_time_ahead` berbeda):

| Pemicu escape | dTf=0.3 (default) | dTf=1.0 (neo-parity) |
|---|---|---|
| Guard 2 (look-ahead) | 8 | **28 (3.5×)** |
| Guard 1 (posisi drone) | 10 | 8 |
| No-progress | 22 | 2 |
| **Laju leg selesai** | **1.06 /menit** | **0.66 /menit** |
| Akhir | 8 round-trip bersih | **PHYSICS EXPLOSION** |

Perhatikan: dTf=1.0 lebih cepat sesaat tapi **throughput misi TURUN** — escape
beruntun membuat drone tidak pernah stabil, altitude melorot (z 1.96 → 1.15 →
1.05), clearance jatuh ke 0.21 m lalu 0.00 m (kontak fisik), berakhir explosion
di (-10.7, -37.4, z=84 m) @ 25.77 m/s.

### 3. Menutupi kandidat model yang buruk

Ini alasan metodologis yang paling penting: escape adalah **mekanisme rescue**.
Kalau model menghasilkan kandidat yang menembak tengah obstacle, escape akan
menyelamatkan drone dan misi tetap "selesai" — kegagalan model jadi tidak
terlihat di metrik akhir. Mode `use_lookahead_guard:=true` sengaja memakai aksi
**invalidate+hover** (berhenti di tempat), BUKAN escape, supaya kandidat buruk
terlihat sebagai kemacetan, bukan tersamarkan sebagai manuver.

---

## Kapan escape BOLEH aktif

| Keperluan | Guard mode | Alasan |
|---|---|---|
| **Menilai kualitas model FM** | `use_safety_guards:=false use_lookahead_guard:=true` | Escape mencemari hasil (3 alasan di atas) |
| **Perbandingan NN vs FM / antar-checkpoint** | sama seperti di atas, **identik untuk kedua model** | Keadilan — mode guard harus sama persis |
| Koleksi dataset (expert) | FULL (default) | Butuh menyelesaikan misi, bukan mengukur model |
| Deployment / demo | FULL (default) | Keselamatan lebih penting daripada kemurnian pengamatan |
| Uji stabilitas pipeline (bukan model) | FULL (default) | Yang diuji memang termasuk mekanisme rescue-nya |

---

## Peringatan: guard penuh-OFF ≠ mode uji yang benar

**JANGAN** memakai `use_safety_guards:=false` SENDIRIAN (tanpa
`use_lookahead_guard:=true`). Itu mematikan look-ahead guard juga, dan hasilnya
bukan "model apa adanya" melainkan **artefak latensi octomap**: peta dibangun
asinkron dari kamera, jadi trajektori yang tampak bersih saat direncanakan bisa
menabrak obstacle yang belum termapping.

Bukti terdokumentasi (`wall_gauntlet`, guard penuh-OFF): **kedua model gagal** —
NN macet (462 replan ditolak), FM **crash** dengan `min_clear = 0.00`. Itu tidak
mendiskriminasi kualitas model sama sekali, cuma menunjukkan keduanya tak bisa
terbang tanpa pengaman peta.

`guard_clearance:=0.60` dipilih = hard barrier, jadi guard TIDAK memveto rencana
aman yang menyerempet pintu/celah sempit.

---

## Catatan terkait: visualisasi RViz saat escape (diperbaiki 2026-07-27)

Sebelum perbaikan, selama escape `_replan()` return lebih awal sehingga **tidak
ada marker dipublikasikan sama sekali** — path kuning membeku di layar sementara
drone terbang ke arah lain. Ditambah `Marker.DELETEALL` yang bersifat global
menghapus `ns="flown"` setiap replan gagal tanpa menggambarnya ulang.

Sudah diperbaiki di `fm_inference_base.py`:
- `DELETEALL` global → `DELETE` ber-namespace, hanya untuk id kandidat yang basi
- saat escape: path lama dihapus + digambar **panah magenta arah escape**
  (`_publish_escape_marker()`)
- queue publisher marker 1 → 10 (dua MarkerArray dikirim berurutan tiap replan)

Artinya di RViz sekarang, **panah magenta = drone sedang escape = bukan keluaran
model**. Kalau panah ini sering muncul saat pengujian, itu tanda konfigurasinya
salah (escape masih aktif) — lihat aturan inti di atas.

---

## Lapis anti-tabrakan baru (2026-07-27, sore) — TIDAK melanggar aturan di atas

Setelah 4 run gauntlet_train2 di mode uji ini tetap MENABRAK (g8/g9/ga/gb —
lihat blok "FIX TABRAKAN INFERENCE" di CLAUDE.md workspace), tiga lubang
ditutup. Semuanya tetap **tanpa escape**:

| Lapis | Aksi | Kenapa ini bukan escape |
|---|---|---|
| Gate biaya rencana (`max_plan_cost`, default 50) | tolak kandidat, coba kandidat berikutnya | menyaring solver yang divergen (gb menerima cost=2007.6 lalu menabrak); tidak menerbangkan apa pun sendiri |
| Lantai post-check (0.45 m dari pusat) + syarat MENJAUH | tolak trajektori menempel dinding | tetap veto, bukan manuver |
| Pembatas kecepatan proporsional-clearance | **skala laju saja** | arah TIDAK PERNAH diubah (diuji: vektor tetap paralel) — model tetap menentukan ke mana, pipeline cuma menentukan seberapa cepat |
| Abort buta (`blind_abort_s`, default 20 s) | mendarat + vonis di log | mengakhiri run yang sudah tak bisa diukur, bukan menyelamatkannya |

**Yang penting untuk metodologi:** pembatas kecepatan tidak bisa menyamarkan
kandidat buruk seperti escape, karena ia tidak pernah menggantikan keluaran
model — kandidat yang menembak tengah obstacle tetap terlihat sebagai
kemacetan/veto, cuma didekati lebih pelan. Kalau ingin membandingkan
antar-model, pastikan ketiga flag ini SAMA untuk kedua model (default sudah
sama); untuk mereproduksi perilaku lama: `max_plan_cost:=0
use_speed_limit:=false blind_abort_s:=0`.

Banner sekarang mencetak ketiganya — cek saat mulai:

```
Guard mode    : GUARDS 1+2, invalidate+hover, NO escape
Plan gate     : cost <= 50 | post-check floor 0.45 m
Speed limit   : clearance-proportional (floor 0.15 m/s) | look-ahead >= 2.5 m
Blind abort   : 20 s off-map -> land
```

> ⚠️ Pelajaran proses: memperpanjang horizon look-ahead sempat ikut
> memperpanjang veto ruang-belum-teramati → drone MERAYAP 0.03-0.12 m/s
> (`path ahead UNOBSERVED` terus-menerus). Uji offline TIDAK menangkapnya;
> hanya terbang SITL yang menangkap. Sudah diperbaiki (dua horizon terpisah).
> Artinya: perubahan pada guard WAJIB diuji terbang, bukan cuma unit test.
