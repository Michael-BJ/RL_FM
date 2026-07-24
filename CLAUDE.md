# neo_planner_ws — Catatan Workspace

## MODIFIKASI: Expert Multi-Homotopy (dataset multimodal) — 2026-07-04

### Apa yang diubah dan mengapa

Semula expert hanya menyimpan **satu** solusi (cost terkecil) per scene ke
`train.csv` → dataset unimodal → FM planner tidak bisa belajar bahwa satu
scene ambigu (obstacle lebar di tengah) punya beberapa jawaban valid
(kiri DAN kanan). Padahal `batch_plan()` memang sudah mengoptimasi 3 kandidat
(lurus / kanan / kiri) setiap replan — dua lainnya dulu dibuang.

Sekarang expert menyimpan **semua solusi yang konvergen, aman, dan berbeda
jalur** sebagai baris CSV terpisah dengan **id & depth image yang sama**
(1 scene = 1 image = 1–3 baris). Perilaku terbang drone TIDAK berubah:
yang dieksekusi tetap solusi cost terkecil.

### Cara kembali ke perilaku lama (tanpa mengubah kode)

```bash
ros2 launch neo_planner expert_planning.launch.py ... save_all_modes:=false
```

`save_all_modes:=false` = persis perilaku semula (hanya cost terkecil,
1 baris per scene). Default: `true`.

### File yang diubah (backup asli: `backups_before_multimodal/`)

1. **`src/neo_planner/neo_planner/min_jerk_planner.py`** — `batch_plan()`:
   - Tambah `self.all_solutions = [(int_wpts, ts, cost), ...]` berisi SEMUA
     kandidat yang konvergen (bukan hanya yang terbaik).
   - Pemilihan solusi terbaik untuk eksekusi tidak berubah.

2. **`src/neo_planner/neo_planner/expert_planner_node.py`**:
   - Parameter baru `save_all_modes` (default `True`).
   - `_save_sample(...)`: parameter `int_wpts, ts` diganti `solutions`
     (list `(int_wpts, ts)`); menulis 1 image + N baris CSV ber-id sama;
     `_n_samples` bertambah sebanyak N (target_samples = jumlah BARIS csv,
     bukan jumlah scene).
   - Call site di `_replan_and_record()`: loop `all_solutions`, buang
     duplikat (semua waypoint < 0.5 m dari solusi yang sudah tercatat =
     basin sama), filter `_check_trajectory_safe()` per solusi.

3. **`src/neo_planner/launch/expert_planning.launch.py`** — argumen
   `save_all_modes` diteruskan ke node.

### Untuk mengembalikan kode 100% ke asli

```bash
cp backups_before_multimodal/min_jerk_planner.py      src/neo_planner/neo_planner/
cp backups_before_multimodal/expert_planner_node.py   src/neo_planner/neo_planner/
cp backups_before_multimodal/expert_planning.launch.py src/neo_planner/launch/
colcon build --packages-select neo_planner   # lalu build ulang
```

### Dataset & model ORIGINAL (sebelum multimodal) — lokasi backup terverifikasi 2026-07-09

Selain kode (`backups_before_multimodal/`), dataset dan model yang dilatih
dari expert versi lama (3-seed, 1 path/scene) juga masih ada dan sudah dicek
langsung (isi file, bukan cuma nama folder):

- **Dataset**: `~/training_data/run_20260704_142244/train.csv` — **5000
  baris = 5000 id unik (rasio 1.0)**, terbukti 1 path per scene (bukan
  hasil `save_all_modes`). Timestamp koleksi (14:22) lebih awal dari
  timestamp backup kode (22:29 hari yang sama) → konsisten dikumpulkan
  SEBELUM kode diubah ke multimodal.
- **Model NN original**: `~/saved_net/planner_net_20260704_161214.pth`
  (+ `.onnx`, `.onnx.data`, `nn_loss_20260704_161214.png`). ⚠️ **Beda
  lokasi dari convention run-run sesudahnya** — file ini ada LANGSUNG di
  root `~/saved_net/`, BUKAN di `~/saved_net/nn/run_.../` (kemungkinan
  command training pertama pakai `--out_dir ~/saved_net` tanpa subfolder
  `nn`). Tidak dilindungi folder unik seperti run lain → rawan tertimpa
  kalau ada training NN lain yang lupa menambahkan `/nn` di `--out_dir`.
  Pertimbangkan copy ke folder terpisah (mis. `~/saved_net_original_baseline/`)
  kalau butuh jaminan lebih aman.
- **Model FM original**: dua checkpoint, sama-sama dari dataset di atas —
  `~/saved_net/fm/run_20260704_163845/` (epochs=50) dan
  `~/saved_net/fm/run_20260704_170810/` (epochs=100, dijalankan lebih
  akhir → kemungkinan besar ini yang dianggap final). Keduanya lengkap
  (`.pth`, `.onnx`, `.onnx.data`, `fm_norm_*.json`, loss curve `.png`).

### Catatan dampak ke downstream

- **Loader training** (`nn_trainer.py` / `fm_trainer.py` di
  `~/Training_model_FM_planner`) TIDAK perlu diubah — mereka membaca CSV per
  baris dan memuat image berdasarkan kolom `id`; id duplikat berarti beberapa
  baris memakai image yang sama. Aman.
- **NN-Planner dilatih pada data multimodal akan tampak lebih buruk** di scene
  ambigu (menebak rata-rata kiri/kanan = menembus tengah). Itu perilaku yang
  diharapkan (mode averaging) dan justru menjadi baseline pembanding FM.
- Jumlah baris per scene: 1 (scene tak ambigu) sampai 3 (ambigu, kiri+kanan+…).
  `target_samples` sekarang mengukur baris; jumlah scene/image bisa lebih sedikit.

## World eksperimen ambiguitas (dibuat 2026-07-04)

- Generator: `src/worlds/generate_wide_worlds.py` (satu daftar obstacle →
  `.sdf` + `.pcd` konsisten; edit daftar → jalankan ulang).
- `poles_wide` (training): 4 kotak lebar 1.2/1.6/2.0/2.4 m TEPAT di sumbu
  terbang y=0 + tiang/kotak kecil off-center. Koridor 0..20 → **wajib
  `goal_x:=20.0`**.
- `wide_test` (test/inference, held-out): 1.4/1.8/2.6 m, layout berbeda.
- `gauntlet_train` (training v2, dibuat 2026-07-07, pasangan frontend
  homotopi): koridor 0..30 → **wajib `goal_x:=30.0 arena_x_max:=33.0`**
  (default arena 25 memotong koridor!). 6 event: kotak lebar 2.0 m, dinding
  berpintu 2.0 m (pintu samping y=+1.35; dinding menutup sumbu — pintu yang
  mengangkangi y=0 terbaca 'free' oleh frontend), dinding 4.4 m, zig-zag
  S-curve (x=16/18.5), dinding 5.4 m pintu off-center 2.0 m (x=22.5,
  y=-1.15, cermin R2), slalom 4 tiang x=26-27 sela 2.0-2.3 m (sengaja
  tak-ambigu; tiang luar (26.5,±4.4) — sela diagonal 1.9 m memicu explosion
  di jalan pulang) + 1 tiang pengisi tersisa (19.9,3.8). Dua tiang pengisi
  lain DIHAPUS: (5.4,-4.9) duduk di koridor bawah R2 (kantong sempit →
  explosion #1); (9.8,-2.6) hanya 1.64/1.70 m dari dinding R2/R3 — lolos
  ambang lama 1.3 m tapi di bawah target fungsional 2.0 m → explosion #2.
  Audit lama HANYA menegakkan 1.3 m (batas absolut planner), bukan 2.0 m
  (target fungsional) untuk tiang pengisi — celah ini sudah diperbaiki
  (lihat catatan frontend v1.2 di bawah).
  **REVISI CELAH 2026-07-07: semua celah fungsional ≥ 2.0 m**
  — versi awal 1.5-1.6 m membuat drone BERPUTAR DI TEMPAT di pintu R5
  (aperture − 2×inflasi 0.3 = pita 0.9 m → edt puncak 0.45 < safe_dis 0.5
  → semua rute kena penalti + penggemukan voxel octomap). Celah 2.0 m →
  edt pusat ~0.7 = bebas penalti. Audit otomatis:
  `src/worlds/training_worlds/audit_gauntlet_train.py` (2026-07-08: dipindah
  dari `src/worlds/` ke `training_worlds/` bersama world training lain —
  lihat bagian "Lap-alternation pilot" di bawah; memodelkan dinding virtual
  arena) — pinch antar-baris (tersempit 1.62 m), passability A* @0.65 m DAN
  @0.80 m bebas-penalti (3 jalur, kini wajib), ≥2 kelas homotopi per event.
  Denah: `src/worlds/training_worlds/gauntlet_train_layout.png`. `wall_gauntlet`
  (4.0/4.8/5.6 m) dan `wide_test` TETAP held-out untuk test.
  Audit rev 2026-07-07 (post-explosion #2): pinch-check kini menegakkan
  GAP_TARGET (2.0 m) untuk pasangan yang melibatkan tiang pengisi, bukan
  cuma PAIR_MIN (1.3 m) — sebelumnya tiang (9.8,-2.6) lolos audit (1.64 m
  dari dinding R2, 1.70 m dari R3) tapi jadi chokepoint nyata di uji
  terbang. Tersempit setelah fix: 2.11 m.
- `test01` (TEST held-out, dibuat 2026-07-07, pasangan uji generalisasi
  gauntlet_train): kelas event SAMA (kompleksitas setara, ambang audit
  identik) tapi layout beda — urutan zig-zag/dinding ditukar, sisi pintu
  DICERMIN (E2 pintu bawah y=-1.6; E5 pintu atas y=+1.35), stagger slalom
  dibalik, semua lebar/posisi digeser (kotak 2.4 m, dinding 4.6 m —
  wall_gauntlet 4.0/4.8/5.6 tetap tak tersentuh). Koridor 0..30 → **wajib
  `goal_x:=30.0 arena_x_max:=33.0`**. HANYA untuk inference/eval — jangan
  dipakai collecting data training (boleh untuk dataset TEST open-loop via
  expert). Audit: `src/worlds/audit_test01.py` (LULUS 2026-07-07: tersempit
  antar-baris 2.10 m, passability 3 jalur @0.65 DAN @0.80 m, ≥2 kelas per
  event ambigu, E6 threading 1 kelas by design; + cek baru pengisi→dinding
  virtual arena ≥ 2.0 m). Denah: `src/worlds/test01_layout.png`. `.sdf`
  sudah dicopy ke `~/PX4-Autopilot/Tools/simulation/gz/worlds/`.

## Frontend homotopi expert (2026-07-07) — AKTIF default

- `homotopy_frontend.py` v1.1: `use_topo_frontend:=true` (default) di
  `expert_planning.launch.py` → `batch_plan` di-monkey-patch jadi
  `topo_batch_plan` (seed dari enumerasi kelas homotopi A* di ESDF; 3-seed
  lama HANYA fallback bila grid belum siap/A* gagal). `false` = perilaku
  3-seed asli. Log konfirmasi: `[Expert] Homotopy front-end ACTIVE`.
- **v1.1 `bounds`**: A* dibatasi arena − inset 0.5 m. Tanpa ini A* memilih
  rute murah MEMUTARI LUAR tiang cermin (grid ESDF seluas pcd, y ±15) →
  drone keluar arena.
- **Dinding virtual arena** (`esdf_ros2.py`, param opsional `arena_bounds`
  — hanya expert; node inference TIDAK berubah): sel di luar arena =
  occupied sebelum EDT. Wajib karena cost MINCO tidak punya suku arena:
  seed A* di dalam pun, optimizer/escape bebas mendorong trajektori keluar
  (out-of-bounds beruntun + 2 explosion di run 2026-07-07 sebelum fix).
  Bukti aktif: log IDLE awal `esdf=1.70` di (0,0) (dinding x=-2 + inflasi
  0.3), bukan ~2.90.
- Timing terukur (grid 400×300): replan lokal 0.24-0.96 s, first-plan
  full-corridor ~1.6 s — muat budget 3 s. **v1.2**: budget internal
  frontend — diversifikasi A* max 0.7 s, optimasi seed sisa dilewati
  setelah 1.5 s total (≥1 seed selalu dioptimasi). Alasan: satu plan_once
  MINCO yang sulit bisa 1-4 s dan tak bisa dipotong; tanpa headroom,
  "Replan failed (budget)" beruntun → drone melacak trajektori basi →
  escape ping-pong → explosion (2x teramati 2026-07-07).
- Bobot MINCO `[1,1,1,10000]` + `collision_cost_tol` JANGAN diubah antar
  run dataset (konsistensi label); yang bermakna rasio bobot & tol pada
  skala sama, bukan angka absolut.
- Denah lama: `src/worlds/wide_worlds_layout.png`. Semua `.sdf` sudah dicopy
  ke `~/PX4-Autopilot/Tools/simulation/gz/worlds/`.

Contoh collecting multimodal di world training v2:

```bash
ros2 launch neo_planner sitl.launch.py world:=gauntlet_train
ros2 launch neo_planner mavros_only.launch.py
ros2 launch neo_planner camera_vfh.launch.py
ros2 launch neo_planner expert_planning.launch.py \
    pcd_path:=$HOME/neo_planner_ws/src/worlds/training_worlds/gauntlet_train.pcd \
    goal_x:=30.0 arena_x_max:=33.0 target_samples:=5000
```
(world lama `poles_wide`: `goal_x:=20.0`, pcd
`src/worlds/training_worlds/poles_wide.pcd`. 2026-07-08: semua world
TRAINING — `poles_wide`, `gauntlet_train`, `gauntlet_train2` — sekarang di
`src/worlds/training_worlds/`, bersama 5 world atom lain. World HELD-OUT
(`wide_test`, `wall_gauntlet`, `test01`) TETAP di `src/worlds/`.)

## Diagnosis mode collapse FM + `training_worlds/` (2026-07-08)

### Temuan: FM checkpoint TIDAK genuinely multimodal

Uji offline (`offline_multimodal_test.py`, sample checkpoint FM 300x pada
scene beku yang diketahui punya 2 label kiri+kanan dari `train.csv`):
**0 dari 17 scene ambigu held-out** menunjukkan sebaran seimbang — hampir
semua collapse >95% ke SATU sisi (mis. 300/300, 299/300), walau label
kedua sisi ada di data training. Dicoba naikkan `n_steps` sampai 64
(32x lipat) — tidak berubah, jadi ini BUKAN artefak diskritisasi Euler,
melainkan mode collapse asli pada vector field yang dipelajari.

Root cause di `fm_trainer.py` (ada di `~/Training_model_FM_planner/`, di
luar workspace ini):
1. Dataset sumber (`training_data/run_20260707_182545`, dikumpulkan dari
   `gauntlet_train`) cuma **9.2% scene ambigu** (419/4558) — sinyal
   gradien untuk target yang saling konflik terlalu jarang dibanding
   mayoritas scene unimodal.
2. `random_split` per-BARIS (bukan per-scene) bisa memisah sepasang label
   kiri/kanan ke train vs val secara acak → validasi tidak representatif.
3. Metrik diagnostik `best_of_K`/`spread_y` dirata-rata ke SELURUH
   dataset → collapse pada minoritas ambigu tertutupi mayoritas unimodal
   yang sehat (loss curve terlihat normal padahal model gagal total di
   titik yang justru jadi tujuan riset).

### Fix di `fm_trainer.py` (2026-07-08)

- `group_split()` — split train/val per-scene (by `id`), bukan per-baris.
- `make_ambig_sampler()` — `WeightedRandomSampler`, baris scene ambigu
  di-oversample (default `--ambig_oversample 8.0`; `1.0` = nonaktif).
- `validate()` sekarang melapor metrik BERSTRATA: `best_of_K_uniq` vs
  `best_of_K_ambig` (bukan cuma agregat), plus `ambig_worst` (worst-case
  best-of-K per scene ambigu — apakah SEMUA label tercakup, bukan cuma
  yang terdekat dengan mode yang sudah collapse). Model-selection
  checkpoint sekarang pakai `best_of_K_ambig`, bukan agregat.
- Backup asli: `~/Training_model_FM_planner/fm_trainer.py.bak_before_multimodal_fix`.

### `src/worlds/training_worlds/` — world atom sederhana (2026-07-08)

Dibuat karena dataset FM SELAMA INI cuma dari 1 world kompleks
(`gauntlet_train`) → variasi visual scene ambigu sempit (cuma beberapa
titik spesifik), plus rasio ambigu rendah (9.2%). Daripada oversampling
mengulang-ulang scene yang itu-itu saja (risiko menghafal bukan
generalisasi), dibuat 5 world SEDERHANA (1 konsep obstacle per world,
2 instance beda skala per world untuk yield), bentuk & dimensi beda-beda,
supaya gampang di-debug per-world kalau ada yang janggal:

| World | Bentuk | Dimensi (2 instance) |
|---|---|---|
| `box_narrow` | box tepat y=0 | lebar 1.3 / 1.6 m (ambang bawah feasibility) |
| `box_wide` | box tepat y=0 | lebar 2.6 / 3.2 m (detour besar) |
| `cylinder_fat` | 1 silinder gemuk | diameter 1.4 / 2.2 m (bentuk BULAT) |
| `twin_pole_gate` | SEPASANG tiang rapat | celah tengah 0.3 / 0.7 m (< 1.3 m PAIR_MIN → wajib memutari pasangan, topologi beda dari 1 objek solid) |
| `plate_wall` | plat tipis lebar | sx 0.2-0.25 m (tipis), sy 2.2 / 3.4 m (rasio aspek beda dari box) |

Semua koridor 0..20, **`goal_x:=20.0`** (arena bounds DEFAULT
`expert_planning.launch.py`, tidak perlu override apa pun — beda dari
`gauntlet_train`/`test01` yang wajib `arena_x_max:=33.0`). Generator:
`src/worlds/training_worlds/generate_training_worlds.py` (reuse
`write_sdf`/`write_pcd` dari `generate_wide_worlds.py`). Denah:
`src/worlds/training_worlds/training_worlds_layout.png`. `.sdf` sudah
dicopy ke `~/PX4-Autopilot/Tools/simulation/gz/worlds/`. Terdaftar di
`flight_monitor.py` (`_find_wide_obstacles()` juga mengenali silinder
tunggal DAN pasangan tiang yang mengangkangi y=0, bukan cuma box).

Contoh collecting (ganti nama world untuk yang lain):
```bash
ros2 launch neo_planner sitl.launch.py world:=box_narrow
ros2 launch neo_planner mavros_only.launch.py
ros2 launch neo_planner camera_vfh.launch.py
ros2 launch neo_planner expert_planning.launch.py \
    pcd_path:=$HOME/neo_planner_ws/src/worlds/training_worlds/box_narrow.pcd \
    goal_x:=20.0 target_samples:=1500
```

**Strategi training**: kumpulkan dari SEMUA 5 world ini + gabung dengan
`gauntlet_train` (pool, BUKAN curriculum/fine-tune bertahap — pooling
sekali jalan menghindari risiko catastrophic forgetting dari fine-tuning
sekuensial). `fm_trainer.py` sudah mendukung multi-folder secara native:
```bash
python3 fm_trainer.py --data_dir \
    ~/training_data/run_<gauntlet_train> \
    ~/training_data/run_<box_narrow> ~/training_data/run_<box_wide> \
    ~/training_data/run_<cylinder_fat> ~/training_data/run_<twin_pole_gate> \
    ~/training_data/run_<plate_wall> \
    --out_dir ~/saved_net/fm --epochs 100 --batch_size 32
```
`wide_test`/`wall_gauntlet`/`test01` TETAP held-out murni untuk evaluasi —
jangan collecting training dari situ.

## Label `mode_sig` + perbaikan yield bimodal v1.5-v1.7 (2026-07-08)

### Masalah awal

User melaporkan: dataset multimodal tidak pernah *benar-benar* bimodal —
dua baris ber-`id` sama seringkali cuma "sisi sama yang bergeser", bukan
kiri-vs-kanan sungguhan. Akar masalah: dedup mode di `expert_planner_node.py`
memakai jarak Euclidean waypoint (< 0.5 m = duplikat) — heuristik yang TIDAK
BISA membedakan "sisi sama, bergeser" dari "sisi berlawanan sungguhan".

### v1.5 — Label `mode_sig` (signature topologis, bukan jarak)

- `homotopy_frontend.py`: `compute_mode_signature()` + `mode_signature_to_str()`
  — signature side (`+1`/`-1`/`free`) dihitung dari trajektori FINAL (bukan
  seed A*), pakai formula sama yang sudah dipakai `_signature_for_path()`
  untuk mendiversifikasi A*. Berlaku untuk solusi dari topo-frontend MAUPUN
  fallback stock 3-seed.
- `expert_planner_node.py`: kolom baru `mode_sig` di CSV; dedup di
  `_replan_and_record()` diganti dari jarak Euclidean 0.5 m menjadi
  kesamaan `mode_sig`. Perilaku terbang TIDAK berubah.
- **Validasi lapangan** (`box_narrow`, `run_20260708_193548`, 300 sampel,
  v1.5 SAJA aktif): 6.0% scene bimodal murni (17/283), 7.4%
  "unimodal-nonfree" (garis lurus head→tail terhalang tapi cuma 1 sisi
  tercatat — peluang yang terlewat), sanity-check 0 anomali (`free`
  bercampur `mode_sig` nyata di `id` sama).

### v1.6 — DUA fix yield (satu di antaranya BERMASALAH, lihat v1.7)

Termotivasi oleh 7.4% "unimodal-nonfree" di atas:

1. **`ENUM_LOOKAHEAD_M=9.0`**: `topo_batch_plan()` query A* frontend ke
   titik yang diperpanjang (kolinear, bukan `tail_state` asli) supaya
   obstacle yang terlewat target lokal (`longitu_step=5m`) tetap terlihat.
   `seeds_from_paths(arc_len=...)` memotong lagi ke jarak eksekusi asli —
   drone tetap hanya menerbangkan segmen pendek yang sama. **Fix ini BAIK,
   tidak direvisi.**
2. **`bounds_for_seed()` (DICABUT DI v1.7)**: bound L-BFGS-B pada
   koordinat Y MENTAH tiap waypoint interior, supaya optimizer tak bisa
   "menyeberang" ke sisi lain saat optimasi. **Pendekatan ini SALAH** —
   mengasumsikan sumbu terbang head→tail selalu mendekati horizontal
   (valid di validasi offline yang selalu head_y=tail_y=0), tapi di
   penerbangan nyata `head_y` sering jauh berbeda dari `tail_y` (drift
   akumulasi dari escape/nudge sebelumnya).

**Validasi lapangan v1.6** (`run_20260708_195555`, 300 sampel): bimodal
**JATUH ke 1.5%** (2/134, lebih buruk dari tanpa fix sama sekali!), dan
**SEMUA 12 scene "unimodal-nonfree" bernilai `mode_sig=-1`** (0x `+1`) —
smoking gun bahwa bound yang salah arah secara sistematis menjegal sisi
`+1`. Root cause dikonfirmasi lewat unit test langsung: `bounds_for_seed`
gagal total begitu `head_y` jauh dari `tail_y`.

### v1.7 — Fix yang benar: `constraint_for_seed` (linear constraint eksak)

- Ganti bound Y-mentah (aproksimasi) dengan **`scipy.optimize.LinearConstraint`**
  pada offset lateral EKSAK relatif sumbu head→tail sesungguhnya (formula
  sama yang mendefinisikan `mode_sig` itu sendiri) — benar untuk sumbu
  manapun, bukan cuma yang mendekati horizontal.
- `plan_once()` di `min_jerk_planner.py` sekarang terima `constraints=`;
  otomatis pindah ke solver **SLSQP** (L-BFGS-B tak mendukung linear
  constraint umum).
- **Bug kedua ditemukan & diperbaiki saat validasi**: SLSQP tanpa bound
  eksplisit pada `tau` (time-allocation, parameterisasi sigmoid) bisa
  meledak numerik (`OverflowError` di `map_tau2T`) — L-BFGS-B punya
  trust-region implisit yang mencegah ini, SLSQP tidak. Fix: bound
  keamanan `tau ∈ [-30,30]` (jauh di luar rentang valid manapun, tidak
  pernah aktif secara fisik — murni higienis numerik, bukan aproksimasi).

**Validasi lapangan v1.7** (`run_20260708_201100`, 300 sampel): bimodal
4.2% (12/288) — **secara statistik TIDAK BERBEDA dari baseline v1.5**
(z=1.00, bukan regresi tapi juga belum terbukti lebih baik pada rate
utama), tapi bug fatal v1.6 (0:12 satu sisi, overflow) **hilang total**.
Unimodal-nonfree naik ke 8.0% (vs 7.4% v1.5) — dalam noise yang sama.

### Ringkasan 3 run (world sama, target sama: `box_narrow`, 300 sampel)

| Run | Versi | Bimodal | Unimodal-nonfree | Sign split (single-side) |
|---|---|---|---|---|
| `run_20260708_193548` | v1.5 (baseline) | 6.0% (17/283) | 7.4% | 14:7 (p=0.19, tak signifikan) |
| `run_20260708_195555` | v1.6 (bug) | **1.5%** (2/134) | 9.0% | **0:12 (rusak total)** |
| `run_20260708_201100` | v1.7 (fix) | 4.2% (12/288) | 8.0% | 3:20 (**p=0.0005, signifikan**) |

### Isu terbuka: asimetri sisi (belum terpecahkan)

Ketiga run menunjukkan A* condong menemukan sisi `-1` lebih dulu/lebih
sering (bukan cuma v1.6 yang rusak total — v1.5 & v1.7 juga skew, v1.7
malah lebih signifikan secara statistik daripada v1.5). Dugaan: tie-breaking
`heapq` di `_astar()`/`find_paths()` saat cost kedua sisi sama persis,
ATAU asimetri halus di geometri/inflasi grid. BELUM diinvestigasi —
independen dari bug collapse yang sudah diperbaiki v1.6→v1.7 (itu soal
"begitu KEDUA sisi ditemukan, apakah keduanya bertahan sampai akhir";
ini soal "sisi mana yang ditemukan/berhasil LEBIH DULU").

### File yang diubah sesi ini

- `neo_planner/homotopy_frontend.py`: v1.5 (`compute_mode_signature`,
  `mode_signature_to_str`), v1.6 (`ENUM_LOOKAHEAD_M`, `seeds_from_paths`
  dapat param `arc_len`, kembalikan 3-tuple `(int_wpts, ts, sig)`), v1.7
  (`constraint_for_seed` menggantikan `bounds_for_seed` yang dihapus).
- `neo_planner/min_jerk_planner.py`: `plan_once()` terima `bounds=`
  (v1.6) lalu `constraints=` (v1.7, otomatis pilih SLSQP + tau safety
  bound saat `constraints` diberikan).
- `neo_planner/expert_planner_node.py`: kolom CSV `mode_sig`, dedup
  berbasis signature (bukan jarak 0.5 m).

### Rekomendasi lanjutan (belum dikerjakan)

1. Investigasi asimetri sisi (`heapq` tie-breaking di `_astar`).
2. Sampel lebih besar (>300) untuk power statistik lebih kuat sebelum
   menyimpulkan v1.7 vs v1.5 pada rate bimodal utama (z=1.00 saat ini
   terlalu dekat ambang untuk konklusif).
3. Lanjutkan ke world lain (`box_wide`, `cylinder_fat`, `twin_pole_gate`,
   `plate_wall`) + `gauntlet_train` setelah v1.7/isu asimetri dianggap
   cukup stabil.

## Lap-alternation pilot + world training kompleks kedua (2026-07-08)

### Masalah yang dijawab

Enumerasi per-replan (`topo_batch_plan`/`find_paths`, v1.0-v1.7) hanya
menjawab "apakah SCENE ini secara lokal punya >1 jawaban valid" — tidak
menjamin kedua sisi itu benar-benar DITERBANGKAN (closed-loop, Gazebo) dan
terekam merata sepanjang MISI. Yang dieksekusi selalu solusi termurah, dan
A*/heapq terbukti bias ke sisi `-1` lebih dulu (lihat "Isu terbuka: asimetri
sisi" di atas) — jadi drone yang benar-benar terbang bisa condong satu sisi
terus meski dataset CSV mencatat kedua sisi sebagai "ditemukan".

### Fix: `forced_sign` (tembok virtual per-obstacle, level lap)

- **`homotopy_frontend.py` v1.8**: `find_paths()` terima `forced_sign`
  (+1/-1/None). Kalau diisi, blokir sisi SEBALIKNYA di sekitar tiap
  blocked-interval (lokal, ± `WALL_X_MARGIN=1.0` m di sumbu x — BUKAN
  separuh korfidor global, supaya event ber-pintu-tunggal-asimetris tidak
  jadi mustahil dilewati) SEBELUM diversifikasi A* mulai — reuse
  `_wall_off_side()` yang sudah ada (dipakai v1.3 di tengah loop). Dibaca
  dari atribut mutable `frontend.forced_sign` oleh `topo_batch_plan()`
  (default `None` = perilaku v1.7 tak berubah) — tidak ada call site
  `planner.batch_plan(esdf, head, tail)` yang perlu diubah. `patch_planner()`
  kini menaruh instance frontend di `planner._topo_frontend` supaya mission
  loop bisa menjangkaunya. Diverifikasi lewat self-test sintetis:
  `forced_sign=1` → cuma temukan `(1,)`; `forced_sign=-1` → cuma `(-1,)`.

- **`expert_planner_node.py`**: parameter baru `force_lap_alternation`
  (default `false`, butuh `use_topo_frontend=true`). Kalau aktif: siklus 3
  leg tetap — leg 1 dipaksa kiri (`forced_sign=1`), leg 2 dipaksa kanan
  (`-1`), leg 3 bebas (`None`) — diterapkan lewat `_advance_lap_leg()` di
  titik balik yang SUDAH ADA (toggle `traj_direction`). Setelah leg 3
  selesai, `_lap_pilot_done=True` menghentikan loop utama (BUKAN menunggu
  `target_samples`) lalu lanjut landing normal. Alasan berhenti di 3 leg,
  bukan looping sampai `target_samples`: mayoritas jarak tempuh 1 lap ada di
  korfidor non-ambigu (jauh dari obstacle) — mengulang siklus lebih banyak
  hanya menambah volume baris unimodal, tidak menaikkan rasio bimodal
  (pelajaran sama dengan dataset `gauntlet_train` yang cuma 9.2% ambigu).
  Pilot ini murni untuk VALIDASI mekanisme dulu (apakah `mode_sig` yang
  terekam benar-benar berganti sesuai paksaan), bukan untuk volume dataset.

- **`expert_planning.launch.py`**: argumen `force_lap_alternation`
  diteruskan ke node.

Jalankan pilot:
```bash
ros2 launch neo_planner expert_planning.launch.py \
    pcd_path:=$HOME/neo_planner_ws/src/worlds/training_worlds/box_narrow.pcd \
    goal_x:=20.0 target_samples:=99999 force_lap_alternation:=true
```
(`target_samples` dibuat besar sengaja — bukan lagi kondisi stop aktif,
pilot berhenti sendiri setelah 3 leg.) Cek log
`[Expert] Lap-alternation leg N/3: forced_sign=... (kiri/kanan/bebas)` di
tiap titik balik, lalu bandingkan kolom `mode_sig` — leg 1 harus semua `1`,
leg 2 semua `-1`, leg 3 campuran alami.

### World training kompleks kedua: `gauntlet_train2`

`training_worlds/` (5 world atom, 1 event/world) dinilai terlalu sederhana
untuk memvalidasi lap-alternation di world multi-event; `test01`/`wide_test`
TETAP held-out (tidak boleh dipakai training — lihat bagian atas). Solusinya:
world training KOMPLEKS KEDUA, `gauntlet_train2` (generator:
`src/worlds/generate_wide_worlds.py`, obstacle `GAUNTLET_TRAIN2`; audit:
`src/worlds/training_worlds/audit_gauntlet_train2.py`, **LULUS** — pinch
tersempit 1.90 m, passability 2 jalur @0.65 m DAN @0.80 m, kelas homotopi
per event sesuai target). Koridor 0..30 → **wajib `goal_x:=30.0
arena_x_max:=33.0`** (sama seperti `gauntlet_train`/`test01`). `.sdf` sudah
dicopy ke `~/PX4-Autopilot/Tools/simulation/gz/worlds/`. Denah:
`src/worlds/training_worlds/gauntlet_train2_layout.png`.

Beda dari `gauntlet_train`/`test01` (bukan cuma reorder/mirror — TIPE event
baru yang belum pernah dipakai kombinasi manapun sebelumnya):
- **G1** x=3.5 — silinder gemuk BULAT dead-center (d=2.2 m) + tiang
  pembatas. `gauntlet_train`/`test01` selalu pakai box untuk obstacle
  dead-center; ini bentuk bulat pertama di world kompleks (2 kelas: kiri/kanan).
- **G2** x=8.0 — dinding+pintu 2.0 m standar (box), pintu offset y=+1.9
  (baru — beda dari train +1.35 dan test01 -1.6) (2 kelas).
- **G3** x=12.3 — GERBANG SEPASANG TIANG (2 tiang celah tengah 2.0 m + 2
  tiang pembatas luar), bukan dinding+pintu — threading 1 kelas by design,
  topologi beda dari slalom 4-tiang (cuma 2 tiang inti) dan dari dinding+pintu.
- **G4** x=16.5 — DINDING TANPA PINTU, asimetris off-center (sy=5.6, pusat
  y=+0.6, gap atas 2.1 m vs bawah 3.3 m) — wajib memutar salah satu ujung;
  `gauntlet_train`/`test01` tidak punya dinding solid tanpa celah di tengah
  korfidor (2 kelas).
- **G5** x=20/22.5/25 — zig-zag 3-SEGMEN (train/test01 cuma 2 segmen).
  Diverifikasi lewat audit: HANYA 1 kelas (swing penuh di luar klaster jauh
  lebih mahal daripada S-weave) → threading ketiga, beda tipe dari G3/G6,
  BUKAN event ambigu L/R (jangan dianggap bug kalau audit lain menunjukkan
  1 kelas juga di event serupa).
- **G6** x=27.5-28.5 — slalom 4-tiang, geometri RELATIF identik dengan R6
  `gauntlet_train` yang sudah teraudit+diterbangkan aman, cuma ditranslasi
  +1.5 m di x (event paling rawan sengaja tidak pakai angka baru dari nol).

Tidak ada tiang pengisi di v1 dunia ini (pelajaran `gauntlet_train`: 2 dari
3 pengisi awal justru jadi chokepoint tak-disengaja, memicu 2x PHYSICS
EXPLOSION saat terbang — lebih aman tanpa pengisi dulu).

Contoh collecting (biasa, tanpa pilot):
```bash
ros2 launch neo_planner sitl.launch.py world:=gauntlet_train2
ros2 launch neo_planner mavros_only.launch.py
ros2 launch neo_planner camera_vfh.launch.py
ros2 launch neo_planner expert_planning.launch.py \
    pcd_path:=$HOME/neo_planner_ws/src/worlds/training_worlds/gauntlet_train2.pcd \
    goal_x:=30.0 arena_x_max:=33.0 target_samples:=5000
```

### Reorganisasi folder: semua world TRAINING pindah ke `training_worlds/` (2026-07-08)

`poles_wide`, `gauntlet_train` (+ `audit_gauntlet_train.py` +
`gauntlet_train_layout.png`), `gauntlet_train2` (+ `audit_gauntlet_train2.py`
+ `gauntlet_train2_layout.png`) dipindah dari `src/worlds/` ke
`src/worlds/training_worlds/` — sekarang SEMUA world training (world atom +
world kompleks) ada di satu folder. World HELD-OUT (`wide_test`,
`wall_gauntlet`, `test01`) TETAP di `src/worlds/` (tidak dipindah — jangan
disatukan dengan folder training, supaya risiko tak sengaja collecting dari
world held-out makin kecil). `generate_wide_worlds.py` (generator, TETAP di
`src/worlds/` — dipakai bersama untuk training & held-out) sekarang menulis
`poles_wide`/`gauntlet_train`/`gauntlet_train2` ke `training_worlds/`
otomatis (`TRAINING_WORLD_NAMES`), sisanya tetap ke `src/worlds/`. Kedua
audit script (`audit_gauntlet_train{,2}.py`) diupdate path relatifnya
(1 level lebih dalam) tapi tetap import `generate_wide_worlds` dari
`src/worlds/` lewat `sys.path`. `poles.sdf`/`poles.pcd`, `bricks*`,
`forest_neo*` TIDAK dipindah — bukan bagian lini riset ambiguitas ini.

**Temuan sampingan saat verifikasi ulang audit pasca-pindah**: `R4 zig-zag`
di `gauntlet_train` sekarang HANYA 1 kelas homotopi (target audit: 2) --
`audit_gauntlet_train.py` GAGAL di bagian ini (pinch & passability tetap
LULUS). Dikonfirmasi BUKAN akibat pemindahan file atau perubahan
`homotopy_frontend.py` v1.8 sesi ini (blok `forced_sign` baru 100% ter-guard
`if forced_sign is not None`, tidak pernah aktif saat audit memanggil
`find_paths()` tanpa argumen itu; hasil deterministik di 3x run berturut).
`gauntlet_train2` TIDAK punya masalah serupa (audit LULUS penuh).

### FIX R4 zig-zag: spacing 2.5 → 3.2 m (2026-07-08)

Root cause ditemukan lewat instrumentasi `_astar` langsung (bukan cuma baca
angka final): pada spacing 2.5 m antar 2 segmen R4 (x=16.0/18.5), jendela
diversifikasi A* (`WALL_X_MARGIN=1.0` m di sekitar TIAP segmen) saling
tumpang-tindih (~0.6 m overlap). Begitu ronde 1 menemukan S-weave dan
memblokir sisi yang dipakai di KEDUA segmen, zona tumpang-tindih itu jadi
terblokir dari kedua sisi sekaligus → tembok penuh yang memutus start↔goal
sama sekali (bukan sekadar mendorong ke rute lebih mahal) → ronde 2 A*
selalu `None`, cuma 1 kelas yang pernah tercatat. Diverifikasi lewat sweep
spacing 2.5/2.8/3.0/3.2/3.5/4.0/4.5 m pada obstacle terisolasi: 2.5 & 2.8 m
gagal (1 kelas), ≥3.0 m selalu 2 kelas. Dipilih **3.2 m** (segmen kedua
x=18.5 → **x=19.2**) untuk margin — overlap 0.4 m di y=0 & pinch diagonal
2.1 m TETAP terjaga (cuma jarak-x antar-segmen yang berubah, bukan
bentuk/offset y). `audit_gauntlet_train.py` sekarang **LULUS PENUH**
(R4: 2 kelas, `sigs=[(-1,1),(1,-1)]`; pinch tersempit tetap 2.11 m;
passability 2 jalur @0.65 m dan @0.80 m tak berubah). `.sdf`/`.pcd`
diregenerasi & `.sdf` sudah dicopy ulang ke
`~/PX4-Autopilot/Tools/simulation/gz/worlds/`.

**Implikasi lebih luas (belum ditindaklanjuti)**: mekanisme wall-off v1.3
bisa gagal SENYAP (bukan error, cuma "1 kelas" tanpa tanda apa pun) kalau
dua blocked-interval berdekatan sehingga jendela `WALL_X_MARGIN`-nya
tumpang-tindih — ini juga akar masalah yang sama dengan `gauntlet_train2`
G5 (zig-zag 3-segmen, HANYA 1 kelas by design/verified, lihat catatan di
atas) walau di sana sudah didokumentasikan sebagai threading yang memang
1 kelas, bukan bug — spacing G5 (2.5 m, sama seperti R4 sebelum fix) belum
dicek apakah menaikkannya juga akan membuka kelas ke-2 di sana. Kalau mau
diperbaiki secara umum (bukan per-world), opsi: turunkan `WALL_X_MARGIN`
secara adaptif berdasar jarak antar blocked-interval, bukan konstanta tetap
1.0 m untuk semua kasus.

### Rekomendasi lanjutan (belum dikerjakan)

1. Jalankan pilot 3-leg di `box_narrow` (dunia sederhana) dulu untuk
   memvalidasi `mode_sig` benar-benar berganti sesuai `forced_sign` di
   penerbangan nyata (bukan cuma self-test sintetis), baru lanjut ke
   `gauntlet_train2`.
2. Kalau pilot terbukti jalan, `force_lap_alternation` bisa dipertimbangkan
   sebagai mode collecting utama (bukan cuma pilot 3-leg) — perlu desain
   ulang kondisi stop (loop N siklus, bukan cuma 1x 3-leg) sebelum dipakai
   volume penuh.
3. World training kompleks TAMBAHAN (`gauntlet_train3`, dst.) bisa disusul
   dengan pola yang sama (generator + audit terpisah per world) setelah
   `gauntlet_train2` terbukti aman diterbangkan.

## REVERT: MINCO solver kembali ke L-BFGS-B saja, v1.6/v1.7 constraint dicabut (2026-07-08)

### Apa yang dicabut dan mengapa

`homotopy_frontend.py` v1.9 mencabut mekanisme constraint per-seed v1.6/v1.7
(`constraint_for_seed()`, import `LinearConstraint`, `LATERAL_BOUND_MARGIN_M`)
yang tadinya memaksa tiap seed homotopi tetap di sisinya sendiri lewat
`scipy.optimize.minimize(method='SLSQP', constraints=...)`. `topo_batch_plan()`
sekarang panggil `planner.plan_once()` TANPA argumen — kembali ke L-BFGS-B
murni tak-terkendala, sama seperti SEMUA call site lain di codebase ini
(`batch_plan`, `warm_start_plan`). `min_jerk_planner.py`'s `plan_once()`
disederhanakan kembali (parameter `bounds`/`constraints` dihapus, tidak ada
cabang SLSQP lagi).

**Alasan (data dari tabel perbandingan 3-run di section "Label `mode_sig`"
di atas)**: v1.7 (SLSQP+constraint) TIDAK terbukti lebih baik dari v1.5
(unconstrained, baseline asli) pada metrik yang jadi tujuannya sendiri —
rate bimodal 4.2% (v1.7) vs 6.0% (v1.5), z=1.00 (secara statistik tidak
berbeda). Sementara itu isu TERPISAH yang belum terpecahkan (skew sisi, A*
condong ke `-1`) justru MEMBURUK dan jadi signifikan secara statistik di
v1.7 (3:20, p=0.0005) dibanding v1.5 (14:7, p=0.19, tak signifikan). Di
atas itu, SLSQP tidak punya trust-region implisit seperti L-BFGS-B (line
search-nya) — perlu tambalan manual (`tau` bound `[-30,30]`) setelah
teramati bisa overflow `map_tau2T()`, kegagalan yang tidak pernah dialami
L-BFGS-B murni. Kesimpulan: kompleksitas tambahan (solver kedua + linear
constraint hasil derivasi manual + tambalan numerik) tidak terbayar oleh
manfaat yang bahkan belum lolos noise statistik.

### Yang TIDAK ikut dicabut (masih aktif, tidak terganggu revert ini)

- v1.3 hard wall-off diversification di dalam `find_paths()` (mekanisme
  BEDA — bekerja di grid A* sebelum MINCO, bukan constraint pada optimizer
  MINCO) — inilah yang masih jadi mekanisme utama untuk menemukan >1 kelas
  homotopi per scene.
- v1.5 `mode_sig`/`compute_mode_signature` (dedup & label CSV).
- v1.6 `ENUM_LOOKAHEAD_M` (perbaikan yield yang TERBUKTI baik, tidak direvisi).
- v1.8 `forced_sign`/lap-alternation pilot (mekanisme independen — sudah
  diverifikasi ulang tetap bekerja normal setelah revert ini, lihat
  self-test `homotopy_frontend.py`).
- Diverifikasi lewat run end-to-end (`topo_batch_plan` + `MinJerkPlanner`
  asli, bukan cuma self-test sintetis): kedua kelas homotopi tetap
  ditemukan & dioptimasi dengan `plan_once()` tak-terkendala.

### Isu terbuka yang TERTINGGAL (belum terpecahkan, sekarang tanpa constraint)

Skew sisi A* (kecenderungan menemukan `-1` lebih dulu) masih belum
diinvestigasi akar masalahnya (dugaan: tie-breaking `heapq` di `_astar`/
`find_paths`). Revert ini TIDAK memperbaikinya — cuma menghapus fix yang
terbukti tidak menolong (dan berkorelasi dengan skew yang lebih parah).
Kalau mau diperbaiki, pendekatan lain (bukan constraint per-seed di MINCO)
kemungkinan lebih tepat — misalnya audit langsung `heapq` tie-breaking,
atau alternating tie-break order antar-replan.

## FIX: `forced_sign` tidak benar-benar dipatuhi MINCO — `_BiasedESDF` (2026-07-09)

### Temuan (pilot 3-leg pertama di `box_narrow`, run_20260709_101942)

Setelah revert ke L-BFGS-B murni (section di atas), pilot `force_lap_alternation`
pertama kali benar-benar diterbangkan (bukan cuma self-test sintetis). Hasilnya
BURUK: **100% sample non-`free` di leg terpaksa (3/3) merekam `mode_sig` di sisi
SEBALIKNYA** dari yang dipaksakan (leg 1 forced kiri dapat `-1`; leg 2 forced
kanan dapat `1` dua kali). Akar masalah: `forced_sign` (v1.8) cuma memblokir
sisi lawan di **grid A* internal `find_paths()`** — SEED yang diusulkan ke
MINCO memang dari sisi benar, tapi `plan_once()` (sekarang unconstrained sejak
revert di atas) memakai **ESDF ASLI** yang tidak tahu apa-apa soal tembok
virtual itu, jadi optimizer bebas menggeser waypoint balik ke sisi lain kalau
itu lebih murah — PERSIS root cause v1.6 yang sama ("nothing in unconstrained
L-BFGS-B stops a waypoint from sliding across the axis"), cuma sekarang tanpa
tambalan constraint apa pun.

### Fix: `_BiasedESDF` (`homotopy_frontend.py` v1.10)

Bukan mengembalikan SLSQP+LinearConstraint (itu yang baru saja dicabut, dan
memang untuk kasus BEDA — diversifikasi per-scene normal, bukan forced_sign).
Sebagai gantinya: `_BiasedESDF` — wrapper baca-saja di sekitar ESDF asli yang
me-nol-kan jarak (= obstacle solid) di sisi SEBALIKNYA `forced_sign`, di jendela
lokal-x yang SAMA persis dengan yang dipakai `_wall_off_side()` untuk A* (reuse
`WALL_X_MARGIN`, reuse formula lateral-offset yang sama yang mendefinisikan
`mode_sig` itu sendiri). `topo_batch_plan()` sekarang memakai `_BiasedESDF` ini
(bukan ESDF asli) sebagai `map` yang dilihat MINCO SELAMA optimasi — jadi
`plan_once()` TETAP unconstrained L-BFGS-B murni (selaras dengan preferensi
kembali ke L-BFGS-B di atas), tapi sekarang benar-benar "terdorong menjauh"
dari sisi yang salah lewat cost collision biasa, sama seperti obstacle
sungguhan — bukan lewat constraint solver terpisah. Seed search (`find_paths`)
dan perekam CSV (`compute_mode_signature`/`_check_trajectory_safe`) TETAP
memakai ESDF ASLI (tidak pernah melihat tembok bias ini), jadi label yang masuk
dataset selalu dinilai terhadap lingkungan sungguhan, bukan tembok buatan.

### Hasil validasi ulang (run_20260709_102718, box_narrow)

Pelanggaran turun dari 3/3 (100%) menjadi 1 kasus (leg 2 kebetulan tidak
menemui scene ambigu sama sekali di run ini, jadi belum teruji di sana). Tidak
ada explosion/skip di kedua run, accept rate 100%. Sisa 1 pelanggaran (leg 1,
sampel `seed=2` di log — retry ke-3 `_set_local_target` sebelum berhasil,
posisi drone saat itu sudah di sisi yang salah) **BELUM diinvestigasi lebih
lanjut** — kemungkinan retry-loop `_set_local_target` di
`expert_planner_node.py` mendarat di local target yang berinteraksi beda
dengan tembok dibanding target pertama. Diputuskan (2026-07-09): lanjut
mengumpulkan data ke 7 world `training_worlds/` lain dulu (perbaikan sudah
signifikan, ini pilot validasi mekanisme bukan produksi volume penuh) — bukan
menghentikan collecting untuk mengejar 0 pelanggaran dulu.

### Verifikasi

- Test sintetis end-to-end (`MinJerkPlanner` asli + `topo_batch_plan`, 1
  obstacle terisolasi): `forced_sign=1` → final flown `mode_sig=(1,)`;
  `forced_sign=-1` → `(-1,)`; `forced_sign=None` → cost-optimal alami. 100%
  sesuai di kasus terisolasi ini.
- 2x run SITL nyata di `box_narrow` (yang punya 2 obstacle terpisah, x=6 &
  x=14 — beda dari test sintetis yang cuma 1 obstacle): 3/3 → 1 pelanggaran.

## Keputusan riset (go/no-go) + audit pilot 3-leg (2026-07-10)

### Review menyeluruh — keputusan: LANJUT dengan 1 eksperimen penentu ber-kill-criteria

Ditinjau atas pertanyaan user ("gap generator↔optimizer, multimodal sulit
dibuktikan, keunggulan FM vs NN tak terlihat — lanjut atau tidak?"). Temuan
kunci dari penelusuran bukti di disk:

- **Checkpoint FM retrain 2026-07-09** (`~/saved_net/fm/run_20260709_113954/`,
  pasangan NN dari data identik: `~/saved_net/nn/run_20260709_114526/`)
  **LOLOS gate bimodal**: log `[FM] GATE two-sided 64/100 (64%)` (checkpoint
  lama yang collapse: 3%) + plot kandidat dua sisi di 2/3 dinding
  wall_gauntlet: `~/flighttest_results/fm_candidates_wallg.png`. Mode
  collapse yang didiagnosis 2026-07-08 SUDAH terperbaiki di level model.
- **SEMUA perbandingan closed-loop NN-vs-FM yang pernah ada memakai
  checkpoint FM yang collapse** (monitor_logs terakhir 2026-07-08 13:15).
  Perbandingan fair dengan pasangan model 2026-07-09 BELUM pernah dijalankan
  → eksperimen penentu di bawah adalah perbandingan fair PERTAMA.
- Penerbangan FM 2026-07-09 15:15 di wall_gauntlet gagal karena masalah
  STACK (drone keluar arena y=5.8, z→0, escape ping-pong ke arah unexplored)
  — sudah dijawab v2.2 `nn_inference_node.py` (arena_bounds ke ESDF +
  center-bias escape + `ESCAPE_UNMAPPED_CAP`), masuk SETELAH penerbangan itu.
- Keputusan user 2026-07-10: perbandingan resmi memakai
  **`use_safety_guards:=false`** (tanpa guard/escape — model diuji apa
  adanya). Protokol lengkap + kriteria go/no-go dibekukan di `command.txt`
  blok "EKSPERIMEN PENENTU"; rekap otomatis:
  `src/neo_planner/neo_planner/decisive_eval.py` (baru, jalankan langsung
  dengan python3, tanpa build). Kill criteria: bila FM-minimal TIDAK lebih
  baik dari NN di wall_gauntlet padahal GATE ≥30% → klaim inti gugur →
  STOP jalur FM, tulis negative result.
- Verifikasi deploy 2026-07-10: install = egg-link ke `build/neo_planner`,
  seluruh `.py` node + launch di build/install IDENTIK dengan `src/`
  (use_safety_guards & arena bounds sudah aktif di kedua launch inference).

### Audit pilot 3-leg lap-alternation (12 run 2026-07-09) — jawaban "sudah berhasil?"

Metode: batas leg diambil dari log ROS expert (`[Expert] Lap-alternation leg
N/3` + timestamp epoch), baris CSV dipetakan ke leg via timestamp `id`.
(Segmentasi berbasis arah CSV SALAH — kolom pos CSV frame relatif-drone;
turnaround/escape membuat flip arah palsu.)

- **Mekanisme: BERJALAN.** 11/12 run menyelesaikan 3 leg dan berhenti
  sendiri; 1 run (`run_20260709_105032`) terputus di leg 2 (tidak ada log
  "PILOT selesai").
- **Kepatuhan sisi: BOCOR ~40%.** Agregat baris non-free di leg terpaksa:
  leg1 (kiri/+1) 62 patuh vs 41 langgar; leg2 (kanan/−1) 57 vs 42. Dihitung
  ulang hanya pada baris-pertama-per-id (≈ solusi yang diterbangkan):
  35/94 dan 40/96 — sama saja → drone benar-benar TERBANG di sisi salah
  ~40%, bukan sekadar label ekstra `dataset_only`. Kebocoran kecil di run
  box_narrow awal (1-3 baris non-free/leg) ternyata artefak sampel kecil;
  di world padat obstacle (run 105439/105820/110227/111837/112254)
  pelanggaran masif.
- **Jalur bocor yang dicurigai (belum diinvestigasi per kasus):**
  (1) fallback stock 3-seed di `topo_batch_plan()` (baris 767/809
  `homotopy_frontend.py`) — TANPA forced_sign/_BiasedESDF dan TIDAK
  ter-log (grep 'fallback' di log = 0 baris; tambahkan logging kalau mau
  diagnosa); (2) jendela `_BiasedESDF` lokal (±WALL_X_MARGIN) — obstacle
  di luar jendela segmen aktif tak tertembok; (3) retry `_set_local_target`
  / escape.
- **Sisi positif: keseimbangan dataset TERCAPAI secara agregat** — total
  12 run: sisi +1 = 157 vs −1 = 163 (leg bebas 53:65). Untuk tujuan
  menyeimbangkan label kiri/kanan, hasil kolektif tercapai (ironisnya
  sebagian justru karena kebocoran itu simetris).
- **Keputusan:** perbaikan kebocoran = optimisasi yield, JANGAN memblokir
  eksperimen penentu; kerjakan setelah go/no-go bila hasilnya "go".

## FIX kebocoran forced_sign via fallback tak ter-bias (v1.12, 2026-07-10)

Ditindaklanjuti dari audit pilot 3-leg (bagian di atas): ~40% sampel leg
terpaksa terbang di sisi SALAH. Akar masalah yang diperbaiki sesi ini:
`topo_batch_plan()` memanggil stock 3-seed `fallback` dengan **`esdf`
ASLI** (bukan `opt_esdf` ter-bias) di DUA titik — `if not seeds` (A* tak
menemukan jalur sisi terpaksa di scene padat) dan `if not solutions` —
dan tanpa log sama sekali (grep 'fallback' di log lama = 0 baris → bocor
tak terlacak).

Perbaikan (`homotopy_frontend.py`):
- `opt_esdf` (`_BiasedESDF`) kini dibangun DI AWAL `topo_batch_plan()`,
  sebelum cek seed — supaya KEDUA fallback bisa meneruskannya ke stock
  `batch_plan`. Saat `forced_sign` aktif, collision cost fallback jadi
  ikut menolak sisi salah (sisi salah = jarak 0 = obstacle solid) atau
  gagal (lebih baik daripada sampel salah-label; recorder tetap menilai
  hasil terbang terhadap ESDF ASLI). `forced_sign=None` → `opt_esdf` IS
  `esdf` → perilaku fallback normal TAK berubah.
- `_log_fallback()` + counter `fallback_count` / `fallback_forced_count`
  di instance frontend. Fallback saat leg terpaksa = WARNING tiap kali
  (`[homotopy] FALLBACK during FORCED leg`, grep-able); fallback biasa =
  info jarang (pertama + tiap 50). Ringkasan akhir expert kini mencetak
  `Topo fallback : N total (M during forced legs)`.

Verifikasi (3 lapis, semua di `src/neo_planner/neo_planner/`):
1. Self-test bawaan `homotopy_frontend.py` tetap lolos (2 kelas box 2.6 m).
2. Unit dispatch `test_forced_sign_fallback.py`: forced-leg fallback
   menerima `_BiasedESDF` dgn sisi salah solid (d=0.00) & sisi terpaksa
   terbuka (d=0.30); unforced fallback tetap terima esdf mentah; counter/log
   benar.
3. A/B end-to-end `test_forced_sign_endtoend.py` dgn `MinJerkPlanner` ASLI
   (120 scene box acak; ukur sisi trajektori yang BENAR-BENAR diterbangkan
   via `compute_mode_signature` vs ESDF asli — sama seperti recorder; hanya
   esdf yang diterima fallback yang beda OLD vs NEW):
   - Paksa bergantian +1/-1 (seperti pilot nyata): sisi-SALAH **OLD 39%
     -> NEW 1%**. OLD 39% ≈ audit lapangan ~40% → test offline mereproduksi
     kebocoran nyata dgn setia. `correct` NEW NAIK (41%->49%); sisanya jadi
     `fail` (tak ada sampel = tak mencemari, BUKAN salah-label) bukan `wrong`.
   - Paksa sisi tersempit (worst case): sisi-SALAH OLD 78% -> NEW 2%.
Edit LIVE tanpa rebuild (build/neo_planner/neo_planner = symlink ke src —
verifikasi inode 2026-07-10; koreksi catatan §2 CLAUDE.md paket yang
menyangka install = copy).

SISA yang BELUM dikerjakan: (1) jendela `_BiasedESDF` masih lokal
±WALL_X_MARGIN — obstacle di luar segmen aktif belum tertembok; (2) uji
terbang SITL ulang pilot 3-leg untuk konfirmasi rasio kebocoran turun di
Gazebo nyata (offline sudah 39%->1%, tapi SITL punya escape/retry yang tak
termodelkan di test). Jalankan pilot 3-leg box_narrow lalu `lap_audit.py`.

## Eksperimen penentu DIJALANKAN + mode fair v2.3 (2026-07-10)

### Orkestrasi SITL otonom terbukti bekerja

Claude menjalankan stack SITL penuh headless dari sesi non-interaktif
(sitl+mavros+camera_vfh+inference+flight_monitor via background + poll log),
4 trial terbang nyata. Bukti: environment mampu (DISPLAY=:1, gz sim, PX4
px4_sitl_default ter-build). Cutoff per-trial berbasis waktu + deteksi
'Goal reached/DONE/PHYSICS' dari log node (dengan guards-OFF drone yang macet
TAK pernah self-terminate -> wajib cutoff manual).

### Temuan: guards penuh-OFF membuat wall_gauntlet "kedua model gagal"

Trial guards-OFF (`use_safety_guards:=false` saja) di wall_gauntlet, goal 22:
- **NN**: 0/1, MACET di dinding terlebar x=17.5 (5.6 m), 433 replan gagal,
  0 escape, min_clear 0.36 (tak menabrak). Deterministik.
- **FM**: 0/1, CRASH/physics-explosion (terlempar ke x=-41, z<0) di dinding
  ke-2 x=12, min_clear 0.00. GATE bimodal 49% (model sehat).

Diagnosis: "replan gagal" masif = HARD POST-CHECK menolak trajektori
grazing (bekerja benar). Tanpa look-ahead guard, tak ada yang mengompensasi
octomap telat -> satu kandidat yang "bersih" saat plan menabrak dinding yang
belum termapping (FM). Ini artefak LATENCY OCTOMAP, bukan sinyal kualitas
model. Kesimpulan metodologi: guards penuh-OFF di world dinding-solid tak
mendiskriminasi model, cuma menunjukkan keduanya tak bisa terbang mentah.

### Fix: mode fair `use_lookahead_guard` (v2.3, nn_inference_node.py)

Param baru `use_lookahead_guard` (default false; hanya aktif saat
`use_safety_guards:=false`). Menyalakan HANYA Guard 2 (look-ahead) dengan
aksi **invalidate+hover** (berhenti sebelum obstacle) BUKAN escape. Jadi:
pengaman-peta minimal yang SAMA untuk NN & FM (cegah crash akibat octomap
telat), TANPA rescue aktif (escape/no-progress/proximity semua tetap OFF)
yang bisa menutupi kandidat buruk. Banner node mencetak `Guard mode :
LOOK-AHEAD only (invalidate+hover, no escape)`. Diteruskan lewat kedua
launch (nn_inference/fm_planning_unknown) + diwarisi fm_inference_node.
Diverifikasi live di SITL (log guard menyala sbg hover, bukan escape).

### Head-to-head di gauntlet_train2, mode fair (goal 30, n=1 masing-masing)

| Model | Goal | Replan ok/gagal | Look-ahead hover | min_clear | Hasil |
|---|---|---|---|---|---|
| **NN** | **0/1** | 11 / 462 | 4x | 0.39 (no crash) | macet di x~22 (zig-zag/dinding G4-5) |
| **FM** | **1/1** | 56 / 1 | 11x | 0.43 (no crash) | **SELESAI x=30, 81 s / 39 m** |

FM (bimodal, GATE 94-95%) commit bersih ke satu sisi & lolos seluruh
gauntlet; NN (mode-averaging) gagal menemukan rencana aman (462 tolakan
post-check = menembak tengah obstacle) lalu macet. Diskriminasi persis
prediksi riset, dalam perbandingan FAIR (mode guard identik). Mode fair
juga mencegah crash di KEDUA model (min_clear >= 0.39, nol explosion).

### CAVEAT (wajib diingat sebelum menyimpulkan)

1. **n=1 masing-masing.** FM STOKASTIK -> wajib >=5 trial sebelum klaim.
   NN deterministik (macet akan berulang).
2. **gauntlet_train2 = world TRAINING, bukan held-out.** Hasil di sini
   mengukur fitting, bukan generalisasi. Klaim generalisasi WAJIB di
   held-out mode-fair: **test01** (flyable, celah ~2 m — beda dari
   wall_gauntlet yang dinding-solid & terlalu keras). wide_test juga.
3. wall_gauntlet held-out tetap berguna sbagai "batas atas kesulitan"
   (karakterisasi kegagalan), bukan perbandingan penyelesaian.

### Langkah berikут (belum dijalankan)

Ulangi head-to-head mode-fair (>=5 trial/model) di **test01** (held-out,
goal 30 arena 33) untuk klaim generalisasi; lalu wide_test (goal 20).
Perintah + kriteria di command.txt blok "EKSPERIMEN PENENTU" (tambah
`use_lookahead_guard:=true`). Rekap: `decisive_eval.py --prefix nnF,fmF`.

### KOREKSI (re-run GUI 2026-07-10): closed-loop STOKASTIK, n=1 tak andal

Re-run mode-fair gauntlet_train2 dengan GUI (2 trial baru): **NN kali ini
MENCAPAI GOAL** (1/1, 62.5s) — kebalikan dari run headless sebelumnya yang
macet 0/1. FM juga selesai lagi (1/1, 57.3s). Jadi klaim awal "NN
deterministik → macet berulang" SALAH: hasil closed-loop bervariasi antar-run
karena octomap dibangun ASINKRON (timing kamera/replan beda tiap kali), jadi
posisi & peta saat tiap replan tak identik walau modelnya deterministik.

Tally mode-fair gauntlet_train2 sejauh ini: **NN 1/2 selesai, FM 2/2 selesai**
(FM GATE bimodal konsisten 94-100%). Terlalu kecil untuk menyimpulkan apa pun
— justru BUKTI LANGSUNG kenapa butuh >=5 trial/model. Diskriminasi "NN macet
vs FM selesai" dari n=1 pertama itu NOISE, bukan temuan. Semua metrik n=1 di
tabel atas harus dibaca sebagai anekdot, bukan hasil.

## Head-to-head wall_gauntlet mode-fair (2026-07-10) — diskriminasi TERJELAS sejauh ini

Menjawab pertanyaan user "kenapa NN tak stuck di gauntlet_train2": karena
world itu tak menghukum mode-averaging (asimetris + celah longgar + pipeline
menyelamatkan). wall_gauntlet (dinding SOLID SIMETRIS 4.0/4.8/5.6 m, koridor
luar 1.4 m) menghukumnya. Diuji mode-fair (`use_lookahead_guard:=true`,
goal 22, n=1 masing-masing):

| Model | Hasil | Macet di | Replan ok/gagal | min_clear |
|---|---|---|---|---|
| NN | 0/1 | dinding-2 (x~11) | 37 / **500** | 0.59 |
| FM | 0/1 | dinding-3/terlebar (x~17) | 65 / **61** | 0.34 |

- **NN**: lolos dinding-1, lalu di dinding-2 BERPUTAR-PUTAR (osilasi kiri/kanan
  y=+3↔−5, tak bisa commit — persis mode averaging tervisualkan) lalu macet.
  Rasio replan 7% sukses = mayoritas rencana menembak tengah dinding → ditolak
  post-check. Plot: `output_path/flown_paths_wall_gauntlet_side.png`.
- **FM**: lolos dinding-1 DAN dinding-2 (commit satu sisi bersih), tembus ke
  dinding terlebar x~17, mentok di sana. Rasio replan 52% sukses. GATE bimodal
  90-92% sepanjang terbang.
- **Diskriminasi JELAS** (FM tembus 2 dinding vs NN 1; 52% vs 7% replan sukses),
  arah persis prediksi riset. TAPI: (1) n=1; (2) KEDUANYA belum sampai goal 22
  — dinding 5.6 m + koridor 1.4 m + tanpa-escape terlalu keras; (3) FM juga
  berosilasi (cuma lebih jauh). Untuk story "FM SELESAI, NN tidak" yang bersih,
  pertimbangkan goal lebih pendek (mis. x~13, tepat lewat dinding-2) atau world
  diskriminator dgn dinding terlebar sedikit lebih ramah — lihat "celah desain
  eksperimen": wall_gauntlet mungkin masih di ambang terlalu keras utuk
  penyelesaian penuh, tapi SANGAT baik utk metrik "seberapa jauh + rasio replan".
- Osilasi FM (pertanyaan user): garis terbang = rantai replan lokal 0.5s;
  MINCO cuma menghaluskan tiap segmen, tak mengoordinasi antar-replan; FM-minimal
  resample tanpa memori → sisi bisa flip antar-replan. Peredam ada tapi OFF di
  mode fair (`use_mode_persistence`/`use_anchor_sampling`). Untuk deploy nyata
  (bukan perbandingan fair) nyalakan itu; untuk fair, terapkan smoothing yang
  SAMA ke NN & FM.

## World `decisive_gap` (NN-trap held-out, dibuat 2026-07-10) + TEMUAN KUNCI

Dirancang menjawab permintaan user: world yang MEMAKSA NN lurus-ke-tengah
(buntu) tapi memberi FM peluang. Generator: `generate_wide_worlds.py`
(`DECISIVE_GAP`). 3 box SIMETRIS tepat di y=0 (sy 2.6/3.0/3.4, span +-1.3/1.5/1.7)
+ sepasang tiang luar simetris → koridor sisi LONGGAR **2.2 m** (vs wall_gauntlet
1.4 m). Koridor 0..24 → goal_x:=24 (arena default, TANPA override). HELD-OUT.
`.sdf` sudah dicopy ke PX4 worlds; terdaftar di flight_monitor + output_path viz.
Layout: `output_path/decisive_gap_layout.png`.

### Hasil (mode-fair, goal 24, n=1 masing-masing) — RANCANGAN GAGAL MENJEBAK NN

| Model | Goal | Waktu | Jalur | Replan ok/gagal | min_clear |
|---|---|---|---|---|---|
| NN | **1/1** | 62.2s | 40.6 m | 43 / 8 | 0.37 |
| FM | **1/1** | 53.5s | 28.6 m | 37 / **0** | 0.61 |

**KEDUANYA sampai goal.** NN TIDAK terjebak lurus-ke-tengah: pada celah 2.2 m
yang longgar, NN **commit ke satu sisi** (walau sempat ragu sesaat di tengah —
terlihat kelokan kecil x~4-5 di plot) lalu lolos. Jadi mode-averaging → buntu
TIDAK termanifestasi di obstacle simetris yang celahnya feasible.

### TEMUAN LINTAS-WORLD (mode fair, semua n=1) — penting untuk go/no-go

| World | Celah sisi | NN | FM |
|---|---|---|---|
| wall_gauntlet | 1.4 m (sempit) | GAGAL (macet dinding-2) | GAGAL (lebih jauh, dinding-3) |
| decisive_gap | 2.2 m (longgar) | LOLOS (berkelok, 0.37) | LOLOS (efisien, 0.61, 0 fail) |
| gauntlet_train2 | ~2 m (asimetris) | LOLOS | LOLOS |

- **TIDAK ADA geometri di mana NN GAGAL tapi FM LOLOS.** Di celah sempit
  keduanya gagal; di celah longgar keduanya lolos. Ini menguatkan intuisi user:
  apa yang membuat NN gagal (tak bisa commit di celah sempit) juga menjegal FM.
- **TAPI keunggulan FM NYATA & konsisten, hanya GRADUAL bukan biner:** di celah
  longgar FM lebih efisien (jalur 28.6 vs 40.6 m, 0 vs 8 replan gagal, clearance
  0.61 vs 0.37, commit dini vs ragu-di-tengah); di celah sempit FM lebih jauh
  (2 dinding vs 1). FM "lebih mulus & lebih tegas", bukan "berhasil di mana NN
  gagal".
- **Implikasi:** klaim riset "FM menyelesaikan misi yang NN tak bisa" BELUM
  didukung di world mana pun. Yang didukung: "FM lebih efisien/robust." Untuk
  klaim biner, geometri diskriminator mungkin memang tak ada (hipotesis user);
  fokus alternatif: (a) kuantifikasi keunggulan gradual (jalur/clearance/replan-
  fail) dengan >=5 trial, atau (b) bandingkan FM vs baseline multimodal sederhana
  (MDN) untuk cek apakah kompleksitas FM terbayar. Plot:
  `output_path/flown_paths_decisive_gap_*`.

## FM encoder visual128 (v1.1) + diagnosis prediksi-vs-GT (2026-07-14)

Diagnosis offline (skrip: scratchpad `eval_pred_vs_gt.py`, checkpoint 0711):
error kedua model ~10x lebih besar pada scene yang jawabannya bergantung isi
depth image dibanding scene free, BAHKAN pada data training sendiri — info
obstacle nyaris tak lolos bottleneck visual (ResNet18 beku, 24 fitur).
Kandidat FM "nyangkut di tengah" (midgap) 18-28% dan TIDAK berubah pada
n_steps 32 → bukan artefak Euler, tapi field yang dipelajari.

Fix (`~/Training_model_FM_planner/fm_planner.py` v1.1, disinkron ke
`src/neo_planner/neo_planner/fm_planner.py`; backup `*.bak_before_visual128`):
`IMG_FEATURE_SIZE` 24 -> 128; layer3/4 ResNet di-unfreeze (layer1/2 tetap
beku); `load_checkpoint()` auto-deteksi ukuran fitur dari state_dict →
checkpoint lama (24) tetap termuat. `fm_trainer.py`: dua param group,
backbone layer3/4 LR x0.1 (`--backbone_lr_mult`). NN sengaja TIDAK diubah.

Hasil (`~/saved_net/fm/run_20260714_142058`, data FM_collecting+FM_collect_2
= 2212 baris / 67 scene ambigu): two-sided coverage naik 16%->53% (train) dan
35%->94% (box_narrow di luar training); under-deflection hilang (|lat| kandidat
2.09 vs label 2.17 m). TAPI checkpoint terpilih epoch 15/100 — overfitting
(train CFM turun 1.05->0.50 sementara val CFM naik 1.12->1.21 setelah ep15):
10.6 jt param trainable vs 1990 baris. Kesimpulan: butuh data jauh lebih
banyak; presisi per-sampel (K1) memburuk, sebaran melebar di semua scene.

## MIGRASI KONTRAK INPUT — kamera Gemini 2 + label antisipatif (2026-07-14)

Motivasi: (1) user deploy nyata dgn Orbbec Gemini 2 (depth 0.15-10 m, H91/V66,
1280x800); (2) temuan perilaku "expert belok mepet": target lokal 5 m + cost
MINCO baru aktif < safe_dis + expert pegang peta penuh (tenang walau kamera
tertutup dinding) → murid meniru belok telat + latensi octomap → escape;
(3) normalisasi depth per-frame (÷ max frame) menghapus skala absolut justru
saat dinding memenuhi pandangan. Kamera sim lama HFOV ternyata 73 deg
(1.274 rad), BUKAN 88 seperti asumsi VFH.

Perubahan (SEMUA live via symlink, tanpa rebuild):
1. `~/PX4-Autopilot/.../models/OakD-Lite/model.sdf` depth camera:
   hfov 1.274->1.5882 (91 deg), near 0.2->0.15, far 19.1->10.0. Resolusi
   tetap 640x480 → VFOV sim ~74.7 vs Gemini 66 deg = mismatch vertikal yang
   DITERIMA. Terverifikasi live: max finite depth 9.17 m, beyond-range = inf.
2. `expert_planner_node.py` + `nn_inference_node.py` `_form_nn_input`:
   normalisasi RENTANG-TETAP `DEPTH_NORM_MAX_M=10.0` (0=invalid, 255=beyond;
   nan/inf ditangani nan_to_num). Identik di kedua node.
3. ~~`longitu_step` 5.0 -> 7.5~~ **DIBATALKAN sore harinya — lihat "REVERT
   longitu_step" di bawah. TETAP 5.0 di kedua node.**
4. `camera_vfh.launch.py` hfov_deg 88 -> 91.

⚠️ KONSEKUENSI: SEMUA dataset & checkpoint sebelum 2026-07-14 = LEGACY —
tidak kompatibel dua arah dgn konvensi baru (normalisasi + FOV + range +
distribusi target beda). Wajib re-collect + retrain dari nol untuk lini baru.
Run parsial `~/training_data/run_20260714_145910` (483 baris gauntlet_train2,
konvensi LAMA, koleksi dihentikan user) ikut legacy. Koleksi ulang BELUM
dijalankan (ditahan atas permintaan user). Skrip orkestrasi koleksi siap:
scratchpad `collect_g2/collect_gauntlet_train2.sh` (babysitter auto-restart).
TODO deploy nyata: node konversi Orbbec 16UC1 mm -> float32 m + resize
1280x800 -> 640x480 (sudah ada pola di `gz_depth_bridge_node`).

## FIX ANTISIPASI EXPERT: seed lanjutan (v1.13, 2026-07-14)

### Masalah (dari user): "drone maju sampai vision tertutup baru berputar"

Prinsipnya: student (FM) TIDAK MUNGKIN lebih pintar dari teacher. Kalau label
expert berisi "maju sampai mepet baru belok", FM meniru itu, lalu di inference
(tanpa peta ground-truth + latensi octomap) jadi lebih telat lagi -> buntu ->
escape. Jadi expert-nya yang harus diperbaiki lebih dulu.

### Diagnosis TERUKUR (harness offline, planner+frontend ASLI, tanpa Gazebo)

Skrip (scratchpad): `sim_expert_approach.py`, `sweep_anticipation.py`,
`sweep2_safedis.py`, `validate_continuity.py`. Simulasi receding-horizon:
drone menerbangkan ~1 m tiap replan (1 Hz @ 1 m/s) lalu replan — persis pola
closed-loop.

- Expert baru menyimpang (|y| >= 0.3 m) pada **0.7-1.7 m** dari muka obstacle
  — saat obstacle sudah menutupi **~100% HFOV** (91 deg). Keluhan user
  TERKONFIRMASI kuantitatif.
- **Akar masalah (bukan kamera, bukan model): PROCRASTINATION receding-horizon.**
  Jejak per-siklus: (a) selama target lokal jatuh DI DEPAN obstacle, garis
  lurus memang bebas -> rencana lurus itu BENAR; (b) begitu target jatuh DI
  BALIK obstacle, garis head->target menembus obstacle tapi target TIDAK
  digeser (LOS-nudge sengaja dimatikan v1.3 demi multimodalitas) -> tail duduk
  persis di belakang dinding -> min-jerk menaruh belokan di TENGAH segmen
  (energi minimum); (c) yang diterbangkan tiap siklus cuma ~1 m PERTAMA yang
  masih lurus, lalu replan menyusun ulang DARI NOL -> belokan digeser "nanti"
  lagi. Belokan tak pernah terlaksana; drone sampai di dinding dengan y~0.5 m
  -> replan gagal -> escape.

### Fix yang DIADOPSI: seed lanjutan (`use_plan_continuity`, default true)

- `homotopy_frontend.py` v1.13: `topo_batch_plan()` membaca
  `frontend.anchor_seed` (disetel node tiap replan) dan menaruhnya sebagai
  seed **TAMBAHAN paling depan** (bukan pengganti) — enumerasi A* tetap jalan
  penuh, jadi `all_solutions` (label multimodal) TIDAK berkurang. Solusi hasil
  seed itu diterbangkan bila biayanya <= `CONTINUITY_COST_FACTOR` (2.5) x
  termurah. Efek: belokan TETAP DI TEMPATNYA dalam koordinat dunia -> benar-
  benar dieksekusi, tidak digeser tiap replan.
- `expert_planner_node.py`: `_set_anchor_seed()` (sisa trajektori aktif pada
  1/3 & 2/3 waktu tersisa di depan head — pola sama dengan `_anchor_from_traj`
  di fm_inference_node), dipanggil sebelum loop replan; `_clear_anchor_seed()`
  saat discontinuity (escape sudah otomatis lewat `_traj.invalidate()`).
- Hasil terukur (ESDF ber-inflasi 0.25 m, setia dengan aslinya): divert
  **1.7 -> 2.7 m** (box 4.8 m); **stuck/replan-gagal HILANG** pada box 2.6 m
  (OFF: 4 gagal & tak lolos -> ON: lolos, 0 gagal). Target lokal TIDAK diubah
  -> scene tetap ambigu -> multimodalitas label aman (self-test frontend tetap
  2 kelas; regresi `forced_sign` v1.12 tetap 1% wrong-side).

### DIUJI tapi TIDAK diadopsi (jebakan yang nyaris terlewat)

`safe_dis` 0.5 -> 1.0 (+ `collision_cost_tol` 10 -> 100) menambah antisipasi
(divert 1.76 -> 2.74 m di box 2.6 m, 0 replan gagal) — TAPI pada ESDF
ber-inflasi ia **MENJEGAL threading celah 2.0 m**: edt pusat pintu 0.70 <
safe_dis 1.0 -> seluruh pintu kena penalti -> 4x replan gagal -> drone menolak
masuk (= bug "berputar di pintu R5"). Uji pertama sempat LOLOS PALSU karena
mock ESDF belum meniru inflasi. **safe_dis TETAP 0.5.** Kalau mau lanjut ke
arah ini: safe_dis harus ADAPTIF (dibatasi lebar koridor lokal), bukan
konstanta global.

### REVERT `longitu_step` 7.5 -> 5.0 (sore 2026-07-14) — MENYEBABKAN CRASH

Uji SITL gauntlet_train2 (arm BASELINE, continuity OFF) dgn longitu_step 7.5:
**PHYSICS EXPLOSION di x=3** (40 s setelah takeoff). Rantainya terbaca jelas di
log: biaya rencana meledak `12 -> 33 -> 66.8` -> "Skip: trajectory unsafe/void"
-> ESCAPE -> escape ping-pong (3x ganti arah dalam 4 s) -> "left the arena
pos=(3.7,-6.1)" -> explosion speed 8.1 m/s. Sebab: gauntlet_train2 punya event
tiap ~4 m; target 7.5 m mendarat DI BALIK 1-2 obstacle sekaligus -> garis
head->tail memotong beberapa event -> MINCO dipaksa manuver besar -> tak aman.
(Pagi harinya world SAMA terbang mulus 483 sampel dgn longitu_step 5.0.)

Perubahan lain hari itu TIDAK menyentuh kendali terbang: ESDF expert dibangun
dari `/pointcloud/map` (**pcd ground-truth**, bukan kamera) -> FOV/range/
normalisasi kamera hanya memengaruhi GAMBAR dataset + VFH, bukan planner.

**longitu_step TETAP 5.0** (kedua node). Justifikasi tambahan: sweep offline
menunjukkan longitu_step nyaris tidak menyumbang antisipasi (divert ~sama di
L=5/7.5/10) — yang menyumbang adalah seed lanjutan v1.13.

Verifikasi ulang (L=5.0, gauntlet_train2, 220 s/trial):

| cont | escape | replan ok/fail | explosion | x_max | sampel |
|---|---|---|---|---|---|
| OFF  | 14 | 65 / 11 | 0 | 28.5 | 94  |
| ON#1 |  6 | 80 / 11 | 0 | 32.0 | 110 |
| ON#2 |  8 | 80 / 13 | 0 | 30.3 | 128 |

0 explosion di semua trial L=5.0 -> revert menyelesaikan crash. Seed lanjutan
(ON) memangkas escape ~separuh (14 -> 6-8), menaikkan replan sukses (65 -> 80)
dan mencapai goal (OFF berhenti di 28.5). n kecil & closed-loop STOKASTIK —
trial tambahan sedang berjalan.

### KOREKSI: harness offline TIDAK robust secara kuantitatif

Angka divert dari harness (`sim_expert_approach.py` dkk) **berayun** tergantung
konfigurasi (mis. di L=5.0 continuity ON sempat terlihat LEBIH BURUK untuk box
2.6 m — berlawanan dgn hasil di L=7.5). MEKANISME-nya (procrastination
receding-horizon, terlihat jelas di jejak per-siklus) tetap sahih dan
dikonfirmasi SITL (escape 14 -> 6-8), tapi **jangan mengutip angka divert
harness sebagai hasil**. Harness hanya menguji 1 obstacle di arena LAPANG —
justru itu sebabnya ia gagal menangkap bahwa longitu_step 7.5 menjebol di dunia
PADAT. Metrik yang dipercaya: escape/replan-fail/explosion dari SITL.

### SISA (belum dikerjakan) — antisipasi belum tuntas

Untuk obstacle LEBAR drone masih mulai belok saat pandangan ~90% tertutup
(divert 2.7 m di box 4.8 m). Kontinuitas menghentikan penundaan BERULANG, tapi
rencana PERTAMA saat obstacle masuk horizon tetap menaruh belokan di tengah
segmen. Menggeser lebih dini butuh salah satu dari:
(a) LOS-nudge target (v1.3 mematikannya demi multimodalitas -> keputusan riset:
antisipasi vs ambiguitas label), atau
(b) suku biaya baru di MINCO yang menghukum "menuju ruang yang akan terhalang"
(target tetap lurus -> ambiguitas aman; lebih sesuai klaim riset, tapi menyentuh
cost+gradien MINCO).
Fix v1.13 BARU divalidasi offline — belum diuji SITL.

## INTEGRASI RL SEED GENERATOR — `use_rl_seeds` (2026-07-17)

### Apa yang ditambahkan

Policy SAC beku (`sac_v1_ft_side`, bimodal ~50% di gym) dipakai sebagai
SEED GENERATOR dual-h di expert (Gazebo), menggantikan enumerasi A* saat
berhasil — fallback otomatis ke `batch_plan` lama (topo/3-seed) saat gagal:

1. **File baru di paket** (dicopy dari `~/rl_gym_v1.0/rl_gym/`):
   `neo_planner/rl_seed_adapter.py` + `neo_planner/policy_export_ft_side.pt`
   (2,8 MB TorchScript; live via symlink build, TANPA rebuild).
2. **`expert_planner_node.py`**: param `use_rl_seeds` (default **false** =
   perilaku lama 100%) + `rl_policy_path`; helper `_try_rl_seed_plan()` —
   2 seed (h=-1/h=+1) dioptimasi 1x `plan_once` per seed TANPA retry
   perturbasi (retry `warm_start_plan` mengganti seed dgn generik, merusak
   struktur dual-h); semua yang konvergen -> `all_solutions` (format sama
   `batch_plan`, jadi pipeline mode_sig/dedup/safety TIDAK berubah);
   termurah dieksekusi. Hook di `_replan_and_record()` sebelum `batch_plan`;
   log replan dapat tag ` RL`; ringkasan akhir mencetak `RL seeds : ok/try`.
3. **`expert_planning.launch.py`**: argumen `use_rl_seeds` diteruskan.

Cara pakai (world lain: ganti pcd+goal seperti biasa):
```bash
ros2 launch neo_planner expert_planning.launch.py \
    pcd_path:=$HOME/neo_planner_ws/src/worlds/training_worlds/poles_wide.pcd \
    goal_x:=20.0 target_samples:=200 use_rl_seeds:=true
```

### Hasil verifikasi

- **Offline** (MinJerkPlanner ASLI + mock ESDF box ala poles_wide, 60 scene;
  skrip scratchpad `verify_rl_seeds_offline.py`): 88% seed konvergen, 100%
  scene >=1 seed (tanpa fallback), 48% scene bimodal murni (2 sig beda),
  ~67 ms/replan total (inferensi policy 1,8 ms).
- **SITL poles_wide trial #1**: 38 sampel lalu PHYSICS EXPLOSION — TAPI
  signature-nya pola LAMA yang terdokumentasi (leg pulang, drone hanyut ke
  tepi arena y~5, stuck, escape ping-pong 3x ganti arah <5 s), bukan crash
  menerbangkan trajektori RL langsung.
- **SITL poles_wide trial #2**: **SELESAI 200/200 sampel, 0 explosion**,
  14 escape (transien, selalu recover), accept rate 81,6%, lag 16 ms.
  `RL seeds: 136/305 replan konvergen (45%), 53 dual-h (17%)`; waktu replan
  RL ~164 ms vs fallback ~278 ms (sampel log; enumerasi topo penuh
  historis 240-960 ms). Dataset `run_20260717_210804`: 200 baris/179 scene,
  **11,7% scene bimodal murni** (vs 4-6% baseline v1.5/v1.7 box_narrow —
  world beda, jangan dibandingkan 1:1), sisi seimbang (+1: 49 vs -1: 66),
  unimodal-nonfree 39,7%.

### Catatan & sisa masalah (BELUM dikerjakan)

- **Jalur RL mem-bypass anchor continuity v1.13 dan enumerasi topo** saat
  seed RL konvergen -> tidak ada seed lanjutan antar-replan; dugaan
  penyumbang escape lebih sering (14/run vs biasanya lebih rendah di world
  mudah). Kalau mau digabung: tawarkan `frontend.anchor_seed` juga di jalur
  RL, atau pakai RL hanya sebagai seed TAMBAHAN di `topo_batch_plan`.
- **Konvergensi seed RL di penerbangan nyata cuma 45%** (offline 88%) —
  scene nyata lebih beragam (multi-obstacle, dinding arena virtual, head
  velocity/target ter-clamp). 55% replan tetap jatuh ke fallback lama
  (aman, tapi manfaat kecepatan hilang di situ).
- **Mask unknown**: `build_crop_from_esdf` menganggap semua cell teramati
  (octomap -1 tidak diteruskan) — probe gym menunjukkan ini melemahkan
  pemecahan sisi oleh h. Kalau seed dual-h sering satu sisi, ini penyebab
  utamanya.
- Model masih rezim sensor gym 88°/6 m — fine-tune adaptasi Gemini 2
  disarankan sebelum koleksi dataset serius (keputusan lama, ditunda).
- Revert: cukup JANGAN set `use_rl_seeds:=true` (default false); tidak ada
  perubahan perilaku lain.

## DIAGNOSA + FIX: escape-spin di G4 gauntlet_train2 — sy 7.6 -> revert 5.6 (2026-07-20)

### Gejala (dilaporkan user)

Saat collecting normal (topo frontend, TANPA RL seeds), drone kadang tidak
menabrak/dianggap gagal lewat karena dimensi, melainkan **escape aktif
berulang dan drone berputar-putar di depan obstacle tanpa pernah
melewatinya** — bukan macet permanen (butuh detik-menit), tapi sangat lama
dan tidak produktif untuk collecting.

### Diagnosa (SITL, `use_rl_seeds:=false`, gauntlet_train2, 400 sampel)

Direproduksi dengan telemetri jelas: pada satu episode, `samples` counter
**diam di 350 selama >40 detik** sementara posisi bolak-balik x=[13,1-15,6]
di depan **G4** (dinding tanpa pintu, x=16,5) — 5 siklus
`Replan failed (budget) -> ESCAPE -> mundur/maju sedikit -> gagal lagi`,
y hanyut perlahan dari +2,7 turun ke -0,4 sebelum akhirnya (setelah ~90
detik total) kebetulan hanyut cukup jauh untuk menemukan celah bawah.
Run selesai 400/400 (0 explosion) tapi `Topo fallback: 455 total` —
rasio fallback/replan jauh lebih tinggi dari world lain (poles_wide:
99/305).

**Akar penyebab ditemukan di kode, bukan dugaan**: `generate_wide_worlds.py`
G4 pernah diperpanjang `sy 5.6 -> 7.6` (+1 m tiap ujung, komentar sendiri:
"atas permintaan") — perubahan ini TIDAK pernah masuk ke CLAUDE.md
(dokumentasi jadi basi). Efeknya: celah atas yang tadinya 2,1 m menjadi
**~1,1 m — di bawah lebar yang bisa dilewati drone sama sekali**. Dinding
jadi efektif satu sisi (paksa bawah, celah 2,3 m), menuntut simpangan
lateral **>3,5 m** dari sumbu tengah untuk memutar ujung bawah.

Konteks tambahan yang memperjelas kenapa ini sulit untuk MEKANISME MANA
PUN (bukan cuma bug spesifik): obstacle terlebar yang pernah dilatih ke
policy RL (`sac_v1_ft_side`, gym `scene_generator.py` level>=3
"wall_gauntlet regime") = **3,0 m**, dan plafon aksi lateral RL
(`q_scale`) juga **3,0 m** — disetel pas kurikulum tersulit itu. G4 versi
diperpanjang (>3,5 m) berada **di luar distribusi training** untuk kedua
mekanisme (klasik maupun RL) — bukan bukti "RL/planner tidak bisa
beradaptasi", melainkan geometri yang memang di luar apa pun yang pernah
divalidasi. Prinsip yang sama seperti fix v1.13 ("student tidak mungkin
lebih pintar dari teacher, perbaiki expert/world dulu") berlaku di sini.

### Fix: revert G4 ke sy=5.6

`generate_wide_worlds.py` GAUNTLET_TRAIN2 G4 dikembalikan ke geometri asli
(`sy=5.6`, celah atas ~2,1 m / bawah ~3,3 m, keduanya feasible & dalam
rentang q_scale 3,0 m). World diregenerasi
(`python3 generate_wide_worlds.py`), `.sdf` dicopy ulang ke
`~/PX4-Autopilot/Tools/simulation/gz/worlds/gauntlet_train2.sdf`.
`audit_gauntlet_train2.py` **LULUS PENUH** pasca-revert: G4 kembali 2
kelas homotopi (`sigs=[(-1,-1),(1,1)]`), pinch tersempit 1,90 m,
passability @0,65 m dan @0,80 m terjaga di semua event.

### Rekomendasi lanjutan

Kalau di masa depan ingin sengaja menguji obstacle di luar kurikulum RL
(>3,0 m / >q_scale), perlakukan sebagai stress-test eksplisit yang
disadari (bukan default collecting), dan pertimbangkan perbesar horizon
target lokal / escape bias khusus di titik itu dulu — bukan mengandalkan
drone hanyut kebetulan lewat escape acak.

## PERLUASAN scene_generator RL: level L4 "gauntlet-chain" (2026-07-20)

### Motivasi (dari diagnosa OOD)

Arena training RL (`~/rl_gym_v1.0/rl_gym/scene_generator.py`) SELAMA INI cuma
L0-L3: koridor 20 m SATU ARAH (start kiri -> goal kanan), maksimal 1 obstacle
lebar frontal (L3). Diagnosa kegagalan RL di gauntlet_train2 (lihat "area
navigasi training RL" + head-to-head 2026-07-20): **78% escape (18/23) di leg
PULANG** — arah yang tak pernah dilatih — dan crop 6.4 m sering memuat 2 event
sekaligus (G1/G2/G3 berdekatan), konfigurasi di luar distribusi L0-L3. Model
tidak rusak; arenanya terlalu sempit untuk deployment multi-event dua-arah.

### Yang ditambahkan (scene_generator.py v1.1, train_sac.py)

- **Level baru L4 "gauntlet-chain"** (`generate(level>=4)`): 3-4 EVENT
  berantai (spasi ~4 m, beberapa masuk 1 crop) meniru gauntlet_train2.
  Method baru `_add_event()` dengan 4 tipe: `wide_center` (G1/G4, bimodal
  L/R), `wall_door` (G2, dinding+pintu ~2 m offset), `twin_gate` (G3,
  sepasang tiang -> memutar), `offset_wide` (asimetris). >=1 event dijamin
  `wide_center` (sinyal side-conditioning h selalu ada). Semua event
  menyisakan celah lolos (obstacle tak melewati y=±3.5 -> koridor luar
  selalu bebas, konsisten gauntlet_train2).
- **DUA-ARAH**: L4 menukar start<->goal 50% (menutup kegagalan leg-pulang).
  L0-L3 TETAP satu-arah, TIDAK diubah sama sekali.
- `train_sac.py`: `--max-level` (default **4**) + diteruskan ke
  `CurriculumTelemetryCallback` (dulu hardcoded 3). Kurikulum promosi
  L0->L4 saat success-rate rolling >= 0.85.

### Verifikasi (self-test `python3 scene_generator.py`)

- L0-L3 tetap passable & tak berubah (perilaku lama utuh).
- **L4 x200 seed: passable 200/200 (100% feasible via BFS konektivitas
  edt>0.35), arah-terbalik 52% (dua-arah bekerja), 4.1 event/scene.**
- Env integrasi: `DronePlannerEnv(level=4)` reset+step 12 episode dua-arah
  tanpa error, bentuk observasi benar (crop 1x64x64, vec 8).
- Denah contoh: `~/rl_gym_v1.0/rl_gym/l4_gauntlet_chain_layout.png`.

### BELUM dikerjakan (keputusan retraining ada di user)

- **Retrain SAC dengan L4** belum dijalankan (~1.3 jt step, berjam-jam).
  Perintah: `python3 train_sac.py --run-name sac_v2_gauntlet --max-level 4`
  (default sudah 4). Setelah itu re-ekspor policy (`rl_seed_adapter.py
  --export runs/sac_v2_gauntlet/final.zip --out policy_export_v2.pt`) dan
  ganti `policy_export_ft_side.pt` di paket bila mau dipakai expert.
- Pilihan koridor: L4 dipertahankan 20 m (grid tak membesar -> L0-L3 tak
  melambat). Kalau mau parity penuh 30 m/6-event: naikkan `x_max`,
  `goal_xy`, `max_steps` — perubahan lebih besar, ditunda.
- Belum ada bukti L4 memperbaiki hasil SITL: itu baru terjawab SETELAH
  retrain + head-to-head ulang di gauntlet_train2/test01.

## HASIL: policy RL v2 (L4 dua-arah) vs lama di gauntlet_train2 (2026-07-21)

### Setup

Retrain SAC L0->L4 (2 juta step, `~/rl_gym_v1.0/rl_gym/runs/sac_v2_gauntlet/`,
final SR L4 ~0.77, side-consistency 0.89) di-ekspor ke
`neo_planner/policy_export_v2.pt` (via `rl_seed_adapter.py --export`) dan
diuji SITL di gauntlet_train2 (G4 sudah sy=5.6, use_rl_seeds:=true,
rl_policy_path baru diteruskan lewat launch). Dataset v2:
`run_20260721_171530`.

### Head-to-head (400 sampel, n=1 masing-masing)

| Metrik | Policy lama (ft_side) | Policy v2 (L4) |
|---|---|---|
| Accept rate | 80.6% | **84.9%** |
| Escape total | 23 | **17** |
| **Escape leg PULANG** | **18 (78%)** | **6 (35%)** |
| Escape leg PERGI | 5 | 11 |
| Keseimbangan sisi | 131:100 (skew) | **97:106 (seimbang)** |
| Bimodal murni | 10.2% | 6.7% |
| Konvergensi seed RL | ~46% | 39% |
| Explosion | 0 | 0 |

### Temuan

Kegagalan leg-PULANG (cacat yang ditargetkan training dua-arah L4) turun
tajam **18 -> 6 (78% -> 35%)**. Pola escape yang dulu timpang (5 pergi : 18
pulang) kini merata (11 : 6) — "titik buta arah pulang" hilang. Total escape
turun, accept naik, keseimbangan sisi dataset membaik (konsisten
side-consistency 0.90 saat training). Ini SEARAH prediksi riset: melatih RL
di arena kompleks dua-arah mengurangi kegagalan yang dulu muncul karena
policy tak pernah melihat arah pulang.

### Caveat (jangan overclaim)

- **n=1 vs n=1**, closed-loop STOKASTIK. Log policy lama sudah terhapus
  (angka lama dari analisis terdokumentasi sesi ini, bukan re-run). Klaim
  kokoh butuh >=5 trial/policy + uji signifikansi.
- Metrik sekunder JUSTRU turun: bimodal murni (6.7% vs 10.2%) & konvergensi
  seed RL (39% vs 46%). Bukan menang di semua lini — menang di yang dituju
  (leg pulang, keseimbangan sisi), campur di lainnya.
- Escape leg-pergi naik (5->11) — total membaik semata dari leg pulang.

### Sisa (belum dikerjakan)

- Multi-trial (>=5) v2 + baseline lama untuk signifikansi statistik.
- Evaluasi lintas-level L0-L4 policy v2 (cek forgetting scene mudah) — BELUM.
- Uji held-out (test01/wide_test) untuk klaim generalisasi, bukan cuma
  training world gauntlet_train2.

## TURUNKAN safety distance (fm_planner): safe_dis 0.5 -> 0.20 (2026-07-21)

### Alasan (usulan user)

User menduga `safe_dis` yang berlaku terlalu besar membuat drone sulit
ber-navigasi di ruang sempit/kompleks, dan mengusulkan ulang skala margin:
jarak aman dari **pusat massa** ke obstacle seharusnya cuma radius bodi
(pusat->mounting motor, ~0.25 m) + margin tambahan kecil, total **~0.5 m**
saat diam; boleh naik ke **~0.70 m** saat melaju (v~1.0 m/s = cruise lurus,
bukan sedang bermanuver ketat) karena kecepatan tinggi biasanya berarti
sedang di lorong terbuka, bukan di celah sempit.

Sanity-check dulu terhadap riwayat: menaikkan `safe_dis` 0.5->1.0 pernah
dicoba (catatan v1.13 di atas) dan JUSTRU merusak threading celah 2.0 m
(bug "berputar di pintu R5") -> **menurunkan** `safe_dis` (arah usulan user)
justru arah yang benar berdasarkan bukti historis proyek ini sendiri,
bukan arah baru yang belum teruji.

### Temuan saat implementasi: inflasi ESDF nyata = 0.30 m, bukan 0.25 m nominal

`esdf_ros2.py` membulatkan `n_cells = ceil(inflation_radius / resolusi)` ->
`ceil(0.25/0.1) = 3` sel -> inflasi FISIK yang benar-benar diterapkan =
**3 x 0.1 = 0.30 m**, bukan 0.25 m yang dikonfigurasi. Jadi angka literal
usulan user (`safe_dis=0.25`) akan menghasilkan total 0.30+0.25=**0.55 m**
(nyaris sama dengan opsi "aman" yang TIDAK dipilih user), bukan 0.5 m yang
dimaksud. Konstanta dikoreksi memakai inflasi RIIL (0.30 m), bukan nominal,
supaya hasil akhir benar-benar 0.5/0.70 m sesuai maksud user:

```
safe_dis (base)  = 0.50 - 0.30(inflasi riil) = 0.20
K_SPEED_MARGIN   = (0.70 - 0.50) / 1.0        = 0.20
```

### Perubahan

- `min_jerk_planner.py` `PlannerConfig.safe_dis`: 0.5 -> **0.20**. `hard_dis`
  (0.30) TIDAK diubah — user memilih opsi "persis angka usulan" setelah
  diberi tahu bahwa ini membuat `safe_dis` (target lunak) < `hard_dis`
  (lantai keras): zona soft-cubic `[hard_dis, safe_dis)` jadi kosong,
  potensi dua-tingkat kolaps jadi satu tingkat (hard barrier saja). Trade-off
  ini SUDAH disetujui secara eksplisit, bukan bug yang terlewat.
- `expert_planner_node.py`: `K_SPEED_MARGIN` 0.25 -> **0.20**;
  `SAFE_DIS_TRAP_MIN` 0.30 -> **0.20** (WAJIB disamakan dengan base baru —
  kalau tetap 0.30, `_dynamic_safe_dis()`'s `max(SAFE_DIS_TRAP_MIN, ...)`
  akan diam-diam membatalkan penurunan base ke 0.30 lagi). Konsekuensi:
  mekanisme shrink-saat-replan-gagal (`SAFE_DIS_TRAP_STEP`) jadi INERT (base
  sudah di titik minimum, tak ada lagi "ruang lunak" tersisa untuk disusutkan)
  — dipertahankan di kode untuk konsistensi struktural/tuning masa depan,
  bukan dihapus.
- `expert_planning.launch.py`: default `safe_dis` 0.5 -> **0.20**, deskripsi
  argumen diperbarui menjelaskan hubungan dengan inflasi riil 0.30 m.

### Hasil (diverifikasi via replikasi formula `_dynamic_safe_dis`+`safe_dis_eff`)

| v (m/s) | replan gagal beruntun | Total dari pusat massa |
|---|---|---|
| 0.0 | 0 atau berapa pun | **0.50 m** |
| 1.0 (v_max cruise) | 0 atau berapa pun | **0.70 m** |

Smoke-test end-to-end (`RLSeedAdapter` + `MinJerkPlanner` asli, `hard_dis=0.30
> safe_dis=0.20`) tetap berjalan normal, seed RL tetap konvergen — kombinasi
`safe_dis < hard_dis` tidak menyebabkan crash/exception di optimizer.

### BELUM diuji SITL — ini perubahan analitis/offline

Perubahan ini HANYA diverifikasi lewat replikasi formula + smoke-test
end-to-end di atas (bukan Gazebo). Efek nyata terhadap perilaku terbang
(apakah benar mengurangi kesulitan navigasi di celah sempit tanpa membuat
drone menempel obstacle di ruang terbuka, mengingat zona soft-cubic kini
kosong) **belum diverifikasi di SITL**. Rekomendasi: jalankan collecting
singkat (mis. `box_narrow`, target_samples kecil) dan bandingkan
escape/replan-fail/min_clear dengan baseline lama sebelum dipakai volume
penuh — pola verifikasi yang sama seperti perubahan `safe_dis` sebelumnya
di v1.13.

### Launch lain (TIDAK disentuh, cek scope-nya)

`nn_planning_unknown.launch.py` (safe_dis default **0.7**), `fm_planning_unknown.launch.py`
dan `nn_inference.launch.py` (keduanya **0.5**) adalah launch INFERENSI
(deployment), terpisah dari `expert_planning.launch.py` (collecting) yang
diubah sesi ini. `K_SPEED_MARGIN`/`SAFE_DIS_TRAP_MIN` juga tidak ada di
`nn_inference_node.py`/`fm_inference_node.py` (mekanisme dinamis ini
khusus expert). Belum ditanyakan ke user apakah perubahan yang sama juga
diinginkan untuk inferensi.
