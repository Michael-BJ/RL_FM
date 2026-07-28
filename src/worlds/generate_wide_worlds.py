#!/usr/bin/env python3
# generate_wide_worlds.py — v1.0
"""
Generator sepasang world Gazebo (.sdf) + peta ground-truth (.pcd) untuk
eksperimen ambiguitas multimodal (FM vs NN planner).

Satu daftar obstacle -> dua file yang DIJAMIN konsisten:
  <name>.sdf : world PX4 SITL (boilerplate identik poles.sdf)
  <name>.pcd : voxel 0.1 m — lantai 40x30 m di z=0.05 + isi solid obstacle
               (format identik poles.pcd, dipakai publish_pcd_node -> ESDF)

World yang dihasilkan (2026-07-08: world TRAINING ditulis ke training_worlds/
subfolder, world HELD-OUT tetap di direktori ini — lihat TRAINING_WORLD_NAMES):
  poles_wide     : TRAINING/collecting [-> training_worlds/] — kotak lebar
                   1.2-2.4 m TEPAT di sumbu terbang y=0 (scene ambigu
                   simetris: kiri/kanan sama valid) + tiang tipis & obstacle
                   off-center sebagai variasi.
  wide_test      : TEST/inference (held-out, tetap di sini) — kelas
                   tantangan sama, layout held-out (lebar & posisi berbeda,
                   beberapa sedikit off-center).
  wall_gauntlet  : TEST diskriminator (held-out, tetap di sini) — 3 dinding
                   simetris lebar menaik.
  gauntlet_train : TRAINING v2 (2026-07-07) [-> training_worlds/] — koridor
                   0..30, lebih kompleks dari wall_gauntlet: dinding
                   berpintu, S-curve, slalom. WAJIB goal_x:=30.0
                   arena_x_max:=33.0.
  test01         : TEST held-out (tetap di sini), pasangan gauntlet_train.
  gauntlet_train2: TRAINING v3 (2026-07-08) [-> training_worlds/] — world
                   kompleks kedua, tipe event berbeda dari gauntlet_train/
                   test01 (silinder gemuk, gerbang 2-tiang, dinding tanpa
                   pintu asimetris, zig-zag 3-segmen). WAJIB goal_x:=30.0
                   arena_x_max:=33.0.

Cara pakai:
    python3 generate_wide_worlds.py            # world training -> ./training_worlds/,
                                                # world held-out -> direktori ini
    cp training_worlds/poles_wide.sdf wide_test.sdf \
        ~/PX4-Autopilot/Tools/simulation/gz/worlds/

Lalu:
    ros2 launch fm_planner sitl.launch.py world:=poles_wide
    ros2 launch fm_planner expert_planning.launch.py \
        pcd_path:=<dir ini>/training_worlds/poles_wide.pcd target_samples:=5000

Catatan desain (arena expert: X 0..33, Y +-13, drone bolak-balik (0,0)<->(goal_x,0)):
  - Spawn (0,0) dan area turnaround dibiarkan kosong (radius ~2.5 m).
  - Tinggi obstacle >= 3.4 m supaya tidak bisa dilewati dari atas.
  - Jarak antar obstacle sumbu ~5-6 m agar drone kembali head-on
    setelah tiap detour.
"""
import math
import os

RES = 0.1                     # resolusi voxel .pcd (identik poles.pcd)
GROUND_X = (-5.0, 35.0)       # lantai .pcd (identik poles.pcd)
GROUND_Y = (-15.0, 15.0)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2026-07-08: world TRAINING (poles_wide, gauntlet_train, gauntlet_train2)
# dipindah ke training_worlds/ (sekumpul dengan 5 world atom lain dari lini
# riset yang sama) supaya semua world training ada di satu folder. World
# HELD-OUT (wide_test, wall_gauntlet, test01) TETAP di sini -- jangan
# dipindah, folder training_worlds/ hanya untuk world yang aman dikumpulkan.
TRAIN_OUT_DIR = os.path.join(OUT_DIR, "training_worlds")
# 2026-07-21: konsolidasi — hanya gauntlet_train, gauntlet_train2, dan maze_train
# (world lain dihapus atas permintaan user). Ketiganya world TRAINING.
TRAINING_WORLD_NAMES = {"gauntlet_train", "gauntlet_train2",
                        "gauntlet_train3", "maze_train"}


# ─────────────────────────────────────────────────────────────────────────────
# Definisi obstacle
#   box     : dict(type="box", x, y, sx, sy, h)   (sx=depth arah terbang, sy=lebar)
#   cylinder: dict(type="cyl", x, y, r, h)
# ─────────────────────────────────────────────────────────────────────────────
# Koridor terbang 0..20 (jalankan dengan goal_x:=20); spawn (0,0) dan
# turnaround (20,0) bersih radius >= 2 m.
POLES_WIDE = [
    # ── Generator ambiguitas: kotak lebar TEPAT di y=0 ──
    dict(type="box", x=4.5,  y=0.0,  sx=0.6, sy=1.2, h=3.6),
    dict(type="box", x=9.0,  y=0.0,  sx=0.6, sy=1.6, h=3.6),
    dict(type="box", x=13.5, y=0.0,  sx=0.8, sy=2.0, h=3.6),
    dict(type="box", x=17.0, y=0.0,  sx=0.8, sy=2.4, h=3.6),
    # ── Variasi: tiang tipis off-center (skill dasar tetap terlatih) ──
    dict(type="cyl", x=6.8,  y=2.8,  r=0.4, h=3.9),
    dict(type="cyl", x=7.0,  y=-3.0, r=0.4, h=3.9),
    dict(type="cyl", x=11.2, y=-2.7, r=0.5, h=3.9),
    dict(type="cyl", x=11.0, y=3.1,  r=0.4, h=3.9),
    dict(type="cyl", x=15.3, y=2.9,  r=0.5, h=3.9),
    dict(type="cyl", x=15.5, y=-3.2, r=0.4, h=3.9),
    # ── Kotak kecil off-center (seperti poles.sdf asli) ──
    dict(type="box", x=2.6,  y=-2.3, sx=1.0, sy=1.0, h=3.4),
    dict(type="box", x=18.6, y=3.0,  sx=1.0, sy=1.0, h=3.4),
]

# ─────────────────────────────────────────────────────────────────────────────
# WALL_GAUNTLET — world diskriminator NN-gagal / FM-lolos (dibuat 2026-07-06).
#
# Prinsip desain (kenapa kotak 2.6 m di wide_test TIDAK cukup membedakan):
# recovery dari komitmen-tengah di kotak sempit itu murah — half-width 1.3 m
# masih mudah ditebus optimizer/escape. Supaya mode-averaging (warm start di
# TENGAH) benar-benar mahal, dinding harus SIMETRIS EKSAK di y=0 (tidak ada
# cue sisi; rata-rata kiri/kanan = tepat tengah) dan LEBAR relatif terhadap
# jangkauan sensing (max_range 4 m, HFOV 88°):
#   dinding 4.0 m : kedua tepi terlihat sampai jarak 2.1 m (jendela keputusan
#                   ~1.9 m perjalanan) — level pemanasan.
#   dinding 4.8 m : tepi terlihat sampai 2.5 m (jendela ~1.5 m).
#   dinding 5.6 m : tepi terlihat sampai 2.9 m (jendela ~1.1 m ≈ 2x replan)
#                   — komitmen dini WAJIB; ragu = terkurung di depan dinding.
# Lebar menaik → satu run menghasilkan kurva ambang (NN lolos dinding mana,
# gagal di mana), bukan sekadar biner. Pasangan tiang cermin di kedua ujung
# menjaga simetri DAN membatasi koridor luar tetap 1.4 m (skala koridor sama
# dengan world training — skill terbang tidak berubah, hanya KEPUTUSAN yang
# diuji). Jarak antar dinding 5.5-6 m agar drone kembali head-on (re-decision
# independen) sebelum dinding berikutnya. goal_x := 22.0.
#
# PRASYARAT eksperimen: checkpoint FM harus benar-benar bimodal (retrain
# --mirror_aug; checkpoint 2026-07-06 TERBUKTI unimodal — kandidat 2 sisi
# hanya 3%). Tanpa itu FM akan gagal persis seperti NN di world ini.
# ─────────────────────────────────────────────────────────────────────────────
WALL_GAUNTLET = [
    # Tiga dinding simetris y=0, lebar menaik (tingkat kesulitan naik)
    dict(type="box", x=6.0,  y=0.0, sx=0.4, sy=4.0, h=3.6),
    dict(type="box", x=12.0, y=0.0, sx=0.4, sy=4.8, h=3.6),
    dict(type="box", x=17.5, y=0.0, sx=0.4, sy=5.6, h=3.6),
    # Tiang cermin: simetri terjaga, koridor luar tepat 1.4 m di semua dinding
    dict(type="cyl", x=6.0,  y=3.8,  r=0.4, h=3.9),
    dict(type="cyl", x=6.0,  y=-3.8, r=0.4, h=3.9),
    dict(type="cyl", x=12.0, y=4.2,  r=0.4, h=3.9),
    dict(type="cyl", x=12.0, y=-4.2, r=0.4, h=3.9),
    dict(type="cyl", x=17.5, y=4.6,  r=0.4, h=3.9),
    dict(type="cyl", x=17.5, y=-4.6, r=0.4, h=3.9),
]

# ─────────────────────────────────────────────────────────────────────────────
# GAUNTLET_TRAIN — world TRAINING v2 (dibuat 2026-07-07), pasangan frontend
# homotopi. Koridor 0..30 (WAJIB goal_x:=30.0 arena_x_max:=33.0 — default
# launch 25.0 memotong koridor!).
#
# Prinsip desain (kenapa lebih kompleks dari wall_gauntlet):
#   wall_gauntlet hanya menguji SATU kelas keputusan (dinding simetris,
#   kiri/kanan). Untuk TRAINING, dataset harus memuat semua kelas keputusan
#   yang akan diuji + variasi struktur pintu/urutan supaya model belajar
#   memilih, bukan menghafal layout:
#   R1 x=3.5  kotak 2.0 m sumbu       -> ambiguitas dasar (2 kelas)
#   R2 x=7.5  dinding + PINTU 2.0 m   -> pintu samping y=+1.35 (dinding
#             (menutup sumbu y=0)        menutup sumbu): pintu vs memutar
#   R3 x=12   dinding 4.4 m + tiang   -> kelas wall_gauntlet (lebar beda:
#             cermin (koridor 2.0 m)      4.0/4.8/5.6 tetap held-out)
#   R4 x=16+18.5 dinding zig-zag      -> S-curve ATAU swing luar (2-3 kelas,
#             (overlap 0.4 m di y=0)      keputusan BERURUTAN, bukan tunggal)
#   R5 x=22.5 dinding 5.4 m + pintu   -> pintu OFF-CENTER 2.0 m @ y=-1.15
#             + tiang cermin             (cermin arah R2, bias sisi seimbang)
#   R6 x=26.5 slalom 4 tiang (2.3 m)  -> threading, garis lurus valid
#             (sengaja TAK-ambigu: dataset juga butuh "lurus itu benar")
#   + 3 tiang pengisi off-path (variasi depth image, bukan chokepoint).
#
# JAMINAN LAIK-LEWAT (audit otomatis: audit_gauntlet_train.py, rev 2026-07-07):
#   Semua celah fungsional >= 2.0 m -> edt pusat ~0.7 >= safe_dis 0.5 =
#   BEBAS PENALTI (pelajaran uji terbang: celah 1.5 m membuat drone berputar
#   di tempat — lihat komentar GAUNTLET_TRAIN). Batas absolut planner tetap
#   1.3 m; margin 0.7 m menyerap penggemukan voxel octomap + tracking error.
#   Audit memodelkan DINDING VIRTUAL arena (esdf_ros2 arena_bounds): sela
#   tiang-cermin -> tepi arena sengaja < 1.3 m (tertutup) di R3/R5 supaya
#   rute wajib lewat koridor yang dirancang.
# ─────────────────────────────────────────────────────────────────────────────
GAUNTLET_TRAIN = [
    # ── REVISI CELAH 2026-07-07 (v2): SEMUA celah >= 2.0 m ──
    # Uji terbang v1 (celah 1.5-1.6 m): drone BERPUTAR DI TEMPAT di pintu R5.
    # Aperture 1.5 m - 2x inflasi 0.3 = pita bebas 0.9 m -> edt puncak 0.45
    # < safe_dis 0.5 -> SEMUA rute kena penalti + penggemukan voxel octomap
    # membuat kandidat saling bergantian. Celah 2.0 m -> edt pusat ~0.7 =
    # bebas penalti (2x(0.3+0.5)=1.6 m minimum, +0.4 m margin voxel/tracking).
    # R1: kotak lebar warm-up, tiang pembatas koridor luar (gap 2.0 m/sisi)
    dict(type="box", x=3.5,  y=0.0,   sx=0.6, sy=2.0, h=3.6),
    dict(type="cyl", x=3.5,  y=3.4,   r=0.4, h=3.9),
    dict(type="cyl", x=3.5,  y=-3.4,  r=0.4, h=3.9),
    # R2: dinding y=-4.0..4.0, PINTU 2.0 m di y=+0.35..2.35 (pusat +1.35).
    #     Dinding MENUTUP sumbu y=0 (pintu tak boleh mengangkangi sumbu:
    #     garis lurus lolos lewat pintu -> signature 'free' -> frontend
    #     melipat semua rute jadi 1 kelas). Kelas: pintu / memutar bawah.
    dict(type="box", x=7.5,  y=3.175, sx=0.4, sy=1.65, h=3.6),
    dict(type="box", x=7.5,  y=-1.825, sx=0.4, sy=4.35, h=3.6),
    # R3: dinding 4.4 m + tiang cermin -> koridor dinding-tiang 2.0 m/sisi;
    #     sela tiang->dinding-virtual-arena 1.0 m = tertutup (wajib koridor)
    dict(type="box", x=12.0, y=0.0,   sx=0.4, sy=4.4, h=3.6),
    dict(type="cyl", x=12.0, y=4.6,   r=0.4, h=3.9),
    dict(type="cyl", x=12.0, y=-4.6,  r=0.4, h=3.9),
    # R4: zig-zag (S-curve): overlap 0.4 m di y=0, pinch diagonal 2.1 m.
    # REVISI 2026-07-08: spacing 2.5 -> 3.2 m (x=18.5 -> x=19.2). Root cause
    # ditemukan lewat audit: pada spacing 2.5 m, jendela diversifikasi A*
    # WALL_X_MARGIN=1.0 m di sekitar TIAP segmen saling tumpang tindih
    # (~0.6 m overlap) -- begitu ronde 1 menemukan S-weave lalu memblokir
    # sisi yang dipakai di kedua segmen, zona tumpang-tindih itu jadi
    # terblokir dari KEDUA sisi sekaligus -> tembok penuh yang memutus
    # start<->goal sama sekali (bukan sekadar mendorong ke rute lebih
    # mahal) -> ronde 2 A* selalu gagal (None), audit cuma dapat 1 kelas.
    # Diverifikasi lewat sweep spacing 2.5/2.8/3.0/3.2/3.5/4.0/4.5 m: 2.5 &
    # 2.8 m gagal (1 kelas), >=3.0 m selalu 2 kelas. Dipilih 3.2 m untuk
    # margin. Overlap 0.4 m di y=0 & pinch diagonal TETAP terjaga (cuma
    # jarak x antar-segmen yang berubah, bukan bentuk/offset y).
    dict(type="box", x=16.0, y=1.4,   sx=0.4, sy=3.2, h=3.6),
    dict(type="box", x=19.2, y=-1.4,  sx=0.4, sy=3.2, h=3.6),
    # R5: dinding 5.4 m (y -2.7..2.7) PINTU 2.0 m di y=-2.15..-0.15
    #     (pusat -1.15, cermin arah pintu R2 -> bias sisi seimbang);
    #     koridor dinding-tiang 2.0 m/sisi.
    dict(type="box", x=22.5, y=1.275, sx=0.4, sy=2.85, h=3.6),
    dict(type="box", x=22.5, y=-2.425, sx=0.4, sy=0.55, h=3.6),
    dict(type="cyl", x=22.5, y=5.1,   r=0.4, h=3.9),
    dict(type="cyl", x=22.5, y=-5.1,  r=0.4, h=3.9),
    # R6: slalom 4 tiang (celah tengah 2.0 m, sela dalam-luar 2.3 m).
    #     Tiang luar ±4.4 sejajar x=26.5: sela diagonal 1.9 m versi lama
    #     memicu explosion di jalan pulang (run_20260707_161432 — drone
    #     kembali off-axis pasca-turnaround dan harus menembus diagonal
    #     tersempit sambil replan budget-fail).
    dict(type="cyl", x=26.0, y=1.3,   r=0.4, h=3.9),
    dict(type="cyl", x=27.0, y=-1.3,  r=0.4, h=3.9),
    dict(type="cyl", x=26.5, y=4.4,   r=0.4, h=3.9),
    dict(type="cyl", x=26.5, y=-4.4,  r=0.4, h=3.9),
    # Tiang pengisi off-path (variasi persepsi, bukan chokepoint).
    # Tiang (5.4,-4.9) DIHAPUS 2026-07-07: duduk di tengah pita koridor
    # bawah R2 (band -4.3..-5.7 dgn dinding virtual) -> kantong sempit ->
    # replan budget-fail beruntun -> escape ping-pong -> explosion
    # (run_20260707_160427, 146 sampel).
    # Tiang (9.8,-2.6) DIHAPUS 2026-07-07 (bukan sekadar digeser): hanya
    # 1.64/1.70 m dari dinding R2/R3 -> lolos ambang lama (1.3 m) tapi di
    # bawah target fungsional 2.0 m -> koridor efektif sempit -> explosion
    # ke-2 (run_20260707_162311, 834 sampel). audit_gauntlet_train.py kini
    # menguji SEMUA pasangan yang melibatkan tiang pengisi terhadap
    # GAP_TARGET (2.0 m), bukan cuma PAIR_MIN (1.3 m).
    dict(type="cyl", x=19.9, y=3.8,   r=0.4, h=3.9),
]

# ─────────────────────────────────────────────────────────────────────────────
# TEST01 — world TEST held-out (dibuat 2026-07-07), pasangan uji generalisasi
# untuk gauntlet_train. HANYA untuk inference/eval — TIDAK BOLEH dipakai
# mengumpulkan data TRAINING (kecuali dataset test open-loop via expert).
# Koridor 0..30 (WAJIB goal_x:=30.0 arena_x_max:=33.0, sama dgn gauntlet_train).
#
# Prinsip: KELAS event sama dengan gauntlet_train (kompleksitas setara,
# diaudit ambang yang sama) tapi layout berbeda — urutan event ditukar
# (zig-zag sebelum dinding lebar), sisi pintu DICERMIN (R2 pintu atas ->
# E2 pintu bawah; R5 pintu bawah -> E5 pintu atas), stagger slalom dibalik,
# semua lebar/posisi digeser. Model yang menghafal layout training akan
# salah sisi di sini; model yang belajar KEPUTUSAN akan tetap benar.
#   E1 x=3.8  kotak 2.4 m sumbu        (train: 2.0 m @ x=3.5)
#   E2 x=8.0  dinding + pintu 2.0 m di y=-1.6 (train: +1.35)
#   E3 x=12.5/15 zig-zag CERMIN (bawah dulu; train: atas dulu @ 16/18.5)
#   E4 x=19   dinding 4.6 m + tiang cermin (train: 4.4 m; held-out
#             wall_gauntlet 4.0/4.8/5.6 tetap tidak tersentuh)
#   E5 x=23.5 dinding 5.4 m pintu off-center y=+1.35 (train: -1.15)
#   E6 x=26.4-27.4 slalom stagger dibalik (celah tengah 2.1 m)
#   + 1 tiang pengisi (10.7, 3.4) — >= 2.0 m dari SEMUA obstacle DAN dari
#     dinding virtual arena (pelajaran tiang (5.4,-4.9) & (9.8,-2.6) yang
#     dihapus dari gauntlet_train: pengisi dekat koridor = chokepoint).
#
# JAMINAN LAIK-LEWAT: semua celah fungsional >= 2.0 m (edt pusat ~0.7 >=
# safe_dis 0.5 = bebas penalti). Audit otomatis: audit_test01.py (ambang
# identik audit_gauntlet_train.py + cek pengisi->dinding arena).
# ─────────────────────────────────────────────────────────────────────────────
TEST01 = [
    # E1: kotak 2.4 m sumbu, tiang pembatas (gap 2.0 m/sisi, luar 2.0 m)
    dict(type="box", x=3.8,  y=0.0,    sx=0.6, sy=2.4, h=3.6),
    dict(type="cyl", x=3.8,  y=3.6,    r=0.4, h=3.9),
    dict(type="cyl", x=3.8,  y=-3.6,   r=0.4, h=3.9),
    # E2: dinding y=-4.0..4.0, PINTU 2.0 m di y=-2.6..-0.6 (pusat -1.6,
    #     CERMIN R2 train yang pintunya di +1.35). Dinding menutup sumbu.
    dict(type="box", x=8.0,  y=1.7,    sx=0.4, sy=4.6, h=3.6),
    dict(type="box", x=8.0,  y=-3.3,   sx=0.4, sy=1.4, h=3.6),
    # E3: zig-zag CERMIN (bawah dulu), overlap 0.4 m di y=0, pinch 2.1 m
    dict(type="box", x=12.5, y=-1.5,   sx=0.4, sy=3.4, h=3.6),
    dict(type="box", x=15.0, y=1.5,    sx=0.4, sy=3.4, h=3.6),
    # E4: dinding 4.6 m + tiang cermin (koridor 2.0 m/sisi; sela tiang->
    #     dinding arena 0.9 m = tertutup, wajib lewat koridor)
    dict(type="box", x=19.0, y=0.0,    sx=0.4, sy=4.6, h=3.6),
    dict(type="cyl", x=19.0, y=4.7,    r=0.4, h=3.9),
    dict(type="cyl", x=19.0, y=-4.7,   r=0.4, h=3.9),
    # E5: dinding 5.4 m (y -2.7..2.7), PINTU 2.0 m di y=+0.35..+2.35
    #     (pusat +1.35, cermin R5 train); koridor dinding-tiang 2.0 m/sisi
    dict(type="box", x=23.5, y=2.525,  sx=0.4, sy=0.35, h=3.6),
    dict(type="box", x=23.5, y=-1.175, sx=0.4, sy=3.05, h=3.6),
    dict(type="cyl", x=23.5, y=5.1,    r=0.4, h=3.9),
    dict(type="cyl", x=23.5, y=-5.1,   r=0.4, h=3.9),
    # E6: slalom 4 tiang, stagger DIBALIK (bawah dulu; celah tengah 2.1 m,
    #     sela dalam-luar 2.4 m — di atas insiden diagonal 1.9 m train v1)
    dict(type="cyl", x=26.4, y=-1.35,  r=0.4, h=3.9),
    dict(type="cyl", x=27.4, y=1.35,   r=0.4, h=3.9),
    dict(type="cyl", x=26.9, y=4.5,    r=0.4, h=3.9),
    dict(type="cyl", x=26.9, y=-4.5,   r=0.4, h=3.9),
    # Tiang pengisi off-path: 2.1 m dari dinding E2, 2.2 m dari arena y=6
    dict(type="cyl", x=10.7, y=3.4,    r=0.4, h=3.9),
]

# ─────────────────────────────────────────────────────────────────────────────
# GAUNTLET_TRAIN2 — world TRAINING v3 (dibuat 2026-07-08), world kompleks
# KEDUA (selain gauntlet_train) — posisi, dimensi, DAN bentuk event berbeda
# dari gauntlet_train/test01, bukan sekadar reorder/mirror. TRAINING (bukan
# held-out) — aman dikumpulkan. Koridor 0..30 (WAJIB goal_x:=30.0
# arena_x_max:=33.0, sama seperti gauntlet_train/test01).
#
# Kenapa dibuat: training_worlds/ (5 world atom, 1 event/world) dinilai
# terlalu sederhana untuk pilot lap-alternation; gauntlet_train/test01 sudah
# dipakai (train/held-out). Dunia ini menambah variasi TIPE EVENT yang belum
# pernah dipakai gauntlet_train/test01 sama sekali (bukan cuma angka beda):
#   G1 x=3.5   silinder gemuk BULAT dead-center (d=2.2m) + tiang pembatas
#              -> gauntlet_train/test01 SELALU pakai box untuk obstacle
#              dead-center; ini bentuk BULAT pertama di world kompleks.
#   G2 x=8.0   dinding+pintu 2.0m standar (box), offset pintu y=+1.9 (BARU
#              -- beda dari train +1.35 dan test01 -1.6).
#   G3 x=12.3  GERBANG SEPASANG TIANG (bukan dinding+pintu): 2 tiang celah
#              tengah 2.0m + 2 tiang pembatas luar -> 'threading' (1 kelas,
#              gaya R6/E6) tapi topologi BEDA dari slalom 4-tiang (cuma
#              2 tiang inti, bukan 4) DAN dari dinding+pintu (bukan box).
#   G4 x=16.5  DINDING TANPA PINTU, asimetris off-center (sy=5.6, pusat
#              y=+0.6) -> HANYA bisa dilewati memutar salah satu ujung;
#              gauntlet_train/test01 tidak pernah punya dinding solid tanpa
#              celah di tengah korfidor (mirip wall_gauntlet tapi asimetris
#              & training-safe, bukan held-out).
#   G5 x=20/22.5/25  zig-zag 3-SEGMEN (train/test01 cuma 2 segmen) -> chicane
#              lebih panjang. Diverifikasi lewat audit: HANYA 1 kelas
#              (swing penuh di luar klaster jauh lebih mahal daripada
#              S-weave, bukan alternatif kompetitif) -> threading ketiga
#              (beda TIPE dari G3/G6), bukan event ambigu L/R.
#   G6 x=27.5-28.5  slalom 4-tiang, POSISI baru (translasi +1.5m dari pola
#              gauntlet_train R6 yang sudah terbukti aman -- lihat catatan
#              di bawah kenapa translasi, bukan angka baru dari nol).
#
# CATATAN KEAMANAN: semua celah keputusan primer ditarget 2.0 m mentah
# (raw geometric aperture) -- pola SAMA yang sudah divalidasi gauntlet_train/
# test01 (post-inflasi ESDF planner 0.25 m/sisi -> ~1.5 m efektif -> edt
# pusat ~0.7, bebas penalti; celah <1.5 m TERBUKTI membuat drone berputar di
# tempat, lihat catatan gauntlet_train). G6 SENGAJA memakai geometri relatif
# R6 gauntlet_train yang sudah diaudit+diterbangkan (bukan angka baru untuk
# event tersulit) -- cuma ditranslasi ke posisi x lain di world ini, supaya
# risiko numerik event paling rawan tetap minimal. TIDAK ADA tiang pengisi
# di v1 dunia ini (pelajaran gauntlet_train: 2 dari 3 pengisi awal justru
# jadi chokepoint tak-disengaja dan memicu 2x PHYSICS EXPLOSION saat
# terbang -- lebih aman tanpa pengisi dulu, bisa ditambah belakangan setelah
# lokasi kosong diverifikasi lewat audit).
#
# JAMINAN LAIK-LEWAT: audit_gauntlet_train2.py (ambang identik
# audit_gauntlet_train.py/audit_test01.py -- lihat file itu untuk definisi).
# ─────────────────────────────────────────────────────────────────────────────
GAUNTLET_TRAIN2 = [
    # G1: silinder gemuk dead-center (d=2.2 m, BULAT bukan box) + tiang
    #     pembatas (gap silinder->tiang 2.0 m/sisi)
    dict(type="cyl", x=3.5,  y=0.0,  r=1.1, h=3.9),
    dict(type="cyl", x=3.5,  y=3.5,  r=0.4, h=3.9),
    dict(type="cyl", x=3.5,  y=-3.5, r=0.4, h=3.9),
    # G2: dinding y=-4.0..4.0, PINTU 2.0 m di y=0.9..2.9 (pusat +1.9, offset
    #     baru -- beda dari train R2 (+1.35) dan test01 E2 (-1.6)). Dinding
    #     menutup sumbu y=0 (band pintu tidak mengangkangi 0).
    dict(type="box", x=8.0,  y=3.45,  sx=0.4, sy=1.1, h=3.6),
    dict(type="box", x=8.0,  y=-1.55, sx=0.4, sy=4.9, h=3.6),
    # G3: gerbang sepasang tiang (bukan dinding+pintu) -- celah tengah 2.0 m
    #     (threading, 1 kelas by design, gaya R6/E6 tapi cuma 2 tiang inti)
    #     + 2 tiang pembatas luar (gap tiang-inti->pembatas 2.0 m/sisi).
    dict(type="cyl", x=12.3, y=1.4,   r=0.4, h=3.9),
    dict(type="cyl", x=12.3, y=-1.4,  r=0.4, h=3.9),
    # 2026-07-27: tiang pembatas luar y=±4.2 -> ±5.6 (tepi menempel dinding
    #     arena ±6.0). Sebelumnya menyisakan koridor luar 1.4 m: terlalu
    #     sempit untuk diterbangkan (guard_clearance 0.60 dari pusat berarti
    #     butuh >=1.2 m bersih) tapi cukup lebar untuk TERLIHAT bisa dilewati
    #     planner -> drone masuk lalu terjebak. Menempel dinding = jelas
    #     tertutup, sekaligus tetap memenuhi tujuan aslinya (mencegah drone
    #     memutari seluruh event lewat koridor luar).
    dict(type="cyl", x=12.3, y=5.6,   r=0.4, h=3.9),
    dict(type="cyl", x=12.3, y=-5.6,  r=0.4, h=3.9),
    # G4: dinding TANPA pintu, asimetris off-center.
    #     2026-07-27: sy 7.6 -> 5.6. Revert ini SUDAH didokumentasikan di
    #     CLAUDE.md (2026-07-20, "DIAGNOSA + FIX escape-spin di G4") tapi
    #     ternyata TIDAK PERNAH diterapkan ke generator — sy tetap 7.6, dan
    #     komentar lama masih berbunyi "Revert: kembalikan sy=5.6". Pada 7.6
    #     spans y=-3.2..4.4 -> celah ATAS cuma 1.1 m (tak muat drone) sehingga
    #     dinding jadi efektif satu-sisi; drone terbukti macet di depannya
    #     (x=17.65, y=2.95, 10+ replan gagal berturut, 2026-07-27).
    #     Pada 5.6 spans y=-2.2..3.4 -> celah ATAS 2.6 m, BAWAH 3.8 m.
    dict(type="box", x=16.5, y=0.6,   sx=0.4, sy=5.6, h=3.6),
    # G5: zig-zag 3-segmen (train/test01 cuma 2), overlap 0.4 m di y=0 tiap
    #     pasangan berurutan, spacing 2.5 m (sama seperti gauntlet_train R4).
    dict(type="box", x=20.0, y=1.4,   sx=0.4, sy=3.2, h=3.6),
    dict(type="box", x=22.5, y=-1.4,  sx=0.4, sy=3.2, h=3.6),
    dict(type="box", x=25.0, y=1.4,   sx=0.4, sy=3.2, h=3.6),
    # G6: slalom 4-tiang -- geometri RELATIF sama persis dengan R6
    # gauntlet_train (celah tengah diagonal ~2.0 m, sela dalam-luar ~2.3 m,
    # sudah diaudit+diterbangkan aman), ditranslasi +1.5 m di x ke posisi
    # baru supaya event paling rawan di seluruh gauntlet TIDAK memakai
    # angka baru yang belum pernah divalidasi.
    # 2026-07-27 (semua celah >= 2.0 m): geser +0.1 m di x supaya celah ke
    #     G5 segmen-3 (box x=25.0) naik 1.90 -> 2.00 m; renggangkan tiang inti
    #     y ±1.3 -> ±1.35 supaya celah diagonal tengah 1.99 -> 2.08 m; tiang
    #     pembatas luar ±4.4 -> ±5.6 menempel dinding arena (alasan sama
    #     seperti G3 di atas — koridor luar 1.2 m adalah jebakan, bukan jalur).
    dict(type="cyl", x=27.6, y=1.35,  r=0.4, h=3.9),
    dict(type="cyl", x=28.6, y=-1.35, r=0.4, h=3.9),
    dict(type="cyl", x=28.1, y=5.6,   r=0.4, h=3.9),
    dict(type="cyl", x=28.1, y=-5.6,  r=0.4, h=3.9),
]

# DECISIVE_GAP — HELD-OUT "NN-trap" discriminator (dibuat 2026-07-10)
# Dirancang menjawab: "buat world yang membuat NN LURUS ke tengah (buntu) tapi
# FM punya peluang nyata". Tiap event: box SIMETRIS tepat di sumbu terbang
# (y=0) yang menutup jalur lurus, + SEPASANG tiang luar simetris yang membentuk
# koridor sisi LONGGAR 2.2 m (jauh lebih lega dari wall_gauntlet 1.4 m yang
# terbukti nyaris tak muat untuk model apa pun). Logika diskriminasi:
#   - Tengah tertutup + scene SIMETRIS -> prediksi tunggal NN (mean kiri&kanan)
#     jatuh ~di tengah -> ditolak post-check -> NN tak bisa commit -> macet.
#   - FM menyampel 8 kandidat -> ada yang commit bersih ke sisi -> celah 2.2 m
#     cukup lega untuk dilewati -> FM (idealnya) lolos.
# Jika FM JUGA gagal di celah selega ini -> bukti masalahnya OSILASI/sistem
# (mode persistence OFF di fair), bukan multimodalitas model. Menang-kalah
# apa pun = informatif. Central width naik 2.6/3.0/3.4 (detour bertambah)
# tapi GAP tetap 2.2 m (kelayakan sisi konstan). Koridor 0..24 -> goal_x:=24
# (arena default 25, TANPA override). HELD-OUT: jangan collecting data di sini.
DECISIVE_GAP = [
    # E1 x=6  — box y=0 sy=2.6 (span +-1.3) + tiang luar +-3.9 -> gap 2.2 m
    dict(type="box", x=6.0,  y=0.0,  sx=0.4, sy=2.6, h=3.6),
    dict(type="cyl", x=6.0,  y=3.9,  r=0.4, h=3.9),
    dict(type="cyl", x=6.0,  y=-3.9, r=0.4, h=3.9),
    # E2 x=13 — box y=0 sy=3.0 (span +-1.5) + tiang luar +-4.1 -> gap 2.2 m
    dict(type="box", x=13.0, y=0.0,  sx=0.4, sy=3.0, h=3.6),
    dict(type="cyl", x=13.0, y=4.1,  r=0.4, h=3.9),
    dict(type="cyl", x=13.0, y=-4.1, r=0.4, h=3.9),
    # E3 x=20 — box y=0 sy=3.4 (span +-1.7) + tiang luar +-4.3 -> gap 2.2 m
    dict(type="box", x=20.0, y=0.0,  sx=0.4, sy=3.4, h=3.6),
    dict(type="cyl", x=20.0, y=4.3,  r=0.4, h=3.9),
    dict(type="cyl", x=20.0, y=-4.3, r=0.4, h=3.9),
]

# ─────────────────────────────────────────────────────────────────────────────
# MAZE1 — "maze" held-out untuk melihat kemampuan tiap model (dibuat 2026-07-11)
#
# Tujuan: MENGAMPLIFIKASI beda gradual FM(bimodal) vs NN(unimodal) dengan
# merangkai BANYAK titik keputusan dalam satu lintasan berkelok, plus memaksa
# weave nyata (S-curve) supaya lintasan terlihat seperti maze.
#
# BUKAN maze buntu klasik (dinding-dinding + jalan buntu + belok 90°): planner
# ini terbang menuju goal LURUS di depan (goal_y=0) dengan A*/MINCO LOKAL —
# tak punya perencana global maupun kemampuan mundur (backtrack). Maze buntu
# → OOD + butuh global routing → KEDUA model gagal karena alasan yang TIDAK
# ada hubungannya dengan bimodalitas (confound). Jadi "maze" di sini =
# koridor berkelok maju-terus (forward-progress) dengan dua jenis event:
#
#   BIMODAL gate  (G1/G3/G5): kotak SIMETRIS di y=0 + sepasang tiang luar →
#       celah KIRI dan KANAN sama-sama terbuka (~2.1-2.2 m). Dua kelas homotopi
#       → planner HARUS komit satu sisi. NN (mode-averaging) membidik rata-rata
#       = TENGAH (terblokir) → ragu/tolak-replan; FM (bimodal) komit bersih.
#   CHICANE door  (G2/G4): dinding hampir selebar arena dengan PINTU tunggal
#       off-center (G2 pintu ATAS y=+1.5; G4 pintu BAWAH y=-1.5) → 1 kelas,
#       memaksa weave. Transisi G3(atas)→G4(bawah) memaksa S besar =
#       tampak maze + menguji apakah planner memilih sisi G3 yang MENYIAPKAN
#       pintu G4 (look-ahead komitmen — justru keunggulan bimodal).
#
# Semua celah/pintu >= 2.0 m (target fungsional; < 2.0 m drone berputar di
# tempat — lihat catatan frontend). Koridor 0..24 -> goal_x:=24 (arena default,
# TANPA override, sama seperti decisive_gap). HELD-OUT: jangan collect training.
# Audit: src/worlds/audit_maze1.py.
MAZE1 = [
    # G1 BIMODAL x=4.5 — box 2.6 (span +-1.3) + tiang +-3.9 -> celah L/R 2.2 m
    dict(type="box", x=4.5,  y=0.0,   sx=0.4, sy=2.6, h=3.6),
    dict(type="cyl", x=4.5,  y=3.9,   r=0.4, h=3.9),
    dict(type="cyl", x=4.5,  y=-3.9,  r=0.4, h=3.9),
    # G2 CHICANE x=8.5 — pintu ATAS y in [+0.4,+2.6] (lebar 2.2, pusat +1.5)
    dict(type="box", x=8.5,  y=-2.55, sx=0.3, sy=5.9, h=3.6),   # blok y<+0.4
    dict(type="box", x=8.5,  y=4.05,  sx=0.3, sy=2.9, h=3.6),   # blok y>+2.6
    # G3 BIMODAL x=12.5 — box 3.0 (span +-1.5) + tiang +-4.1 -> celah L/R 2.1 m
    dict(type="box", x=12.5, y=0.0,   sx=0.4, sy=3.0, h=3.6),
    dict(type="cyl", x=12.5, y=4.1,   r=0.4, h=3.9),
    dict(type="cyl", x=12.5, y=-4.1,  r=0.4, h=3.9),
    # G4 CHICANE x=16.5 — pintu BAWAH y in [-2.6,-0.4] (lebar 2.2, pusat -1.5)
    dict(type="box", x=16.5, y=2.55,  sx=0.3, sy=5.9, h=3.6),   # blok y>-0.4
    dict(type="box", x=16.5, y=-4.05, sx=0.3, sy=2.9, h=3.6),   # blok y<-2.6
    # G5 BIMODAL x=20.5 — box 2.6 (span +-1.3) + tiang +-3.9 -> celah L/R 2.2 m
    dict(type="box", x=20.5, y=0.0,   sx=0.4, sy=2.6, h=3.6),
    dict(type="cyl", x=20.5, y=3.9,   r=0.4, h=3.9),
    dict(type="cyl", x=20.5, y=-3.9,  r=0.4, h=3.9),
]

WIDE_TEST = [
    # Held-out: lebar & posisi beda; satu sedikit off-center (+0.3 m).
    # Kotak TERLEBAR (2.6 m) sengaja ditaruh PALING DEPAN (x=5.5, pertemuan
    # pertama) — supaya jelas terlihat sejak dini apakah NN/FM mengantisipasi
    # obstacle lebar atau tidak, tanpa harus menunggu dua kotak lain dulu.
    dict(type="box", x=5.5,  y=0.0,  sx=0.9, sy=2.6, h=3.6),
    dict(type="box", x=10.5, y=0.3,  sx=0.7, sy=1.8, h=3.6),
    dict(type="box", x=15.0, y=0.0,  sx=0.7, sy=1.4, h=3.6),
    # Tiang penyela
    dict(type="cyl", x=8.0,  y=-2.9, r=0.4, h=3.9),
    dict(type="cyl", x=7.8,  y=3.0,  r=0.5, h=3.9),
    dict(type="cyl", x=12.8, y=2.7,  r=0.4, h=3.9),
    dict(type="cyl", x=13.0, y=-3.2, r=0.4, h=3.9),
    dict(type="cyl", x=17.8, y=-2.8, r=0.5, h=3.9),
    dict(type="cyl", x=17.6, y=3.1,  r=0.4, h=3.9),
]


# ─────────────────────────────────────────────────────────────────────────────
# SDF
# ─────────────────────────────────────────────────────────────────────────────
SDF_HEADER = """<?xml version="1.0" ?>
<!--
  {name}.sdf — dibuat oleh generate_wide_worlds.py
  Eksperimen ambiguitas multimodal: obstacle lebar di sumbu terbang y=0.
  Koordinat IDENTIK dengan {name}.pcd (peta ground-truth expert).

  Pakai:
    cp {name}.sdf ~/PX4-Autopilot/Tools/simulation/gz/worlds/
    ros2 launch fm_planner sitl.launch.py world:={name}
-->
<sdf version="1.9">
  <world name="{name}">

    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- WAJIB untuk magnetometer PX4 SITL -->
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397742</latitude_deg>
      <longitude_deg>8.545594</longitude_deg>
      <elevation>488.0</elevation>
    </spherical_coordinates>

    <plugin filename="gz-sim-physics-system"            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system"            name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-scene-broadcaster-system"  name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system"      name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-imu-system"                name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system"       name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-altimeter-system"          name="gz::sim::systems::Altimeter"/>
    <plugin filename="gz-sim-magnetometer-system"       name="gz::sim::systems::Magnetometer"/>
    <!-- WAJIB: tanpa NavSat -> "Compass Sensor 0 missing" -->
    <plugin filename="gz-sim-navsat-system"             name="gz::sim::systems::NavSat"/>
    <plugin filename="gz-sim-contact-system"            name="gz::sim::systems::Contact"/>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <cast_shadows>false</cast_shadows>
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""

SDF_BOX = """
    <model name="{name}">
      <pose>{x:.2f} {y:.2f} {zc:.3f} 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx:.2f} {sy:.2f} {h:.2f}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx:.2f} {sy:.2f} {h:.2f}</size></box></geometry>
          <material>
            <ambient>{amb} 1</ambient>
            <diffuse>{dif} 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""

SDF_CYL = """
    <model name="{name}">
      <pose>{x:.2f} {y:.2f} {zc:.3f} 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><cylinder><radius>{r:.2f}</radius><length>{h:.2f}</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>{r:.2f}</radius><length>{h:.2f}</length></cylinder></geometry>
          <material>
            <ambient>0.6 0.3 0.1 1</ambient>
            <diffuse>0.7 0.4 0.15 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""

SDF_FOOTER = """
  </world>
</sdf>
"""


# ─────────────────────────────────────────────────────────────────────────────
# GAUNTLET_TRAIN3 (2026-07-21) — varian gauntlet ke-3 untuk TRAINING
#   Sama-jenis dengan gauntlet_train (deret event ambigu), TAPI:
#   - SEMUA 6 event BIMODAL (kiri/kanan) — beda kompleksitas: gauntlet_train
#     punya 1 event threading (R6 slalom); di sini NOL threading -> sinyal
#     multimodal (2 jawaban valid per scene) semaksimal mungkin, persis yang
#     dibutuhkan riset FM (maze_train ditolak justru karena mayoritas 1-jawaban).
#   - Posisi & lebar BERBEDA dari gauntlet_train/2; menonjolkan obstacle BULAT
#     (silinder gemuk) sebagai penghalang tengah (H2 d=2.0, H6 d=2.4).
#   SEMUA celah >= 2.0 m. Koridor 0..30 -> WAJIB goal_x:=30 arena_x_max:=33.
#   Audit: training_worlds/audit_gauntlet_train3.py
GAUNTLET_TRAIN3 = [
    # H1 x=4 — box tengah 2.2 (span +-1.1) + tiang +-3.5 -> celah L/R 2.0
    dict(type="box", x=4.0,  y=0.0,   sx=0.5, sy=2.2, h=3.6),
    dict(type="cyl", x=4.0,  y=3.5,   r=0.4, h=3.9),
    dict(type="cyl", x=4.0,  y=-3.5,  r=0.4, h=3.9),
    # H2 x=9 — silinder gemuk d=2.0 (BULAT) di tengah + tiang +-3.7 -> L/R 2.3
    dict(type="cyl", x=9.0,  y=0.0,   r=1.0, h=3.6),
    dict(type="cyl", x=9.0,  y=3.7,   r=0.4, h=3.9),
    dict(type="cyl", x=9.0,  y=-3.7,  r=0.4, h=3.9),
    # H3 x=13.5 — dinding y=[-4,4] + PINTU 2.2 off-axis y=-1.4 (bimodal: pintu / memutar)
    dict(type="box", x=13.5, y=1.85,  sx=0.4, sy=4.3, h=3.6),   # blok y=[-0.3,4]
    dict(type="box", x=13.5, y=-3.25, sx=0.4, sy=1.5, h=3.6),   # blok y=[-4,-2.5]
    # H4 x=18 — dinding SOLID 4.6 (span +-2.3) tanpa pintu -> memutar salah satu ujung
    dict(type="box", x=18.0, y=0.0,   sx=0.4, sy=4.6, h=3.6),
    # H5 x=22.5 — box tengah 3.0 (lebih lebar, span +-1.5) + tiang +-4.0 -> L/R 2.1
    dict(type="box", x=22.5, y=0.0,   sx=0.5, sy=3.0, h=3.6),
    dict(type="cyl", x=22.5, y=4.0,   r=0.4, h=3.9),
    dict(type="cyl", x=22.5, y=-4.0,  r=0.4, h=3.9),
    # H6 x=27 — silinder gemuk d=2.4 (BULAT besar) + tiang +-4.2 -> L/R 2.6
    dict(type="cyl", x=27.0, y=0.0,   r=1.2, h=3.6),
    dict(type="cyl", x=27.0, y=4.2,   r=0.4, h=3.9),
    dict(type="cyl", x=27.0, y=-4.2,  r=0.4, h=3.9),
]


# ─────────────────────────────────────────────────────────────────────────────
# MAZE_TRAIN (2026-07-21) — world TRAINING berbentuk MAZE (labirin berliku)
#   Beda dari gauntlet (deret event terpisah): ini KORIDOR MENGANYAM — deretan
#   pintu ber-offset yang berselang-seling (atas/bawah) memaksa drone meliuk
#   terus-menerus seperti menyusuri lorong labirin, disisipi 2 titik pilihan
#   bimodal (kiri/kanan). SEMUA celah >= 2.2 m (di atas target 1.8 m), jarak
#   antar-event 4 m. Koridor 0..30 -> WAJIB goal_x:=30 arena_x_max:=33
#   (amplop sama dengan gauntlet_train). Audit: training_worlds/audit_maze_train.py.
MAZE_TRAIN = [
    # D1 x=4 — pintu 2.2 m di y=+1.6 (anyam ke ATAS): dua box sisakan celah
    dict(type="box", x=4.0,  y=-2.75, sx=0.3, sy=6.5, h=3.6),   # blok y<+0.5
    dict(type="box", x=4.0,  y=4.35,  sx=0.3, sy=3.3, h=3.6),   # blok y>+2.7
    # D2 x=8 — pintu 2.2 m di y=-1.6 (anyam ke BAWAH)
    dict(type="box", x=8.0,  y=-4.35, sx=0.3, sy=3.3, h=3.6),   # blok y<-2.7
    dict(type="box", x=8.0,  y=2.75,  sx=0.3, sy=6.5, h=3.6),   # blok y>-0.5
    # C1 x=12 — BIMODAL: box lebar 2.8 (span +-1.4) + tiang +-4.0 -> celah L/R 2.2
    dict(type="box", x=12.0, y=0.0,   sx=0.4, sy=2.8, h=3.6),
    dict(type="cyl", x=12.0, y=4.0,   r=0.4, h=3.9),
    dict(type="cyl", x=12.0, y=-4.0,  r=0.4, h=3.9),
    # D3 x=16 — pintu 2.2 m di y=+1.6 (anyam ke ATAS)
    dict(type="box", x=16.0, y=-2.75, sx=0.3, sy=6.5, h=3.6),
    dict(type="box", x=16.0, y=4.35,  sx=0.3, sy=3.3, h=3.6),
    # D4 x=20 — pintu 2.2 m di y=-1.6 (anyam ke BAWAH)
    dict(type="box", x=20.0, y=-4.35, sx=0.3, sy=3.3, h=3.6),
    dict(type="box", x=20.0, y=2.75,  sx=0.3, sy=6.5, h=3.6),
    # C2 x=24 — BIMODAL: box lebar 3.2 (span +-1.6) + tiang +-4.2 -> celah L/R 2.2
    dict(type="box", x=24.0, y=0.0,   sx=0.4, sy=3.2, h=3.6),
    dict(type="cyl", x=24.0, y=4.2,   r=0.4, h=3.9),
    dict(type="cyl", x=24.0, y=-4.2,  r=0.4, h=3.9),
    # D5 x=28 — pintu 2.2 m di y=+1.6 (anyam ke ATAS, terakhir sebelum goal)
    dict(type="box", x=28.0, y=-2.75, sx=0.3, sy=6.5, h=3.6),
    dict(type="box", x=28.0, y=4.35,  sx=0.3, sy=3.3, h=3.6),
]


def write_sdf(name, obstacles, path):
    parts = [SDF_HEADER.format(name=name)]
    n_box = n_cyl = 0
    for ob in obstacles:
        if ob["type"] == "box":
            n_box += 1
            wide = ob["sy"] >= 1.2
            # kotak lebar (generator ambiguitas) merah, kotak kecil biru
            amb = "0.7 0.2 0.2" if wide else "0.4 0.5 0.7"
            dif = "0.8 0.3 0.3" if wide else "0.5 0.6 0.8"
            parts.append(SDF_BOX.format(
                name=f"box_{n_box}", x=ob["x"], y=ob["y"], zc=ob["h"] / 2,
                sx=ob["sx"], sy=ob["sy"], h=ob["h"], amb=amb, dif=dif))
        else:
            n_cyl += 1
            parts.append(SDF_CYL.format(
                name=f"pole_{n_cyl}", x=ob["x"], y=ob["y"], zc=ob["h"] / 2,
                r=ob["r"], h=ob["h"]))
    parts.append(SDF_FOOTER)
    with open(path, "w") as f:
        f.write("".join(parts))
    print(f"[sdf] {path}  ({n_box} box, {n_cyl} cylinder)")


# ─────────────────────────────────────────────────────────────────────────────
# PCD — voxel solid 0.1 m, format identik poles.pcd
# ─────────────────────────────────────────────────────────────────────────────
def _grid(lo, hi):
    """Pusat sel 0.1 m di [lo, hi): lo+0.05, lo+0.15, ..."""
    n = int(round((hi - lo) / RES))
    return [lo + RES / 2 + i * RES for i in range(n)]


def write_pcd(obstacles, path):
    # kumpulkan kolom xy obstacle dulu supaya titik lantai di bawah obstacle
    # tidak ditulis dua kali (layer z=0.05 milik obstacle)
    obs_pts, obs_cols = [], set()
    for ob in obstacles:
        zs = _grid(0.0, ob["h"])
        if ob["type"] == "box":
            xs = _grid(ob["x"] - ob["sx"] / 2, ob["x"] + ob["sx"] / 2)
            ys = _grid(ob["y"] - ob["sy"] / 2, ob["y"] + ob["sy"] / 2)
            cols = [(x, y) for x in xs for y in ys]
        else:
            r = ob["r"]
            xs = _grid(ob["x"] - r, ob["x"] + r)
            ys = _grid(ob["y"] - r, ob["y"] + r)
            cols = [(x, y) for x in xs for y in ys
                    if math.hypot(x - ob["x"], y - ob["y"]) <= r]
        for (x, y) in cols:
            obs_cols.add((round(x, 2), round(y, 2)))
            for z in zs:
                obs_pts.append((x, y, z))

    pts = []
    for x in _grid(*GROUND_X):
        for y in _grid(*GROUND_Y):
            if (round(x, 2), round(y, 2)) not in obs_cols:
                pts.append((x, y, RES / 2))
    pts.extend(obs_pts)

    with open(path, "w") as f:
        f.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\n"
                "TYPE F F F\nCOUNT 1 1 1\n"
                f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 0 0 0 1\n"
                f"POINTS {len(pts)}\nDATA ascii\n")
        for x, y, z in pts:
            f.write(f"{x:.2f} {y:.2f} {z:.2f}\n")
    print(f"[pcd] {path}  ({len(pts)} points)")


def main():
    # 2026-07-21: hanya 3 world tersisa (gauntlet_train, gauntlet_train2,
    # maze_train). Definisi world lama (POLES_WIDE/WIDE_TEST/.../MAZE1) sengaja
    # DIBIARKAN di file ini sebagai referensi, tapi TIDAK digenerasi lagi.
    for name, obstacles in [("gauntlet_train", GAUNTLET_TRAIN),
                            ("gauntlet_train2", GAUNTLET_TRAIN2),
                            ("gauntlet_train3", GAUNTLET_TRAIN3),
                            ("maze_train", MAZE_TRAIN)]:
        out_dir = TRAIN_OUT_DIR if name in TRAINING_WORLD_NAMES else OUT_DIR
        write_sdf(name, obstacles, os.path.join(out_dir, f"{name}.sdf"))
        write_pcd(obstacles, os.path.join(out_dir, f"{name}.pcd"))


if __name__ == "__main__":
    main()
