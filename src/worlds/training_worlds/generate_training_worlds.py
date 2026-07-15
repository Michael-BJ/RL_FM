#!/usr/bin/env python3
# generate_training_worlds.py — v1.0
"""
generate_training_worlds.py
=============================================================================
5 world "atom" sederhana untuk mengumpulkan data multimodal — SATU konsep
obstacle per world (bukan course kompleks seperti gauntlet_train), tapi
dimensi/bentuk berbeda antar world. Tujuan (lihat diskusi 2026-07-08):

  1. gauntlet_train (satu-satunya sumber dataset FM saat ini) cuma 9.2% scene
     ambigu, dan variasi visualnya sempit (cuma beberapa titik spesifik di 1
     world) -> dicurigai jadi penyebab utama model collapse ke 1 mode saat
     disampel berulang pada scene yang sama (lihat offline_multimodal_test.py
     di scratchpad sesi ini: 0/17 scene ambigu held-out menunjukkan sebaran
     bimodal seimbang).
  2. Kalau HANYA pakai 1 world sederhana (mis. poles_wide lama), model bisa
     overfit ke bentuk/posisi obstacle itu-itu saja -> makanya di sini dibuat
     BEBERAPA world dengan bentuk & dimensi berbeda (box sempit, box lebar,
     silinder gemuk, sepasang tiang rapat, plat tipis lebar), tiap world
     tetap simple (2 instance obstacle sejenis per world, skala beda) supaya
     gampang di-debug per-world kalau ada yang janggal.

World yang dihasilkan (semua koridor 0..20, goal_x:=20.0, arena bounds
DEFAULT expert_planning.launch.py — tidak perlu override apa pun):
  box_narrow      : box sempit TEPAT y=0, lebar 1.3/1.6 m (ambang bawah —
                     margin kiri-kanan tipis, uji batas feasibility)
  box_wide        : box lebar TEPAT y=0, 2.6/3.2 m (detour besar)
  cylinder_fat     : SATU silinder gemuk (bentuk BULAT, bukan kotak),
                     diameter 1.4/2.2 m
  twin_pole_gate  : SEPASANG tiang rapat (topologi beda — 2 objek terpisah
                     membentuk satu penghalang, celah tengah < 1.3 m jadi
                     TIDAK bisa ditembus, drone wajib memutari SELURUH
                     pasangan, bukan lewat tengah)
  plate_wall      : "plat" tipis tapi SANGAT lebar (sx kecil, sy besar) —
                     rasio aspek beda dari box biasa, mendekati dinding tipis

Pakai (identik generate_wide_worlds.py):
    python3 generate_training_worlds.py
    cp *.sdf ~/PX4-Autopilot/Tools/simulation/gz/worlds/

Lalu collecting (contoh box_narrow; ganti nama world untuk yang lain):
    ros2 launch fm_planner sitl.launch.py world:=box_narrow
    ros2 launch fm_planner mavros_only.launch.py
    ros2 launch fm_planner camera_vfh.launch.py
    ros2 launch fm_planner expert_planning.launch.py \
        pcd_path:=<dir ini>/box_narrow.pcd goal_x:=20.0 target_samples:=1500

REVISI 2026-07-08a — tiang cermin pembatas (DICOBA, lalu DIBATALKAN)
=============================================================================
Kalibrasi pertama (box_narrow, 100 sampel, obstacle polos di ruang
terbuka): 0% scene ambigu (0 baris duplikat id), padahal sisi kiri/kanan
bervariasi ANTAR scene (58 positif/42 negatif) — jadi drone TIDAK bias satu
sisi, tapi setiap replan tunggal cuma menemukan SATU kelas homotopi.

Percobaan pertama: apit tiap obstacle dengan tiang cermin (pola
gauntlet_train, celah fungsional 2.0 m) supaya lebar detour dibatasi fisik.
Ini SEBENARNYA bekerja, tapi cuma menyamarkan bug yang sesungguhnya ada di
`homotopy_frontend.py`: `_paint_corridor()` cuma menandai jalur yang SUDAH
ditemukan dengan penalti LUNAK (radius CORRIDOR_RADIUS=0.8 m) — bukan
larangan keras. Kalau ruang di sekitar obstacle terbuka, A* ronde
berikutnya SELALU bisa "geser sedikit lebih lebar di sisi yang SAMA" dengan
biaya lebih murah daripada benar-benar pindah ke sisi seberang -> signature
sisi yang sama terus, tak pernah dianggap kelas baru.

FIX SEBENARNYA (2026-07-08b, di homotopy_frontend.py, bukan di sini):
`find_paths()` sekarang mem-BLOKIR KERAS (bukan menalti) wilayah pada sisi
yang baru ditemukan, dalam jendela-x lokal di sekitar obstacle
(`_wall_off_side()`) — ronde berikutnya secara fisik tidak bisa lewat situ
lagi, jadi TERPAKSA lompat ke sisi seberang. Diverifikasi lewat tes sintetik
(obstacle terisolir di ruang terbuka, PERSIS skenario kalibrasi yang
gagal) -> kedua kelas ditemukan tanpa tiang cermin apa pun.

Karena akar masalah sudah diperbaiki di algoritmanya, tiang cermin di sini
DIHAPUS lagi — worlds kembali ke desain polos/simple semula (1 obstacle
per event, tanpa pembatas tambahan), sesuai permintaan awal "tetap simple".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_wide_worlds import write_sdf, write_pcd  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Konvensi sama seperti poles_wide: koridor terbang 0..20 (goal_x:=20),
# spawn (0,0) & turnaround (20,0) bersih radius >= 2.5 m, obstacle tinggi
# >= 3.4 m (tidak bisa dilewati dari atas), 2 instance per world di x=6 dan
# x=14 (spacing 8 m -> drone kembali head-on sebelum event berikutnya).

BOX_NARROW = [
    dict(type="box", x=6.0,  y=0.0, sx=0.6, sy=1.3, h=3.6),
    dict(type="box", x=14.0, y=0.0, sx=0.6, sy=1.6, h=3.6),
]

BOX_WIDE = [
    dict(type="box", x=6.0,  y=0.0, sx=0.8, sy=2.6, h=3.6),
    dict(type="box", x=14.0, y=0.0, sx=0.8, sy=3.2, h=3.6),
]

CYLINDER_FAT = [
    dict(type="cyl", x=6.0,  y=0.0, r=0.7, h=3.9),   # diameter 1.4 m
    dict(type="cyl", x=14.0, y=0.0, r=1.1, h=3.9),   # diameter 2.2 m
]

# Celah tengah = 2*y - 2*r. Event 1: 1.1-0.8=0.3 m; Event 2: 1.5-0.8=0.7 m.
# Keduanya < PAIR_MIN (1.3 m, ambang absolut planner) -> pasti tak tertembus,
# drone wajib memutari SELURUH pasangan (topologi beda dari 1 box/cyl solid).
TWIN_POLE_GATE = [
    dict(type="cyl", x=6.0,  y=0.55, r=0.4, h=3.9),
    dict(type="cyl", x=6.0,  y=-0.55, r=0.4, h=3.9),
    dict(type="cyl", x=14.0, y=0.75, r=0.4, h=3.9),
    dict(type="cyl", x=14.0, y=-0.75, r=0.4, h=3.9),
]

PLATE_WALL = [
    dict(type="box", x=6.0,  y=0.0, sx=0.25, sy=2.2, h=3.6),
    dict(type="box", x=14.0, y=0.0, sx=0.2,  sy=3.4, h=3.6),
]

WORLDS = {
    "box_narrow": BOX_NARROW,
    "box_wide": BOX_WIDE,
    "cylinder_fat": CYLINDER_FAT,
    "twin_pole_gate": TWIN_POLE_GATE,
    "plate_wall": PLATE_WALL,
}


def main():
    for name, obstacles in WORLDS.items():
        write_sdf(name, obstacles, os.path.join(OUT_DIR, f"{name}.sdf"))
        write_pcd(obstacles, os.path.join(OUT_DIR, f"{name}.pcd"))


if __name__ == "__main__":
    main()
