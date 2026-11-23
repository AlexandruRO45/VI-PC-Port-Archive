# Leonida — Panduan Mulai Cepat (`id`)

> [!NOTE]
> Dokumentasi dalam bahasa Inggris ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md), dan
> [`docs/README.md`](../../README.md)) **bersifat otoritatif** (menjadi rujukan utama). Panduan ini hanya mencakup instruksi instalasi
> dan kompilasi yang stabil (disinkronkan dengan commit `dde82f27e`). Untuk status perkembangan proyek terbaru yang
> diperbarui setiap hari, silakan merujuk ke [`README.md`](../../../README.md).

## Apa itu Leonida

Leonida adalah proyek rekompilasi statis untuk Grand Theft Auto VI (versi Xbox Series) agar dapat berjalan secara
asli (*native*) di Windows. Proyek ini dibangun **di PC Anda sendiri, dari salinan sah game milik Anda sendiri**.
Shader RDNA2 bawaan konsol diterjemahkan ke SPIR-V (Vulkan), aset (*model, texture, audio*) dibaca langsung dari
paket instalasi Anda, dan perangkat lunak sistem konsol diimplementasikan ulang dalam bentuk *shims*.

> [!IMPORTANT]
> Repositori ini **tidak memuat game, kunci enkripsi, maupun berkas executable**, dan tidak menyediakan instruksi
> tentang cara mendapatkannya. Anda memerlukan salinan sah game Anda sendiri, dengan berkas yang sudah dapat dibaca
> di media penyimpanan. Proyek ini tidak mendekripsi atau membypass perlindungan apa pun. Semua yang dihasilkan oleh
> proses build (shader hasil terjemahan, aset yang diekstrak, folder runtime, serta save file) berasal dari game Anda
> dan **harus tetap berada di PC Anda sendiri**: jangan pernah mengunggah atau membagikan berkas tersebut dalam *issue*
> maupun *pull request* (lihat [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

Proyek ini masih dalam tahap awal: **game belum dapat di-boot**.

## Platform yang Didukung

Saat ini game belum dapat dimainkan di platform mana pun. Status keterjalanan komponen saat ini:

- **Windows**: Seluruh komponen berjalan (`make_game.py`, perangkat *tools*, dan *runtime* `gta6rt` beserta lapisan D3D12 → Vulkan dan pengujiannya).
- **Linux**: Penerjemah shader, pemroses aset, dan alat uji GPU Vulkan berjalan secara *native* (sudah diuji pada RADV dan NVIDIA). Runtime adalah kode Windows: dikompilasi silang (*cross-compile*) menggunakan clang-cl dan xwin, serta pengujiannya dijalankan di bawah Wine atau Proton. Game dirancang untuk dijalankan dengan cara serupa (#217).
- **macOS (Apple Silicon)**: Alat Python dan pengujian Vulkan berjalan secara *native* di atas MoltenVK (Homebrew) atau KosmicKrisp. Runtime berjalan di bawah Wine, mirip seperti di Linux. Keterbatasan saat ini: belum mendukung *geometry stage* dan `VK_EXT_mesh_shader`, subgroup berukuran 32 (shader wave64 dijalankan dengan emulasi), dan penanganan float MoltenVK; lihat rinciannya di [`docs/testing.md`](../../testing.md).
- **AMD RDNA1 (RX 5000)**: Target yang didukung karena arsitekturnya mirip dengan RDNA2 konsol; subgroup 64 miliknya menjalankan shader wave64 secara *native*.

Hasil pengujian GPU dari seluruh platform dikumpulkan pada [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

## Prasyarat

- **Salinan sah milik Anda sendiri** dari paket Xbox Series, versi `1.0.2.1`: direktori yang memuat `MicrosoftGame.Config` dan arsip `.rpf` sebagai berkas biasa yang dapat dibaca di disk.
- **Windows 10/11 x64**, dengan ruang kosong sekitar **400 GB** untuk berkas kerja (ekspor aset membutuhkan sekitar 337 GB sebelum tahap audio). Lokasi penyimpanan ini dapat dialihkan ke drive lain menggunakan parameter `--work`.
- [Python 3.12+](https://www.python.org/downloads/) (pastikan mencentang "Add python.exe to PATH" saat instalasi).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/) (untuk `spirv-as`, `spirv-val`, dan `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) dengan **target AMDGPU** untuk `llvm-mc`:
  - Paket resmi LLVM untuk Windows **tidak menyertakan** target AMDGPU.
  - Untuk mengompilasi `llvm-mc.exe` secara *native* dengan target AMDGPU di Windows, jalankan `.\scripts\build_llvm_mc.ps1` dari PowerShell di direktori *root* repositori. Skrip ini membutuhkan `git` dan Visual Studio C++ tools. Skrip akan mengatur variabel `LLVM_MC` untuk pengguna Anda (buka terminal baru setelahnya agar variabel terbaca). Proses kompilasi memakan waktu sekitar 20–40 menit.
  - Alternatif lebih cepat jika Anda menggunakan [conda](https://docs.conda.io/): jalankan `conda install -c conda-forge llvm-tools` (versi Windows ini sudah mencakup target AMDGPU), lalu atur variabel lingkungan `LLVM_MC` ke jalur lengkap `llvm-mc.exe` yang terpasang (`where.exe llvm-mc` akan menampilkannya saat lingkungan aktif).
  - Melalui WSL: instal `llvm-21` di Ubuntu WSL dan atur `LLVM_MC` langsung ke perintah WSL dalam sesi PowerShell:
    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```
    Lihat rinciannya di [`docs/design.md`](../../design.md#disassembling-shaders).
  - Di Linux, `llvm-mc` yang sudah terpasang di `PATH` sudah mencukupi.
- **Visual Studio 2022** dengan beban kerja *Desktop development with C++* dan **Windows SDK `10.0.26100` atau yang lebih baru**, untuk tahap `runtime` (runtime menggunakan *root signatures* versi 1.2 yang tidak tersedia di versi SDK lama).
- [ffmpeg](https://ffmpeg.org/download.html) pada `PATH` (*opsional*: mendekode *stream* XMA2, dan wajib jika menggunakan `--wav`).

## Cara Melakukan Build

### Cara Mudah (Windows)

1. Pasang seluruh prasyarat di atas.
2. Clone repositori ini **beserta submodulnya**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
   *Catatan: Tombol "Download ZIP" dari GitHub tidak menyertakan `third_party/rpf9-explorer`. Proses build akan mengambilnya jika `git` terpasang, atau memberikan petunjuk jika belum.*
3. Klik dua kali pada **`Build.bat`**.

Skrip ini akan mencari salinan game Anda secara otomatis: mencari di Downloads, Desktop, Documents (termasuk OneDrive), Games, folder profil pengguna, serta *root* setiap drive hingga kedalaman dua folder. Skrip mendeteksi versi game berdasarkan ukuran beberapa berkas kecil tanpa membaca berkas besar. Jika menemukan lebih dari satu salinan, skrip akan meminta Anda memilih; jika tidak ditemukan, skrip akan meminta Anda menempelkan jalur folder game. Sebelum memulai, skrip akan memverifikasi prasyarat dan kapasitas ruang kosong. Hasil build akan disimpan di `Documents\Leonida`. Proses build yang terhenti dapat dilanjutkan kembali tanpa mengulang dari awal.

Folder game yang benar adalah folder yang memuat `MicrosoftGame.Config` dan arsip `.rpf` sebagai berkas biasa.

### Ruang Disk Terbatas

Kebutuhan ruang ~400 GB berlaku untuk proses build lengkap. Untuk sekadar *menjalankan* port hasil build, tahap `--stages check,shaders,runtime` hanya membutuhkan sekitar 30 GB; tahap `assets` (~370 GB) hanya dibutuhkan untuk alat riset/ekspor aset, dan opsi `--wav` akan menambahkan ukuran berkas beberapa kali lipat dari ukuran Ogg Opus. Parameter `--work` dan `--out` dapat dialihkan ke drive penyimpanan lain.

Runtime memetakan paket game langsung di tempat: jika drive paket game bersifat *read-only*, gunakan parameter `gta6rt --overlay <folder_dapat_ditulis>` (opsi ini milik `gta6rt`, bukan `make_game.py`). Jangan hapus berkas `.s` di `work/shaders/dxbc/disasm` agar tahap shader tidak mengulang pembongkaran dari nol.

### Melalui Terminal (`make_game.py`)

Anda juga dapat menjalankan proses build melalui antarmuka baris perintah (CLI) dengan kendali penuh:

```powershell
python -I make_game.py --find                          # menampilkan daftar salinan yang ditemukan di PC ini
python -I make_game.py --pick                          # membuka jendela dialog untuk memilih folder game
python -I make_game.py --check                         # memeriksa integritas salinan tanpa melakukan build
python -I make_game.py --check --full-check            # memeriksa hash SHA-256 seluruh berkas (membutuhkan waktu lebih lama)
python -I make_game.py --game "D:\jalur\ke\paket" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # menjalankan tahapan tertentu saja
python -I make_game.py --game "D:\jalur\ke\paket" --wav # mendekode audio Opus ke PCM WAV (berukuran sangat besar)
python -I make_game.py --wave64-emulate off            # auto (default), on, atau off
```

Tahapan build dieksekusi dengan urutan: `check` → `shaders` → `assets` → `runtime` → `pipelines`. Tahap `pipelines` memvalidasi setiap shader hasil terjemahan menggunakan driver GPU Anda, mencatat shader yang dilewati, serta menyimpan *pipeline cache* ke `pipeline_cache.bin` di direktori `--out`.

- **`--wave64-emulate`**: Menentukan apakah shader *compute* dan *mesh* wave64 diterjemahkan secara presisi untuk ukuran subgroup GPU apa pun. GPU NVIDIA dan Intel menggunakan subgroup 32 sehingga memerlukan emulasi ini. Nilai bawaan, `auto`, mendeteksi konfigurasi via `vulkaninfo` dan mengaktifkan emulasi jika GPU tidak mendukung subgroup 64. Pada driver Windows AMD, `auto` mempertahankan wave64 *native* untuk RDNA1 dan RDNA2 guna mencegah kendala crash pada kode emulasi (#857).
- **Opsi Fallback**: Tersedia opsi untuk perangkat keras yang tidak memiliki fitur Vulkan tertentu: `--no-float-controls`, `--bary-fallback` (tanpa `VK_KHR_fragment_shader_barycentric`), `--texel-offset-fallback` (tanpa `VK_KHR_maintenance8`), dan `--mesh-compute` (tanpa `VK_EXT_mesh_shader`). Dokumentasi lengkap tersedia di [`docs/design.md`](../../design.md).
- **Deteksi Pembaruan Otomatis**: Build mencatat versi `llvm-mc`, sumber kode penerjemah, dan sumber *decoder* aset (`third_party/rpf9-explorer` dan `tools/rpf9dump`). Jika salah satunya diperbarui, hanya tahap terkait yang akan diproses ulang.
- **`--work`**: Tentukan drive dengan ruang kosong minimal 350–400 GB. Penerjemahan shader merupakan proses yang paling membebani CPU; secara default proses ini menggunakan separuh jumlah *threads* prosesor (dapat disesuaikan dengan `--jobs`).
- **`--wav`**: Dinonaktifkan secara bawaan karena PCM WAV 16-bit berukuran jauh lebih besar daripada Ogg Opus.
- **Linux dan Wine/Proton**: `make_game.py` juga dapat dijalankan di Linux. Loader merupakan aplikasi Windows dan dijalankan menggunakan Wine atau Proton; tahap `runtime` mengompilasi silang (*cross-compile*) menggunakan clang-cl, lld-link, dan SDK yang disiapkan oleh [xwin](https://github.com/Jake-Shadle/xwin). Anda juga dapat menyediakan hasil kompilasi Windows dengan `--runtime-build <folder>`.

## Masalah Umum (Troubleshooting)

- **Mengunduh via "Download ZIP" di GitHub**: ZIP tidak menyertakan submodul `third_party/rpf9-explorer`. Gunakan `git clone --recursive` atau jalankan `git submodule update --init --recursive` di folder repositori.
- **`llvm-mc` menolak `gfx1013` atau target AMDGPU**: Anda mungkin menggunakan penginstal resmi LLVM untuk Windows yang belum menyertakan target AMDGPU. Jalankan `.\scripts\build_llvm_mc.ps1` atau arahkan `LLVM_MC` ke instalasi LLVM di WSL.
- **CMake berhenti dengan pesan "needs the Windows SDK 10.0.26100 or newer"** atau `Desc_1_2` tidak ditemukan: Perbarui Windows SDK ke versi `10.0.26100` atau lebih baru melalui Visual Studio Installer atau jalankan `winget install Microsoft.WindowsSDK.10.0.26100`, lalu hapus folder `runtime/build` dan jalankan konfigurasi ulang.
- **Kehabisan ruang di drive `C:`**: Gunakan argumen `--work "D:\leonida-work"` untuk menempatkan direktori kerja sementara di drive lain.
- **Windows memblokir berkas uji yang baru dikompilasi** (*"An Application Control policy has blocked this file"*): Hal ini disebabkan oleh fitur Smart App Control di Windows 11. Lihat panduannya di [`docs/testing.md`](../../testing.md#known-problems).

## Berkontribusi dan Pengujian GPU

1. Periksa pull request yang sedang terbuka dan [issue koordinasi][coord] agar tidak terjadi duplikasi pekerjaan. Lakukan reservasi pekerjaan dengan mengirim pesan `claiming: <ID>` atau deskripsi singkat jika belum terdaftar.
2. Jalankan pengujian terkait menggunakan data sintetis (tanpa berkas game), misalnya: `python -I tests/test_manifest_verifier.py`.
3. Dalam pull request, cantumkan ID pekerjaan yang diklaim, apa saja yang telah diuji, spesifikasi GPU serta versi driver (jika menjalankan tes GPU), dan sebutkan jika Anda dibantu oleh AI sesuai aturan di [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

Panduan lengkap pengujian GPU dapat dibaca di [docs/testing.md](../../testing.md). Bagikan hasil pengujian GPU Anda di [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166) (satu komentar untuk setiap kombinasi GPU dan driver).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
