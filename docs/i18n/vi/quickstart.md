# Leonida — Hướng dẫn nhanh (`vi`)

> [!NOTE]
> Tài liệu tiếng Anh ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) và
> [`docs/README.md`](../../README.md)) là **bản chuẩn chính thức**. Trang này chỉ tóm tắt phần hướng dẫn cài đặt
> và build ổn định (đồng bộ theo commit `867d922f0`). Để xem tiến độ thay đổi hằng ngày của dự án, hãy đọc
> [`README.md`](../../../README.md).

## Leonida là gì

Leonida là dự án tái biên dịch tĩnh (static recompilation) đưa *Grand Theft Auto VI* (phiên bản Xbox Series) chạy
trực tiếp trên Windows, được build **ngay trên máy của bạn từ chính bản game bạn sở hữu**. CPU của Xbox Series là
x86-64 tương tự PC nên mã của game được nạp và chạy trực tiếp, phần mềm hệ thống của console được tái hiện bằng các
shim, shader RDNA2 được dịch sang SPIR-V cho Vulkan, và model, texture, âm thanh được đọc từ gói game của bạn.

> [!IMPORTANT]
> Kho mã nguồn này **không chứa game, khóa giải mã hay tệp thực thi nào**, và cũng không hướng dẫn cách lấy chúng.
> Không có bất kỳ cơ chế giải mã hay vượt qua bảo vệ nào ở đây. Mọi thứ được tạo ra trong quá trình build (shader đã
> dịch, tài nguyên trích xuất, thư mục `runtime` và file save) đều lấy từ bản game của bạn và **chỉ được phép giữ
> trên máy cá nhân**: tuyệt đối không đính kèm hoặc chia sẻ lên issue hay pull request (xem
> [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

Dự án đang ở giai đoạn rất sớm: **game chưa thể khởi động được**.

## Các nền tảng

Hiện chưa có gì chơi được trên bất kỳ nền tảng nào. Mức độ chạy được hiện tại:

- **Windows**: Đầy đủ mọi thành phần (`make_game.py`, các công cụ và `runtime` gồm `gta6rt` cùng lớp D3D12 → Vulkan
  và bộ test).
- **Linux**: Trình dịch shader, công cụ tài nguyên và bộ test GPU Vulkan chạy trực tiếp (đã kiểm thử trên RADV và
  NVIDIA). Phần runtime là mã Windows: được biên dịch chéo bằng `clang-cl` và `xwin`, rồi chạy test qua Wine hoặc
  Proton (cách dự kiến để chạy game trên Linux, #217).
- **macOS (Apple Silicon)**: Các công cụ Python và bộ test Vulkan chạy trực tiếp qua MoltenVK (Homebrew) hoặc
  KosmicKrisp; runtime chạy qua Wine như trên Linux. Giới hạn đã biết: thiếu geometry stage và `VK_EXT_mesh_shader`,
  subgroup 32-lane (shader wave64 chạy ở chế độ mô phỏng), và cách xử lý số thực dấu phẩy động của MoltenVK; xem chi
  tiết trong [`docs/testing.md`](../../testing.md).
- **AMD RDNA1 (RX 5000)**: Mục tiêu được hỗ trợ, gần với RDNA2 của console; subgroup 64-lane chạy trực tiếp các
  shader wave64.

Kết quả kiểm thử GPU trên mọi nền tảng được tổng hợp tại [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

## Yêu cầu tiên quyết

- **Bản sao hợp lệ của chính bạn** gói Xbox Series phiên bản `1.0.2.1`: thư mục chứa `MicrosoftGame.Config` và các
  tệp lưu trữ `.rpf` ở dạng tệp thông thường đọc được trên đĩa.
- **Windows 10/11 x64**, còn trống khoảng **400 GB** (400 GiB, tức khoảng 430 GB theo đơn vị thập phân) cho các tệp
  làm việc (có thể trỏ sang ổ đĩa khác bằng `--work`).
- [Python 3.12+](https://www.python.org/downloads/) (nhớ đánh dấu "Add python.exe to PATH" khi cài).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/) (`spirv-as`, `spirv-val`, `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) có hỗ trợ **AMDGPU target** (`llvm-mc`):
  - Bản cài đặt LLVM chính thức cho Windows **không** kèm AMDGPU target.
  - Để tự build `llvm-mc.exe` bản Windows hỗ trợ AMDGPU (mất khoảng 20–40 phút, cần `git` và bộ công cụ C++ của
    Visual Studio), chạy `scripts\build_llvm_mc.ps1` từ thư mục gốc kho mã nguồn; script sẽ tự đặt biến `LLVM_MC`.
  - Cách nhanh hơn nếu bạn dùng [conda](https://docs.conda.io/): chạy `conda install -c conda-forge llvm-tools` (bản
    Windows của gói này có sẵn AMDGPU target), sau đó đặt `LLVM_MC` trỏ tới đường dẫn đầy đủ của `llvm-mc.exe`
    (chạy `where.exe llvm-mc` sau khi kích hoạt môi trường; trong PowerShell, `where` đứng một mình là bí danh của
    `Where-Object`).
  - Hoặc cài `llvm-21` trong WSL Ubuntu và gán toàn bộ câu lệnh cho `LLVM_MC`:
    `LLVM_MC="wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"` (xem
    [`docs/design.md`](../../design.md#disassembling-shaders)).
  - Trên Linux, chỉ cần có `llvm-mc` trong `PATH`.
- **Visual Studio 2022** với workload **Desktop development with C++** và **Windows SDK `10.0.26100` trở lên** (bắt
  buộc cho bước `runtime` vì runtime dùng root signature phiên bản 1.2).
- [ffmpeg](https://ffmpeg.org/download.html) trong `PATH` *(tùy chọn: dùng để giải mã luồng XMA2, và bắt buộc nếu
  dùng `--wav`)*.

Để kiểm tra nhanh cấu hình máy trước khi build, chạy:

```powershell
python -I tools/pc_checker.py --cli
```

## Cách build

### Cách đơn giản (Windows)

1. Cài đặt đầy đủ các yêu cầu tiên quyết ở trên.
2. Clone kho mã nguồn **kèm theo submodule**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
   Nếu dùng "Download ZIP" từ GitHub sẽ thiếu `third_party/rpf9-explorer`; quá trình build sẽ tự tải nếu có `git`,
   hoặc báo hướng dẫn cho bạn.
3. Nhấp đúp vào **`Build.bat`**.

Script sẽ tự tìm bản game của bạn trong Downloads, Desktop, Documents (kể cả thư mục được Windows chuyển hướng như
OneDrive), Games, thư mục người dùng và gốc mỗi ổ đĩa (sâu tối đa 2 cấp thư mục). Nó nhận diện phiên bản qua kích
thước một vài tệp mà không đọc dữ liệu lớn. Nếu tìm thấy nhiều bản, nó sẽ hỏi bạn chọn; nếu không thấy, nó yêu cầu
dán đường dẫn thư mục. Trước khi chạy, script kiểm tra toàn bộ công cụ và dung lượng trống rồi liệt kê một lần những
gì còn thiếu. Kết quả xuất ra `Documents\Leonida`, và nếu bị gián đoạn có thể chạy lại để tiếp tục từ chỗ dở.

### Khi ổ đĩa ít dung lượng

Mức ~400 GB ở trên là cho toàn bộ quá trình build. Để *chạy* bản port, `--stages check,shaders,runtime` chỉ cần
khoảng 30 GB; bước `assets` (~370 GB) chỉ phục vụ các công cụ xuất tài nguyên, còn `--wav` sẽ chiếm thêm dung lượng
gấp nhiều lần Ogg Opus. Có thể đặt `--work` và `--out` trên ổ đĩa khác. Runtime ánh xạ trực tiếp gói game tại chỗ,
nên nếu ổ chứa game ở chế độ chỉ đọc hãy dùng `gta6rt --overlay <thư mục ghi được>` (đây là tùy chọn của `gta6rt`,
không phải của `make_game.py`). Hãy giữ nguyên các tệp `.s` trong `work/shaders/dxbc/disasm` để lần chạy sau chỉ cần
dịch ngược lại những phần còn thiếu rồi dịch toàn bộ shader.

### Từ terminal (`make_game.py`)

```powershell
python -I make_game.py --find                          # liệt kê các bản sao tìm thấy trên máy
python -I make_game.py --pick                          # mở hộp thoại chọn thư mục game
python -I make_game.py --check                         # kiểm tra bản sao, không build
python -I make_game.py --check --full-check            # so sánh thêm SHA-256 mọi tệp (chậm)
python -I make_game.py --game "D:\path\to\your\package" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # chỉ chạy một số bước
python -I make_game.py --game "D:\path\to\your\package" --wav   # giải mã thêm Opus sang PCM WAV (rất lớn)
python -I make_game.py --wave64-emulate off            # auto (mặc định), on hoặc off
```

Thứ tự các bước: `check` chạy trước; sau đó `shaders`, `assets` và `runtime` chạy song song nếu được chọn; `pipelines`
bắt đầu ngay khi shader dịch xong (và `vk_pipeline_check` build xong) để tạo thử từng shader trên driver GPU của bạn
và lưu cache vào `pipeline_cache.bin` trong `--out` (runtime hiện chưa nạp cache này). Quá trình dịch shader dùng
`--jobs` tiến trình (mặc định một nửa số luồng CPU). Trên terminal tương tác, giao diện TUI cho phép xem log từng
bước bằng phím `1`–`4` / `Tab`; dùng `--no-tui` để in log dạng văn bản thuần.

- **`--wave64-emulate`**: Quyết định có dịch các shader compute và mesh wave64 để chạy chính xác trên mọi kích thước
  subgroup hay không (NVIDIA và Intel có subgroup 32-lane nên cần bật). Mặc định `auto` hỏi `vulkaninfo`; trên driver
  Windows của AMD, `auto` giữ wave64 thuần trên RDNA1 và RDNA2 khi GPU của runtime hỗ trợ vì driver này có thể crash
  với mã LDS mô phỏng (#857). Chế độ này được lưu cạnh các shader đã dịch và đổi cờ sẽ kích hoạt dịch lại.
- **Các cờ dự phòng (fallback)** lưu và tự động dịch lại tương tự: `--no-float-controls`, `--bary-fallback` (thiết bị
  không có `VK_KHR_fragment_shader_barycentric`), `--texel-offset-fallback` (thiết bị không có `VK_KHR_maintenance8`),
  và `--mesh-compute` (chạy mesh shader dưới dạng compute khi thiếu `VK_EXT_mesh_shader`, hoặc ép bật lại bằng
  `--native-mesh`). Chi tiết xem [`docs/design.md`](../../design.md).
- **Tự động build lại khi công cụ thay đổi**: Quá trình build ghi nhận lệnh/phiên bản `llvm-mc`, mã nguồn trình dịch
  và phiên bản bộ giải mã tài nguyên (`third_party/rpf9-explorer` và `tools/rpf9dump`), chỉ chạy lại bước bị ảnh
  hưởng khi có thay đổi.
- **Linux và Wine/Proton**: `make_game.py` chạy được trên Linux. Bước `pipelines` dùng `vk_pipeline_check` (build vào
  `runtime/build/linux` bằng CMake 3.24+, C++20 và Vulkan headers). Phần loader được biên dịch chéo sang Windows trong
  `runtime/build/windows` bằng `clang-cl` và `lld-link` (LLVM 19+, tránh 22.1.7 và 22.1.8) cùng bộ MSVC CRT / Windows
  SDK từ [xwin](https://github.com/Jake-Shadle/xwin), hoặc truyền `--runtime-build <thư mục>` (ví dụ artifact
  `runtime-windows` từ GitHub Actions). Script sinh ra `gta6rt.sh` dùng `wine` hoặc biến `WINE`.

## Xử lý sự cố thường gặp

- **Tải mã nguồn bằng "Download ZIP" trên GitHub**: Gói ZIP thiếu submodule `third_party/rpf9-explorer`. Hãy clone
  bằng `git clone --recursive`, hoặc chạy `git submodule update --init --recursive` trong thư mục repo.
- **`llvm-mc` từ chối `gfx1013` hoặc AMDGPU target**: Có thể bạn đang dùng bản LLVM Windows chính thức. Hãy chạy
  `scripts\build_llvm_mc.ps1`, dùng `conda-forge` (`llvm-tools`), hoặc trỏ `LLVM_MC` vào `llvm-mc` trong WSL (`llvm-21`).
- **CMake báo lỗi thiếu Windows SDK `10.0.26100` trở lên** (hoặc thiếu `Desc_1_2` trong
  `D3D12_VERSIONED_ROOT_SIGNATURE_DESC`): Cài đặt Windows SDK `10.0.26100`+ qua Visual Studio Installer hoặc
  `winget install Microsoft.WindowsSDK.10.0.26100`, xóa `runtime/build` rồi cấu hình lại.
- **Thiếu dung lượng trên ổ `C:`**: Truyền `--work "D:\leonida-work"` để chuyển tệp tạm sang ổ đĩa khác.
- **Windows chặn chạy các tệp test vừa biên dịch** ("An Application Control policy has blocked this file"): Do Smart
  App Control hoặc trình diệt virus trên Windows 11. Xem cách xử lý trong [`docs/testing.md`](../../testing.md#known-problems).

## Đóng góp và chạy kiểm thử GPU

1. Kiểm tra các pull request đang mở và [issue điều phối][coord] trước khi bắt đầu để tránh làm trùng. Nhận việc
   bằng bình luận `claiming: <ID>` (hoặc mô tả một dòng nếu chưa có mã).
2. Chạy các bài kiểm thử liên quan với dữ liệu tổng hợp (ví dụ `python -I tests/test_manifest_verifier.py` từ thư mục
   gốc kho mã nguồn).
3. Trong pull request, ghi rõ mã công việc, những gì bạn đã kiểm thử, tên GPU và phiên bản driver (nếu có chạy GPU
   test), và nêu rõ nếu có sự hỗ trợ của AI. Xem đầy đủ quy tắc tại [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

[Hướng dẫn kiểm thử GPU](../../testing.md) trình bày cách chạy bộ test GPU không cần tệp game. Gửi kết quả của bạn
vào [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166) (mỗi GPU và driver một bình luận).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
