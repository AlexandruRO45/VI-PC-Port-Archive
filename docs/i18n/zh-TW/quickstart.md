# Leonida — 快速上手（`zh-TW`）

> [!NOTE]
> 英文檔案（[`README.md`](../../../README.md)、[`CONTRIBUTING.md`](../../../CONTRIBUTING.md) 與
> [`docs/README.md`](../../README.md)）是**權威版本**。本文只彙整穩定的安裝與建置說明（與 `867d922` 同步）。
> 專案的每日進展請查看 [`README.md`](../../../README.md)。

## Leonida 是什麼

Leonida 是一個**靜態重編譯**（static recompilation）專案。Xbox Series 的 CPU 與 PC 一樣是 x86-64，因此遊戲的程式碼
會被直接載入並**原生執行**，而主機的系統軟體則重新實作。主機上已預先編譯好的 RDNA2 GPU 程式會翻譯成 SPIR-V 供
Vulkan 使用；遊戲的模型、紋理與音訊則從你的套件中讀取。這一切發生在**你自己的電腦上，用你自己的遊戲拷貝**。

> [!IMPORTANT]
> 本倉庫**不包含**遊戲、金鑰或可執行檔，也不說明如何取得。你需要自己的拷貝，而且它的檔案在磁碟上已經可讀。
> 這裡沒有任何解密或繞過保護的內容。建置所產生的一切（生成的著色器、擷取的資源、runtime 資料夾與你的儲存檔）
> 都來自你的遊戲，並**留在你的 PC 上**。切勿分享這些檔案——也不要在 issue 或 pull request 中附上它們
> （見 [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)）。

專案仍處於早期：**遊戲目前還無法啟動**。所有不需要遊戲可執行檔就能完成的工作都已就緒，因此一旦開發者手上有一份可執行檔，
就能開始啟動方面的開發。

## 平台

任何平台目前都還無法遊玩。各平台現在能執行的部分：

| 平台 | 現況 |
|---|---|
| **Windows** | 全部：`make_game.py`、各項工具，以及 runtime（`gta6rt` 與 D3D12 → Vulkan 層）和它的測試。 |
| **Linux** | 翻譯器、資源工具與 Vulkan GPU 測試工具可原生建置並執行（已在 RADV 與 NVIDIA 上測試過）。runtime 是 Windows 程式碼：以 clang-cl 與 xwin 交叉編譯，其測試在 Wine 或 Proton 下執行。遊戲預計也以同樣方式執行——在 Wine 或 Proton 之下，讓 D3D12 層走原生 Vulkan（#217）。 |
| **macOS（Apple Silicon）** | Python 工具與 Vulkan 測試工具可原生執行，使用 MoltenVK（Homebrew）或 KosmicKrisp。runtime 是 Windows 程式碼，與 Linux 一樣在 Wine 下執行：CrossOver、arm64 搭配 FEX，或在 Rosetta 下跑 x86-64（#1706、#1707、#1924；#3119 修掉了那邊最後幾個 loader 測試）。已知落差：沒有 geometry 階段也沒有 `VK_EXT_mesh_shader`（這些測試會跳過）、子群組寬度為 32（wave64 著色器以模擬方式執行），以及 MoltenVK 的浮點處理（`PerVertexKHR`、NaN 比較、-0、`NoContraction` 的代價）；[`docs/testing.md`](../../testing.md) 逐項列出（#2947）。 |
| **AMD RDNA1（RX 5000）** | 受支援的目標：與主機的 RDNA2 相近，且其 64 寬子群組可原生執行 wave64 著色器。RX 5700 XT 的結果通過 GPU 測試套件（#146）；在 RADV 上，不含 fragment barycentrics 的 pixel shader 有一處已知差異（[`docs/testing.md`](../../testing.md) 的 *Known problems*，#330）。 |

所有平台的 GPU 結果都彙整在 [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166)；如何執行測試見
[`docs/testing.md`](../../testing.md)。

## 前置需求

- **你自己的拷貝**：Xbox Series 套件，版本 `1.0.2.1`。遊戲目錄就是那個持有 `MicrosoftGame.Config` 與 `.rpf`
  封存檔、且都是普通檔案（plain files）的目錄。
- **Windows 10/11 x64**，並預留約 **400 GB** 空間給工作檔（單是資源匯出在音訊步驟之前實測約 337 GB，耗時約 2 小時 35 分）。
  這裡的磁碟數字以 Windows 檔案總管的顯示為準，也就是 GiB：400 GiB 約等於十進位的 430 GB。可用 `--work` 把它們放到另一顆硬碟。
- [Python 3.12+](https://www.python.org/downloads/)（勾選 “Add python.exe to PATH”）。
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)。
- [Vulkan SDK](https://vulkan.lunarg.com/)，提供 `spirv-as` / `spirv-val`。
- [LLVM 21+](https://github.com/llvm/llvm-project/releases)，且需要附 **AMDGPU target** 的 `llvm-mc`。
  官方 Windows 版**不含** AMDGPU target。
  - 在 Windows 上，`scripts\build_llvm_mc.ps1` 會建置含該 target 的原生 `llvm-mc.exe`（不需要 WSL；需要下文的
    git 與 Visual Studio C++ 工具）並設定 `LLVM_MC`；大約耗時 20-40 分鐘。
  - 較快的做法（如果你使用 [conda](https://docs.conda.io/)）：`conda install -c conda-forge llvm-tools`
    （其 Windows 版本包含 AMDGPU target；[docs/testing.md](../../testing.md) 使用的是 23.1.6 版），然後把 `LLVM_MC`
    設為它所安裝之 `llvm-mc.exe` 的完整路徑（環境啟用後以 `where.exe llvm-mc` 查看；在 PowerShell 中單獨的 `where`
    是 `Where-Object`）。
  - 或在 WSL 中安裝 `llvm-21`，並把整段命令設給 `LLVM_MC`：
    `LLVM_MC="wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"`（見 [`docs/design.md`](../../design.md#disassembling-shaders)）。
  - Linux 上只要 `PATH` 裡有 `llvm-mc` 即可。
- **Visual Studio 2022**，含「使用 C++ 的桌面開發」工作負載與 **Windows SDK `10.0.26100` 或更新**，供 runtime
  階段使用（runtime 使用 1.2 版 root signature，較舊的 SDK 沒有宣告它）。
- [ffmpeg](https://ffmpeg.org/download.html) 置於 `PATH` 中（選填：解碼 XMA2 串流；使用 `--wav` 時為必要）。

**這台 PC 夠力嗎？** `python -I tools/pc_checker.py --cli` 會偵測 CPU、GPU 與 RAM，並在長時間建置前與參考設定檔比較：
`--profile 1`（預設）是「等同 Xbox Series X 的 PC」，`--profile 2` 是社群對 1080p 低畫質的估計。
`python -I tools/pc_checker.py --check --json` 以 JSON 輸出同樣的判定，並把它當成結束碼回傳（0 足夠、1 偏弱、2 無法判斷）。

## 如何建置

這個移植版不是下載好就能用的成品，而是在你的 PC 上、用你自己的拷貝建置出來的。

### 簡易方式

1. 安裝上述前置需求。
2. **連同子模組**下載本專案：`git clone --recursive https://github.com/ItsNotRexed/Leonida`。
   用 GitHub 的 “Download ZIP” 會漏掉 `third_party/rpf9-explorer`；若已安裝 `git`，建置時會自動取得，否則會告訴你該怎麼辦。
3. 雙擊 **`Build.bat`**。

它會自己找到你的遊戲拷貝：搜尋 Downloads、Desktop、Documents（也包括 Windows 實際存放它們的位置，例如 OneDrive）、
Games、使用者目錄，以及每個磁碟頂層往下兩層。它僅依據幾個檔案的大小辨識版本，不會讀取任何大型檔案。若找到多份拷貝，
會詢問你要用哪一份；若一份都找不到，會讓你貼上目錄。開始之前，它會檢查下列各項是否都已安裝、以及是否有足夠的可用空間，
並一次列出所有缺少的項目。結果寫入 `Documents\Leonida`。中止的建置會從中斷處接續執行。

### 磁碟空間不足

上面的約 400 GB 是完整建置的數字。只想**執行**這個移植版時，`--stages check,shaders,runtime` 約需 30 GB；
`assets` 階段（約 370 GB）只供資源工具使用，而 `--wav` 會再增加數倍於 Ogg Opus 的體積（當裝不下的是 `assets` 階段時，
事前檢查會把這個選項印出來）。`--work` 與 `--out` 都可以放在其他磁碟。runtime 會就地映射遊戲套件，因此套件所在磁碟為唯讀時
需要 `gta6rt --overlay <可寫目錄>`；`--overlay` 是 `gta6rt` 的選項，不是 `make_game.py` 的。
請保留 `work/shaders/dxbc/disasm` 裡的 `.s` 檔案：下一次執行會拿它們與 `disasm.done` 中的清單比對，重新反組譯缺少的部分
（若該目錄的 `disasm.done` 比這份清單還舊，則全部重新反組譯），接著再把所有著色器重新翻譯一次。當該階段必須清空 `disasm`
目錄、而它是一個符號連結時，建置會拒絕執行。

### 命令列

行為相同、但控制更細：

```powershell
python -I make_game.py --find                             # 列出本機找到的拷貝
python -I make_game.py --pick                             # 在資料夾對話方塊中選擇遊戲目錄
python -I make_game.py --check                            # 驗證你的拷貝，不建置任何東西
python -I make_game.py --check --full-check               # 另比對每個檔案的 SHA-256（較慢）
python -I make_game.py --game "D:\path\to\your\package" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets            # 只執行指定階段
python -I make_game.py --game "D:\path\to\your\package" --wav   # 另外把 Ogg Opus 解碼為 PCM WAV（很大）
python -I make_game.py --wave64-emulate auto              # auto（預設）、on 或 off
```

**階段順序**：`check` 一定先跑；接著被選取的 `shaders`、`assets`、`runtime` 會**一起平行執行**（彼此沒有相依關係）。
`pipelines` 在翻譯後的著色器就緒時啟動，且若這次執行包含 runtime，會等該階段完成 `vk_pipeline_check` 的重建之後才開始，
因此它能與 `assets` 剩下的部分重疊；它會在你的 GPU 驅動程式上建立每個翻譯後的著色器，回報任何必須跳過的項目，
並把管線快存取到 `--out` 中的 `pipeline_cache.bin`（runtime 目前還不會載入這個快取，所以它不會加快首次啟動）。
著色器翻譯是最耗 CPU 的步驟：以 `--jobs` 個程序執行（預設為 CPU 執行緒數的一半）。當 shaders 與 assets 共用同一台機器時，
這個額度會在兩者之間分配，並作為 `--jobs` 傳給 `rpf9dump`；單獨執行 assets 時，仍會讓 `rpf9dump` 用滿所有核心。
進度列會顯示百分比與概略 ETA（翻譯分片、資源步驟，以及整體階段完成度）。

在互動式終端機中，會出現一個 LEONIDA 設定畫面讓你設定 jobs 與階段是否重疊，接著可用按鍵切換的各階段日誌檢視
（`1`–`4` / `Tab`）讓平行輸出分流顯示；完整日誌放在 `work/logs/` 下。要純主控台輸出就用 `--no-tui`（或把 stdout 導向別處）。

`--wav` 預設關閉，因為 16-bit PCM WAV 的體積是 Ogg Opus 的數倍。它會在音訊銀行步驟之後，把重打包的 Ogg Opus 串流
解碼到 `work\assets\audio\wav`；建置接續執行時會跳過已存在的 WAV。

### `--wave64-emulate`

它決定是否把 wave64 的 compute 與 mesh 著色器翻譯成**在任意子群組寬度下都精確**的版本
（見 [`docs/design.md`](../../design.md) 的「Wave64 emulation」）。NVIDIA 與 Intel GPU 的子群組是 32 寬，需要這個模擬。
預設的 `auto` 會詢問 `vulkaninfo`（屬於 Vulkan SDK），只要有任何 GPU 缺少 64 寬子群組、**或其子群組大小無法判斷**，
就啟用模擬。它也會檢查 runtime 所使用的 GPU 能否以 64 寬執行 compute 與 mesh 著色器；使用 `--mesh-compute` 時，
只需要 compute 的子群組支援。

在 AMD 的 Windows 驅動程式上，只要 runtime 的 GPU 支援，`auto` 會在 **RDNA1 與 RDNA2** 上使用原生 wave64，
即使另一張 GPU 是 32 寬、或 `vulkaninfo` 不可用也一樣。因為該驅動在模擬的 LDS 程式碼上可能崩潰（#857）。
對於不支援 mesh 的 RDNA1 顯示卡，判定改用 compute 階段——它可以在沒有原生 mesh 支援的情況下跑 wave64。

此模式會儲存在翻譯後的著色器旁，改變它會重新翻譯。使用 `auto` 時通常可以沿用之前以模擬方式翻譯的著色器；上述的 AMD
Windows 驅動例外會把它們以原生模式重新翻譯。要強制模擬用 `--wave64-emulate on`，強制原生 wave64 用 `--wave64-emulate off`。

### 四個 fallback 旗標

它們同樣會轉給翻譯器、儲存在翻譯後的著色器旁，改變它們會重新翻譯：

- `--no-float-controls`：移除 `SignedZeroInfNanPreserve`，用於那些在缺少 `shaderSignedZeroInfNanPreserveFloat32`
  時管線檢查會跳過著色器的裝置；需要 PR #664 的翻譯器旗標。
- `--bary-fallback`：從插值輸入讀取原始重心座標，用於沒有 `VK_KHR_fragment_shader_barycentric` 的裝置。
- `--texel-offset-fallback`：把 `image_sample*_o` 的偏移折疊進座標，用於沒有 `VK_KHR_maintenance8` 的裝置。
- `--mesh-compute`：把 mesh 著色器當成 compute 處理，用於沒有 `VK_EXT_mesh_shader` 的裝置；在 Intel 的 Windows Vulkan
  驅動上會自行啟用，因為該驅動的原生 mesh draw 會崩潰（BUILD-2202），除非指定 `--native-mesh`。

每個旗標的說明見 [`docs/design.md`](../../design.md)。

### 建置的快取與重做規則

建置會記錄反組譯是由哪個 `llvm-mc` 命令與版本產生。改變其中任一項，或沿用沒有這份記錄的舊工作目錄，都會重新執行反組譯與翻譯，
但會保留已擷取的著色器輸入。

建置也會記錄著色器是由哪些翻譯器原始碼產生的。這些檔案有變動時，會沿用已儲存的反組譯結果重新翻譯。升級到會記錄這項資訊的版本之後，
較舊的工作目錄需要再翻譯一次。

建置另會記錄每個封存檔的模型與紋理、以及每個紋理池是由哪個解碼器（`third_party/rpf9-explorer` 的 commit 加上
`tools/rpf9dump` 的原始碼）匯出的。它改變時，`assets` 階段會移除該匯出並重做。較舊的工作目錄沒有這份記錄，因此升級後
其模型、紋理與紋理池會被再匯出一次。

### Linux 上的 Vulkan 診斷

只要有 CMake 3.24+、C++20 編譯器，以及已安裝的 Vulkan 開發標頭與 loader：

```sh
cmake -S runtime -B runtime/build/linux
cmake --build runtime/build/linux --parallel
./runtime/build/linux/vk_pipeline_check /path/to/spirv
```

把持有 `index.json` 與翻譯後 SPIR-V 檔案的那個目錄傳給 `vk_pipeline_check`。`make_game.py` 也能在 Linux 上執行：
`pipelines` 階段使用上面建置出的 `vk_pipeline_check`（在 `runtime/build/linux` 或 `runtime/build`），若它不存在，
腳本會在任何其他工作之前自行建置到該處。

### Linux 上透過 Wine 或 Proton 執行 loader

loader 是 Windows 程式，在 Wine 或 Proton 下執行；只要推送到 `main` 的變更動到 `runtime/`，CI 就會在 Wine 下跑它的測試
（Runtime workflow 的 `wine` job），也可從 Actions 分頁手動觸發。在 Linux 上，`runtime` 階段會用 clang-cl 與 lld-link
（LLVM 19+，但 22.1.7 與 22.1.8 有已知問題，見 [docs/testing.md](../../testing.md#known-problems)；命令名用不帶版本後綴的，
或五個 LLVM 工具都加同一個版本後綴，例如 `clang-cl-21`）把它交叉建置到 `runtime/build/windows`，並連結 `xwin`
佈局好的 MSVC CRT 與 Windows SDK，位置在 `XWIN_DIR` 或 `~/.local/share/xwin-sdk`
（[docs/testing.md](../../testing.md#the-windows-runtime-on-linux-under-wine-or-proton) 有一次性的 `xwin` 命令）。
缺少這些條件時，該階段不會列入預設 stages；明確要求它時，會在任何工作開始之前先列出缺少了什麼。
在別處建置好的 Windows 版本也可以使用，例如該 workflow 的 `windows` job 產出、解壓後的 `runtime-windows` artifact：

```sh
python3 -I make_game.py --runtime-build ~/Downloads/runtime-windows
```

兩種做法都會把 `gta6rt.exe` 與 `gta6rt_tls_host.dll` 複製到 `--out`，並在旁邊寫出 `gta6rt.sh`，後者用 `wine`
啟動 `gta6rt.exe`，或用 `WINE` 指定的那套 Wine（若是 Proton，就是它的 `files/bin/wine`）。同一個選項在 Windows 上會跳過建置。

在 Linux 上，先執行 `<out>/gta6rt.sh --dry-run` 檢視該標題的匯入，再執行 `<out>/gta6rt.sh --init` 檢查 TLS callback
與 DLL 初始化，啟動則用 `<out>/gta6rt.sh --run`。這些命令使用 `game_path.txt` 記錄的套件、把寫入導向移植版的 overlay，
並預設使用 Vulkan。當以 `LEONIDA_D3D12=vulkan` 選取 Vulkan 後端時，打包好的 runtime 會透過 `gta6rt.exe` 旁的
`shader_path.txt`，在 `<work>/shaders/spirv` 中找到翻譯後的著色器；要改用其他目錄請設定 `LEONIDA_SHADERS`。

## 疑難排解

- **用 GitHub 的 “Download ZIP” 下載**：壓縮檔不含 `third_party/rpf9-explorer` 子模組。請改用
  `git clone --recursive`，或在倉庫目錄中執行 `git submodule update --init --recursive`。
- **`llvm-mc` 拒絕 `gfx1013` / AMDGPU**：多半是用了 Windows 官方 LLVM 安裝套件，它不含 AMDGPU target。請改用
  `scripts\build_llvm_mc.ps1`、conda 的 `llvm-tools`，或把 `LLVM_MC` 指向 WSL 裡的 `llvm-mc`（`llvm-21`）。
- **建置 runtime 時回報 `D3D12_ROOT_SIGNATURE_DESC2` 錯誤**：把 Windows SDK 升級到 `10.0.26100` 或更新
  （可透過 Visual Studio Installer 或 `winget install Microsoft.WindowsSDK.10.0.26100` 安裝），刪除建置目錄後重新設定。
- **`C:` 磁碟空間不足**：用 `--work "D:\leonida-work"` 把工作檔放到其他分割區。

## 參與貢獻與執行 GPU 測試

- **最大的未解項目**（依英文 `README.md`）：**著色器翻譯器在硬體上的驗證**（vertex shader 的輸入佈局，若主機以 NGG
  執行 VS 則 `v0 = VertexIndex` 尚未驗證；以及 32 寬子群組上 VS/PS/GS 的 wave64 跨半波程式碼）、
  **Xbox D3D12 → Vulkan**（mesh shader 繪製、間接繪製狀態、stream output；#313 追蹤剩下的方法介面，且尚未用真實可執行檔驗證）、
  以及**模型復原**（`levels.rpf` 中 71 個尚未復原的 fragment 候選，見 [`docs/models.md`](../../models.md)）。
- GPU 測試指南（不需要遊戲檔案）：[`docs/testing.md`](../../testing.md)。
- 貢獻規則、零遊戲資料政策與任務協調：[`CONTRIBUTING.md`](../../../CONTRIBUTING.md)。
- **切勿在 issue 或 pull request 中附上遊戲檔案、擷取的資源或生成的著色器。**
