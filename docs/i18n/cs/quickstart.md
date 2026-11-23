# Leonida — rychlý start (`cs`)

> [!NOTE]
> Závazná je anglická dokumentace ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)
> a [`docs/README.md`](../../README.md)). Tento návod obsahuje jen stálé kroky k přípravě a sestavení (sladěno
> s commitem `dde82f2`). Aktuální stav projektu najdeš v [`README.md`](../../../README.md).

## Co je Leonida?

Leonida je projekt statické rekompilace (static recompilation), jehož cílem je spustit Grand Theft Auto VI
(Xbox Series) nativně ve Windows. Port se sestavuje **na tvém PC z tvé vlastní kopie hry**. Procesor Xbox Series je
stejně jako PC x86-64, takže se kód hry načte a spustí přímo a systémový software konzole je napsaný znovu jako
takzvané shimy. Předkompilované RDNA2 shadery konzole se překládají do SPIR-V (Vulkan) a modely, textury a zvuk se
čtou přímo z tvého balíčku.

> [!IMPORTANT]
> Tento repozitář **neobsahuje hru, klíče ani spustitelný soubor** a ani nevysvětluje, jak je získat. Nic z něj
> nedešifruje data ani neobchází žádnou ochranu. Všechno, co sestavení vytvoří (přeložené shadery, extrahované
> assety, pracovní složka a tvoje uložené pozice), pochází z tvé kopie a **musí zůstat na tvém PC**. Tyto soubory
> nikdy nepřikládej k issues ani pull requestům a nesdílej je (viz [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

Projekt je zatím na začátku: **hra se ještě nespustí.**

## Platformy

Zatím se nedá hrát na žádné platformě. Co dnes kde běží:

- **Windows**: všechno (`make_game.py`, nástroje a runtime `gta6rt` s vrstvou D3D12 → Vulkan a jejími testy).
- **Linux**: překladač, nástroje na assety a nástroje pro testy GPU ve Vulkanu se sestaví a běží nativně (testováno
  na RADV a NVIDIA). Runtime je kód pro Windows: křížově se kompiluje pomocí clang-cl a xwin a jeho testy běží pod
  Wine nebo Protonem. Hra má později běžet stejně (#217).
- **macOS (Apple Silicon)**: nástroje v Pythonu a testovací nástroje pro Vulkan běží nativně, na MoltenVK (Homebrew)
  nebo KosmicKrisp. Runtime běží jako na Linuxu, pod Wine. Známá omezení: chybí geometry stage a
  `VK_EXT_mesh_shader`, subgroups jsou široké 32 (wave64 shadery běží emulovaně) a MoltenVK jinak zachází
  s desetinnými čísly; [`docs/testing.md`](../../testing.md) je vypisuje všechna.
- **AMD RDNA1 (RX 5000)**: podporovaný cíl, blízký RDNA2 v konzoli; jeho 64 široké subgroups spouštějí wave64 shadery
  nativně.

Výsledky testů GPU ze všech platforem se sbírají v [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

## Co budeš potřebovat

- **Vlastní kopii** balíčku pro Xbox Series ve verzi `1.0.2.1` (složka s `MicrosoftGame.Config` a archivy `.rpf`,
  jako obyčejné čitelné soubory na disku).
- **Windows 10/11 x64** a zhruba **400 GB volného místa** na pracovní soubory. Velikosti jsou uvedené tak, jak je
  ukazuje Průzkumník Windows, tedy v GiB: 400 GiB je asi 430 GB v desítkových jednotkách. Pomocí `--work` mohou ležet
  i na jiném disku.
- [Python 3.12+](https://www.python.org/downloads/) (při instalaci zaškrtni *„Add python.exe to PATH“*).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/) (`spirv-as`, `spirv-val` a `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) s **AMDGPU targetem** (`llvm-mc`):
  - Oficiální build LLVM pro Windows AMDGPU target **neobsahuje**.
  - Nativní `llvm-mc.exe` s AMDGPU sestaví `.\scripts\build_llvm_mc.ps1`, spuštěný v PowerShellu v hlavní složce
    repozitáře. Skript potřebuje `git` a C++ nástroje Visual Studia, nastaví tvému uživateli proměnnou `LLVM_MC`
    a požádá tě, abys pak otevřel nový terminál, aby ji viděl. Trvá to asi 20–40 minut.
  - Rychleji, pokud používáš [condu](https://docs.conda.io/): `conda install -c conda-forge llvm-tools` (build pro
    Windows AMDGPU target obsahuje), pak nastav `LLVM_MC` na plnou cestu k nainstalovanému `llvm-mc.exe`
    (`where.exe llvm-mc` ji ukáže, když je prostředí aktivní; samotné `where` je v PowerShellu cmdlet
    `Where-Object`).
  - Přes WSL: nainstaluj v Ubuntu `llvm-21` a proměnnou nastav ve stejné relaci PowerShellu, ve které spouštíš
    `make_game.py`:

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Víc v [`docs/design.md`](../../design.md#disassembling-shaders).
  - Na Linuxu stačí `llvm-mc` v `PATH`.
- **Visual Studio 2022** s workloadem *Vývoj desktopových aplikací pomocí C++* a **Windows SDK `10.0.26100` nebo
  novějším** (pro stage `runtime`, která používá root signatures verze 1.2).
- [ffmpeg](https://ffmpeg.org/download.html) v `PATH` *(volitelně: dekóduje XMA2 streamy a je potřeba pro `--wav`)*.

**Stačí na to tvoje PC?** `python -I tools/pc_checker.py --cli` zjistí procesor, grafiku a RAM a před dlouhým
sestavením je porovná s referenčními profily: `--profile 1` (výchozí) odpovídá PC srovnatelnému s Xbox Series X,
`--profile 2` je komunitní odhad pro 1080p na nízké detaily.

## Jak to sestavit

### Jednoduchá cesta (Windows)

1. Nainstaluj všechno ze seznamu výše.
2. Naklonuj repozitář **i se submodulem**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
   „Download ZIP“ z GitHubu vynechá `third_party/rpf9-explorer`. Sestavení si ho stáhne samo, pokud je nainstalovaný
   `git`, jinak ti řekne, co dělat.
3. Dvakrát klikni na **`Build.bat`**.

Sestavení si tvou kopii najde samo: prohledá Stažené soubory, Plochu, Dokumenty (i tam, kam je Windows ukládá,
třeba na OneDrive), Games, tvou uživatelskou složku a kořen každého disku, vždy dvě složky do hloubky. Verzi pozná
podle velikosti několika souborů, aniž by četlo velké soubory. Když najde víc kopií, zeptá se, kterou použít, když
žádnou, požádá tě o vložení cesty ke složce. Před startem zkontroluje, že je všechno nainstalované a že je dost
místa, a vypíše najednou všechno, co chybí. Výsledek skončí v `Documents\Leonida`. Přerušené sestavení při dalším
spuštění pokračuje tam, kde skončilo.

### Málo místa na disku

Údaj o místě výše platí pro úplné sestavení. Pro samotné *spuštění* portu potřebuje
`--stages check,shaders,runtime` asi 30 GB; stage `assets` (~370 GB) slouží jen nástrojům na assety a `--wav` přidá
ještě několikanásobek velikosti Ogg Opus. `--work` a `--out` mohou být na jiném disku. Runtime čte balíček na místě;
pokud leží na disku jen pro čtení, potřebuješ `gta6rt --overlay <zapisovatelná složka>` (je to volba `gta6rt`,
ne `make_game.py`). Soubory `.s` v `work/shaders/dxbc/disasm` nemaž: další spuštění je porovná se seznamem
v `disasm.done`, chybějící disassembluje znovu a pak znovu přeloží všechny shadery.

### Z terminálu (`make_game.py`)

```powershell
python -I make_game.py --find                          # vypíše kopie nalezené na tomto PC
python -I make_game.py --pick                          # vybere složku hry v dialogu
python -I make_game.py --check                         # zkontroluje tvou kopii, nic nesestavuje
python -I make_game.py --check --full-check            # navíc porovná SHA-256 každého souboru (pomalé)
python -I make_game.py --game "D:\cesta\k\balicku" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # jen vybrané stages
python -I make_game.py --game "D:\cesta\k\balicku" --wav   # navíc dekóduje Opus do PCM WAV (velké)
python -I make_game.py --wave64-emulate off            # auto (výchozí), on nebo off
```

Pořadí stages: `check` běží vždy první; pak běží `shaders`, `assets` a `runtime` souběžně (nezávisí na sobě).
`pipelines` začne, jakmile jsou přeložené shadery hotové.

- **`--wave64-emulate`**: emuluje wave64 compute a mesh shadery na grafikách se subgroups širokými 32 (NVIDIA
  a Intel). Výchozí `auto` se zeptá `vulkaninfo` a emulaci zapne sám, když je potřeba. Karty AMD (RDNA) spouštějí
  wave64 nativně. S ovladačem AMD pro Windows zůstává `auto` na RDNA1 a RDNA2 u nativního wave64, protože tento
  ovladač může u emulovaného LDS kódu spadnout (#857). Režim se ukládá vedle přeložených shaderů a jeho změna je
  přeloží znovu.
- **`--work`**: ideálně disk s aspoň 400 GB volného místa. Stage `shaders` trvá nejdéle a ve výchozím stavu
  používá polovinu vláken procesoru (dá se změnit přes `--jobs`).
- **`pipelines`**: vytvoří každý přeložený shader na ovladači tvé grafiky, ověří, že se zkompiluje, nahlásí ty,
  které musel přeskočit, a uloží pipeline cache do `pipeline_cache.bin` v `--out`. Runtime tuto cache zatím
  nenačítá, takže první spuštění nezrychlí.
- **Záložní volby** pro zařízení, kterým chybí některá funkce Vulkanu; ukládají se vedle přeložených shaderů a jejich
  změna je přeloží znovu: `--no-float-controls`, `--bary-fallback` (bez `VK_KHR_fragment_shader_barycentric`),
  `--texel-offset-fallback` (bez `VK_KHR_maintenance8`) a `--mesh-compute` (bez `VK_EXT_mesh_shader`). Viz
  [`docs/design.md`](../../design.md).
- **Co se přestaví samo**: sestavení si pamatuje příkaz a verzi `llvm-mc`, zdrojáky překladače a dekodér assetů
  (`third_party/rpf9-explorer` a `tools/rpf9dump`). Když se něco z toho změní, zopakuje jen dotčený krok
  (disassembly, překlad, nebo export modelů a textur).
- **V interaktivním terminálu** se nejdřív ukáže obrazovka nastavení LEONIDA (počet jobs a zda se stages mají
  překrývat) a pak log rozdělený podle stages, mezi kterými přepínáš klávesami `1`–`4` / `Tab`. Úplné logy jsou ve
  `work/logs/`. S `--no-tui` dostaneš obyčejný výpis do konzole.
- **Linux a Wine/Proton**: `make_game.py` běží i na Linuxu. Stage `pipelines` používá `vk_pipeline_check` (sestavený
  CMakem v `runtime/build/linux`; když chybí, skript si ho sestaví sám). Loader je program pro Windows a běží pod
  Wine nebo Protonem: stage `runtime` ho křížově kompiluje pomocí clang-cl a lld-link (LLVM 19+, ale ne 22.1.7 ani
  22.1.8) proti Windows SDK, které připraví [xwin](https://github.com/Jake-Shadle/xwin); viz
  [`docs/testing.md`](../../testing.md). Případně přes `--runtime-build <složka>` zadáš build pro Windows udělaný
  jinde (`gta6rt.exe` a `gta6rt_tls_host.dll`). V obou případech skript vytvoří `gta6rt.sh`, který program spustí
  přes `wine` (nebo přes Wine z proměnné `WINE`, třeba z Protonu).

## Časté problémy

- **Repozitář stažený přes „Download ZIP“ na GitHubu**: ZIP neobsahuje submodul `third_party/rpf9-explorer`. Použij
  `git clone --recursive`, nebo ve složce repozitáře spusť `git submodule update --init --recursive`.
- **`llvm-mc` odmítá `gfx1013` / AMDGPU**: nejspíš máš oficiální instalátor LLVM pro Windows. Použij
  `scripts\build_llvm_mc.ps1`, nebo nastav `LLVM_MC` na `llvm-mc` z WSL (`llvm-21`).
- **Chyba `D3D12_ROOT_SIGNATURE_DESC2` při sestavování runtime**: aktualizuj Windows SDK na `10.0.26100` nebo novější
  (přes Visual Studio Installer nebo `winget install Microsoft.WindowsSDK.10.0.26100`), smaž složku buildu
  a nakonfiguruj znovu.
- **Málo místa na `C:`**: s `--work "D:\leonida-work"` dáš pracovní soubory na jiný disk.

## Jak pomoct a spustit testy GPU

1. Projdi otevřené pull requesty a [koordinační issue][coord], ať se práce nedělá dvakrát. Úkol si tam zaber
   pomocí `claiming: <ID>`, nebo krátkým popisem, pokud ještě ID nemá.
2. Spusť odpovídající testy se syntetickými vstupy, například kontrolu manifestu pomocí
   `python -I tests/test_manifest_verifier.py` v hlavní složce repozitáře.
3. V pull requestu uveď ID úkolu, popiš, co jsi testoval, a u testů GPU i grafickou kartu a verzi ovladače. Napiš,
   jestli jsi použil pomoc AI. Pravidla jsou v [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

[Návod k testům GPU](../../testing.md) ukazuje, jak je spustit bez herních souborů.
Výsledky testů GPU patří do [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
