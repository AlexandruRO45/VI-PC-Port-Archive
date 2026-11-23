# Leonida — Szybki start (`pl-PL`)

> [!NOTE]
> Dokumentacja w języku angielskim ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) i
> [`docs/README.md`](../../README.md)) jest **wersją obowiązującą**. Ten przewodnik zawiera tylko stabilne instrukcje
> konfiguracji i budowania (zsynchronizowany z commitem `9f0ebbb3c`). Aktualny stan projektu znajdziesz w
> [`README.md`](../../../README.md).

## Czym jest Leonida

Leonida to projekt statycznej rekompilacji Grand Theft Auto VI (Xbox Series), który uruchamia grę natywnie na
Windowsie. Port jest budowany **na Twoim komputerze z Twojej własnej kopii gry**. Shadery RDNA2 z konsoli są
tłumaczone na SPIR-V (Vulkan), zasoby gry są odczytywane bezpośrednio z Twojej paczki, a oprogramowanie systemowe
konsoli jest zaimplementowane od nowa w postaci tzw. shimów.

> [!IMPORTANT]
> To repozytorium **nie zawiera gry, kluczy ani pliku wykonywalnego** i nie wyjaśnia, jak je zdobyć. Nic tutaj nie
> odszyfrowuje danych ani nie obchodzi zabezpieczeń. Wszystko, co tworzy build (przetłumaczone shadery, wyodrębnione
> zasoby, folder roboczy i zapisy gry), pochodzi z Twojej kopii i **musi zostać wyłącznie na Twoim komputerze** —
> nigdy nie załączaj ani nie udostępniaj tych plików w issues ani pull requestach (zobacz
> [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

Projekt jest na wczesnym etapie: **gra jeszcze się nie uruchamia**.

## Platformy

Na żadnej platformie nie da się jeszcze zagrać (stan projektu opisuje [`README.md`](../../../README.md)).
Co dziś działa gdzie:

- **Windows**: wszystko — `make_game.py`, narzędzia i runtime (`gta6rt` wraz z warstwą D3D12 → Vulkan) z testami.
- **Linux**: tłumacz, narzędzia zasobów i testowe narzędzia Vulkan budują się i działają natywnie (testowane na RADV
  i NVIDII). Runtime to kod windowsowy: cross-kompiluje się go clang-cl i xwin, a jego testy działają pod Wine lub
  Protonem. Gra ma docelowo działać tak samo — pod Wine lub Protonem, z warstwą D3D12 na natywnym Vulkanie (#217).
- **macOS (Apple Silicon)**: narzędzia Python i testowe narzędzia Vulkan działają natywnie, na MoltenVK (Homebrew)
  lub KosmicKrisp. Runtime działa jak na Linuksie — pod Wine. Znane braki: brak etapu geometry i
  `VK_EXT_mesh_shader` (te testy są pomijane), podgrupy 32-wątkowe (shadery wave64 działają emulowane) oraz
  arytmetyka zmiennoprzecinkowa MoltenVK; pełną lista ma [`docs/testing.md`](../../testing.md).
- **AMD RDNA1 (RX 5000)**: wspierany cel — bliski konsolowemu RDNA2, a jego 64-wątkowe podgrupy uruchamiają
  shadery wave64 natywnie.

Wyniki testów GPU ze wszystkich platform zbierane są w [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

## Wymagania

- **Twoja własna kopia** paczki Xbox Series w wersji `1.0.2.1` (folder z plikiem `MicrosoftGame.Config` i archiwami `.rpf` czytelnymi na dysku jako zwykłe pliki).
- **Windows 10/11 x64** z około **400 GB wolnego miejsca** na pliki robocze (sam eksport zasobów zmierzył ok. 337 GB przed krokiem audio, w ok. 2 h 35 min). Rozmiary dysków podajemy tak, jak pokazuje je Eksplorator Windows, czyli w GiB: 400 GiB to ok. 430 GB w jednostkach dziesiętnych. Mogą trafić na inny dysk dzięki `--work`.
- [Python 3.12+](https://www.python.org/downloads/) (zaznacz *„Add python.exe to PATH”*).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/) (`spirv-as`, `spirv-val` i `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) z **targetem AMDGPU** (`llvm-mc`):
  - Oficjalna wersja LLVM dla Windowsa **nie** zawiera targetu AMDGPU.
  - Aby zbudować natywny `llvm-mc.exe` z AMDGPU, uruchom `.\scripts\build_llvm_mc.ps1` w PowerShellu w głównym
    folderze repozytorium. Skrypt wymaga `git` i narzędzi C++ z Visual Studio, ustawia zmienną `LLVM_MC` dla
    Twojego użytkownika i prosi o otwarcie nowego terminala, żeby zmienna była widoczna. Budowanie trwa ok. 20-40
    minut.
  - Szybsza droga, jeśli używasz [condy](https://docs.conda.io/): `conda install -c conda-forge llvm-tools`
    (windowsowy build zawiera target AMDGPU), a potem ustaw `LLVM_MC` na pełną ścieżkę instalowanego przez nią
    `llvm-mc.exe` (`where.exe llvm-mc` pokaże ją po aktywowaniu środowiska; w PowerShellu samo `where` to
    `Where-Object`).
  - Jeśli używasz WSL, zainstaluj `llvm-21` w Ubuntu i ustaw zmienną w tej samej sesji PowerShella, w której
    uruchamiasz `make_game.py`:

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Więcej szczegółów w [`docs/design.md`](../../design.md#disassembling-shaders).
  - Na Linuksie wystarczy `llvm-mc` w `PATH`.
- **Visual Studio 2022** z pakietem *Programowanie aplikacji klasycznych w języku C++* i **Windows SDK `10.0.26100` lub nowszym** (potrzebne do etapu `runtime`, który używa root signatures w wersji 1.2).
- [ffmpeg](https://ffmpeg.org/download.html) w `PATH` *(opcjonalnie: dekoduje strumienie XMA2 i jest wymagany przy `--wav`)*.

## Jak zbudować

### Prosty sposób (Windows)

1. Zainstaluj powyższe wymagania.
2. Sklonuj repozytorium **razem z submodułem**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
   „Download ZIP” z GitHuba nie zawiera `third_party/rpf9-explorer`. Build sam go dociągnie, jeśli masz `git`, a w
   przeciwnym razie powie, co zrobić.
3. Kliknij dwukrotnie **`Build.bat`**.

Skrypt sam znajdzie Twoją kopię gry (Pobrane, Pulpit, Dokumenty — także tam, gdzie Windows je trzyma, np. w
OneDrive — Games, folder użytkownika i główne foldery wszystkich dysków, do dwóch poziomów w głąb). Rozpoznaje
wersję po rozmiarach kilku plików, bez czytania dużych danych. Jeśli znajdzie kilka kopii, zapyta, której użyć, a
jeśli żadnej — poprosi o wklejenie ścieżki do folderu. Przed startem sprawdza wymagania i wolne miejsce, wypisując
wszystkie braki naraz, a wynik zapisuje w `Documents\Leonida`. Przerwany build można uruchomić ponownie — będzie
kontynuowany od miejsca, w którym się zatrzymał.

**Czy ten komputer wystarczy?** `python -I tools/pc_checker.py --cli` rozpoznaje CPU, GPU i RAM i porównuje je
z profilami referencyjnymi przed długim buildem: `--profile 1`, domyślny, to odpowiednik peceta na poziomie Xbox
Series X, a `--profile 2` to społecznościowe oszacowanie dla 1080p Low. `python -I tools/pc_checker.py --check
--json` wypisuje ten sam werdykt jako JSON i zwraca go jako kod wyjścia (0 dobry, 1 słaby, 2 nieznany).

### Mało miejsca na dysku

Podane wyżej ~400 GB dotyczy pełnego builda. Samo *uruchomienie* portu wymaga `--stages check,shaders,runtime`
i około 30 GB; etap `assets` (~370 GB) jest potrzebny tylko narzędziom zasobów, a `--wav` dokłada kilkukrotność
rozmiaru Ogg Opus (preflight wypisuje ten wybór, gdy to właśnie etap `assets` się nie mieści). `--work` i `--out`
mogą trafić na inny dysk. Runtime mapuje paczkę w miejscu, więc dysk z paczką tylko do odczytu wymaga
`gta6rt --overlay <folder zapisywalny>`; `--overlay` to opcja `gta6rt`, nie `make_game.py`. Nie usuwaj plików `.s`
z `work/shaders/dxbc/disasm`: kolejny run porównuje je z listą w `disasm.done`, ponownie robi disassembly
brakujących (wszystkich, jeśli `disasm.done` jest starsze niż ta lista), a potem ponownie tłumaczy wszystkie
shadery. Folderu `disasm`, który jest symlinkiem, etap odmówi wyczyścić.

### Z terminala (`make_game.py`)

```powershell
python -I make_game.py --find                          # lista kopii gry znalezionych na tym komputerze
python -I make_game.py --pick                          # wybór folderu gry w oknie dialogowym
python -I make_game.py --check                         # sprawdza Twoją kopię, niczego nie buduje
python -I make_game.py --check --full-check            # dodatkowo porównuje SHA-256 każdego pliku (wolno)
python -I make_game.py --game "D:\sciezka\do\paczki" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # tylko wybrane etapy
python -I make_game.py --game "D:\sciezka\do\paczki" --wav  # dodatkowo dekoduje Ogg Opus do PCM WAV (duże pliki)
python -I make_game.py --wave64-emulate auto           # auto (domyślnie), on lub off
```

Etapy: `check` zawsze idzie pierwszy; potem `shaders`, `assets` i `runtime` (jeśli wybrane) idą równolegle — nie
zależą od siebie. `pipelines` startuje, gdy przetłumaczone shadery są gotowe, a gdy ten run obejmuje runtime, po
tym, jak ten etap przebuduje `vk_pipeline_check`, więc może nakładać się na resztę assets; tworzy każdy
przetłumaczony shader na Twoim sterowniku GPU, zgłasza te, które musi pominąć i zapisuje pipeline cache do
`pipeline_cache.bin` w `--out` (runtime nie wczytuje tego cache, więc nie przyspiesza pierwszego startu).
Tłumaczenie shaderów to najdłuższy etap CPU: działa na `--jobs` procesach (domyślnie połowa wątków procesora).
Gdy shadery i assets dzielą maszynę, ten budżet jest dzielony między nie i przekazywany do `rpf9dump` jako
`--jobs`; samodzielny run assets pozwala `rpf9dump` użyć wszystkich rdzeni. Linie postępu pokazują procent
i przybliżony ETA dla shardów tłumaczenia, kroków assets i całego etapu.

W interaktywnym terminalu ekran konfiguracji LEONIDA pozwala ustawić liczbę zadań i to, czy etapy mają się
nakładać, a potem widok logów per etap przełączany klawiszami (`1`–`4` / `Tab`) rozdziela równoległy output;
pełne logi etapów trafiają do `work/logs/`. Użyj `--no-tui` (albo przekieruj stdout), żeby dostać zwykłą konsolę.

- **`--wave64-emulate`**: emuluje shadery compute/mesh wave64 na kartach z podgrupami 32-wątkowymi (NVIDIA i Intel).
  Domyślny tryb `auto` pyta `vulkaninfo` i sam włącza emulację, gdy jest potrzebna, a także sprawdza, czy GPU
  runtime'u uruchomi shadery compute i mesh 64-wątkowo. AMD (RDNA) wykonuje wave64 natywnie: na windowsowym
  sterowniku AMD `auto` zostaje przy natywnym wave64 na RDNA1 i RDNA2, gdy GPU runtime'u to obsługuje — na tym
  sterowniku emulowany kod LDS potrafi wywalić program (#857). Tryb zapisywany jest obok przetłumaczonych
  shaderów, a jego zmiana tłumaczy je od nowa. `auto` zwykle ponownie używa wcześniej przetłumaczonych pod
  emulację shaderów; obejście dla windowsowego sterownika AMD przebudowuje je w trybie natywnym.
- **`--work`**: najlepiej wskazać dysk z co najmniej 400 GB wolnego miejsca. Etap `shaders` trwa najdłużej i
  domyślnie używa połowy wątków procesora (można to zmienić przez `--jobs`).
- **`pipelines`**: tworzy każdy przetłumaczony shader na Twoim sterowniku GPU i zgłasza te, które musi pominąć;
szczegóły (kolejność etapów, pipeline cache) w akapicie o etapach niżej.
- **Flagi fallback** dla urządzeń bez danej funkcji Vulkan — przekazywane do tłumacza, zapisywane obok
  przetłumaczonych shaderów, a ich zmiana tłumaczy je od nowa: `--no-float-controls`, `--bary-fallback` (bez
  `VK_KHR_fragment_shader_barycentric`), `--texel-offset-fallback` (bez `VK_KHR_maintenance8`) i
  `--mesh-compute` (bez `VK_EXT_mesh_shader`). Szczegóły w [`docs/design.md`](../../design.md).
- **Co jest robione od nowa automatycznie**: build zapisuje polecenie i wersję `llvm-mc`, które zrobiły
  disassembly, źródła tłumacza oraz decoder zasobów (commit submodułu `third_party/rpf9-explorer` i źródła
  `tools/rpf9dump`). Zmiana dowolnego z nich powtarza tylko dotyczący jej krok (disassembly, tłumaczenie albo
  eksport modeli i tekstur); starsze foldery robocze, w których nic nie zapisano, zrobią go raz jeszcze po
  ulepszeniu.
- **Linux i Wine/Proton**: `make_game.py` działa też na Linuksie. Etap `pipelines` używa `vk_pipeline_check`
  (zbudowanego cmake w `runtime/build/linux`; jeśli go nie ma, skrypt buduje go sam przed całą resztą). Loader to
  program windowsowy i działa pod Wine lub Protonem: etap `runtime` cross-kompiluje go clang-cl i lld-link
  (LLVM 19+, ale nie 22.1.7 ani 22.1.8) pod MSVC CRT i Windows SDK przygotowane przez
  [xwin](https://github.com/Jake-Shadle/xwin); bez nich etap wypada z domyślnych, a prośba o niego wypisze, czego
  brakuje. Build windowsowy zrobiony gdzie indziej też wystarczy, np. artefakt `runtime-windows` z CI:

  ```sh
  python3 -I make_game.py --runtime-build ~/Downloads/runtime-windows
  ```

  W obu przypadkach skrypt kopiuje `gta6rt.exe` i `gta6rt_tls_host.dll` do `--out` i pisze obok nich `gta6rt.sh`,
  który uruchamia program przez `wine` (albo przez Wine wskazane zmienną `WINE`, np. z Protona). Na Windowsie ta
  sama opcja pomija build.

Na Linuksie uruchom `<out>/gta6rt.sh --dry-run`, żeby przejrzeć importy spakowanego programu, a potem
`<out>/gta6rt.sh --init`, żeby sprawdzić callbacki TLS i inicjalizację DLL. Do startu służy `<out>/gta6rt.sh
--run`. Te polecenia korzystają z paczki wpisanej w `game_path.txt`, z `overlay` portu do zapisów i domyślnie
z Vulkana.

## Częste problemy

- **Pobrałeś repozytorium przyciskiem „Download ZIP” na GitHubie**: archiwum ZIP nie zawiera submodułu `third_party/rpf9-explorer`. Użyj `git clone --recursive` albo uruchom `git submodule update --init --recursive` w folderze repozytorium.
- **`llvm-mc` odrzuca `gfx1013` / AMDGPU**: najpewniej masz oficjalny instalator LLVM dla Windowsa. Użyj `scripts\build_llvm_mc.ps1` albo ustaw `LLVM_MC` na `llvm-mc` z WSL (`llvm-21`).
- **Błąd `D3D12_ROOT_SIGNATURE_DESC2` przy budowaniu runtime**: zaktualizuj Windows SDK do `10.0.26100` lub nowszego przez Visual Studio Installer albo `winget install Microsoft.WindowsSDK.10.0.26100`, usuń folder builda i skonfiguruj go ponownie.
- **Brak miejsca na dysku `C:`**: podaj `--work "D:\leonida-work"`, aby przenieść pliki robocze na inną partycję.

## Jak pomóc i uruchomić testy GPU

1. Przejrzyj otwarte PR-y i [issue koordynacyjne][coord], żeby nie dublować pracy. Zarezerwuj zadanie wpisem
   `claiming: <ID>` albo jednym zdaniem opisu, jeśli zadanie nie ma jeszcze ID.
2. Uruchom odpowiednie testy na syntetycznych danych wejściowych. Na przykład weryfikator manifestów testujesz
   poleceniem `python -I tests/test_manifest_verifier.py` w głównym folderze repozytorium.
3. W PR podaj ID zadania, opisz, co przetestowałeś, a jeśli uruchamiałeś testy GPU — podaj kartę i wersję
   sterownika. Zaznacz, czy korzystałeś z pomocy AI. Zasady znajdziesz w [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

[Przewodnik po testach GPU](../../testing.md) pokazuje, jak uruchomić testy bez plików gry.
Wyniki testów GPU publikuj w [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
