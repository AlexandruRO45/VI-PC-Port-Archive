# Jak przetestować gta6rt na własnej, legalnie posiadanej grze

Ten dokument opisuje **wyłącznie** pracę z plikami, które już posiadasz i masz prawo skopiować na swój PC
(własna konsola w trybie deweloperskim, własna płyta). Nie opisuje i nie wspiera obchodzenia zabezpieczeń:
żadnych kluczy, odszyfrowywania, patchowania DRM ani pobierania treści z serwerów. Jeśli pliki okażą się
zaszyfrowane, zatrzymaj się i zapisz to w `docs/design.md` — tak mówi `AGENTS.md` tego projektu.

Repozytorium nie zawiera żadnych danych gry: pliki leżą wyłącznie u Ciebie, w katalogu, który wskażesz.

---

## 1. Co będzie potrzebne

| Rzecz | Po co |
|---|---|
| Konsola Xbox z włączonym trybem deweloperskim (albo własna płyta z grą) | źródło plików, do których masz prawo |
| Przeglądarka + sieć lokalna | Device Portal konsoli (kopiowanie plików) |
| `gta6rt.exe` zbudowany z tego repo (`runtime/build/`) | uruchamia grę |
| Windows 10/11 x64, GPU z Vulkan 1.3, sterownik NVIDIA/AMD | warstwa D3D12→Vulkan |

Sprawdź, że runtime w ogóle działa, **zanim** weźmiesz grę do ręki — sekcja 5 (harness testowy).

## 2. Skąd wziąć pliki (legalnie)

### 2.1 Konsola w trybie deweloperskim (zalecane, np. darmowe demo z Microsoft Store)

1. Włącz **Developer Mode** na konsoli (aplikacja Xbox Dev Mode; jednorazowa konfiguracja, wymaga konta
   Microsoft — to oficjalna funkcja konsoli, nie modyfikacja systemu).
2. Zainstaluj na konsoli grę/demo **ze sklepu Microsoft Store** (normalny zakup/pobranie).
3. Otwórz **Device Portal**: `https://<IP-konsoli>:11443` w przeglądarce, zaloguj się kontem deweloperskim.
4. Przejdź do **File Explorer** i znajdź katalog pakietu zainstalowanej gry — zwykle pod
   `D:\DevelopmentFiles\` albo w drzewie pakietu widocznym w portalu (nazwa zaczyna się od identyfikatora
   pakietu, np. `Microsoft.<Gra>_<wersja>_x64__<hash>`).
5. Skopiuj na PC **pliki gry**: główny `.exe`, towarzyszące `.dll` oraz pliki danych, które są częścią tego
   samego pakietu — przyciskiem *Download* w portalu albo przez sieciowy udział (`\\<IP>\...`, jeśli portal
   go udostępnia). Kopiuj **tylko to, do czego masz prawo** i tylko na swój komputer.

### 2.2 Własna płyta

Jeśli potrafisz odczytać zawartość własnej płyty na swoim PC (napęd/urządzenie, którym legalnie dysponujesz),
skopiuj z niej `.exe` i `.dll` gry. Projekt nie dostarcza narzędzi do omijania zabezpieczeń nośnika; jeśli
zawartość jest zaszyfrowana — koniec drogi, zapisz to w `docs/design.md`.

### 2.3 Czego NIE robić

- nie pobieraj „gotowych” plików gry z internetu,
- nie używaj kluczy, patchy DRM, loaderów omijających zabezpieczenia,
- nie wrzucaj plików gry do repozytorium ani do zgłoszeń/PR-ów (zakaz z `AGENTS.md`).

## 3. Ułóż katalog gry

Utwórz np. `C:\Games\moja-gra\` i zachowaj strukturę z konsoli: główny `.exe` i wszystkie `.dll`, które leżały
obok niego (to „bundled modules”, które loader mapuje sam). Pliki danych zostaw tam, gdzie były — runtime ich
nie tłumaczy, ale gra może ich szukać po ścieżkach względnych.

```
C:\Games\moja-gra\
    <nazwa>.exe          <- główny obraz gry
    <moduł1>.dll
    <moduł2>.dll
    ...                  <- dane gry w oryginalnym układzie
```

## 4. Uruchom

```powershell
cd C:\Users\konta\Documents\deepseek-harness\default-workspace\Leonida\runtime\build

# 1) Najpierw sam raport: mapowanie, relokacje, bindowanie importów. Żaden kod gry się nie wykonuje.
.\gta6rt.exe --dry-run "C:\Games\moja-gra\<nazwa>.exe" --game-dir "C:\Games\moja-gra"

# 2) DllMain/TLS każdego modułu (bez entry pointu gry).
.\gta6rt.exe --init "C:\Games\moja-gra\<nazwa>.exe" --game-dir "C:\Games\moja-gra"

# 3) Pełny start: --init, a potem entry point obrazu (to już uruchamia kod gry).
.\gta6rt.exe --run "C:\Games\moja-gra\<nazwa>.exe" --game-dir "C:\Games\moja-gra"
```

Co wypisuje raport:

```
shim tables linked: dxgi.dll=yes d3d12_xs.dll=yes
<nazwa>.exe   base=... size=... entry=+0x...  shim=N host=N module=N unimplemented=N
    shim          d3d12.dll!D3D12CreateDevice
    host          kernel32.dll!VirtualAlloc
    UNIMPLEMENTED dxgi.dll!CreateDXGIFactory1
N module(s) mapped, K unimplemented import(s)
```

- `host` — załatwia system Windows (prawdziwy `LoadLibrary` + `GetProcAddress`),
- `shim` — nasza implementacja (tabela shimów lub eksport `leonida_shims.dll`),
- `module` — inny moduł gry skopiowany obok (bundled),
- `UNIMPLEMENTED` — nie mamy odpowiednika: gra dostaje thunk, który **loguje i zwraca 0**.

## 5. Najpierw sprawdź runtime bez gry (harness testowy)

To ten sam łańcuch (loader → tabela shimów → DXGI/D3D12 → Vulkan → okno), ale bez żadnych danych gry:

```powershell
cd C:\Users\konta\Documents\deepseek-harness\default-workspace\Leonida\runtime\build
.\gta6rt.exe --dry-run leonida_test_harness.exe --game-dir . --shims leonida_shims.dll
.\gta6rt.exe --run     leonida_test_harness.exe --game-dir . --shims leonida_shims.dll
```

Oczekiwane: `0 unimplemented import(s)`, okno z **animowanym trójkątem**, w `leonida_window.log` linia
`wsi=1 shared_device=1 frames=...` (u nas ~220 klatek w 4 s). To dowód, że `gta6rt.exe` mapuje obraz,
binduje importy i prezentuje przez Vulkan, zanim dołożysz do tego prawdziwą grę.

## 6. Gdzie patrzeć, gdy gra się wysypie

| Miejsce | Co mówi |
|---|---|
| Konsola `gta6rt` (raport importów) | których importów brakuje (`UNIMPLEMENTED`) jeszcze **przed** startem |
| `C:\Games\moja-gra\missing_imports.log` | **tu zaglądaj w pierwszej kolejności**: jedna linia `dll!nazwa` na każdy import, którego nie udało się zbindować w tym uruchomieniu; plik jest przepisywany przy każdym `load()` |
| `stderr`, linie `[rt] unimplemented: dll!nazwa` | API, które gra **wywołała** w czasie działania i dostała stub (0) — zwykle tuż przed wysypką |
| `stderr`, linie `[rt] dxgi:`, `[rt] d3d12:`, `[wsi]` | ostrzeżenia warstwy: nieobsługiwane metody COM, brakujące funkcje, problemy z presentem |
| `error: ...` na końcu | loader nie zamapował obrazu (uszkodzony plik, brak modułu, zły PE) |
| exit code / zawieszenie | najczęściej brak shimu dla wątku/pliku/audio albo oczekiwanie na usługę sieciową; kolejność: `missing_imports.log` → log `[rt] unimplemented` → raport importów |

Praktyczna pętla pracy:

1. `--dry-run` → poprawić/uzupełnić najczęstszy `UNIMPLEMENTED` w tabeli shimów,
2. `--run` → gdy gra padnie, w logu wywołań widać kolejny brakujący `dll!nazwa`,
3. dopisać shim (albo stub z logiem, jeśli na tym etapie wystarczy), zbudować, powtórzyć.

## 7. Czego na pewno jeszcze brakuje prawdziwej grze

Kolejność wynika z tego, co już mamy i czego wymaga pierwsza klatka (docs/d3d12_vulkan.md, sekcja 6):

- **DXGI** — jest factory/adapter/swapchain/`Present`/`ResizeBuffers`; brakuje `SetFullscreenState`,
  enumeracji outputów, kompozycji `PresentX` i `Present` na buforze innym niż 0,
- **PresentX** — ciało prezentacji gotowe (`rt::d3d12::present_x`); brakuje zmierzonego slotu w vtable kolejki
  Xbox i kompozycji płaszczyzn (src/dst rect),
- **XGameRuntime / XSystem / XTaskQueue** — shimy są, ale nie wszystkie ścieżki (async, kolejki zadań) są
  przechodzone przez grę,
- **wejście/wyjście**: XInput, klawiatura/mysz, gamepad,
- **audio**: biniki Bink/audio i wyjście dźwięku,
- **pamięć/wątki/pliki**: shimy `xmem`, `VirtualAlloc`, `CreateThread` — pokryte w testach, ale gra może chcieć
  więcej (`CreateFile2`, IOCP, `WaitOnAddress`),
- **MRT**: pipeline'y z wieloma `SV_Target` działają (GBuffer), ale testowy shader ma jeden output — warto
  dołożyć fixture z prawdziwym GBufferem.

## 8. Uwagi bezpieczeństwa i higieny

- uruchamiaj grę w katalogu, do którego masz prawo zapisu (runtime zapisuje `missing_imports.log` w katalogu gry),
- nie udostępniaj plików gry ani logów z ich zawartością (nazwy shaderów, adresy) w publicznych zgłoszeniach —
  `AGENTS.md` tego zabrania,
- `--run` wykonuje kod gry w Twoim procesie: rób to na maszynie, którą możesz w razie potrzeby zrestartować,
- commity i PR-y nie mogą zawierać plików gry; wszystko, co potrzebne do reprodukcji, opisuj nazwami API.
