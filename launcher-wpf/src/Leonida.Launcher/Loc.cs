using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Markup;

namespace Leonida.Launcher;

/// <summary>
/// The launcher's languages: English (the text as written) and Polish. Static text in the XAML is translated when a
/// window loads (<see cref="Apply"/> walks it and swaps any string it has a translation for; bound values are left
/// alone); text made in code goes through <see cref="T"/> and <see cref="F"/>. Changing the language reloads the
/// window. README pages, make_game.py's output and other people's names stay as they are.
/// </summary>
public static class Loc
{
    public static readonly (string Code, string Name)[] Languages = [("en", "English"), ("pl", "Polski")];
    static string language = "en";

    /// <summary>"en" or "pl". Numbers, dates and sizes follow it too (1.0 GB / 1,0 GB), not the Windows locale.</summary>
    public static string Language
    {
        get => language;
        set
        {
            language = value;
            var culture = System.Globalization.CultureInfo.GetCultureInfo(value == "pl" ? "pl-PL" : "en-US");
            System.Globalization.CultureInfo.DefaultThreadCurrentCulture = culture;
            System.Globalization.CultureInfo.DefaultThreadCurrentUICulture = culture;
            System.Globalization.CultureInfo.CurrentCulture = culture;
            System.Globalization.CultureInfo.CurrentUICulture = culture;
        }
    }
    public static bool Polish => Language == "pl";

    public static string T(string en) => Polish && Pl.TryGetValue(en, out var p) ? p : en;
    public static string F(string en, params object?[] args) => string.Format(T(en), args);

    /// <summary>Translates every static string under <paramref name="root"/> (logical tree, tooltips, context menus,
    /// list columns).</summary>
    public static void Apply(DependencyObject root)
    {
        if (!Polish) return;
        var seen = new HashSet<DependencyObject>();
        Walk(root, seen);
    }

    static void Walk(DependencyObject d, HashSet<DependencyObject> seen)
    {
        if (!seen.Add(d)) return;
        switch (d)
        {
            case TextBlock tb when !BindingOperations.IsDataBound(tb, TextBlock.TextProperty):
                // Text set as an attribute leaves Inlines empty, so the whole text is looked up first; text made of
                // several runs (some bold, some bound) is translated run by run.
                if (Pl.TryGetValue(tb.Text, out var t)) tb.Text = t;
                else foreach (var i in tb.Inlines.ToList()) Walk(i, seen);
                break;
            case Run r when !BindingOperations.IsDataBound(r, Run.TextProperty):
                if (Pl.TryGetValue(r.Text, out var rt)) r.Text = rt;
                break;
            case HeaderedItemsControl h when h.Header is string hs && !BindingOperations.IsDataBound(h, HeaderedItemsControl.HeaderProperty):
                if (Pl.TryGetValue(hs, out var ht)) h.Header = ht;
                break;
            case HeaderedContentControl hc when hc.Header is string hcs:
                if (Pl.TryGetValue(hcs, out var hct)) hc.Header = hct;
                break;
            case ContentControl cc when cc.Content is string cs && !BindingOperations.IsDataBound(cc, ContentControl.ContentProperty):
                if (Pl.TryGetValue(cs, out var ct)) cc.Content = ct;
                break;
            case Window w when Pl.TryGetValue(w.Title, out var wt):
                w.Title = wt;
                break;
        }
        if (d is FrameworkElement fe)
        {
            if (fe.ToolTip is string tip && Pl.TryGetValue(tip, out var tt)) fe.ToolTip = tt;
            if (fe.ContextMenu != null) Walk(fe.ContextMenu, seen);
            if (fe is ListView { View: GridView gv })
                foreach (var col in gv.Columns)
                    if (col.Header is string ch && Pl.TryGetValue(ch, out var cht)) col.Header = cht;
        }
        foreach (var child in LogicalTreeHelper.GetChildren(d))
            if (child is DependencyObject c) Walk(c, seen);
    }

    /// <summary>{l:T text} for text inside templates, which the walk can't reach.</summary>
    [MarkupExtensionReturnType(typeof(string))]
    public sealed class TExtension(string text) : MarkupExtension
    {
        public override object ProvideValue(IServiceProvider serviceProvider) => T(text);
    }

    // English as written → Polish.
    static readonly Dictionary<string, string> Pl = new()
    {
        // ---- window, sidebar, navbar
        ["Leonida Launcher"] = "Leonida Launcher",
        ["LAUNCHER"] = "LAUNCHER",
        ["Home"] = "Strona główna",
        ["About Leonida"] = "O Leonidzie",
        ["Library"] = "Biblioteka",
        ["Can I Run Leonida"] = "Czy uruchomię Leonidę",
        ["Your PC against the consoles"] = "Twój PC na tle konsol",
        ["Collapse or expand the sidebar"] = "Zwiń lub rozwiń panel boczny",
        ["Minimise"] = "Minimalizuj",
        ["Maximise"] = "Maksymalizuj",
        ["Close"] = "Zamknij",
        ["HOME"] = "START",
        ["PROJECT STATUS"] = "STAN PROJEKTU",
        ["PLATFORMS"] = "PLATFORMY",
        ["HOW IT WORKS"] = "JAK TO DZIAŁA",
        ["CREDITS"] = "TWÓRCY",
        ["BUILD"] = "BUDOWANIE",
        ["HISTORY"] = "HISTORIA",
        ["SETTINGS"] = "USTAWIENIA",
        ["Settings"] = "Ustawienia",
        ["Your profile"] = "Twój profil",
        ["Not built yet"] = "Jeszcze niezbudowana",
        ["Ready to play"] = "Gotowa do gry",
        ["Built"] = "Zbudowana",
        ["Running"] = "Działa",
        ["Leonida folder not found"] = "Nie znaleziono folderu Leonida",
        ["Building · {0}"] = "Budowanie · {0}",

        ["Leonida Launcher is already open."] = "Leonida Launcher jest już otwarty.",
        ["Something went wrong:\n\n{0}\n\nDetails are in {1}."] = "Coś poszło nie tak:\n\n{0}\n\nSzczegóły są w {1}.",
        ["A build is running. Stop it and close the launcher?\n\nA build stops the way Ctrl+C stops it and continues where it left off next time."] =
            "Trwa budowanie. Zatrzymać je i zamknąć launcher?\n\nBuild zatrzyma się jak po Ctrl+C i następnym razem wznowi się w tym miejscu.",
        ["The port is running. Stop it and close the launcher?"] = "Port działa. Zatrzymać go i zamknąć launcher?",

        // ---- Home
        ["SUPPORTED GAME"] = "OBSŁUGIWANA GRA",
        ["YOUR CHECKOUT"] = "TWOJA KOPIA REPOZYTORIUM",
        ["CONTRIBUTORS"] = "TWÓRCY",
        ["LINKS"] = "LINKI",
        ["Project information"] = "Informacje o projekcie",
        ["Project status"] = "Stan projektu",
        ["Platforms"] = "Platformy",
        ["How it works"] = "Jak to działa",
        ["Credits"] = "Twórcy",
        ["PEOPLE"] = "OSÓB",
        ["COMMITS"] = "COMMITÓW",
        ["Find someone"] = "Znajdź osobę",
        ["Loading…"] = "Wczytywanie…",
        ["commit author"] = "autor commitów",
        ["{0} people, {1} commits"] = "{0} osób, {1} commitów",
        ["Unavailable"] = "Niedostępne",
        ["Unknown"] = "Nieznana",
        ["No Leonida folder"] = "Brak folderu Leonida",
        ["Not a git checkout (a source download)"] = "To nie repozytorium git (pobrane źródła)",
        ["This section is not in your checkout's README.md."] = "Tej sekcji nie ma w README.md Twojej kopii.",
        ["From GitHub, as of {0}: everyone with commits in {1}. AI assistants and bots (Claude, Claude Code, Devin, Dependabot) are tools, not contributors, and are not listed."] =
            "Z GitHuba, stan na {0}: wszyscy z commitami w {1}. Asystenci AI i boty (Claude, Claude Code, Devin, Dependabot) to narzędzia, nie twórcy, więc ich tu nie ma.",
        ["GitHub can't be reached, so this is your checkout's git history (no profile pictures). AI assistants and bots are not listed."] =
            "Brak połączenia z GitHubem, więc to historia git Twojej kopii (bez zdjęć profilowych). Asystentów AI i botów tu nie ma.",
        ["Online features are off (Launcher settings), so this is your checkout's git history, without profile pictures."] =
            "Funkcje online są wyłączone (ustawienia launchera), więc to historia git Twojej kopii, bez zdjęć profilowych.",
        ["README.md was not found. Put Leonida.exe next to Build.bat in your Leonida folder, or choose the folder in Settings."] =
            "Nie znaleziono README.md. Umieść Leonida.exe obok Build.bat w folderze Leonida albo wskaż folder w ustawieniach.",

        // ---- Leonida page
        ["PC PORT · BUILT FROM YOUR COPY"] = "PORT NA PC · ZBUDOWANY Z TWOJEJ KOPII",
        ["More"] = "Więcej",
        ["Build options…"] = "Opcje budowania…",
        ["Update the build"] = "Zaktualizuj build",
        ["Run the build again: finished steps are kept, changed inputs (a new translator, other options) are redone."] =
            "Uruchom budowanie jeszcze raz: gotowe kroki zostają, a zmienione dane wejściowe (nowy translator, inne opcje) są robione od nowa.",
        ["Check my game files"] = "Sprawdź pliki gry",
        ["Check the game's imports (dry run)"] = "Sprawdź importy gry (próbnie)",
        ["Open the port's folder"] = "Otwórz folder portu",
        ["Open build logs"] = "Otwórz logi budowania",
        ["Open the Leonida folder"] = "Otwórz folder Leonida",
        ["Having trouble? Check your setup"] = "Masz problem? Sprawdź konfigurację",
        ["SHADERS"] = "SHADERY",
        ["RUNTIME"] = "RUNTIME",
        ["LAST BUILD"] = "OSTATNI BUILD",
        ["ready"] = "gotowe",
        ["not yet"] = "jeszcze nie",
        ["never"] = "nigdy",
        ["Building…"] = "Budowanie…",
        ["PLAY"] = "GRAJ",
        ["STOP"] = "ZATRZYMAJ",
        ["UPDATE BUILD"] = "ZAKTUALIZUJ",
        ["BUILDING{0}"] = "BUDOWANIE{0}",
        ["Starting"] = "Start",
        ["{0} · {1}. Press the button to follow it."] = "{0} · {1}. Naciśnij przycisk, aby śledzić postęp.",
        ["The port is running. Its output is in the BUILD tab."] = "Port działa. Jego wyjście jest w zakładce BUDOWANIE.",
        ["Built. The loader runs a title end to end only with a stand-in executable so far, so expect it to stop early."] =
            "Zbudowany. Loader na razie przechodzi grę od początku do końca tylko z zastępczym plikiem wykonywalnym, więc spodziewaj się wczesnego zatrzymania.",
        ["Leonida.exe needs to be next to Build.bat in your Leonida folder (or choose the folder in Settings)."] =
            "Leonida.exe musi leżeć obok Build.bat w folderze Leonida (albo wskaż folder w ustawieniach).",
        ["Builds the PC port from your own copy of the game, on this PC."] = "Buduje port na PC z Twojej własnej kopii gry, na tym komputerze.",
        ["just now"] = "przed chwilą",
        ["{0} min ago"] = "{0} min temu",
        ["{0} h ago"] = "{0} godz. temu",
        ["{0} d ago"] = "{0} dni temu",
        ["Not ready to start:"] = "Nie można jeszcze uruchomić:",
        ["{0} does not exist yet: the first build makes it."] = "{0} jeszcze nie istnieje: utworzy go pierwszy build.",
        ["Succeeded"] = "Udany", ["Failed"] = "Nieudany", ["Stopped"] = "Zatrzymany", ["Cancelled"] = "Anulowany",
        ["CouldNotStart"] = "Nie wystartował", ["Running "] = "W toku",

        // ---- Port problems (Core)
        ["Build the runtime stage first: it puts gta6rt.exe in the output folder."] = "Najpierw zbuduj etap runtime: umieszcza gta6rt.exe w folderze wyjściowym.",
        ["The output folder has no game_path.txt; build the runtime stage again."] = "W folderze wyjściowym nie ma game_path.txt; zbuduj etap runtime jeszcze raz.",
        ["{0} is missing from your game copy, so there is nothing to run yet."] = "W Twojej kopii gry brakuje {0}, więc na razie nie ma czego uruchomić.",
        ["The shaders are not translated yet (shaders stage)."] = "Shadery nie są jeszcze przetłumaczone (etap shaders).",
        ["A build or the port is already running."] = "Budowanie albo port już działa.",

        // ---- Build popup
        ["BUILD LEONIDA"] = "ZBUDUJ LEONIDĘ",
        ["From your own copy of the game, on this PC. A stopped build continues where it left off."] =
            "Z Twojej własnej kopii gry, na tym komputerze. Przerwane budowanie wznawia się tam, gdzie się zatrzymało.",
        ["YOUR GAME COPY"] = "TWOJA KOPIA GRY",
        ["Browse"] = "Przeglądaj",
        ["Find on this PC"] = "Znajdź na tym PC",
        ["Search where Build.bat searches: Downloads, Desktop, Documents, Games, your home folder and the top of each drive."] =
            "Szuka tam, gdzie Build.bat: w Pobranych, na Pulpicie, w Dokumentach, w Games, w folderze użytkownika i na początku każdego dysku.",
        ["Your own Xbox Series package, version 1.0.2.1: the folder that holds MicrosoftGame.Config, with its files readable on disk."] =
            "Twoja własna paczka Xbox Series w wersji 1.0.2.1: folder z plikiem MicrosoftGame.Config, z plikami czytelnymi na dysku.",
        ["WHERE THE PORT GOES"] = "GDZIE TRAFI PORT",
        ["Left empty: Documents\\Leonida. Recently used folders are in the list."] = "Puste: Dokumenty\\Leonida. Ostatnio używane foldery są na liście.",
        ["WHAT TO BUILD"] = "CO ZBUDOWAĆ",
        ["READY TO RUN"] = "GOTOWE DO URUCHOMIENIA",
        ["  recommended"] = "  zalecane",
        ["Check, shaders and the runtime: what running the port needs. About 30 GB."] = "Sprawdzenie, shadery i runtime: to, czego potrzeba do uruchomienia portu. Około 30 GB.",
        ["EVERYTHING"] = "WSZYSTKO",
        ["Adds the asset export (models, textures, audio) and the pipeline check. About 330 GB."] = "Dochodzi eksport zasobów (modele, tekstury, dźwięk) i sprawdzanie potoków. Około 330 GB.",
        ["CHECK MY GAME FILES"] = "SPRAWDŹ PLIKI GRY",
        ["Only check the copy against the file list; build nothing (--check)."] = "Tylko sprawdź kopię z listą plików; nic nie buduj (--check).",
        ["CHOOSE STAGES"] = "WYBIERZ ETAPY",
        ["Pick the stages yourself."] = "Wybierz etapy samodzielnie.",
        ["File names and sizes against game\\xbox-1.0.2.1.json. The copy is always checked first."] = "Nazwy i rozmiary plików według game\\xbox-1.0.2.1.json. Kopia zawsze jest sprawdzana najpierw.",
        ["Carve, unpack, disassemble with llvm-mc and translate to SPIR-V. The longest step (about 30 GB)."] = "Wycięcie, rozpakowanie, deasemblacja przez llvm-mc i tłumaczenie na SPIR-V. Najdłuższy krok (około 30 GB).",
        ["Models (OBJ), texture dictionaries and pools (DDS), audio banks. About 300 GB; running the port does not need it."] =
            "Modele (OBJ), słowniki i pule tekstur (DDS), banki dźwięku. Około 300 GB; do uruchomienia portu niepotrzebne.",
        ["Build the loader (gta6rt.exe) with Visual Studio and copy it to the output folder."] = "Zbuduj loader (gta6rt.exe) w Visual Studio i skopiuj go do folderu wyjściowego.",
        ["Create every translated shader as a pipeline on your GPU's driver and save a pipeline cache (not loaded by the runtime yet)."] =
            "Utwórz potok dla każdego przetłumaczonego shadera w sterowniku karty i zapisz cache potoków (runtime jeszcze go nie wczytuje).",
        ["Advanced options"] = "Opcje zaawansowane",
        ["WORK FOLDER  (--work)"] = "FOLDER ROBOCZY  (--work)",
        ["Intermediate files: hundreds of GB for a full build. Empty: 'work' inside the output folder."] = "Pliki pośrednie: setki GB przy pełnym buildzie. Puste: 'work' w folderze wyjściowym.",
        ["Also compare SHA-256 of every file  (--full-check, slow)"] = "Porównaj też SHA-256 każdego pliku  (--full-check, wolne)",
        ["Also decode the audio to WAV  (--wav; much larger, needs ffmpeg)"] = "Zdekoduj też dźwięk do WAV  (--wav; dużo większe, wymaga ffmpeg)",
        ["WAVE64 EMULATION"] = "EMULACJA WAVE64",
        ["auto (recommended)"] = "auto (zalecane)",
        ["auto checks your GPU: 32-wide GPUs (NVIDIA, Intel) need emulation; AMD runs wave64 natively. Changing it translates the shaders again."] =
            "auto sprawdza kartę: karty 32-wątkowe (NVIDIA, Intel) potrzebują emulacji; AMD wykonuje wave64 natywnie. Zmiana tłumaczy shadery od nowa.",
        ["PARALLEL JOBS  "] = "RÓWNOLEGŁE ZADANIA  ",
        ["Default"] = "Domyślnie",
        ["Half your CPU threads, as make_game.py chooses."] = "Połowa wątków procesora, tak jak wybiera make_game.py.",
        ["FALLBACKS FOR GPUS WITHOUT A FEATURE (leave off unless yours lacks it)"] = "OBEJŚCIA DLA KART BEZ DANEJ FUNKCJI (zostaw wyłączone, chyba że Twojej jej brakuje)",
        ["Without SignedZeroInfNanPreserve and DenormPreserve, for devices lacking shaderSignedZeroInfNanPreserveFloat32 or shaderDenormPreserveFloat16."] =
            "Bez SignedZeroInfNanPreserve i DenormPreserve, dla urządzeń bez shaderSignedZeroInfNanPreserveFloat32 lub shaderDenormPreserveFloat16.",
        ["Raw barycentrics from an interpolated input, for devices without VK_KHR_fragment_shader_barycentric."] = "Surowe barycentryki z interpolowanego wejścia, dla urządzeń bez VK_KHR_fragment_shader_barycentric.",
        ["Fold image_sample*_o offsets into the coordinate, for devices without VK_KHR_maintenance8."] = "Wlicza przesunięcia image_sample*_o do współrzędnej, dla urządzeń bez VK_KHR_maintenance8.",
        ["16-bit interpolation without PerVertexKHR; on by itself where the GPU needs it (MoltenVK)."] = "16-bitowa interpolacja bez PerVertexKHR; włącza się sama, gdy karta tego wymaga (MoltenVK).",
        ["Mesh shaders as compute shaders, for devices without VK_EXT_mesh_shader."] = "Shadery mesh jako shadery compute, dla urządzeń bez VK_EXT_mesh_shader.",
        ["128 push-constant bytes plus a uniform block; on by itself when the GPU's maxPushConstantsSize is too small."] = "128 bajtów push constants plus blok uniform; włącza się samo, gdy maxPushConstantsSize karty jest za małe.",
        ["EXISTING RUNTIME BUILD  (--runtime-build)"] = "GOTOWY BUILD RUNTIME  (--runtime-build)",
        ["A folder with gta6rt.exe and gta6rt_tls_host.dll, used instead of building them."] = "Folder z gta6rt.exe i gta6rt_tls_host.dll, używany zamiast ich budowania.",
        ["EXTRA ARGUMENTS"] = "DODATKOWE ARGUMENTY",
        ["Passed to make_game.py unchanged, for options this launcher does not know yet."] = "Przekazywane do make_game.py bez zmian, dla opcji, których launcher jeszcze nie zna.",
        ["THE COMMAND (THE SAME AS BUILD.BAT WITH THESE OPTIONS)"] = "POLECENIE (TO SAMO, CO BUILD.BAT Z TYMI OPCJAMI)",
        ["Copy command"] = "Kopiuj polecenie",
        ["Load saved options"] = "Wczytaj zapisane opcje",
        ["Load a preset…"] = "Wczytaj preset…",
        ["Save as preset…"] = "Zapisz jako preset…",
        ["Delete preset"] = "Usuń preset",
        ["Save as preset"] = "Zapisz jako preset",
        ["Name:"] = "Nazwa:",
        ["Save"] = "Zapisz",
        ["Cancel"] = "Anuluj",
        ["Choose the preset in the list first."] = "Najpierw wybierz preset z listy.",
        ["START BUILD"] = "ZACZNIJ BUDOWAĆ",
        ["CHECK FILES"] = "SPRAWDŹ PLIKI",
        ["Default: "] = "Domyślnie: ",
        ["  ·  {0} free"] = "  ·  wolne {0}",
        ["  ·  needs about {0} GB (less when an earlier build is reused)"] = "  ·  potrzeba ok. {0} GB (mniej, gdy wcześniejszy build zostanie użyty ponownie)",
        ["That drive looks too small for this choice: pick a folder on a bigger drive, or Ready to run."] = "Ten dysk wygląda na za mały dla tego wyboru: wybierz folder na większym dysku albo „Gotowe do uruchomienia”.",
        ["Looking for your copy where Build.bat looks…"] = "Szukam Twojej kopii tam, gdzie szuka Build.bat…",
        ["No copy found automatically: choose the folder that holds MicrosoftGame.Config."] = "Nie znaleziono kopii automatycznie: wskaż folder z MicrosoftGame.Config.",
        ["Found: {0} {1}."] = "Znaleziono: {0} {1}.",
        ["Found {0} copies: choose one below."] = "Znaleziono {0} kopie: wybierz jedną poniżej.",
        [" Packaged containers (.xvc/.msixvc) can't be used: the build needs the unpacked folder."] = " Spakowanych kontenerów (.xvc/.msixvc) nie da się użyć: build potrzebuje rozpakowanego folderu.",
        ["The search failed: "] = "Wyszukiwanie nie powiodło się: ",
        ["default ({0})"] = "domyślnie ({0})",

        // ---- Validation (Core)
        ["Choose your game folder (the one that holds MicrosoftGame.Config), or use Find copies."] = "Wskaż folder gry (ten z MicrosoftGame.Config) albo użyj wyszukiwania.",
        ["Select at least one stage."] = "Zaznacz co najmniej jeden etap.",
        ["Python 3.12 or newer was not found (Settings)."] = "Nie znaleziono Pythona 3.12 lub nowszego (ustawienia).",
        ["The Leonida folder was not found (Settings)."] = "Nie znaleziono folderu Leonida (ustawienia).",
        ["The port is running; close it before building."] = "Port działa; zamknij go przed budowaniem.",
        ["A build is already running."] = "Budowanie już trwa.",

        // ---- BUILD tab
        ["NO BUILD RUNNING"] = "NIC SIĘ NIE BUDUJE",
        ["Start one and follow every stage here."] = "Uruchom budowanie i śledź tu każdy etap.",
        ["BUILD…"] = "ZBUDUJ…",
        ["CANCEL"] = "ANULUJ",
        ["Stop the way Ctrl+C stops it. Every stage continues where it stopped the next time you build."] =
            "Zatrzymaj tak jak Ctrl+C. Każdy etap wznowi się w tym miejscu przy następnym budowaniu.",
        ["End the build and every process it started at once. Use Stop unless the build no longer reacts."] =
            "Natychmiast kończy budowanie i wszystkie uruchomione przez nie procesy. Używaj Zatrzymaj, chyba że budowanie przestało reagować.",
        ["ELAPSED"] = "CZAS",
        ["CPU"] = "CPU",
        ["BUILD RAM"] = "RAM BUILDU",
        ["PC RAM USED"] = "RAM PC W UŻYCIU",
        ["PROCESSES"] = "PROCESY",
        ["OUTPUT SIZE"] = "ROZMIAR WYJŚCIA",
        ["WARNINGS"] = "OSTRZEŻENIA",
        ["ERRORS"] = "BŁĘDY",
        ["Memory of every process the build runs"] = "Pamięć wszystkich procesów uruchomionych przez build",
        ["Processes running in the build: Python, translator shards, llvm-mc, compilers"] = "Procesy w buildzie: Python, procesy translatora, llvm-mc, kompilatory",
        ["Measured every minute while the build runs"] = "Mierzony co minutę podczas budowania",
        ["CPU · BUILD RAM (5 MIN)"] = "CPU · RAM BUILDU (5 MIN)",
        ["SHADERS · RDNA2 → SPIR-V"] = "SHADERY · RDNA2 → SPIR-V",
        ["TOTAL"] = "RAZEM",
        ["COMPILED"] = "SKOMPILOWANE",
        ["FAILED"] = "NIEUDANE",
        ["COMPILING NOW"] = "W KOMPILACJI",
        ["REMAINING"] = "POZOSTAŁO",
        ["SPEED"] = "TEMPO",
        ["AVG / SHADER"] = "ŚR. / SHADER",
        ["ETA (THIS STEP)"] = "DO KOŃCA (TEN KROK)",
        ["Translated and passed spirv-val: the .spv module is written"] = "Przetłumaczone i przeszły spirv-val: moduł .spv jest zapisany",
        ["Unsupported, a translator error, or rejected by spirv-val"] = "Nieobsługiwane, błąd translatora albo odrzucone przez spirv-val",
        ["Translator processes on a shader right now, one shader each"] = "Procesy translatora pracujące teraz nad shaderem, po jednym na proces",
        ["Shaders finished per second over the last minute, all processes together"] = "Shadery ukończone na sekundę w ostatniej minucie, wszystkie procesy razem",
        ["The time the translator processes ran, summed, over the shaders they finished"] = "Łączny czas pracy procesów translatora podzielony przez ukończone shadery",
        ["The remaining shaders at the current speed; this step only"] = "Pozostałe shadery przy obecnym tempie; tylko ten krok",
        ["CONSOLE"] = "KONSOLA",
        ["Search the output"] = "Szukaj w wyjściu",
        ["All lines"] = "Wszystkie linie",
        ["Warnings"] = "Ostrzeżenia",
        ["Errors"] = "Błędy",
        ["Steps only"] = "Tylko kroki",
        ["Follow"] = "Śledź",
        ["Copy, clear, save, open the log"] = "Kopiuj, wyczyść, zapisz, otwórz log",
        ["Copy (selected lines, or all shown)"] = "Kopiuj (zaznaczone linie albo wszystkie widoczne)",
        ["Clear the view (the log file keeps everything)"] = "Wyczyść widok (plik logu zachowuje wszystko)",
        ["Save the whole log as…"] = "Zapisz cały log jako…",
        ["Open the log file"] = "Otwórz plik logu",
        ["Open the logs folder"] = "Otwórz folder logów",
        ["Make the console bigger"] = "Powiększ konsolę",
        ["BUILDING"] = "BUDOWANIE", ["RUNNING"] = "DZIAŁA", ["SUCCEEDED"] = "UDANE", ["ENDED"] = "ZAKOŃCZONE",
        ["STOPPED"] = "ZATRZYMANE", ["CANCELLED"] = "ANULOWANE", ["COULD NOT START"] = "NIE WYSTARTOWAŁO",
        ["FAILED "] = "NIEUDANE", ["EXIT {0}"] = "KOD {0}",
        ["THE PORT"] = "PORT",
        ["STAGE {0} OF {1}: {2}"] = "ETAP {0} Z {1}: {2}",
        ["STARTING"] = "START",
        ["{0} OF {1} STAGES DONE"] = "UKOŃCZONO {0} Z {1} ETAPÓW",
        ["started {0}"] = "start {0}",
        [", ended {0}"] = ", koniec {0}",
        ["not run"] = "pominięty",
        ["waiting"] = "czeka",
        ["peak "] = "szczyt ",
        ["The last build {0} on {1} after {2} (HISTORY has its log)."] = "Ostatni build: {0}, {1}, po {2} (log jest w HISTORII).",
        ["{0} translator processes"] = "Procesy translatora: {0}",
        [" · now {0}"] = " · teraz {0}",
        [" · {0} of {1} shards kept from the run before (not counted again)"] = " · {0} z {1} części zachowanych z poprzedniego uruchomienia (nieliczone ponownie)",
        [" · disassembly {0}/{1}"] = " · deasemblacja {0}/{1}",
        [" ({0} reused)"] = " (ponownie użyte: {0})",
        ["DISASSEMBLY {0}%"] = "DEASEMBLACJA {0}%",
        ["Disassembling the native RDNA2 code with llvm-mc: {0} of {1}"] = "Deasemblacja natywnego kodu RDNA2 przez llvm-mc: {0} z {1}",
        [" ({0} reused from the run before)"] = " ({0} z poprzedniego uruchomienia)",
        [". The translation to SPIR-V follows; its numbers come from the translator."] = ". Potem tłumaczenie na SPIR-V; jego liczby pochodzą z translatora.",
        ["From the translator's summaries: {0} shaders, {1} compiled."] = "Z podsumowań translatora: {0} shaderów, {1} skompilowanych.",
        ["Nothing to translate: the shaders from the run before were reused."] = "Nic do tłumaczenia: użyto shaderów z poprzedniego uruchomienia.",
        ["The numbers appear when the translation starts; they come from the translator itself."] = "Liczby pojawią się, gdy ruszy tłumaczenie; pochodzą z samego translatora.",
        [" · pipelines {0} ok, {1} failed, {2} skipped"] = " · potoki: {0} ok, {1} nieudanych, {2} pominiętych",
        ["{0} lines"] = "{0} linii",
        [" · {0} shown"] = " · widocznych {0}",
        ["End the build and every process it started now?\n\nStop lets it finish its step cleanly."] =
            "Zakończyć teraz budowanie i wszystkie uruchomione przez nie procesy?\n\nZatrzymaj pozwala czysto dokończyć krok.",

        // ---- HISTORY
        ["Clear history"] = "Wyczyść historię",
        ["How long builds took (last 20)"] = "Ile trwały buildy (ostatnie 20)",
        ["  ·  {0} succeeded, fastest {1}"] = "  ·  udane: {0}, najszybszy {1}",
        ["No builds yet."] = "Jeszcze nie było buildów.",
        ["Started"] = "Start", ["Result"] = "Wynik", ["Took"] = "Czas", ["Stages done"] = "Ukończone etapy",
        ["Shaders compiled"] = "Skompilowane shadery", ["Peak RAM"] = "Szczyt RAM",
        ["Open its log"] = "Otwórz jego log",
        ["Build again with these options…"] = "Zbuduj ponownie z tymi opcjami…",
        ["Clear the build history? The log files stay."] = "Wyczyścić historię buildów? Pliki logów zostaną.",
        ["That log file is gone."] = "Tego pliku logu już nie ma.",
        ["failed in {0}"] = "nieudany w {0}",
        [" (failed in {0})"] = " (nieudany w {0})",
        ["{0} of {1}"] = "{0} z {1}",

        // ---- Settings page
        ["Back"] = "Wróć",
        ["Launcher settings"] = "Ustawienia launchera",
        ["Leonida settings"] = "Ustawienia Leonidy",
        ["LANGUAGE"] = "JĘZYK",
        ["The launcher's language. README pages and the build's output stay in English."] = "Język launchera. Strony README i wyjście budowania zostają po angielsku.",
        ["START PAGE"] = "STRONA STARTOWA",
        ["Where you left off"] = "Tam, gdzie skończyłeś",
        ["Start with the sidebar collapsed"] = "Uruchamiaj ze zwiniętym panelem bocznym",
        ["Appearance"] = "Wygląd",
        ["Your avatar"] = "Twój awatar",
        ["New random avatar"] = "Nowy losowy awatar",
        ["Builds"] = "Budowanie",
        ["Stop builds with Ctrl+C first (they resume next time)"] = "Zatrzymuj buildy najpierw przez Ctrl+C (wznowią się następnym razem)",
        ["then end them after"] = "a potem zakończ je po",
        ["seconds"] = "sekundach",
        ["Flash the taskbar when a build ends"] = "Migaj paskiem zadań, gdy build się skończy",
        ["Play a sound when a build ends"] = "Odtwórz dźwięk, gdy build się skończy",
        ["CONSOLE TEXT SIZE"] = "ROZMIAR TEKSTU KONSOLI",
        ["DELETE BUILD LOGS OLDER THAN"] = "USUWAJ LOGI BUILDÓW STARSZE NIŻ",
        ["days (0 keeps them all)"] = "dni (0 zachowuje wszystkie)",
        ["Online"] = "Online",
        ["Allow online features (GitHub profile pictures in Credits, the Can I Run checker)"] = "Zezwól na funkcje online (zdjęcia z GitHuba w Twórcach, sprawdzarka Czy uruchomię)",
        ["Off: nothing is fetched; Credits uses your git history and Can I Run shows only your hardware."] =
            "Wyłączone: nic nie jest pobierane; Twórcy korzystają z historii git, a Czy uruchomię pokazuje tylko Twój sprzęt.",
        ["Files"] = "Pliki",
        ["Settings file:"] = "Plik ustawień:",
        ["Restore defaults"] = "Przywróć domyślne",
        ["Open logs folder"] = "Otwórz folder logów",
        ["Restore every launcher setting and build option to its default? Presets and recent folders go too; the build history and logs stay."] =
            "Przywrócić domyślne wszystkie ustawienia launchera i opcje budowania? Presety i ostatnie foldery też znikną; historia buildów i logi zostaną.",
        ["Your setup"] = "Twoja konfiguracja",
        ["Check again"] = "Sprawdź ponownie",
        ["Checking…"] = "Sprawdzanie…",
        ["Something is missing."] = "Czegoś brakuje.",
        ["Everything the build needs is installed."] = "Wszystko, czego potrzebuje build, jest zainstalowane.",
        ["{0} thing(s) to install: see the red lines."] = "Do zainstalowania: {0}. Zobacz czerwone pozycje.",
        ["Asked from make_game.py itself: the checks its preflight makes before a build. Some are only needed for some stages."] =
            "Odpowiedź samego make_game.py: te same sprawdzenia, które robi przed buildem. Niektóre są potrzebne tylko dla części etapów.",
        ["for"] = "dla",
        ["   for "] = "   dla ",
        ["Play"] = "Gra",
        ["PLAY runs gta6rt.exe --run on game_scarlett_master.exe from your copy (runtime/README.md)."] = "GRAJ uruchamia gta6rt.exe --run na game_scarlett_master.exe z Twojej kopii (runtime/README.md).",
        ["OVERLAY FOLDER  (--overlay)"] = "FOLDER NAKŁADKI  (--overlay)",
        ["Where whatever the title writes goes, so the package is never modified. Empty: <game>\\overlay. Needed when the package is on a read-only drive."] =
            "Tu trafia wszystko, co zapisuje gra, więc paczka nigdy się nie zmienia. Puste: <gra>\\overlay. Potrzebne, gdy paczka jest na dysku tylko do odczytu.",
        ["MODS FOLDER  (--mods)"] = "FOLDER MODÓW  (--mods)",
        ["Mods to load (docs/modding.md). Empty: the mods folder next to gta6rt.exe."] = "Mody do wczytania (docs/modding.md). Puste: folder mods obok gta6rt.exe.",
        ["Vulkan D3D12 backend  (LEONIDA_D3D12=vulkan)"] = "Backend D3D12 na Vulkanie  (LEONIDA_D3D12=vulkan)",
        ["Trace D3D12 calls  (--trace-d3d12)"] = "Śledź wywołania D3D12  (--trace-d3d12)",
        ["STARTUP TRACE FILE  (--trace, a new file)"] = "PLIK ŚLADU STARTU  (--trace, nowy plik)",
        ["GAME ARGUMENTS  (after --)"] = "ARGUMENTY GRY  (po --)",
        ["Leonida folder, Python and environment"] = "Folder Leonida, Python i środowisko",
        ["LEONIDA FOLDER (WITH BUILD.BAT)"] = "FOLDER LEONIDA (Z BUILD.BAT)",
        ["Change"] = "Zmień",
        ["Found by itself when Leonida.exe is next to Build.bat."] = "Znajdowany automatycznie, gdy Leonida.exe leży obok Build.bat.",
        ["PYTHON"] = "PYTHON",
        ["Empty: found as Build.bat finds it (py -3.14 … py -3, then python)."] = "Puste: znajdowany tak jak przez Build.bat (py -3.14 … py -3, potem python).",
        ["ENVIRONMENT VARIABLES FOR BUILDS AND PLAY"] = "ZMIENNE ŚRODOWISKOWE DLA BUILDÓW I GRY",
        ["One NAME=value per line, e.g. LLVM_MC=C:\\path\\to\\llvm-mc.exe. A variable left out keeps the value Windows gives it."] =
            "Jedna NAZWA=wartość na linię, np. LLVM_MC=C:\\ścieżka\\do\\llvm-mc.exe. Pominięta zmienna zachowuje wartość z Windows.",
        ["Apply and check again"] = "Zastosuj i sprawdź ponownie",
        ["Python {0}: {1}\n(found as `{2}`, the way Build.bat finds it)"] = "Python {0}: {1}\n(znaleziony jako `{2}`, tak jak przez Build.bat)",
        ["Not found: install Python 3.12 or newer."] = "Nie znaleziono: zainstaluj Pythona 3.12 lub nowszego.",
        ["Port: {0}{1}\nWork files: {2}{3}"] = "Port: {0}{1}\nPliki robocze: {2}{3}",
        ["Restart Leonida Launcher to use this folder."] = "Uruchom ponownie Leonida Launcher, aby użyć tego folderu.",

        // ---- Can I Run
        ["CAN I RUN LEONIDA?"] = "CZY URUCHOMIĘ LEONIDĘ?",
        ["Checking your PC…"] = "Sprawdzam Twój komputer…",
        ["YOUR PC"] = "TWÓJ PC",
        ["Detect again"] = "Wykryj ponownie",
        ["Read the processor, graphics and memory from Windows again and compare once more"] = "Odczytaj ponownie procesor, grafikę i pamięć z Windows i porównaj jeszcze raz",
        ["PROCESSOR"] = "PROCESOR",
        ["GRAPHICS"] = "GRAFIKA",
        ["MEMORY"] = "PAMIĘĆ",
        ["COMPARED WITH EVERY CONSOLE ON THE LIST"] = "PORÓWNANIE ZE WSZYSTKIMI KONSOLAMI Z LISTY",
        ["Open the checker in your browser"] = "Otwórz sprawdzarkę w przeglądarce",
        ["Asking the checker…"] = "Pytam sprawdzarkę…",
        ["Comparing…"] = "Porównuję…",
        ["Your PC is at or above every console on the list."] = "Twój PC dorównuje lub przewyższa każdą konsolę z listy.",
        ["Your PC matches the {0}."] = "Twój PC dorównuje: {0}.",
        ["Your PC is below the consoles on the list."] = "Twój PC jest poniżej konsol z listy.",
        ["{0} references, as the checker lists them. Hover a line for the reference hardware."] = "Odniesień: {0}, tak jak w sprawdzarce. Najedź na linię, aby zobaczyć sprzęt odniesienia.",
        ["The checker can't be reached right now."] = "Sprawdzarka jest teraz niedostępna.",
        ["Your PC is detected above, but {0}. Are you online? Open the checker in your browser to compare by hand."] =
            "Twój PC jest wykryty powyżej, ale {0}. Czy masz internet? Otwórz sprawdzarkę w przeglądarce, aby porównać ręcznie.",
        ["Online features are off in Launcher settings, so only your hardware is shown."] = "Funkcje online są wyłączone w ustawieniach launchera, więc widać tylko Twój sprzęt.",
        ["Verdicts and reference ranges: the community checker by assttro, run live. They compare benchmark bands, not a Leonida requirement; nothing is playable yet."] =
            "Werdykty i zakresy odniesienia: społecznościowa sprawdzarka autorstwa assttro, uruchamiana na żywo. Porównują przedziały wydajności, a nie wymagania Leonidy; na razie nic nie da się grać.",
        ["{0} cores · {1} threads"] = "{0} rdzeni · {1} wątków",
        ["{0} threads"] = "{0} wątków",
        [" video memory"] = " pamięci wideo",
        [" · driver {0}"] = " · sterownik {0}",
        ["{0} module(s)"] = "moduły: {0}",
        [" · type not reported, DDR4 assumed"] = " · typ niepodany, przyjęto DDR4",
        // the checker's own words (its page is in English)
        ["Above"] = "Powyżej", ["In range"] = "W zakresie", ["Below"] = "Poniżej", ["Uncertain"] = "Niepewne",
        ["Community estimate for 1080p Low"] = "Szacunek społeczności dla 1080p Low",
        ["Your PC could run Leonida"] = "Twój PC mógłby uruchomić Leonidę",
        ["Your PC may struggle to run Leonida"] = "Twój PC może mieć problem z Leonidą",
        ["Your PC’s outlook is unclear"] = "Trudno ocenić Twój PC",
        ["Above the {0} hardware range"] = "Powyżej zakresu sprzętu {0}",
        ["Around the {0} hardware range"] = "Na poziomie sprzętu {0}",
        ["Below the {0} hardware range"] = "Poniżej zakresu sprzętu {0}",
        ["Partial {0} comparison"] = "Częściowe porównanie z {0}",
        ["Uncertain {0} comparison"] = "Niepewne porównanie z {0}",
        ["Your PC is outside Leonida’s x64 target"] = "Twój PC nie jest platformą x64, na którą celuje Leonida",
    };
}
