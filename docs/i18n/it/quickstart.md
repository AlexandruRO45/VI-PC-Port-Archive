# Leonida — Guida rapida (`it`)

> [!NOTE]
> La documentazione in inglese ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) e
> [`docs/README.md`](../../README.md)) è la **fonte di riferimento**. Questa guida raccoglie solo le istruzioni
> stabili di configurazione e compilazione (sincronizzata con il commit `aa2988cc`). Per lo stato quotidiano del
> progetto consulta il [`README.md`](../../../README.md).

## Che cos'è Leonida

Leonida è un progetto di ricompilazione statica di Grand Theft Auto VI (Xbox Series) per eseguirlo nativamente su
Windows, **costruito sul tuo PC a partire dalla tua copia**. Il codice del gioco viene caricato ed eseguito
nativamente (la CPU di Xbox Series è x86-64 come quella di un PC), il software di sistema della console è
reimplementato in shim e i programmi GPU RDNA2 della console vengono tradotti in SPIR-V per Vulkan. Modelli,
texture e audio sono letti dal tuo pacchetto.

> [!IMPORTANT]
> Questo repository **non contiene il gioco, chiavi né eseguibili**, e non spiega come ottenerli. Nulla qui
> decifra o aggira protezioni. Tutto ciò che la build produce (shader tradotti, asset estratti, cartella di
> lavoro e salvataggi) deriva dalla tua copia e **deve restare solo sul tuo computer**: non allegare né
> condividere questi file in issue o pull request (vedi [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

Il progetto è in fase iniziale: **il gioco non si avvia ancora**.

## Prerequisiti

- **La tua copia** del pacchetto Xbox Series, versione `1.0.2.1` (la cartella che contiene `MicrosoftGame.Config`
  e gli archivi `.rpf` come file normali su disco).
- **Windows 10/11 x64** con circa **400 GB liberi** per i file di lavoro (possono stare su un altro disco con
  `--work`).
- [Python 3.12+](https://www.python.org/downloads/) (spunta «Add python.exe to PATH»).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/), per `spirv-as` / `spirv-val` (e `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) con il **target AMDGPU**, per `llvm-mc`:
  - La build ufficiale di LLVM per Windows **non** include il target AMDGPU.
  - Per compilare un `llvm-mc.exe` nativo con AMDGPU, esegui questo comando dalla radice del repository
    (non serve WSL):

    ```powershell
    powershell -ExecutionPolicy Bypass -File scripts\build_llvm_mc.ps1
    ```

    Lo script richiede `git` e gli strumenti C++ di Visual Studio (vedi sotto), imposta `LLVM_MC` per il tuo
    utente e ti chiede di aprire un nuovo terminale per caricare la variabile. Richiede circa 20-40 minuti.
  - Più rapido, se usi [conda](https://docs.conda.io/): `conda install -c conda-forge llvm-tools` (la sua build
    per Windows include il target AMDGPU; [`docs/testing.md`](../../testing.md) ha usato la versione 23.1.6), poi
    imposta `LLVM_MC` con il percorso completo del `llvm-mc.exe` installato (con l'ambiente attivo, `where llvm-mc`
    te lo mostra).
  - In alternativa, installa `llvm-21` in WSL (Ubuntu) e imposta la variabile con il comando completo, nella
    stessa sessione PowerShell da cui avvierai `make_game.py`:

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Vedi [`docs/design.md`](../../design.md#disassembling-shaders) per i dettagli.
  - Su Linux basta avere `llvm-mc` nel `PATH`.
- **Visual Studio 2022** con il carico di lavoro **Sviluppo di applicazioni desktop con C++** e il **Windows SDK
  `10.0.26100` o successivo**, necessari per la fase `runtime` (il runtime usa le root signature versione 1.2,
  che gli SDK più vecchi non dichiarano).
- [ffmpeg](https://ffmpeg.org/download.html) nel `PATH` *(facoltativo: decodifica gli stream XMA2 ed è
  obbligatorio con `--wav`)*.

### Poco spazio su disco

I circa 400 GB servono per la build completa. Per *eseguire* il port, `--stages check,shaders,runtime` richiede
circa 30 GB: la fase `assets` (circa 370 GB) serve solo per gli strumenti sugli asset, e `--wav` aggiunge diverse
volte la dimensione degli Ogg Opus (il controllo iniziale te lo propone quando è la fase `assets` a non entrare).
`--work` e `--out` possono stare su un altro disco. Il runtime usa il pacchetto direttamente dove si trova, quindi
se il disco del pacchetto è in sola lettura serve `gta6rt --overlay <cartella scrivibile>` (`--overlay` è
un'opzione di `gta6rt`, non di `make_game.py`). Non cancellare i file `.s` in `work/shaders/dxbc/disasm`: se li
elimini insieme a `work/shaders/spirv/index.json`, la fase `shaders` può risultare completata con zero shader. Una
cartella `disasm` che è un collegamento simbolico viene rifiutata quando la fase deve svuotarla.

## Come compilare

### Modo semplice (Windows)

1. Installa i prerequisiti qui sopra.
2. Scarica il progetto **con il suo sottomodulo**:

   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```

   Il «Download ZIP» di GitHub non include `third_party/rpf9-explorer`. La build lo scarica da sola se `git` è
   installato, altrimenti ti dice cosa fare.
3. Fai doppio clic su **`Build.bat`**.

Lo script cerca da solo la tua copia del gioco: Download, Desktop, Documenti (anche dove Windows li sposta, per
esempio in OneDrive), Giochi, la cartella utente e la radice di ogni disco, fino a due livelli di profondità.
Riconosce la versione dalle dimensioni di pochi file, senza leggere niente di pesante. Se hai più copie ti chiede
quale usare, e se non ne trova nessuna ti chiede di incollare il percorso della cartella. Prima di partire
verifica che tutto il necessario sia installato e che ci sia abbastanza spazio libero, e ti elenca in un colpo
solo ciò che manca. Il risultato va in `Documents\Leonida`. Se la build viene interrotta, riprende da dove si era
fermata.

### Dal terminale (`make_game.py`)

Stesso comportamento, con più controllo:

```powershell
python -I make_game.py --find                          # elenca le copie trovate su questo PC
python -I make_game.py --pick                          # scegli la cartella del gioco in una finestra
python -I make_game.py --check                         # verifica la tua copia, senza compilare
python -I make_game.py --check --full-check            # confronta anche lo SHA-256 di ogni file (lento)
python -I make_game.py --game "D:\percorso\del\pacchetto" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # solo alcune fasi
python -I make_game.py --game "D:\percorso\del\pacchetto" --wav   # decodifica anche Opus in PCM WAV (grande)
python -I make_game.py --wave64-emulate off            # auto (predefinito), on oppure off
```

Le fasi vengono eseguite in quest'ordine: `check` → `shaders` → `assets` → `runtime` → `pipelines`.
`pipelines` crea ogni shader tradotto sul driver della tua GPU, controlla che venga compilato, segnala quelli che
deve saltare e salva una cache delle pipeline in `pipeline_cache.bin`, accanto a `gta6rt.exe` in `--out`. All'avvio
il runtime la usa per precaricare la propria cache delle pipeline Vulkan, se è stata creata sullo stesso
dispositivo. La traduzione degli shader è il passaggio più lungo: usa `--jobs` processi (per impostazione
predefinita metà dei thread della CPU).

- **`--wave64-emulate`**: decide se gli shader compute e mesh wave64 vengono tradotti in modo da essere esatti
  con qualunque dimensione di subgruppo. Le GPU NVIDIA e Intel hanno subgruppi da 32 lane e ne hanno bisogno.
  Il valore predefinito `auto` interroga `vulkaninfo` (parte del Vulkan SDK) e attiva l'emulazione se anche una
  sola GPU non ha subgruppi da 64 o se la loro dimensione è sconosciuta. Controlla anche che la GPU del runtime
  possa eseguire gli shader compute e mesh a 64 lane; con `--mesh-compute` basta il supporto per i compute. Con il
  driver AMD per Windows, `auto` usa wave64 nativo su RDNA1 e RDNA2 quando la GPU del runtime lo supporta, anche
  se un'altra GPU è a 32 lane o se `vulkaninfo` non è disponibile, perché quel driver può andare in crash con il
  codice LDS emulato (#857). Sulle RDNA1 senza mesh shader la scelta si basa sulla fase compute, che può eseguire
  wave64 anche senza il supporto nativo ai mesh shader. La modalità viene salvata accanto agli shader tradotti, e
  cambiarla li traduce di nuovo; di solito `auto` riesce a riusare gli shader già emulati, mentre l'eccezione per
  il driver AMD su Windows li ricostruisce in modalità nativa. Usa `--wave64-emulate on` per forzare l'emulazione
  oppure `off` per forzare wave64 nativo. Vedi «Wave64 emulation» in [`docs/design.md`](../../design.md).
- **Opzioni di ripiego** (inoltrate al traduttore, salvate accanto agli shader; cambiarle li traduce di nuovo):
  `--no-float-controls` (toglie `SignedZeroInfNanPreserve`, per i dispositivi su cui il controllo delle pipeline
  salta uno shader senza `shaderSignedZeroInfNanPreserveFloat32`), `--bary-fallback` (GPU senza
  `VK_KHR_fragment_shader_barycentric`), `--texel-offset-fallback` (GPU senza `VK_KHR_maintenance8`) e
  `--mesh-compute` (GPU senza `VK_EXT_mesh_shader`). Sono documentate in [`docs/design.md`](../../design.md).
- **Ripetizioni automatiche**: la build registra quale comando e quale versione di `llvm-mc` hanno prodotto il
  disassemblato, quali sorgenti del traduttore hanno prodotto gli shader e quale decoder (il commit di
  `third_party/rpf9-explorer` e i sorgenti di `tools/rpf9dump`) ha esportato modelli e texture di ogni archivio.
  Se uno di questi cambia, rifà solo il lavoro interessato: disassemblaggio e traduzione, solo la traduzione,
  oppure l'export di quell'archivio. Le cartelle di lavoro più vecchie, senza queste informazioni, vengono
  aggiornate una volta dopo l'aggiornamento del progetto.
- **`--work`**: conviene puntarlo a un disco con almeno 400 GB liberi.
- **`--wav`**: disattivato di default perché il WAV PCM a 16 bit occupa diverse volte più dell'Ogg Opus. Decodifica
  gli stream Ogg Opus in `work\assets\audio\wav` dopo l'elaborazione dei banchi audio; quando una build riprende,
  i file WAV già presenti vengono saltati.
- **Linux e Wine/Proton**: `make_game.py` funziona anche su Linux. Per la fase `runtime` fuori da Windows passa
  `--runtime-build <cartella>` con una build Windows (`gta6rt.exe` e `gta6rt_tls_host.dll`, per esempio
  l'artefatto `runtime-windows` della CI scompattato): lo script copia i file in `--out` e scrive `gta6rt.sh`,
  che avvia `gta6rt.exe` con `wine` (o con il Wine indicato dalla variabile `WINE`; per Proton, il suo
  `files/bin/wine`). La stessa opzione salta la compilazione anche su Windows. In alternativa, su Linux la fase
  `runtime` cross-compila il loader in `runtime/build/windows` con clang-cl e lld-link (LLVM 19+, esclusi 22.1.7 e
  22.1.8; nomi semplici, oppure tutti e cinque gli strumenti LLVM con lo stesso suffisso di versione, come
  `clang-cl-21`) contro il CRT MSVC e l'SDK di Windows preparati da [xwin](https://github.com/Jake-Shadle/xwin), in
  `XWIN_DIR` oppure in `~/.local/share/xwin-sdk`: vedi
  [`docs/testing.md`](../../testing.md#the-windows-runtime-on-linux-under-wine-or-proton). Senza questi strumenti
  la fase viene esclusa da quelle predefinite, e se la richiedi ti elenca cosa manca prima di iniziare. Su Linux la
  fase `pipelines` usa `vk_pipeline_check` (in `runtime/build/linux` o `runtime/build`) e, se manca, lo compila da
  sola prima di qualsiasi altro lavoro.

## Problemi comuni

- **Hai scaricato con «Download ZIP» di GitHub**: lo ZIP non include il sottomodulo `third_party/rpf9-explorer`.
  Usa `git clone --recursive` oppure esegui `git submodule update --init --recursive` nella cartella del
  repository.
- **`llvm-mc` rifiuta `gfx1013` / AMDGPU**: probabilmente hai l'installer ufficiale di LLVM per Windows. Segui
  una delle opzioni LLVM descritte nei prerequisiti.
- **Errore su `D3D12_ROOT_SIGNATURE_DESC2` durante la build del runtime**: aggiorna il Windows SDK alla versione
  `10.0.26100` o successiva dal Visual Studio Installer oppure con
  `winget install Microsoft.WindowsSDK.10.0.26100`, elimina la cartella di build e riconfigura.
- **Spazio esaurito sul disco `C:`**: passa `--work "D:\leonida-work"` per spostare i file temporanei su un'altra
  partizione.

## Come contribuire ed eseguire i test GPU

1. Controlla le pull request aperte e l'[issue di coordinamento][coord] per evitare lavoro duplicato. Riserva
   l'attività scrivendo `claiming: <ID>` (oppure una descrizione di una riga se non esiste ancora un ID) prima
   di aprire la pull request.
2. Esegui i test pertinenti con input sintetici. Per esempio, per il verificatore dei manifest:
   `python -I tests/test_manifest_verifier.py` dalla radice del repository. La suite del traduttore si esegue
   con `python -m unittest discover -s tests/semantic -v`. I file di test senza blocco `__main__` o con
   `import pytest` (richiedono i pacchetti in `tests/requirements.txt`) si eseguono invece con
   `python -m pytest tests/test_<nome>.py`, come descritto in [`AGENTS.md`](../../../AGENTS.md).
3. Nella pull request indica l'ID dell'attività, cosa hai provato e, se hai eseguito test GPU, quale GPU e
   quale driver. Le regole sono in [`CONTRIBUTING.md`](../../../CONTRIBUTING.md). Se hai usato un assistente
   IA, dichiaralo nella pull request, come richiesto da [`AGENTS.md`](../../../AGENTS.md).

**Mai** allegare a issue o pull request file del gioco, asset estratti o shader generati: condividi solo
risultati (conteggi, esito dei test, classi di errore).

La [guida ai test GPU](../../testing.md) spiega come eseguirli senza i file del gioco. I risultati dei test GPU
vanno pubblicati nell'[issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
