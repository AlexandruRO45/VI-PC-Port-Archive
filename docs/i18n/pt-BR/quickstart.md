# Leonida — Guia rápido (`pt-BR`)

> [!NOTE]
> A documentação em inglês ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) e
> [`docs/README.md`](../../README.md)) é a **referência oficial**. Este guia reúne apenas instruções estáveis de
> configuração e compilação (sincronizado com o commit `3ca3c472`). Para o status diário do projeto, consulte o
> [`README.md`](../../../README.md).

## O que é o Leonida

O Leonida é um projeto de recompilação estática do Grand Theft Auto VI (Xbox Series) para rodar nativamente no
Windows, construído **no seu próprio PC a partir da sua própria cópia**. Os shaders RDNA2 do console são
traduzidos para SPIR-V (Vulkan), os assets são lidos diretamente do seu pacote e o software de sistema do
console é reimplementado em shims.

> [!IMPORTANT]
> Este repositório **não contém o jogo, chaves nem executável**, e não explica como obtê-los. Nada aqui
> descriptografa ou contorna proteções. Tudo o que o build gera (shaders traduzidos, assets extraídos, pasta de
> trabalho e saves) vem da sua cópia e **deve ficar apenas na sua máquina** — nunca anexe nem compartilhe esses
> arquivos em issues ou pull requests (veja [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

O projeto está em fase inicial: **o jogo ainda não dá boot**.

## Pré-requisitos

- **Sua própria cópia** do pacote Xbox Series, versão `1.0.2.1` (pasta com `MicrosoftGame.Config` e os arquivos `.rpf` legíveis em disco).
- **Windows 10/11 x64** com cerca de **30 GB livres** para preparar só o runtime ou **400 GB** para o build completo.
  Os tamanhos são os que o Explorador do Windows mostra (GiB): 400 GiB equivalem a cerca de 430 GB decimais.
- [Python 3.12+](https://www.python.org/downloads/) (marque *"Add python.exe to PATH"*).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/) (`spirv-as`, `spirv-val` e `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) com o **target AMDGPU** (`llvm-mc`):
  - O binário oficial do LLVM para Windows **não** inclui o target AMDGPU.
  - Para compilar um `llvm-mc.exe` nativo com AMDGPU, rode este comando a partir da raiz do repositório.

    ```powershell
    powershell -ExecutionPolicy Bypass -File scripts\build_llvm_mc.ps1
    ```

    O script requer `git` e as ferramentas C++ do Visual Studio (sem WSL), configura `LLVM_MC` para o seu
    usuário e pede que você abra um novo terminal para carregar a variável. Leva cerca de 20-40 minutos.
  - Caminho mais rápido, se você usa [conda](https://docs.conda.io/): `conda install -c conda-forge llvm-tools`
    (o build Windows dele inclui o target AMDGPU; o [`docs/testing.md`](../../testing.md) usou a versão 23.1.6).
    Depois defina `LLVM_MC` com o caminho completo do `llvm-mc.exe` instalado. Com o ambiente ativo,
    `where.exe llvm-mc` mostra o caminho (no PowerShell, `where` sozinho é o `Where-Object`).
  - Se usar o WSL, instale `llvm-21` no Ubuntu e defina a variável na mesma sessão do PowerShell em que vai
    executar `make_game.py`.

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Veja [`docs/design.md`](../../design.md#disassembling-shaders) para mais detalhes.
  - No Linux, basta ter `llvm-mc` no `PATH`.
- **Visual Studio 2022** com *Desenvolvimento para desktop com C++* e **Windows SDK `10.0.26100` ou mais recente** (necessário para o estágio `runtime`, que usa root signatures 1.2).
- [ffmpeg](https://ffmpeg.org/download.html) no `PATH` *(opcional: decodifica streams XMA2 e é exigido ao usar `--wav`)*.

**Este PC aguenta?** `python -I tools/pc_checker.py --cli` detecta CPU, GPU e RAM e compara com os perfis de
referência antes de um build longo: `--profile 1`, o padrão, equivale a um PC no nível do Xbox Series X, e
`--profile 2` é uma estimativa da comunidade para 1080p com gráficos no Low. `python -I tools/pc_checker.py --check --json`
imprime o mesmo veredito em JSON e o devolve como código de saída (0 bom, 1 fraco, 2 desconhecido).

Pouco espaço em disco? Veja [Build com pouco espaço em disco](#build-com-pouco-espaço-em-disco).

## Como compilar

### Modo simples (Windows)

1. Instale os pré-requisitos acima.
2. Clone o repositório **com o submódulo**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
   O "Download ZIP" do GitHub não inclui `third_party/rpf9-explorer`. O build baixa esse submódulo sozinho se o
   `git` estiver instalado e, se não estiver, explica o que fazer.
3. Abra a pasta `Leonida` e dê duplo clique em **`Build.bat`**.

O script procura automaticamente sua cópia (Downloads, Desktop, Documentos, Games, pasta de usuário e raiz dos
discos até dois níveis), valida os pré-requisitos e o espaço livre e grava o resultado em `Documents\Leonida`.
Se o build for interrompido, basta rodar novamente para retomar de onde parou.

### Pelo terminal (`make_game.py`)

```powershell
python -I make_game.py --find                          # lista as cópias encontradas neste PC
python -I make_game.py --pick                          # escolhe a pasta do jogo em uma janela de diálogo
python -I make_game.py --check                         # verifica os arquivos do jogo sem compilar
python -I make_game.py --check --full-check            # também compara o SHA-256 de cada arquivo (lento)
python -I make_game.py --game "D:\caminho\do\pacote" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # roda apenas os estágios escolhidos
python -I make_game.py --game "D:\caminho\do\pacote" --wav  # também decodifica Ogg Opus para PCM WAV (grande)
python -I make_game.py --wave64-emulate off            # auto (padrão), on ou off; veja abaixo
```

Estágios: `check` sempre roda primeiro; depois `shaders`, `assets` e `runtime` rodam juntos quando selecionados (um
não depende do outro). `pipelines` começa quando os shaders traduzidos ficam prontos (e, se a execução inclui
`runtime`, depois que ele recompila o `vk_pipeline_check`), então pode rodar em paralelo com o resto de `assets`.
O `pipelines` cria cada shader traduzido no driver da sua GPU, informa os que precisou pular e salva um cache em
`pipeline_cache.bin` dentro de `--out` (o runtime ainda não carrega esse cache, então ele não acelera o primeiro boot).

Num terminal interativo, uma tela de configuração do LEONIDA permite ajustar os jobs e se os estágios rodam em
paralelo; depois, `1`–`4` / `Tab` alternam o log de cada estágio, e os logs completos ficam em `work/logs/`. Use
`--no-tui` (ou redirecione a saída) para o console simples.

- **`--wave64-emulate`**: emula shaders compute/mesh wave64 em GPUs com subgrupos de 32 lanes (NVIDIA e Intel). O padrão `auto` consulta `vulkaninfo` e ativa a emulação se alguma GPU não tiver subgrupos de 64 lanes ou se o tamanho for desconhecido, verificando também se a GPU do runtime suporta compute e mesh em 64 lanes (com `--mesh-compute`, apenas o suporte em compute é necessário). No driver Windows da AMD, `auto` usa wave64 nativo em GPUs RDNA1 e RDNA2 quando a GPU do runtime suporta (para placas RDNA1 sem suporte a mesh shaders, a decisão usa o estágio compute), mesmo que outra GPU seja de 32 lanes ou que `vulkaninfo` não esteja disponível, pois o compilador desse driver pode falhar em código LDS emulado (#857). O modo escolhido fica salvo junto aos shaders traduzidos, e mudá-lo traduz os shaders de novo; use `--wave64-emulate on` para forçar a emulação ou `--wave64-emulate off` para forçar wave64 nativo.
- **Flags de fallback do tradutor**: `--no-float-controls` (omite `SignedZeroInfNanPreserve`), `--bary-fallback` (lê baricêntricas de um input interpolado sem `VK_KHR_fragment_shader_barycentric`), `--texel-offset-fallback` (incorpora offsets `image_sample*_o` na coordenada sem `VK_KHR_maintenance8`) e `--mesh-compute` (traduz mesh shaders como compute sem `VK_EXT_mesh_shader`; liga sozinho no driver Vulkan da Intel para Windows, cujos mesh draws nativos travam, a menos que você passe `--native-mesh`) são repassadas ao tradutor e salvas junto aos shaders; alterá-las retraduz os shaders (veja [`docs/design.md`](../../design.md)).
- **`--work`**: pode ficar em outro disco (veja [Build com pouco espaço em disco](#build-com-pouco-espaço-em-disco)); o estágio `shaders` é o mais demorado e usa metade das threads da CPU por padrão (ajustável com `--jobs`).
- **`--wav`**: fica desativado por padrão porque WAV PCM de 16 bits é várias vezes maior que Ogg Opus; quando ativado, decodifica os streams Ogg Opus para `work\assets\audio\wav` e pula arquivos WAV já existentes ao retomar um build.

### Build com pouco espaço em disco

Para preparar o runtime e os shaders com cerca de **30 GB livres**, execute na raiz do repositório:

```powershell
python -I make_game.py --stages check,shaders,runtime
```

O valor de ~400 GB é para o build completo. O estágio `assets` (~370 GB) serve apenas às ferramentas de extração, e
`--wav` adiciona várias vezes o tamanho dos arquivos Ogg Opus (a verificação inicial mostra essa escolha quando é o
estágio `assets` que não cabe). Você pode apontar `--work` e `--out` para outro disco. O runtime mapeia o pacote no
lugar, então um pacote em unidade somente leitura precisa de `gta6rt --overlay <pasta gravável>`; `--overlay` é uma
opção do `gta6rt`, não do `make_game.py`.

Mantenha os arquivos `.s` em `work/shaders/dxbc/disasm`: a próxima execução os confere com a lista em
`disasm.done`, desmonta de novo os que faltarem (todos, se o `disasm.done` for anterior a essa lista) e depois
traduz todos os shaders de novo. Uma pasta `disasm` que seja link simbólico é recusada quando o estágio precisa
limpá-la.

### Diagnóstico do Vulkan no Linux

Com CMake 3.24+, um compilador C++20, os cabeçalhos de desenvolvimento e o carregador do Vulkan instalados,
rode na raiz do repositório:

```sh
cmake -S runtime -B runtime/build/linux
cmake --build runtime/build/linux --parallel
./runtime/build/linux/vk_pipeline_check /caminho/para/spirv
```

Passe a pasta que contém `index.json` e os shaders SPIR-V traduzidos. O estágio `pipelines` de `make_game.py`
usa esse verificador em `runtime/build/linux` ou `runtime/build`. Se ainda não existir, o script o compila
antes de iniciar os estágios.

### Runtime no Linux com Wine ou Proton

No Linux, `make_game.py` pode compilar o runtime Windows em `runtime/build/windows` com `clang-cl`,
`lld-link` e o SDK Windows com o CRT do MSVC instalado via [xwin](https://github.com/Jake-Shadle/xwin).
O script procura o SDK em `XWIN_DIR` ou `~/.local/share/xwin-sdk`. Consulte o
[guia de configuração](../../testing.md#the-windows-runtime-on-linux-under-wine-or-proton) para instalar
as ferramentas e conferir as versões do LLVM conhecidas por causar problemas. Sem essas dependências,
o estágio `runtime` fica fora da execução padrão no Linux.

Você também pode reutilizar um build Windows já extraído, como o artefato `runtime-windows` do workflow
Runtime. Passe a pasta que contém `gta6rt.exe` e `gta6rt_tls_host.dll`:

```sh
python3 -I make_game.py --runtime-build ~/Downloads/runtime-windows
```

Nos dois casos, o script copia os arquivos do runtime para o diretório indicado por `--out` e gera `gta6rt.sh` nessa pasta.
O script inicia o runtime com `wine` por padrão. Para usar o Proton, aponte `WINE` para o `files/bin/wine`
da instalação. No Windows, `--runtime-build` também permite reutilizar um build existente.

## Problemas comuns

- **Baixou pelo botão "Download ZIP" do GitHub**: o ZIP não inclui o submódulo `third_party/rpf9-explorer`. Com o `git` instalado, o build o baixa sozinho; sem ele, use `git clone --recursive` ou execute `git submodule update --init --recursive` dentro da pasta do repositório.
- **`llvm-mc` recusa `gfx1013` / AMDGPU**: você provavelmente está com o instalador oficial do LLVM para Windows. Siga um dos passos de configuração do LLVM acima.
- **Erro de `D3D12_ROOT_SIGNATURE_DESC2` ao compilar o runtime**: atualize o Windows SDK para `10.0.26100` ou mais recente pelo Visual Studio Installer ou `winget install Microsoft.WindowsSDK.10.0.26100`, apague a pasta de build e reconfigure.
- **Falha por falta de espaço no drive `C:`**: passe `--work "D:\leonida-work"` para mover os arquivos
  temporários ou siga o [build com pouco espaço em disco](#build-com-pouco-espaço-em-disco).

## Como contribuir e rodar os testes de GPU

1. Consulte os PRs abertos e a [issue de coordenação][coord] para evitar trabalho duplicado. Reserve a
   tarefa com `claiming: <ID>` ou com uma descrição de uma linha, se ainda não houver ID.
2. Rode os testes relevantes com entradas sintéticas. Por exemplo, para testar o verificador de manifestos,
   execute `python -I tests/test_manifest_verifier.py` na raiz do repositório.
3. No PR, inclua o ID da tarefa, descreva o que testou e informe a GPU e o driver se rodou testes de GPU.
   Confira as regras em [`CONTRIBUTING.md`](../../../CONTRIBUTING.md). Se usou assistência de IA, indique
   isso no PR, conforme [`AGENTS.md`](../../../AGENTS.md).

O [guia de testes em GPU](../../testing.md) mostra como executar os testes sem arquivos do jogo.
Publique os resultados de GPU na [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
