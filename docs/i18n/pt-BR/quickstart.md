# Leonida — Guia rápido (`pt-BR`)

> [!NOTE]
> A documentação em inglês ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) e
> [`docs/README.md`](../../README.md)) é a **referência oficial**. Este guia reúne apenas instruções estáveis de
> configuração e compilação (sincronizado com o commit `3faac557b`). Para o status diário do projeto, consulte o
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

- **Sua própria cópia** do pacote Xbox Series, versão `1.0.2.1` (a pasta que contém `MicrosoftGame.Config` e os arquivos `.rpf`, como arquivos legíveis em disco).
- **Windows 10/11 x64**, com cerca de **350 GB livres** para os arquivos de trabalho. Eles podem ficar em outro disco com `--work`.
- [Python 3.12+](https://www.python.org/downloads/) (marque *"Add python.exe to PATH"*).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/), para `spirv-as` / `spirv-val`.
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) com o **target AMDGPU** para o `llvm-mc`:
  - O binário oficial do LLVM para Windows **não** inclui o target AMDGPU.
  - No Windows, `scripts\build_llvm_mc.ps1` compila um `llvm-mc.exe` nativo com AMDGPU (sem WSL; requer `git` e as ferramentas C++ do Visual Studio abaixo) e configura `LLVM_MC`. Leva cerca de 20 a 40 minutos.

    ```powershell
    powershell -ExecutionPolicy Bypass -File scripts\build_llvm_mc.ps1
    ```

  - Uma forma mais rápida, se você usa [conda](https://docs.conda.io/): `conda install -c conda-forge llvm-tools`
    (o build do conda-forge para Windows inclui o target AMDGPU; [`docs/testing.md`](../../testing.md) usou a
    versão 23.1.6). Depois aponte `LLVM_MC` para o caminho completo do `llvm-mc.exe` instalado (`where llvm-mc`
    mostra o caminho com o ambiente ativo).
  - Se usar o WSL, instale `llvm-21` no Ubuntu e defina `LLVM_MC` com o comando inteiro, na mesma sessão do
    PowerShell em que vai executar `make_game.py`:

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Veja [`docs/design.md`](../../design.md#disassembling-shaders) para mais detalhes.
  - No Linux, basta ter `llvm-mc` no `PATH`.
- **Visual Studio 2022** com *Desenvolvimento para desktop com C++* e **Windows SDK `10.0.26100` ou mais recente** (necessário para o estágio `runtime`, que usa root signatures 1.2, que SDKs mais antigos não declaram).
- [ffmpeg](https://ffmpeg.org/download.html) no `PATH` *(opcional: decodifica streams XMA2 e é exigido ao usar `--wav`)*.

**Pouco espaço em disco:** o valor de ~350 GB refere-se ao build completo. Para *executar* o port,
`--stages check,shaders,runtime` precisa de cerca de **30 GB**; o estágio `assets` (~300 GB) serve apenas às
ferramentas de extração de assets, e `--wav` adiciona várias vezes o tamanho dos arquivos Ogg Opus (a verificação
inicial avisa quando é o estágio `assets` que não cabe). `--work` e `--out` podem apontar para outro disco. O
runtime mapeia o pacote no próprio lugar, então uma unidade de pacote somente leitura precisa de
`gta6rt --overlay <pasta gravável>`; `--overlay` é uma opção do `gta6rt`, não do `make_game.py`. Mantenha os
arquivos `.s` em `work/shaders/dxbc/disasm` no lugar: apagá-los, junto com o `work/shaders/spirv/index.json`
traduzido, pode fazer o estágio `shaders` parecer concluído com zero shaders. Uma pasta `disasm` que é um link
simbólico é recusada quando o estágio precisa limpá-la.

## Como compilar

O port não é baixado pronto: ele é compilado no seu PC, a partir da sua própria cópia.

### Modo simples (Windows)

1. Instale os pré-requisitos acima.
2. Clone o repositório **com o submódulo**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
   Um "Download ZIP" do GitHub deixa de fora `third_party/rpf9-explorer`. O build o baixa se `git` estiver
   instalado; senão ele diz o que fazer.
3. Abra a pasta `Leonida` e dê duplo clique em **`Build.bat`**.

O script procura automaticamente sua cópia (Downloads, Desktop, Documentos — incluindo onde o Windows mantém
essas pastas, como o OneDrive —, Games, pasta de usuário e a raiz de cada disco, até dois níveis de profundidade).
Ele reconhece a versão por alguns tamanhos de arquivo, sem ler nada grande. Se houver mais de uma cópia, ele
pergunta qual usar; se não encontrar nenhuma, pede que você cole a pasta. Antes de começar, verifica se tudo
acima está instalado e se há espaço livre suficiente, listando o que estiver faltando de uma vez. O resultado
vai para `Documents\Leonida`. Se o build for interrompido, rodar de novo retoma de onde parou.

### Pelo terminal (`make_game.py`)

Mesmo comportamento, com mais controle:

```powershell
python -I make_game.py --find                          # lista as cópias encontradas neste PC
python -I make_game.py --pick                          # escolhe a pasta do jogo em uma janela de diálogo
python -I make_game.py --check                         # verifica os arquivos do jogo sem compilar
python -I make_game.py --check --full-check            # também compara o SHA-256 de cada arquivo (lento)
python -I make_game.py --game "D:\caminho\do\pacote" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # roda apenas os estágios escolhidos
python -I make_game.py --game "D:\caminho\do\pacote" --wav  # também decodifica Opus para PCM WAV (grande)
python -I make_game.py --wave64-emulate off            # auto (padrão), on ou off; veja abaixo
```

`--wave64-emulate` decide se os shaders compute e mesh wave64 são traduzidos para serem exatos em qualquer
tamanho de subgrupo (veja "Wave64 emulation" em [`docs/design.md`](../../design.md)). GPUs NVIDIA e Intel têm
subgrupos de 32 lanes e precisam disso. O padrão `auto` consulta o `vulkaninfo` (parte do Vulkan SDK) e ativa a
emulação se alguma GPU não tiver subgrupos de 64 lanes ou se o tamanho for desconhecido. Ele também verifica se
a GPU do runtime consegue executar compute e mesh em 64 lanes. Com `--mesh-compute`, basta o suporte em compute.

No driver Windows da AMD, `auto` usa wave64 nativo em GPUs RDNA1 e RDNA2 quando a GPU do runtime suporta, mesmo
se outra GPU for de 32 lanes ou o `vulkaninfo` não estiver disponível, porque esse driver pode falhar em código
LDS emulado (#857). Para placas RDNA1 sem suporte a mesh shaders, a decisão usa o estágio compute, que pode
executar wave64 sem suporte nativo a mesh.

O modo escolhido fica salvo junto aos shaders traduzidos, e mudá-lo os retraduz. `auto` normalmente reaproveita
shaders emulados anteriormente; o contorno para o driver AMD Windows recompila esses shaders em modo nativo.
Use `--wave64-emulate on` para forçar a emulação ou `--wave64-emulate off` para forçar wave64 nativo.

Quatro flags de fallback são repassadas ao tradutor da mesma forma — salvas junto aos shaders traduzidos, e
alterá-las os retraduz: `--no-float-controls` (omite `SignedZeroInfNanPreserve`, para dispositivos cuja
verificação de pipeline pula um shader sem `shaderSignedZeroInfNanPreserveFloat32`; requer a flag do tradutor do
PR #664), `--bary-fallback` (lê baricêntricas brutas de um input interpolado, para dispositivos sem
`VK_KHR_fragment_shader_barycentric`), `--texel-offset-fallback` (incorpora os offsets de `image_sample*_o` na
coordenada, para dispositivos sem `VK_KHR_maintenance8`) e `--mesh-compute` (mesh shaders como compute, para
dispositivos sem `VK_EXT_mesh_shader`). Cada uma está documentada em [`docs/design.md`](../../design.md).

O build registra qual comando e versão do `llvm-mc` produziram a desmontagem. Mudar qualquer um dos dois, ou
reusar uma pasta de trabalho anterior a esse registro, refaz a desmontagem e a tradução, mantendo os inputs de
shader extraídos.

O build também registra quais arquivos-fonte do tradutor produziram os shaders. Uma mudança nesses arquivos refaz
a tradução usando a desmontagem salva. Pastas de trabalho antigas precisam de uma tradução depois da atualização
para registrar essa informação.

O build registra ainda qual decoder (o commit do `third_party/rpf9-explorer` e os fontes de `tools/rpf9dump`)
exportou os modelos e texturas de cada arquivo. Quando ele muda, o estágio `assets` remove aquela exportação e a
refaz. Pastas de trabalho antigas não têm esse registro, então seus modelos e texturas são exportados mais uma vez
depois da atualização.

Os estágios executam nesta ordem: `check` → `shaders` → `assets` → `runtime` → `pipelines` (cria cada shader
traduzido no driver da sua GPU e verifica se ele compila, reporta os que precisa pular e salva um cache de
pipelines em `pipeline_cache.bin` dentro de `--out`; o runtime ainda não carrega esse cache, então ele não
acelera o primeiro boot). A tradução dos shaders é o passo mais longo: ela roda em `--jobs` processos (metade
das threads da CPU por padrão).

`--wav` fica desativado por padrão porque WAV PCM de 16 bits é várias vezes maior que Ogg Opus. Quando ativado,
decodifica os streams Ogg Opus reempacotados para `work\assets\audio\wav` depois do passo dos bancos de áudio;
arquivos WAV já existentes são pulados ao retomar um build.

### Build com pouco espaço em disco

Para preparar o runtime e os shaders com cerca de **30 GB livres**, execute na raiz do repositório:

```powershell
python -I make_game.py --stages check,shaders,runtime
```

O estágio `assets` serve às ferramentas de extração e ocupa cerca de 300 GB. A opção `--wav` aumenta ainda mais
o espaço usado pelos áudios. Você pode apontar `--work` e `--out` para outro disco. Se o disco do pacote for
somente leitura, passe `gta6rt --overlay <pasta gravável>` ao iniciar o runtime (`--overlay` é uma opção do
`gta6rt`, não do `make_game.py`).

### Diagnóstico do Vulkan no Linux

Com CMake 3.24+, um compilador C++20, os cabeçalhos de desenvolvimento e o carregador do Vulkan instalados,
rode na raiz do repositório:

```sh
cmake -S runtime -B runtime/build/linux
cmake --build runtime/build/linux --parallel
./runtime/build/linux/vk_pipeline_check /caminho/para/spirv
```

Passe ao `vk_pipeline_check` a pasta que contém `index.json` e os shaders SPIR-V traduzidos.

O `make_game.py` também roda no Linux. O estágio `pipelines` usa o `vk_pipeline_check` compilado acima
(`runtime/build/linux`, ou `runtime/build`) e o compila ali antes de qualquer outro trabalho, se ele estiver
faltando.

### Runtime no Linux com Wine ou Proton

O loader é um programa Windows e roda sob Wine ou Proton. O CI executa seus testes sob Wine em cada push para
`main` que altera `runtime/`, e sob demanda na aba Actions (o job `wine` do workflow Runtime). No Linux, o
estágio `runtime` o compila em `runtime/build/windows` com `clang-cl` e `lld-link` (LLVM 19+, mas não 22.1.7 nem
22.1.8 — veja [`docs/testing.md`](../../testing.md#known-problems); nomes simples, ou as cinco ferramentas LLVM
com um sufixo de versão, como `clang-cl-21`), contra o CRT do MSVC e o Windows SDK instalados pelo
[xwin](https://github.com/Jake-Shadle/xwin), em `XWIN_DIR` ou `~/.local/share/xwin-sdk`
([`docs/testing.md`](../../testing.md#the-windows-runtime-on-linux-under-wine-or-proton) tem o comando único do
`xwin`). Sem essas dependências, o estágio fica fora da execução padrão no Linux, e pedi-lo lista o que falta
antes de qualquer trabalho começar.

Você também pode reutilizar um build Windows feito em outro lugar, como o artefato `runtime-windows` do job
`windows` desse workflow, descompactado:

```sh
python3 -I make_game.py --runtime-build ~/Downloads/runtime-windows
```

Nos dois casos, o script copia `gta6rt.exe` e `gta6rt_tls_host.dll` para `--out` e gera `gta6rt.sh` ao lado
deles, que inicia `gta6rt.exe` com `wine`, ou com o Wine indicado por `WINE` (para o Proton, o `files/bin/wine`
da instalação). A mesma opção pula a compilação no Windows.

## Problemas comuns

- **Baixou pelo botão "Download ZIP" do GitHub**: o ZIP não inclui o submódulo `third_party/rpf9-explorer`. Use `git clone --recursive` ou execute `git submodule update --init --recursive` dentro da pasta do repositório.
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
