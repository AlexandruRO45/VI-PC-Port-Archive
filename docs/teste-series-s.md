# Teste: perfil Series S (branch `teste/series-s-32gb`)

Build de teste para rodar o **próprio dump** de GTA VI (Xbox Series, 1.0.2.1) no perfil **Series S**.
Ideal para PCs com **32 GB de RAM** ou mais: o Windows limita a memória compartilhada da GPU a metade da RAM,
e com 16 GB o jogo estoura (~8 GB) ao entrar no mundo.

## O que este branch tem (tudo opt-in)
- `VICTORY_RGB9E5_AS_R11G11B10=1`: o Series S pede um alvo HDR 1920x1080 `R9G9B9E5_SHAREDEXP` como render target/UAV.
  O PC não aceita; ele vira `R11G11B10_FLOAT` (mesmo tamanho por pixel).
- `VICTORY_SHOW_TARGET=3840x2160-f28`: o Series S compõe a tela final em planos que o LumeGX ainda não junta.
  O buffer apresentado sai preto. Este contorno mostra o alvo 4K onde o jogo desenha. **Conhecido:** cutscenes
  podem aparecer pretas (só legendas). A correção definitiva (compor os planos) está em andamento.
- `VICTORY_OWN_UPSCALER=1` (opcional): mantém o upscaler próprio do jogo em vez do FSR do Xbox.
- Requer build **diag** (`--features diagnostics`), porque o `SHOW_TARGET` é um recurso de diagnóstico.

## Requisitos
- Windows 10/11 e GPU DX12 (com ou sem mesh shaders).
- Rust (o `rust-toolchain.toml` escolhe a versão; instale via https://rustup.rs).
- O seu dump do jogo, **fora** do repositório. Nada do jogo entra no Git.
- Opcional e recomendado: AMD FidelityFX SDK para PC (`amd_fidelityfx_upscaler_dx12.dll`, licença MIT, assinado pela AMD),
  de GPUOpen `FidelityFX-SDK`, `Kits/FidelityFX/signedbin`. Coloque a DLL numa pasta fora do repositório.

## Como rodar
```powershell
git clone -b teste/series-s-32gb https://github.com/gabztoo/LumeGX.git
cd LumeGX
copy victory.example.toml victory.toml   # opcional: ajuste os caminhos
.\scripts\teste-series-s.ps1 -Game 'D:\seu\dump\game_scarlett_master.exe' -FidelityFx 'D:\ffx'
```
O primeiro build leva alguns minutos. Na tela inicial, espere uns 30 s e aperte um botão do controle.
O loading demora. No mundo, o fps é baixo: as cópias CPU<->GPU ainda não foram otimizadas.

## O que reportar
- Até onde chegou (menu, loading, mundo, cutscene) e prints.
- A pasta de logs que o script mostra no fim (`stderr.log`, `stdout.log`).
- RAM, GPU e se apareceu `0x8007000E` (falta de memória).
