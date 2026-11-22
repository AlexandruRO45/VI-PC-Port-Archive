(() => {
  const REPO = "FelipeMzero/Leonida";
  const DIRECT_BASE = `https://github.com/${REPO}/releases/latest/download`;
  const DIRECT_URLS = {
    windows: `${DIRECT_BASE}/Leonida-windows-x64.exe`,
    linux: `${DIRECT_BASE}/Leonida-linux-x64.tar.gz`,
    macos: `${DIRECT_BASE}/Leonida-macos-arm64.zip`,
    sha256: `${DIRECT_BASE}/SHA256SUMS.txt`,
  };

  const I18N = {
    "pt-BR": {
      nav_downloads: "Downloads",
      nav_gui: "Interface GUI",
      nav_validation: "Validação",
      nav_pipeline: "Pipeline",
      badge_loading: "Carregando última release...",
      hero_title: "Recomp GTA 6 com interface gráfica nativa.",
      hero_desc:
        "Fork do projeto Leonida feito por FelipeMzero. Construa, valide e acompanhe todo o pipeline no Windows, Linux e macOS — sem precisar decorar comandos de terminal. Tudo roda localmente a partir da sua própria cópia.",
      hero_cta: "Baixar Recomp GTA 6",
      hero_repo: "Ver Código-Fonte",
      stat_shaders: "Shaders RDNA2 → SPIR-V validados",
      stat_os: "Executáveis Windows, Linux e macOS",
      stat_val: "Validação integrada de pacote e código",
      stat_local: "Local na sua máquina · Sem arquivos do jogo",
      dl_title: "Downloads por Plataforma",
      dl_sub: "Links diretos sempre apontando para o último release disponível (Windows, Linux e macOS) com verificação SHA-256.",
      dl_win_desc: "Duplo clique para abrir a GUI com detecção automática de pasta e validador embutido.",
      dl_linux_desc: "Inclui executável Qt6 com interface gráfica completa e integração com Wine/Proton.",
      dl_mac_desc: "App bundle nativo com suporte a detecção de iCloud Drive e validação local.",
      dl_checksums: "Verifique a integridade dos binários baixados:",
      val_title: "Motor de Validação Integrado",
      val_sub: "Valide o ambiente, o código-fonte e a integridade da sua cópia antes de iniciar um build longo.",
      val_f1_t: "Pré-requisitos do Sistema",
      val_f1_d: "Verifica Python 3.12+, .NET 8 SDK, Vulkan SDK (spirv-as/spirv-val), LLVM llvm-mc com target AMDGPU, Visual Studio C++, FFmpeg e espaço livre em disco.",
      val_f2_t: "Integridade do Pacote (SHA-256)",
      val_f2_d: "Compara nomes, tamanhos e hashes SHA-256 contra game/xbox-1.0.2.1.json em paralelo, relatando arquivos ausentes ou corrompidos.",
      val_f3_t: "Validação de Código & CI",
      val_f3_d: "tools/validate.py e a suíte de testes unitários verificam sintaxe Python, esquemas de manifesto e garantem zero dados do jogo no repositório.",
      pipe_title: "Pipeline Completo",
      pipe_sub: "Cada estágio salva marcadores de progresso: se o build parar, ele continua exatamente de onde parou.",
      legal_title: "Aviso Legal Importante",
      legal_body:
        "Este repositório e seus executáveis não contêm arquivos do jogo, chaves ou executáveis proprietários, e não descriptografam nem contornam proteções. Você precisa da sua própria cópia em disco. Tudo o que o build gera permanece exclusivamente no seu computador.",
      footer_credit: "Recomp GTA 6 · Fork do projeto Leonida feito por FelipeMzero · GPLv3",
    },
    en: {
      nav_downloads: "Downloads",
      nav_gui: "Desktop GUI",
      nav_validation: "Validation",
      nav_pipeline: "Pipeline",
      badge_loading: "Loading latest release...",
      hero_title: "Recomp GTA 6 with a native desktop GUI.",
      hero_desc:
        "A fork of the Leonida project by FelipeMzero. Build, validate, and monitor the entire pipeline on Windows, Linux, and macOS — no CLI memorization needed. Everything runs locally from your own copy.",
      hero_cta: "Download Recomp GTA 6",
      hero_repo: "View Source Code",
      stat_shaders: "RDNA2 → SPIR-V shaders validated",
      stat_os: "Windows, Linux & macOS executables",
      stat_val: "Built-in package & code validation",
      stat_local: "100% local · Zero game data bundled",
      dl_title: "Downloads by Platform",
      dl_sub: "Direct links always pointing to the latest available release (Windows, Linux, and macOS) with SHA-256 verification.",
      dl_win_desc: "Double-click to launch the GUI with automatic game discovery and built-in validator.",
      dl_linux_desc: "Includes Qt6 binary with full desktop UI and Wine/Proton integration.",
      dl_mac_desc: "Native app bundle with iCloud Drive detection and local validation.",
      dl_checksums: "Verify downloaded binaries:",
      val_title: "Built-in Validation Engine",
      val_sub: "Validate your environment, codebase, and game package integrity before starting a long build.",
      val_f1_t: "System Prerequisites",
      val_f1_d: "Checks Python 3.12+, .NET 8 SDK, Vulkan SDK (spirv-as/spirv-val), LLVM llvm-mc with AMDGPU target, Visual Studio C++, FFmpeg, and free disk space.",
      val_f2_t: "Package Integrity (SHA-256)",
      val_f2_d: "Verifies file names, sizes, and parallel SHA-256 digests against game/xbox-1.0.2.1.json, reporting missing or corrupted files.",
      val_f3_t: "Codebase & CI Validation",
      val_f3_d: "tools/validate.py and the unit test suite check Python compilation, manifest schemas, and enforce zero game data in the repository.",
      pipe_title: "Complete Pipeline",
      pipe_sub: "Every stage writes completion markers so an interrupted build resumes exactly where it left off.",
      legal_title: "Important Legal Notice",
      legal_body:
        "This repository and its executables contain no game files, keys, or proprietary binaries, and never decrypt or bypass any protection. You need your own copy on disk. Everything generated stays strictly on your PC.",
      footer_credit: "Recomp GTA 6 · Fork of Leonida by FelipeMzero · GPLv3",
    },
  };

  let lang = "pt-BR";

  function detectOS() {
    const ua = navigator.userAgent.toLowerCase();
    if (ua.includes("win")) return "windows";
    if (ua.includes("mac")) return "macos";
    if (ua.includes("linux")) return "linux";
    return "windows";
  }

  function applyLang() {
    const dict = I18N[lang];
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (dict[key]) el.textContent = dict[key];
    });
    const toggle = document.getElementById("lang-toggle");
    if (toggle) toggle.textContent = lang === "pt-BR" ? "EN" : "PT";
  }

  async function hydrateReleases() {
    const os = detectOS();
    const card = document.querySelector(`.dl-card[data-os="${os}"]`);
    if (card) card.classList.add("recommended");
    const hint = document.getElementById("os-hint");
    if (hint) {
      const label = os === "windows" ? "Windows x64" : os === "macos" ? "macOS" : "Linux x64";
      hint.textContent = `Sistema detectado: ${label}`;
    }

    const primary = document.getElementById("primary-download");
    if (primary) primary.href = DIRECT_URLS[os] || DIRECT_URLS.windows;

    const badge = document.getElementById("release-badge");
    try {
      const res = await fetch(`https://api.github.com/repos/${REPO}/releases/latest`, {
        headers: { Accept: "application/vnd.github+json" },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const rel = await res.json();
      const assets = Array.isArray(rel.assets) ? rel.assets : [];

      const findAsset = (re, fallback) => assets.find((a) => re.test(a.name))?.browser_download_url || fallback;
      const winUrl = findAsset(/windows.*\.exe$/i, DIRECT_URLS.windows);
      const linuxUrl = findAsset(/linux.*\.tar\.gz$/i, DIRECT_URLS.linux);
      const macUrl = findAsset(/macos.*\.zip$/i, DIRECT_URLS.macos);
      const shaUrl = findAsset(/SHA256SUMS\.txt$/i, DIRECT_URLS.sha256);

      document.getElementById("dl-windows").href = winUrl;
      document.getElementById("dl-linux").href = linuxUrl;
      document.getElementById("dl-macos").href = macUrl;
      document.getElementById("dl-sha256").href = shaUrl;

      if (primary) primary.href = os === "windows" ? winUrl : os === "macos" ? macUrl : linuxUrl;
      if (badge) badge.textContent = `Release ${rel.tag_name || "latest"} · Fork por FelipeMzero`;
    } catch {
      if (badge) badge.textContent = "Fork do Leonida por FelipeMzero · Última Release";
    }
  }

  document.getElementById("lang-toggle")?.addEventListener("click", () => {
    lang = lang === "pt-BR" ? "en" : "pt-BR";
    applyLang();
  });

  const menuBtn = document.getElementById("menu-toggle");
  const navLinks = document.getElementById("nav-links");
  if (menuBtn && navLinks) {
    menuBtn.addEventListener("click", () => {
      const open = navLinks.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", String(open));
    });
    navLinks.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        navLinks.classList.remove("open");
        menuBtn.setAttribute("aria-expanded", "false");
      });
    });
  }

  applyLang();
  hydrateReleases();
})();
