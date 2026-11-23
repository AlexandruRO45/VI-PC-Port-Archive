"""Leonida Project Dashboard - Real-time monitoring interface."""
import streamlit as st
import subprocess
import json
import os
import time
from datetime import datetime, timedelta

# Configuration
st.set_page_config(
    page_title="Leonida Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Platform detection
# ============================================================
import platform

PLATFORM = platform.system()
if PLATFORM == "Windows":
    LEONIDA_REPO = r"D:\Leonida"
elif PLATFORM == "Linux":
    LEONIDA_REPO = os.path.expanduser("~/Leonida")
else:
    LEONIDA_REPO = os.path.join(os.path.dirname(__file__), "..")

TOOLS_DIR = os.path.join(LEONIDA_REPO, "tools")
MONITOR_DIR = os.path.join(LEONIDA_REPO, ".leonida-monitor")

# Configuration serveur distant (Ubuntu)
REMOTE_MONITOR = {
    "enabled": True,
    "host": "192.168.1.150",
    "user": "claude",
    "state_file": "/opt/leonida-repo/.leonida-monitor/monitor-state.json",
    "log_file": "/opt/leonida-repo/.leonida-monitor/monitor-log.txt",
}

# ============================================================
# Data Fetching Functions
# ============================================================

def get_git_status(repo_path: str = LEONIDA_REPO) -> dict:
    """Get current git repository status."""
    status = {
        "branch": "N/A",
        "head_commit": "N/A",
        "head_hash": "N/A",
        "remote": "N/A",
        "has_uncommitted": False,
        "has_staged": False,
        "uncommitted_files": [],
        "last_commit_time": None,
        "commit_message": "",
    }
    try:
        # Branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_path, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            status["branch"] = result.stdout.strip() or "detached"

        # HEAD hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            status["head_hash"] = result.stdout.strip()

        # Last commit message
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=repo_path, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            status["commit_message"] = result.stdout.strip()

        # Last commit time
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=repo_path, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            status["last_commit_time"] = result.stdout.strip()

        # Remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            status["remote"] = result.stdout.strip()

        # Uncommitted files
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            status["has_uncommitted"] = len(lines) > 0
            status["uncommitted_files"] = [
                {"status": line[:2], "file": line[3:]} for line in lines
            ]

        # Staged changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo_path, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            status["has_staged"] = True

    except Exception as e:
        status["error"] = str(e)
    return status


def get_recent_commits(repo_path: str = r"D:\Leonida", count: int = 15) -> list:
    """Get recent commit log."""
    commits = []
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{count}",
             "--format=%h %s (%cr)", "--date=relative"],
            cwd=repo_path, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            commits = result.stdout.strip().split("\n")
    except Exception:
        pass
    return commits


def get_monitor_status() -> dict:
    """Get current monitor/agent status."""
    # Check for running background tasks via lock files
    task_dir = os.path.join(MONITOR_DIR, "tasks")
    tasks = []

    if os.path.exists(task_dir):
        for item in os.listdir(task_dir):
            item_path = os.path.join(task_dir, item)
            if os.path.isfile(item_path):
                try:
                    with open(item_path, 'r') as f:
                        tasks.append(json.load(f))
                except Exception:
                    pass

    # Check for running monitor processes (cross-platform)
    monitor_running = False
    monitor_info = {}
    try:
        if PLATFORM == "Windows":
            ps_args = ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"]
        else:
            ps_args = ["ps", "aux"]

        result = subprocess.run(ps_args, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.split("\n"):
                if "commit_monitor" in line.lower() or "monitor.py" in line.lower():
                    monitor_running = True
                    if PLATFORM == "Linux":
                        parts = line.split()
                        if len(parts) >= 1:
                            monitor_info["pid"] = parts[0]
    except Exception:
        pass

    return {
        "monitor_running": monitor_running,
        "last_check": None,
        "tasks": tasks
    }


def get_project_stats(repo_path: str = LEONIDA_REPO) -> dict:
    """Get project statistics."""
    stats = {
        "total_files": 0,
        "python_files": 0,
        "tool_count": 0,
        "total_lines": 0,
        "size_bytes": 0,
        "tools_dir": os.path.join(repo_path, "tools"),
    }

    try:
        # Count Python files
        tools_dir = stats["tools_dir"]
        if os.path.exists(tools_dir):
            py_files = []
            for f in os.listdir(tools_dir):
                if f.endswith(".py"):
                    py_files.append(f)
                    filepath = os.path.join(tools_dir, f)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as fh:
                            lines = len(fh.readlines())
                            stats["total_lines"] += lines
                    except Exception:
                        pass
            stats["tool_count"] = len(py_files)

        # Count all files recursively
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden dirs and large binaries
            dirs[:] = [d for d in dirs if not d.startswith('.') or d == '.leonida-monitor']
            stats["total_files"] += len(files)
            stats["size_bytes"] += sum(
                os.path.getsize(os.path.join(root, f))
                for f in files if not os.path.islink(os.path.join(root, f))
            )
    except Exception:
        pass

    return stats


def get_remote_monitor_state() -> dict:
    """Récupérer l'état du monitor depuis le serveur Ubuntu via SSH/scp, ou localement en fallback."""
    if not REMOTE_MONITOR.get("enabled", False):
        return None

    state = {
        "available": False,
        "error": None,
        "source": "remote"
    }

    # Essayer de lire depuis le serveur distant
    remote_success = False

    try:
        # Récupérer le fichier d'état via SSH
        ssh_cmd = [
            "ssh",
            f"{REMOTE_MONITOR['user']}@{REMOTE_MONITOR['host']}",
            f"cat {REMOTE_MONITOR['state_file']}"
        ]

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode == 0 and result.stdout.strip():
            import json
            try:
                data = json.loads(result.stdout)
                state["available"] = True
                state.update(data)
                remote_success = True
            except json.JSONDecodeError:
                state["error"] = "Fichier d'état corrompu sur le serveur"
            except RecursionError:
                state["error"] = "Fichier d'état corrompu sur le serveur"
    except FileNotFoundError:
        pass  # SSH non installé, on essaie plink ensuite
    except Exception as e:
        state["error"] = f"Erreur SSH: {e}"

    if not remote_success:
        # SSH a échoué, essayer avec plink (Windows)
        try:
            plink_cmd = [
                "plink",
                "-ssh",
                f"{REMOTE_MONITOR['user']}@{REMOTE_MONITOR['host']}",
                f"cat {REMOTE_MONITOR['state_file']}"
            ]
            result = subprocess.run(
                plink_cmd,
                capture_output=True,
                text=True,
                timeout=15,
                env={**os.environ, "PLINK_PROTOCOL": "ssh"}
            )
            if result.returncode == 0 and result.stdout.strip():
                import json
                data = json.loads(result.stdout)
                state["available"] = True
                state.update(data)
                remote_success = True
        except Exception:
            pass

    if not remote_success:
        # Lecture distante échouée, essayer le fichier local syncé
        local_state_file = os.path.join(MONITOR_DIR, "monitor-state.json")
        if os.path.exists(local_state_file):
            try:
                with open(local_state_file, 'r') as f:
                    data = json.load(f)
                    state["available"] = True
                    state.update(data)
                    state["source"] = "local"
                    state["error"] = None
            except Exception:
                state["error"] = "Erreur de lecture du fichier local"
        else:
            state["error"] = "Lecture distante échouée - aucun fichier local disponible"

    return state if state["available"] or state.get("error") else None


def get_remote_monitor_log(lines: int = 50) -> list:
    """Récupérer les dernières lignes du log du monitor distant."""
    if not REMOTE_MONITOR.get("enabled", False):
        return []

    try:
        # Essayer ssh d'abord
        tail_cmd = [
            "ssh",
            f"{REMOTE_MONITOR['user']}@{REMOTE_MONITOR['host']}",
            f"tail -n {lines} {REMOTE_MONITOR['log_file']}"
        ]

        result = subprocess.run(
            tail_cmd,
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')

    except FileNotFoundError:
        # SSH non disponible, essayer plink
        try:
            plink_cmd = [
                "plink",
                "-ssh",
                f"{REMOTE_MONITOR['user']}@{REMOTE_MONITOR['host']}",
                f"tail -n {lines} {REMOTE_MONITOR['log_file']}"
            ]
            result = subprocess.run(
                plink_cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')
        except Exception:
            pass
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    # Fallback: lire le log local si disponible
    local_log = os.path.join(MONITOR_DIR, "monitor-log.txt")
    if os.path.exists(local_log):
        try:
            with open(local_log, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                return [line.rstrip('\n') for line in all_lines[-lines:]]
        except Exception:
            pass

    return []


def execute_remote_command(command: str) -> dict:
    """Exécuter une commande distante sur le serveur Ubuntu via SSH et retourner le résultat."""
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "returncode": -1,
        "error": None
    }

    if not REMOTE_MONITOR.get("enabled", False):
        result["error"] = "Moniteur distant désactivé"
        return result

    # Construire la commande SSH (clé claude_debian pour ce serveur)
    ssh_cmd = [
        "ssh",
        "-i", r"C:\Users\Tweek\.ssh\claude_debian",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        f"{REMOTE_MONITOR['user']}@{REMOTE_MONITOR['host']}",
        command
    ]

    try:
        proc = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=60,
            # Utiliser le PATH de l'utilisateur pour trouver ssh.exe de Git
            env={**os.environ, "PATH": r"C:\Program Files\Git\usr\bin;" + os.environ.get("PATH", "")}
        )
        result["stdout"] = proc.stdout or ""
        result["stderr"] = proc.stderr or ""
        result["returncode"] = proc.returncode
        result["success"] = proc.returncode == 0
    except subprocess.TimeoutExpired:
        result["error"] = "Délai d'attente dépassé (60s)"
    except FileNotFoundError:
        # SSH non trouvé, essayer plink
        try:
            plink_cmd = [
                "plink",
                "-ssh",
                f"{REMOTE_MONITOR['user']}@{REMOTE_MONITOR['host']}",
                command
            ]
            proc = subprocess.run(
                plink_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "PLINK_PROTOCOL": "ssh"}
            )
            result["stdout"] = proc.stdout or ""
            result["stderr"] = proc.stderr or ""
            result["returncode"] = proc.returncode
            result["success"] = proc.returncode == 0
        except Exception as e:
            result["error"] = f"SSH/Plink non disponible: {e}"
    except Exception as e:
        result["error"] = f"Erreur SSH: {e}"

    return result


def get_shaders_rpf_status() -> dict:
    """Get shader analysis status."""
    status = {
        "analyzed": False,
        "total_shaders": 0,
        "vs_count": 0,
        "ps_count": 0,
        "cs_count": 0,
        "coverage_pct": 0,
    }

    # Check for shader data files
    coverage_file = os.path.join(os.path.dirname(__file__), "..", "shader_coverage.json")
    if os.path.exists(coverage_file):
        try:
            with open(coverage_file, 'r') as f:
                data = json.load(f)
            status["analyzed"] = True
            status.update(data)
        except Exception:
            pass

    return status


# ============================================================
# Dashboard UI
# ============================================================

st.title("🔬 Leonida Project Dashboard")
st.markdown("---")

# --- Top KPI Row ---
col1, col2, col3, col4, col5 = st.columns(5)

git_status = get_git_status()
project_stats = get_project_stats()
shaders_status = get_shaders_rpf_status()

with col1:
    st.metric("Branch", git_status.get("branch", "N/A"))
with col2:
    hash_display = git_status.get("head_hash", "N/A")
    st.metric("Commit HEAD", f"`{hash_display}`" if hash_display != "N/A" else "N/A")
with col3:
    st.metric("Outils Python", project_stats.get("tool_count", 0))
with col4:
    st.metric("Lignes de code", f"{project_stats.get('total_lines', 0):,}")
with col5:
    coverage = shaders_status.get("coverage_pct", 0)
    st.metric("Couverture Shaders", f"{coverage}%")

st.markdown("---")

# --- Recent Commits Section ---
st.header("📝 Derniers Commits")

commits = get_recent_commits()
if commits:
    for commit in commits:
        st.text(commit)
else:
    st.info("Aucun commit trouvé.")

st.markdown("---")

# --- Two Column Layout: Git Status + Monitor ---
col_left, col_right = st.columns(2)

with col_left:
    st.header("📊 Statut du Dépôt Git")

    for key, label in [
        ("head_message", "Dernier message"),
        ("last_commit_time", "Date du dernier commit"),
        ("remote", "Remote URL"),
    ]:
        value = git_status.get(key, "N/A")
        st.text(f"**{label}:** {value}")

    st.text(f"**Branch:** `{git_status.get('branch', 'N/A')}`")

    # Uncommitted changes
    if git_status.get("has_uncommitted"):
        st.warning("⚠️ Modifications non commitées détectées")
        for file_info in git_status.get("uncommitted_files", [])[:20]:
            status_icon = {"M": "✏️", "A": "➕", "D": "❌", "R": "🔄"}.get(file_info["status"][0], "•")
            st.text(f"  {status_icon} `{file_info['file']}` ({file_info['status']})")

        if len(git_status.get("uncommitted_files", [])) > 20:
            st.caption(f"... et {len(git_status['uncommitted_files']) - 20} autres fichiers")
    else:
        st.success("✅ Dépôt propre - Aucune modification non commitée")

st.markdown("---")

with col_right:
    st.header("🔍 Monitoring Distant (Ubuntu)")

    # --- Statut du moniteur distant ---
    remote_state = get_remote_monitor_state()

    if remote_state and remote_state.get("available"):
        st.success("✅ Moniteur distant actif")

        # Métriques clés
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Branch", remote_state.get("branch", "N/A"))
        with m2:
            head_val = remote_state.get("head", "N/A")
            st.metric("Head", f"`{head_val[:8]}`" if len(str(head_val)) > 7 else f"`{head_val}`")
        with m3:
            remote_val = remote_state.get("remote", "N/A")
            branch_remote = remote_val.split("/")[-1] if remote_val and remote_val != "N/A" else "N/A"
            st.metric("Branch remote", branch_remote)
        with m4:
            has_uncommitted = remote_state.get("has_uncommitted", False)
            status_icon = "🔴" if has_uncommitted else "🟢"
            st.metric("Status", status_icon)

        st.caption(remote_state.get("head_message", "Pas de message"))

        # Timestamp dernier check
        ts = remote_state.get("timestamp", "N/A")
        st.caption(f"Dernière MàJ: {ts}")
    else:
        st.error("❌ Moniteur distant inactif")
        if remote_state and remote_state.get("error"):
            st.warning(remote_state["error"])
        st.info("Le moniteur n'est pas encore déployé sur le serveur.")

    st.markdown("---")

    # --- Console logs ---
    st.subheader("📜 Logs du Monitor Distant")

    log_lines = get_remote_monitor_log(100)
    if log_lines and any(l.strip() for l in log_lines):
        # Filtrer les lignes vides et afficher dans un expander
        filtered_lines = [l for l in log_lines if l.strip()]
        with st.expander(f"Voir les {len(filtered_lines)} dernières lignes", expanded=False):
            st.text_area("Logs complets", value="\n".join(filtered_lines), height=300, language="log")
    else:
        st.info("📭 Aucun log disponible (le monitor n'a pas encore écrit)")

    st.markdown("---")

    # --- Contrôles du moniteur distant ---
    st.subheader("🎛️ Contrôles Monitor")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

    with ctrl_col1:
        if st.button("▶️ Démarrer", use_container_width=True):
            res = execute_remote_command(
                "cd /opt/leonida-dashboard && nohup venv/bin/python tools/commit_monitor.py --interval 300 > logs/monitor-stdout.log 2>&1 & echo $!"
            )
            if res["success"]:
                st.success("✅ Monitor démarré")
                st.caption(f"PID: {res['stdout'].strip() or 'en arrière-plan'}")
            else:
                st.error(f"❌ Erreur: {res.get('error', 'succès inconnu')}")
            st.rerun()

    with ctrl_col2:
        if st.button("⏹️ Arrêter", use_container_width=True):
            res = execute_remote_command(
                "pkill -f 'commit_monitor.py' || echo 'Aucun processus trouvé'"
            )
            if res["success"]:
                st.success("✅ Monitor arrêté")
                st.caption(res["stdout"].strip() or "")
            else:
                st.error(f"❌ Erreur: {res.get('error', '')}")
            st.rerun()

    with ctrl_col3:
        if st.button("🔄 Redémarrer", use_container_width=True):
            st.warning("Arrêt en cours...")
            st.rerun()


st.markdown("---")

# --- Uncommitted Files Detail ---
if git_status.get("has_uncommitted"):
    with st.expander("📂 Détail des fichiers modifiés"):
        for file_info in git_status.get("uncommitted_files", []):
            status_icon = {
                "M": "✏️ Modifié",
                "A ": "➕ Ajouté",
                "D ": "❌ Supprimé",
                "R ": "🔄 Renommé",
            }.get(file_info["status"], f"• {file_info['status']}")
            st.text(f"{status_icon} → `{file_info['file']}`")

# --- Auto-refresh ---
st.sidebar.header("⚙️ Paramètres")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
refresh_interval = st.sidebar.selectbox(
    "Intervalle", [15, 30, 60, 120], index=1
) if auto_refresh else None

# Sidebar quick actions
st.sidebar.header("🚀 Actions rapides")
if st.sidebar.button("🔄 Rafraîchir maintenant"):
    st.rerun()

if st.sidebar.button("📊 Afficher tous les commits"):
    st.session_state.show_all_commits = True

if git_status.get("has_uncommitted") and st.sidebar.button("⚡ Commit & Push"):
    try:
        subprocess.run(["git", "add", "."], cwd=git_status.get("_repo_path", r"D:\Leonida"), check=True)
        subprocess.run(
            ["git", "commit", "-m", "Dashboard auto-sync"],
            cwd=git_status.get("_repo_path", r"D:\Leonida"), check=True
        )
        subprocess.run(["git", "push"], cwd=git_status.get("_repo_path", r"D:\Leonida"), check=True)
        st.sidebar.success("✅ Commit & push réussi")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur: {e}")

# Footer
st.caption(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Loop for auto-refresh
if auto_refresh and refresh_interval:
    time.sleep(refresh_interval)
    st.rerun()
