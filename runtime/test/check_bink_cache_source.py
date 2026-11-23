"""Check native bink_play's choice of name table without a game DLL or movie."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from bink_play import stamp


def main(player):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        selected, other = root / "selected package", root / "other package"
        for package in (selected, other):
            package.mkdir()
            for name in ("movies.rpf", "common.rpf"):
                (package / name).write_bytes(b"synthetic")
        cache = root / "profile" / "Leonida" / "bink_names.csv"
        cache.parent.mkdir(parents=True)
        env = dict(os.environ, LOCALAPPDATA=str(cache.parent.parent))
        env.pop("LEONIDA_GAME_DIR", None)
        env.pop("LEONIDA_MOVIE", None)

        def check(label, archive_dir, mount_dir, cli_dir, env_dir, expect_lookup, *, after_stamp=None):
            key = stamp(archive_dir / "movies.rpf") + "|" + stamp(mount_dir / "common.rpf")
            cache.write_text(f"# {key}\npath,offset,width,height,frames\nprobe.bk2,0,16,16,1\n",
                             encoding="utf-8", newline="\n")
            if after_stamp:
                after_stamp()
            args = [player, "--headless", "probe"]
            if cli_dir:
                args += ["--game-dir", str(cli_dir)]
            child = dict(env, LEONIDA_GAME_DIR=str(env_dir)) if env_dir else env
            run = subprocess.run(args, cwd=root, env=child, capture_output=True, timeout=30)
            # On a cache hit the CLI resolves and prints the synthetic movie, then finds no decoder.
            # On a miss these synthetic archives cannot produce a usable replacement table.
            resolved = b"probe.bk2" in run.stdout.splitlines()
            if expect_lookup:
                reason = b"does not contain Bink2Scarlett.dll"
            elif cli_dir or env_dir:
                reason = b"could not build the movie name table"
            else:
                reason = b"needs the game folder once"
            if run.returncode != 1 or resolved != expect_lookup or reason not in run.stderr:
                raise AssertionError(f"{label}: exit {run.returncode}, stdout {run.stdout!r}, stderr {run.stderr!r}")

        check("command line overrides environment", selected, selected, selected, other, True)
        check("foreign archive", other, selected, selected, other, False)
        check("foreign mount", selected, other, None, selected, False)
        check("no selection uses the cached package", other, other, None, None, True)

        def change_mtime(path):
            st = path.stat()
            path.write_bytes(b"different")  # same size as b"synthetic"
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - 2_000_000_000))

        def change_size(path):
            st = path.stat()
            path.write_bytes(b"larger synthetic source")
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))

        check("same-size archive update", selected, selected, selected, None, False,
              after_stamp=lambda: change_mtime(selected / "movies.rpf"))
        check("same-size mount update", selected, selected, None, selected, False,
              after_stamp=lambda: change_mtime(selected / "common.rpf"))
        check("mount size changed with original mtime", selected, selected, selected, None, False,
              after_stamp=lambda: change_size(selected / "common.rpf"))
        check("no selection with missing mount", other, other, None, None, False,
              after_stamp=(other / "common.rpf").unlink)


if __name__ == "__main__":
    main(str(Path(sys.argv[1]).resolve(strict=True)))
