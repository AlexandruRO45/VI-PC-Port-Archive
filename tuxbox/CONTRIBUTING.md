# Contributing

TuxBox is licensed under the Apache License 2.0. A contribution in a pull request is licensed under the same terms. Third-party materials keep their own terms. [License](LICENSE) · [Third-party notices](docs/REFERENCES.md#shader-compiler-attribution)

Open pull requests against `main`.

## Prepare a checkout

Use Python 3.10 or newer and Rust 1.95 or newer. CI uses Python 3.12 and Rust 1.95.0. From the repository root, install the Python dependency and the Windows target used by the D3D12 typecheck:

```sh
python -m pip install -r requirements.txt
rustup target add x86_64-pc-windows-msvc
```

Building Lanes or a provider needs the desktop compiler and Proton setup in the [Lanes build requirements](examples/lanes/README.md#build-requirements).

## Checks

Run these from the repository root before opening a pull request. GitHub runs the same commands on every pull request and on every push to `main`. [Workflow](.github/workflows/ci.yml)

```sh
python -m unittest discover -s tests -q
cargo test --locked --target x86_64-unknown-linux-gnu --manifest-path tools/shader-compiler/Cargo.toml
cargo test --locked --target x86_64-unknown-linux-gnu -p tuxbox-dxbc --features native-bindings
cargo test --locked --target x86_64-unknown-linux-gnu -p tuxbox-xg-layout
cargo check --locked --target x86_64-pc-windows-msvc -p tuxbox-d3d12
```

The D3D12 command typechecks the Windows provider. The Python tests check that `docs/progress.svg` and `docs/RESULTS.md` match the recorded evidence.

## Evidence

When a change adds or edits a receipt under `evidence/`, regenerate both views and include them in the pull request:

```sh
python tuxbox.py status > docs/RESULTS.md
python tuxbox.py status --svg docs/progress.svg
```

The results table and the progress map are generated. Editing either file by hand makes the unit tests fail.

## Files that stay on your machine

Retail games, Proton prefixes, build output, crash dumps, and packaged binaries stay outside the repository. `.gitignore` excludes `build/`, `games/`, `prefixes/`, `target/`, and the usual package and binary suffixes. The unit tests fail when a tracked file matches that ignore list.

Measured homebrew receipts in `evidence/` belong in the repository. Keep title dumps, captures, and native libraries on your machine.
