# AMD64 runtime function inventory

Run these commands with a local PE32+ image. Keep both generated files local.

    python -I tools/runtime_function_inventory.py image.exe > ranges.csv
    python -I tools/rtti_scan.py image.exe --out classes.json

The CSV uses image-relative addresses and exclusive end RVAs. Size is in bytes.
The flags column gives the upper five bits of the first UNWIND_INFO byte.
A single source function can own several ranges. Leaf functions may have no entry.

## Retail measurement

Image SHA-256 e95020b2fa4a721db3df59fdecf75621730ac559d8835252f6c3b658d5a08993
has 299,257 exception-directory entries. All ranges fit in .text.
The histogram of raw unwind flags is:

| Flags | Ranges |
|---:|---:|
| 0 | 298,146 |
| 1 | 3 |
| 2 | 97 |
| 3 | 496 |
| 4 | 515 |

On the same file, the existing RTTI scanner reports 209 type descriptors,
98 classes with RTTI, and 98 vtables. These counts describe structures
recognized by the scanner. They do not establish the total number of C++ classes.
