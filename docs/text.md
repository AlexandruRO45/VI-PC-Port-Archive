# Text: string tables and dialogue subtitles

**Status: decoded.** Every string table the descriptor search finds in `data.rpf`, `story000.rpf` and
`shared_minigame000.rpf` is read, and every dialogue line is joined to its subtitles. Labels are stored only as
hashes. Dialogue labels are named by the conversation data; other labels are named only when a dictionary holds them.
The tables have no language field, so the language is a guess (below).

Measured on the Xbox Series 1.0.2.1 package on 2026-11-22. The outputs quote game text, so they are local work
products: do not commit them or attach them to an issue. The test fixtures are synthetic.

## String tables

A string table is a `0DRP` logical resource ([page_directory.md](page_directory.md)): the `0DRP` stream joined with
the 64 KiB streams that continue it. Its payload holds a hash table, an index array and a string blob. A descriptor
in the resource locates the three. All pointers are `0x5xxxxxxx` resource pointers; the low 28 bits are offsets
into the logical resource.

```
descriptor   ptr index | u32 | u32 0x10 | ptr | 01 01 00 00 | u32 hash_bytes | ptr hash
             | u32 0x10 | u32 blob_bytes | ptr blob
ptr          u32 0x5xxxxxxx, u32 0
hash entry   16 bytes: u32 joaat(lowercase label), u32 0, u8 1, u8[2], u24 index (at +11), u8
index entry  16 bytes: u64, u64 (blob offset << 20 | flags)
blob         NUL-terminated UTF-8 strings
```

A hash entry whose first word is 0, whose second word is not 0, or whose byte at +8 is not 1 is an empty slot. Text
keeps the game's inline markup (`~o~`, `~s~`, `~n~`, `~sl:0.05:0.9~`, `~COLOR_SUBTITLE_FOREIGN_DIALOGUE~`, ...).

| Archive | Tables |
|---|---:|
| `data.rpf` | 14,482 |
| `story000.rpf` | 15,831 |
| `shared_minigame000.rpf` | 78 |
| **Total** | **30,391** tables, 7,252,163 strings |

The same 14,482 `data.rpf` tables come out of a directory carved by `rpf9_carve.py`, the archive read directly, and
an older carve that wrote each resource as one file.

### Language

The tables carry no language field. Each table's language is guessed from the first 400 strings:

1. **Script.** Kana is `ja`, hangul is `ko`, CJK is Chinese and Cyrillic is `ru`.
2. **Latin script.** Common words and distinctive letters (`ąćęłńśźż`, `äöüß`, `ãõ`, `ñ¿¡`, ...) score each of `en`,
   `de`, `es`, `fr`, `it`, `pl` and `pt`. No score, or a tie, gives `xx`.
3. **Variants.** Spanish and Chinese ship two variants each:
   - Chinese splits by characters that differ between Traditional and Simplified script (`這`/`这`).
   - Spanish splits by Spain's second-person plural (`vosotros`, `habéis`) and vocabulary against Latin America's
     (`ustedes`, `celular`, `ahorita`).
   - A table without a marker keeps the plain `es` or `zh` code.

| Language | Tables | Distinct label hashes | Hashes with a second, different text |
|---|---:|---:|---:|
| `en` | 2,190 | 545,145 | 337 |
| `de` | 2,236 | 552,563 | 127 |
| `es-ES` | 1,649 | 525,316 | 4,542 |
| `es-419` | 1,568 | 509,876 | 5,220 |
| `es` (no marker) | 1,214 | 37,802 | 18,852 |
| `fr` | 2,031 | 535,263 | 210 |
| `it` | 2,063 | 544,165 | 56 |
| `ja` | 2,326 | 552,111 | 35 |
| `ko` | 2,334 | 554,576 | 35 |
| `pl` | 2,280 | 547,423 | 58 |
| `pt` | 2,395 | 548,445 | 5,392 |
| `ru` | 2,327 | 545,820 | 33 |
| `zh-Hans` | 2,221 | 553,483 | 81 |
| `zh-Hant` | 2,206 | 554,441 | 34 |
| `zh` (no marker) | 250 | 839 | 252 |
| `xx` (undecided) | 1,059 | 3,652 | 2,246 |

The last column counts label hashes whose text differs between two tables of the same language. A clean split
leaves a few dozen of them (about 35 for `ja`, `ko`, `ru` and `zh-Hant`). Above that there are two causes:

- **Variants mixed.** The plain `es`, `zh` and `xx` rows hold small UI tables without enough text to decide. In the
  marked Spanish rows about 1% of hashes still disagree. The `pt` row holds some tables of another language.
- **Labels with two texts.** Some labels exist in two tables of one language with different text. For example, a
  line spoken in Spanish has a plain subtitle in one table and an `In Spanish: …` subtitle under
  `~COLOR_SUBTITLE_FOREIGN_DIALOGUE~` in another. This accounts for most of the `en` and `fr` counts.

The dialogue table keeps one text per label and language: the last table read wins. `--strings` writes every table
separately, so nothing is lost.

## Dialogue

Conversation data (PSO streams) pairs a dialogue key with its line labels as `<ROOT>AU_<XXXX>\0<LABEL>\0`. A key with N
distinct labels has the audio waves `<key>_01` .. `<key>_<N>`, in order: wave `_01` speaks the first label, `_02` the
second. These are AWC stream names. A label's `joaat` is its hash in the string tables, so each wave gets its text
in every language. A label ending `_<A-Z{1,3}>_<n>` names the speaker code.

| Where the pairs are | Streams |
|---|---:|
| `story000.rpf` | 594 |
| `story000_restofworld.rpf` | 1 |
| `shared_minigame000.rpf` | 4 |
| `data.rpf` | 2 |

Together they give **118,351 keys and 138,104 lines**. Labels can contain lowercase letters (`..._90s_1`). Some
keys reuse another mission's lines, so a key's root and its labels' prefix can differ. Lines with text:

| `en` | `de` | `es-ES` | `es-419` | `fr` | `it` | `ja` | `ko` | `pl` | `pt` | `ru` | `zh-Hans` | `zh-Hant` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 125,240 | 126,451 | 123,738 | 121,624 | 126,423 | 128,101 | 128,063 | 125,083 | 128,060 | 126,391 | 121,400 | 126,483 | 125,072 |

About 7-12% of lines have no text in a given language: the label's hash is in none of that language's tables.

## Grammatical variants (`0x956a9df5` / `0x10075946`)

In `data.rpf`, `62` non-English string tables (`10 it`, `9 fr`, `8 pl`, `6 de`, `6 ko`, `6 ru`, `6 zh-Hant`, `5 es-419`,
`3 pt`, `2 es-ES`, `1 ja`) include `3,026` hash-table entries whose `byte@+8` is greater than `1` (`5`, `7` or `8`
instead of `1`), pointing to `7,345` localized text variants across `3,026` groups.

- **Why `byte@+8` is `5`, `7` or `8` and why a 32-bit record index failed on 28 of 62 tables:** in the enclosing `0DRP`
  page's `q38` block table (`+0x38`), `byte@+8` is the **1-based `q38` block index** of the `0x956a9df5` group array
  (`5` on single-blob tables, `7` on 3-chunk tables, `8` on 4-chunk tables), and `u24@+11` is the 16-byte unit index
  (`2 * k`, since each group record is `32` bytes). Each `32`-byte `0x956a9df5` record holds:
  - `+0x08..+0x0f`: `u32 count, u32 count2` (`1..7`, equal on `3,026 / 3,026` records),
  - `+0x10..+0x17`: `u64 ptr4` = `(byte_offset << 20) | block_id`, a standard `0DRP` block pointer where `ptr4 & 0xFFFFF`
    is the 1-based `q38` block index of `0x10075946` (`6`, `8` or `9`) and **`ptr4 >> 20` is the byte offset** into
    that `0x10075946` array (`first_record * 32`),
  - `+0x1a..+0x1b`: two `u8` source selectors (`rage__fwTextGender` / `rage__fwTextPlurality` source).
  Across all `62 / 62` tables (`3,026 / 3,026` groups, `7,345 / 7,345` variant records), reading `ptr4 >> 20` as a byte
  offset tiles the `0x10075946` block with **0 mismatches** and every `0x10075946` string reference (`+0x00` `u64`,
  `chunk = ref & 3`, `off = ref >> 20`) lands on a NUL-terminated string (`7,345 / 7,345`).
- **Embedded `0DRP` schema (`q28`/`q30`, case-sensitive joaat32 verified):**
  - `rage__fwTextEntrySimple` (`0xa8478889`, `16` B): standard string entry (`Text` `0xb5a7cd47` at `+0x08`).
  - `rage__fwTextEntryVariant` (`0x10075946`, `32` B): `Text` (`+0x00`), `Gender` (`0x159902a6`, `+0x08`), `Plurality`
    (`0x3efaa23f`, `+0x09`), `Case` (`0x42db2391`, `+0x0a`), and a 9-element `2`-byte parameter requirement array
    (`+0x0b..+0x1c`).
  - `rage__fwTextGender` (`0x2a053c90`): `None` (`0`), `Male` (`1`), `Female` (`2`), `Other` (`3`).
  - `rage__fwTextPlurality` (`0xb46aea0a`): `None` (`0`), `Singular` (`1`), `Few` (`2`), `Many` (`3`).
  - `rage__fwTextCase` (`0x4ef01f30`): `None` (`0`), `Case1`..`Case15` (`1..15`).
  - Source enum `0x12b29290`: `None` (`0`), `Param1`..`Param9` (`1..9`), `Player` (`10`), `Speaker` (`11`),
    `Target` (`12`).

## Tables and label names

`--labels` takes word lists or a `hashdict.py` JSON. Each table is then grouped by the longest `_`-prefix (up to three
parts) that at least half of its resolved labels share, usually a mission or feature code. With about 3 million
harvested identifiers as labels, 1,869,883 of 7,172,826 strings (26%) resolve, and 9,409 of 30,349 tables get a
group.

## Usage

```sh
python -I tools/text_tables.py <carve>/data <carve>/story000 <carve>/story000_restofworld <carve>/shared_minigame000 \
    --labels <temp>/hashes.json \
    --dialogue <temp>/dialogue.csv --tables <temp>/tables.csv --strings <temp>/strings.csv
```

A source is an `rpf9_carve.py` directory, an `.rpf` archive (read through `gamedata_index.py`; nothing is written),
or files and directories of whole resources. All sources are read before anything is written, because dialogue
keys and their text sit in different archives.

| Output | Columns |
|---|---|
| `--dialogue` | `wave`, `key`, `label`, `speaker`, `text_<lang>` per language |
| `--tables` | `source`, `offset`, `lang`, `group`, `strings`, `resolved` |
| `--strings` | `source`, `offset`, `lang`, `hash`, `label`, `text` |

Standard library only. The four archives above take about 2 minutes on one core from a carve.

`tools/opus_tags.py` writes the `--dialogue` rows into the exported Ogg Opus files as `LABEL`, `SPEAKER` and
`SUBTITLE` comments (see [audio.md](audio.md#tags-names-and-subtitles)).

## Open questions

- **Language per table.** The game must know each table's language. A field or a naming rule outside the tables
  (the TOC or the executable) would replace the guess.
- **Which text the game picks.** When a label has two texts in one language, which table wins at run time
  (load order, mission scope, the foreign-dialogue subtitle option) is not established.
