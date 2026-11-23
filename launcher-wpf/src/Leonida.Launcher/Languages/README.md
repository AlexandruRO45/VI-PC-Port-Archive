# Translating Leonida Launcher

Each file here is one language. English is the text as written in the source, so it has no file.

```json
{
  "code": "de",
  "name": "Deutsch",
  "culture": "de-DE",
  "strings": {
    "Home": "Startseite",
    "The build finished in {0}.": "Der Build war nach {0} fertig."
  }
}
```

- **`code`**: the language code, the same as the file name (`de.json`). Use `pt-BR` for a regional variant.
- **`name`**: the language's name in that language. Settings › General › Language lists it.
- **`culture`**: the .NET culture for numbers, dates and sizes (`de-DE` shows `1,5 GB`).
- **`strings`**: the English text as written, mapped to your translation.
  - Copy `pl.json` and replace the Polish values; the keys show every string the launcher has.
  - A string you leave out, or leave empty, shows in English.
  - Keep each `{0}`, `{1}`, … where the value goes. The core tests check that a translation has the same placeholders as its English text.
  - `\n` is a line break.
  - A key with `|` (`"BUILD|button"`) is a word that translates two ways; translate the part before the `|`.

## Try a translation without building

Put the file in `%APPDATA%\Leonida\Launcher\Languages\` and restart the launcher; it shows up under
Settings › General › Language. A file there wins over a built-in one with the same code.

## Add it to the launcher

Put the file in this folder. The build embeds every `*.json` here. Then run the core tests:

```powershell
dotnet run --project launcher-wpf\tests\Leonida.Core.Tests -c Release
```

They read every file here and check that it parses, that `code` matches the file name, that `culture` exists and
that every translation keeps its placeholders.

README pages, `make_game.py`'s output and people's names are not translated.
