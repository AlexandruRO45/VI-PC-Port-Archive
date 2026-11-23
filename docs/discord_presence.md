# Local Discord Rich Presence

An optional bridge observes the game's call to `XblPresenceSetPresenceAsync`, copies its SCID, presence ID and
ordered token IDs, resolves them using a private local catalogue, and sends the result to the Discord desktop
client through Windows named pipes. It never calls Xbox Live or Rockstar services. It requires no Discord user
token, OAuth login, server, or additional library. Discord itself controls whether your activity is visible.

The activity is Playing (`type: 0`), named **Grand Theft Auto VI**, with the resolved sentence in `details`.
Its large image asset is `xbs_logo_480`, and its **Leonida** button links to
<https://github.com/ItsNotRexed/Leonida>.

## Set up before building

1. In the [Discord Developer Portal](https://discord.com/developers/applications), create an application named
   **Grand Theft Auto VI**. Copy its Application ID. The portal application name must match: Discord may display
   that registered name instead of the activity's `name` field.
2. Upload your own `xbs_logo_480.png` in that application's Rich Presence art assets, using the asset key
   **xbs_logo_480**. No artwork is distributed here. Wait for Discord to finish processing the asset.
3. Create a private configuration outside the repository and game package, following the schema below.
   Use the SHA-256 and verified callsite/setter RVAs for your exact executable. RVAs are hexadecimal strings.
   Populate `strings` with verified local text in your chosen language. Do not upload the resulting catalogue.
4. Build normally, adding `--discord-presence C:\private\discord_presence.json` to `make_game.py`.
   The runtime stage checks the executable hash and bundles that configuration beside `gta6rt.exe`.
5. Start the Discord desktop client and enable activity sharing. Double-click **Play.cmd** in the finished
   Windows output. This runs the game with its output overlay. `gta6rt --run` automatically loads the bundled
   `discord_presence.json`; no presence flags or setup are needed at each launch.

Existing build options, including `--runtime-build`, still apply. The launcher passes its arguments to the game.
The Wine launcher also reaches the same automatic configuration loading, but Windows named-pipe access to a host
Discord client under Wine has not been verified. Native Windows is the supported integration target.

To use another configuration explicitly, pass `gta6rt --run <exe> --discord-presence <file>`.
The bridge never activates during `--dry-run` or `--init`. Remove the bundled JSON to disable automatic activation.
Rebuilding an output without the option preserves an existing private configuration.

## Configuration schema (synthetic example only)

```json
{
  "application_id": "123456789012345678",
  "image_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "call_rva": "0x1010",
  "setter_rva": "0x1100",
  "strings": {
    "synthetic-service": {
      "presences": { "explore": "Exploring {0}", "menu": "In the menu" },
      "tokens": { "island": "Synthetic Island" }
    }
  }
}
```

These values do not describe a retail executable. The hook validates the executable hash, mapped x64 PE sections,
and the single `E8 rel32` instruction's original target before changing any code. Installation happens before
image initialization. A verified instruction boundary is required; this does not scan arbitrary bytes for calls.
The original setter still receives all four arguments and retains its original result and asynchronous behavior.

Xbox presence IDs are not formatted dashboard text. The catalogue must map each SCID and presence ID to its
actual sentence, with positional `{0}` through `{15}` substitutions using the corresponding ordered token IDs.
Repeated placeholders and literal braces (`{{` and `}}`) work. No retail catalogue is included, and no runtime
connection retrieves one. Unknown IDs, missing tokens, invalid input, inactive presence, and text outside
Discord's 2–128 Unicode character details limit clear or withhold activity rather than publish a guess.

## Behavior and limits

The hook copies bounded input immediately; a worker resolves and publishes it at most once every five seconds.
Only acknowledged updates are deduplicated. Discord disconnects and errors trigger retries; restarting Discord
resends the current activity. Local pipe clients use identification-only security to prevent server impersonation.
IPC reads and writes have a one-second deadline and support cancellation. Shutdown stops the worker and attempts
to clear activity before the runtime's explicit `ExitProcess` call, which skips C++ destructors.

This observes attempted Xbox presence updates. The original offline setter can still fail, and earlier game
guards can prevent any call from occurring. It does not bypass those guards. Synthetic tests verify the bridge;
they do not prove that a retail game reaches this call or that Discord renders the artwork and button. Discord
buttons may be hidden on your own profile; verify visibility from another account after setup.

Run the focused check with:

```powershell
cmake --build runtime/build --config RelWithDebInfo --target discord_presence_test gta6rt
ctest --test-dir runtime/build -C RelWithDebInfo -R '^discord_presence$' --output-on-failure
```

Protocol and ABI references: [Discord RPC](https://docs.discord.com/developers/topics/rpc),
[Discord Rich Presence setup](https://docs.discord.com/developers/discord-social-sdk/development-guides/setting-rich-presence),
[XblPresenceRichPresenceIds](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/live/xsapi-c/presence_c/structs/xblpresencerichpresenceids).
