# Local Discord Rich Presence

The runtime observes attempted Xbox presence updates, resolves their IDs using a verified local catalogue, and
sends the resolved sentence to the Discord desktop client over local IPC. It never contacts Xbox Live or Rockstar.
The activity is Playing, named **Grand Theft Auto VI**, with the resolved sentence in `details`, image asset
`xbs_logo_480`, and a **Leonida** button linking to https://github.com/ItsNotRexed/Leonida.

## Optional user setting

Every runtime build creates `discord_presence.json` beside `gta6rt.exe` with exactly one field:

```json
{
  "richpresence_application_id": ""
}
```

The runtime reads this file automatically. Users can enter their own public Discord Application ID.
A missing file, missing field or blank ID leaves presence disabled. Invalid settings, unavailable Discord or a
missing/invalid hook profile do not prevent game startup. Rebuilds preserve an entered ID and its formatting.
There are no setup comments, hashes, hook RVAs or catalogues in this user-facing file.

The builder copies `xbs_logo_480.png` from the user's game package into the output when available. The package is
never modified. Discord requires this image registered on the user's application under the asset key
`xbs_logo_480`; it cannot display a local PNG directly. The application name should be Grand Theft Auto VI.

## Internal profile

A builder can supply a complete verified private profile with `make_game.py --discord-presence FILE`.
The builder stores this separately as `leonida/presence_profile.json` inside the output. Its internal schema
contains `application_id` (the legacy build-time ID), `image_sha256`, hexadecimal `call_rva` and `setter_rva`, and
`strings` mapping each SCID to `presences` and `tokens` dictionaries. The runtime always uses the user's adjacent
`richpresence_application_id` instead of the legacy ID. Migrating an old complete configuration preserves its
profile in the internal location and reduces the user file to the single setting.

The profile is not a user-editable setup template. No zero RVAs or empty catalogues are generated for users to
fill in. A verified retail hook profile and text catalogue are not distributed. Entering an Application ID alone
cannot supply missing profile data; the runtime disables presence and continues starting the game in that case.

The hook validates the executable SHA-256, mapped executable sections and a single verified direct-call target
before patching. It does not scan arbitrary bytes. All four original arguments, the result and async behavior
are preserved. Local templates support ordered `{0}` through `{15}` token substitutions and escaped braces.
Unknown IDs, missing tokens, inactive presence or invalid details clear or withhold activity rather than guess.

A worker publishes at most once every five seconds, deduplicates acknowledgements, reconnects and handles
PING/PONG. IPC uses identification-only pipe security, bounded I/O and cancellation. Shutdown attempts to clear
activity before the runtime's explicit ExitProcess call. No game-service guard or offline failure is bypassed.

## Validation and limits

Synthetic tests cover packaging, migration, optional user settings, profile loading, resolution, IPC and the
native observer ABI. Actual retail gameplay and Discord rendering remain unverified. Native Windows is the
integration target; Wine host-Discord IPC is unverified.

```powershell
cmake --build runtime/build --config RelWithDebInfo --target discord_presence_test gta6rt
ctest --test-dir runtime/build -C RelWithDebInfo -R '^discord_presence$' --output-on-failure
```

References: [Discord RPC](https://docs.discord.com/developers/topics/rpc),
[Discord Rich Presence assets](https://docs.discord.com/developers/discord-social-sdk/development-guides/setting-rich-presence),
[XblPresenceRichPresenceIds](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/live/xsapi-c/presence_c/structs/xblpresencerichpresenceids).
