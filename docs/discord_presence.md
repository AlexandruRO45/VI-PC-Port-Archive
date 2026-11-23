# Optional Discord Rich Presence

The runtime observes attempted Xbox presence updates, resolves their IDs using a verified local catalogue, and
sends the resolved sentence to the Discord desktop client over local IPC. It never contacts Xbox Live or Rockstar.
The only activity field supplied is `details`. The user's Application ID selects their Discord application;
name, icon and other application defaults are left to Discord. No assets, buttons or hard-coded game name are sent.

## Privacy and informed opt-in

Local IPC describes the transport to the desktop client, not the visibility of the activity. Discord can publish
the description to other users according to the user's Discord activity privacy settings. An Application ID is
not consent by itself: before installing the hook or connecting to Discord, the runtime asks whether to enable
sharing for this run. The default is No. Declining or closing the prompt leaves the game running without presence.
Blank IDs do not show the prompt and never connect. Approval is not saved across game runs.

Only use descriptions you authored or have permission to share, and which contain no sensitive information.
Do not supply protected game-derived catalogue text. No image is read, copied, uploaded or referenced by this
integration; use only your own artwork on your Discord application page. Catalogue files, raw presence/token IDs,
SCIDs and game binaries are not sent to Discord. After opt-in, the selected resolved description is sent as
`details`; it may be visible externally. The integration cannot determine the rights or sensitivity of text a
user supplies, so no protected catalogue or extraction step is bundled.

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

No icon file is copied to the output. The game package is never modified. The application name and default
application icon come from the user's Discord application page. The generated Play.cmd invokes PowerShell
without changing its execution policy; user and system policies remain in force. If policy blocks Play.ps1,
it is not bypassed. The runtime can also be invoked directly with its normal --run and --overlay arguments.

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
native observer ABI. The consent prompt is compiled and statically reviewed, not automated in the synthetic
suite. Actual retail gameplay and Discord rendering remain unverified. Native Windows is the
integration target; Wine host-Discord IPC is unverified.

```powershell
cmake --build runtime/build --config RelWithDebInfo --target discord_presence_test gta6rt
ctest --test-dir runtime/build -C RelWithDebInfo -R '^discord_presence$' --output-on-failure
```

References: [Discord RPC](https://docs.discord.com/developers/topics/rpc),
[Discord Rich Presence assets](https://docs.discord.com/developers/discord-social-sdk/development-guides/setting-rich-presence),
[XblPresenceRichPresenceIds](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/live/xsapi-c/presence_c/structs/xblpresencerichpresenceids).
