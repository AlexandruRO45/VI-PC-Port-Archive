# Live switches: A/B in one session

Comparing two settings by starting the game twice costs about three minutes
per run just to load the same scene again. Each run also lands on a slightly
different moment of the scene, so the difference between two runs is often
smaller than the spread between runs of the same build.

Live switches change while the title runs. The scene is loaded once, and the
two settings take turns in it, every 20 seconds or so, several times.

## Using them

Set `VICTORY_LIVE` to a file. A thread reads it four times a second. Each
`NAME=VALUE` line sets a live switch: `0` is off, anything else on. A switch
the file does not name keeps its environment variable's value (or its default).
Lines starting with `#` are ignored.

```
VICTORY_LIVE=C:\tmp\live.txt
VICTORY_METER_MS=1000
```

Writing `VICTORY_HEAP_SYNC=0` to `C:\tmp\live.txt` turns the descriptor sync
off within a quarter second. Removing the line turns it back to the default.

Each change is logged, so a log can be cut into the stretches each value held:

```
live switch VICTORY_HEAP_SYNC=0
```

A switch is also logged when the file first names it with the value it
already had, so the state is known from the first line that names it.

## Frame lines every second

The frame line ("fps, ms/frame, preparing ...") holds the last second's
numbers, and is logged every 5 seconds by default. `VICTORY_METER_MS=1000`
logs it every second, so a 20-second stretch gives about 17 usable samples
after the first seconds of each change are left out.

## Which switches are live

| Switch | Default | What it does |
|---|---|---|
| `VICTORY_HEAP_SYNC` | on | raw descriptor sync before each submission ([details](descriptor-sync.md)). A heap made while it is off is not tracked, so turning it on later reaches only heaps made since. |
| `VICTORY_WIDE_REFRESH` | off | a submission's host copy refreshes spread over the snapshot threads |

Only switches whose code reads them each time, and whose state stays
consistent when they change in the middle, are made live. A switch becomes
live by reading it through `live::Live` instead of a `OnceLock`:

```rust
static ON: crate::live::Live = crate::live::Live::new("VICTORY_NAME", false);
if ON.on() { /* ... */ }
```

Once read, a check is one atomic load.

## Method

1. Start the game once with `VICTORY_LIVE` and `VICTORY_METER_MS=1000`, and
   reach a steady scene.
2. Write `NAME=1`, wait 20 s; write `NAME=0`, wait 20 s; repeat three times.
3. Give each frame line the state set by the last "live switch NAME=" line
   before it, leave out the lines within 3 s of a change, and compare the
   medians per state and per cycle.

## Results

`VICTORY_HEAP_SYNC` on `77c3019` with this branch, first room, three cycles of
20 s each way, 49 one-second samples per state:

| State | FPS | ms/frame |
|---|---|---|
| on (default) | 6.7 | 148.5 |
| off | 8.2 | 121.8 |

Per cycle, FPS on/off: 6.2/8.5, 7.1/6.8, 6.9/8.2. The whole comparison took
about five minutes, against about eight for two separate runs, and compares
the same stretch of the scene.

`VICTORY_WIDE_REFRESH`, measured with separate runs before this existed, on
`77e1ac7`: 6.1 FPS (164 and 165 ms) with it against 7.0 FPS (143 ms) without.
It stays off by default.
