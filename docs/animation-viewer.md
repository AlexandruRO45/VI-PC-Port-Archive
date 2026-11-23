# Local animation viewer

Run `python -I viewer/serve.py` from the repository root and open
<http://127.0.0.1:8765/animation.html>. Select a JSON manifest and all its referenced
GLBs together in the file picker. The browser reads the selected files directly.
The loopback server serves the viewer code, and the page loads the same pinned
Three.js release as the map viewer. Keep the manifest and exports in a private
work directory outside the checkout.

## Export character and clips

Use [the animated GLB exporter](animation.md) with your own input files and
select LOD 0 and `--max-influences 4` for Three.js. Export a skinned character
GLB with one or more animations. For clips from other dictionaries, export
additional GLBs from the same skeleton. Animation-only exports can omit
`--part`. The manifest maps viewer roles to exact GLB animation names. This is
especially important when different dictionaries use the same clip name.

```json
{
  "version": 1,
  "character": "actor.glb",
  "clips": {
    "idle": "Stand",
    "walk": {"file": "walk.glb", "name": "Movement"},
    "run": {"file": "run.glb", "name": "Movement"}
  }
}
```

A string selects a clip in the character file. An object selects a clip in
another selected GLB. Keep the skeleton and uniquely named animated nodes
consistent across files. GLBs must embed their buffers and any image resources.
The exporter currently emits skinned geometry and animation, so surface
appearance depends on the materials and UVs included in your input GLB.
The viewer uses Y-up coordinates in metres and treats +Z as forward.

## Optional vehicle

Add the `vehicle` object below and map three more character clips under
`clips`: `"enter": "Get In"`, `"drive": "Seated"`, and `"exit": "Get Out"`.
Supply a separate, already exported Y-up vehicle GLB. `anim_glb.py` is the
character and animation exporter. The vehicle GLB can come from any exporter
that preserves the model and the node pivots you want to animate.

```json
{
  "file": "vehicle.glb",
  "entry": [-1.1, 0, 0.3],
  "seat": [-0.35, 0.48, 0.15],
  "entryYaw": 90,
  "seatYaw": 0,
  "door": "DriverDoor",
  "doorAngle": 65,
  "wheels": [
    {"node": "FrontLeft", "steer": true},
    {"node": "FrontRight", "steer": true},
    {"node": "RearLeft"},
    {"node": "RearRight"}
  ],
  "wheelRadius": 0.34,
  "wheelbase": 2.7
}
```

Set `entry` and `seat` in the vehicle GLB's scene coordinates. They are the
character's model origin while standing at the door and seated, respectively.
The example values and names describe no particular asset. Adjust them to
your files. Yaw angles are in degrees around +Y. The door is optional and uses
its node's local +Y hinge axis. Wheels are optional and spin around their local
+X axes, with marked steering wheels turning about local +Y. Omitted angles,
radius and wheelbase use the defaults shown, except `entryYaw` defaults to 0.

The viewer drives walk/run displacement and wheel movement from the controls.
It consumes the exporter's `mover` tracks in the enter/exit clips as visual
root motion, correcting both ends to the anchors. Clip playback drives the
duration of an interaction. Door motion follows a timed hinge curve, so the
pivot, angle and anchors may need adjustment for a particular export. The
driving model is a preview on a grid.

Use W to move, Shift to run or accelerate, A/D to turn, and S to brake. Walk to
the entry anchor and press E to enter. Once stopped, press E to exit. The
camera follows the character on foot, frames the door during transitions, and
trails the vehicle while driving. Drag to orbit, scroll to zoom, and press R to
recenter.
