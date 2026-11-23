const LOCOMOTION = ['idle', 'walk', 'run'];
const DRIVING = ['enter', 'drive', 'exit'];

function requireValue(valid, message) {
  if (!valid) throw new Error(message);
}

function localGlb(name) {
  return typeof name === 'string' && /^[^<>:"/\\|?*\x00-\x1f]+\.glb$/i.test(name)
    && name.trim() === name;
}

function vector(value, label) {
  requireValue(Array.isArray(value) && value.length === 3 && value.every(Number.isFinite),
    `${label} must contain three finite numbers`);
  return value;
}

export function readManifest(json) {
  requireValue(json && typeof json === 'object' && json.version === 1, 'Expected manifest version 1');
  const { character, vehicle, clips } = json;
  requireValue(localGlb(character), 'character must be a local .glb filename');
  requireValue(clips && typeof clips === 'object' && !Array.isArray(clips), 'clips must be an object');
  const roles = vehicle ? [...LOCOMOTION, ...DRIVING] : LOCOMOTION;
  const clipMap = Object.fromEntries(roles.map(role => {
    const value = clips[role];
    const ref = typeof value === 'string' ? { name: value } : value;
    requireValue(ref && typeof ref.name === 'string' && ref.name.trim(), `Missing clip for ${role}`);
    const file = ref.file ?? character;
    requireValue(localGlb(file), `Clip ${role} must use a local .glb filename`);
    return [role, { name: ref.name, file }];
  }));
  const result = { character, clips: clipMap };
  if (vehicle) {
    requireValue(localGlb(vehicle.file), 'vehicle.file must be a local .glb filename');
    const { file, door, doorAngle = 65, entryYaw = 0, seatYaw = 0 } = vehicle;
    requireValue(door === undefined || typeof door === 'string' && door.length, 'vehicle.door must name a node');
    requireValue([doorAngle, entryYaw, seatYaw].every(Number.isFinite), 'Vehicle angles must be finite numbers');
    const wheels = vehicle.wheels ?? [];
    requireValue(Array.isArray(wheels) && wheels.every(wheel => typeof wheel?.node === 'string' &&
      wheel.node.length && (wheel.steer === undefined || typeof wheel.steer === 'boolean')),
      'vehicle.wheels must contain named nodes');
    requireValue(new Set(wheels.map(wheel => wheel.node)).size === wheels.length, 'Wheel nodes must be distinct');
    const wheelRadius = vehicle.wheelRadius ?? .34;
    const wheelbase = vehicle.wheelbase ?? 2.7;
    requireValue([wheelRadius, wheelbase].every(value => Number.isFinite(value) && value > 0),
      'Vehicle wheel radius and wheelbase must be positive');
    result.vehicle = {
      file, door, doorAngle, entryYaw, seatYaw, wheels, wheelRadius, wheelbase,
      entry: vector(vehicle.entry, 'vehicle.entry'), seat: vector(vehicle.seat, 'vehicle.seat'),
    };
  }
  return result;
}

export function readGlbJson(buffer) {
  requireValue(buffer.byteLength >= 20, 'Truncated GLB header');
  const header = new DataView(buffer);
  requireValue(header.getUint32(0, true) === 0x46546c67 && header.getUint32(4, true) === 2,
    'Expected a glTF 2.0 binary (.glb)');
  const length = header.getUint32(12, true);
  requireValue(header.getUint32(8, true) === buffer.byteLength && length <= buffer.byteLength - 20 &&
    header.getUint32(16, true) === 0x4e4f534a, 'Invalid GLB JSON chunk');
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, length)));
  for (const item of [...(json.buffers ?? []), ...(json.images ?? [])]) {
    requireValue(item.uri === undefined || typeof item.uri === 'string' && item.uri.startsWith('data:'),
      'GLB resources must be embedded');
  }
  return json;
}

export function uniqueNamed(items, name, label) {
  const matches = items.filter(item => item.name === name);
  requireValue(matches.length === 1, `${label}: expected one match for “${name}”, found ${matches.length}`);
  return matches[0];
}

export function smoothStep(t) {
  t = Math.min(1, Math.max(0, t));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

export function angleDelta(from, to) {
  return Math.atan2(Math.sin(to - from), Math.cos(to - from));
}
