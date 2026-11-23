import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { angleDelta, readGlbJson, readManifest, smoothStep, uniqueNamed } from './animation-data.mjs';

const element = id => document.getElementById(id);
const viewport = element('viewport');
const status = element('status');
const fileInput = element('files');
const interactButton = element('interact');
const followButton = element('follow');
const speedLabel = element('speed');
const keys = new Set();
const keyLabels = [...element('keys').querySelectorAll('kbd')];
const gaitSpeeds = { idle: 0, walk: 1.3, run: 4.6 };
const radians = THREE.MathUtils.degToRad;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
viewport.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x293239);
scene.fog = new THREE.FogExp2(0x293239, .027);
scene.add(new THREE.HemisphereLight(0xf8faff, 0x777367, 2.2));
const sun = new THREE.DirectionalLight(0xfff4dd, 2.1);
sun.position.set(4, 6, 3);
scene.add(sun);
const grid = new THREE.GridHelper(100, 100, 0x536168, 0x455158);
scene.add(grid);
const camera = new THREE.PerspectiveCamera(46, 1, .04, 100);
new ResizeObserver(() => {
  const { clientWidth, clientHeight } = viewport;
  if (!clientWidth || !clientHeight) return;
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight, false);
}).observe(viewport);

const manager = new THREE.LoadingManager();
manager.setURLModifier(url => {
  if (!/^(blob:|data:)/.test(url)) throw new Error('GLB resources must be embedded');
  return url;
});
const loader = new GLTFLoader(manager);
const up = new THREE.Vector3(0, 1, 0);
const right = new THREE.Vector3(1, 0, 0);
const aim = new THREE.Vector3();
const target = new THREE.Vector3();
const drivePosition = new THREE.Vector3();
const pathOffset = new THREE.Vector3();
const euler = new THREE.Euler(0, 0, 0, 'YXZ');

function nodesByName(root) {
  const nodes = [];
  root.traverse(node => nodes.push(node));
  return nodes;
}

function dispose(root) {
  const resources = new Set();
  root.traverse(node => {
    if (node.geometry) resources.add(node.geometry);
    if (node.skeleton) resources.add(node.skeleton);
    for (const material of [].concat(node.material ?? [])) {
      resources.add(material);
      for (const value of Object.values(material)) if (value?.isTexture) resources.add(value);
    }
  });
  const images = new Set();
  for (const resource of resources) {
    resource.dispose();
    if (resource.isTexture && resource.image?.close) images.add(resource.image);
  }
  for (const image of images) image.close();
}

function animation(source, incoming, actorNodes, role) {
  const tracks = [];
  const motion = {};
  for (const track of incoming.tracks) {
    const binding = THREE.PropertyBinding.parseTrackName(track.name);
    const node = THREE.PropertyBinding.findNode(source, binding.nodeName);
    if (!node) throw new Error(`Cannot resolve animation target “${track.name}”`);
    if (node.name === 'mover' && ['position', 'quaternion'].includes(binding.propertyName)) {
      const basis = node.parent.getWorldQuaternion(new THREE.Quaternion());
      if (role === 'enter' || role === 'exit') {
        const curve = track.createInterpolant();
        if (binding.propertyName === 'position') {
          motion.position = t => new THREE.Vector3().fromArray(curve.evaluate(t)).applyQuaternion(basis);
        } else {
          const inverse = basis.clone().invert();
          motion.yaw = t => euler.setFromQuaternion(basis.clone()
            .multiply(new THREE.Quaternion().fromArray(curve.evaluate(t))).multiply(inverse)).y;
        }
      }
      continue;
    }
    const destination = actorNodes.includes(node) ? node : uniqueNamed(actorNodes, node.name, 'Character target');
    const copy = track.clone();
    copy.name = track.name.replace(binding.nodeName, destination.uuid);
    tracks.push(copy);
  }
  if (!tracks.length) throw new Error(`Clip ${role} contains no character tracks`);
  return { clip: new THREE.AnimationClip(role, incoming.duration, tracks), motion };
}

async function prepare(files) {
  const manifests = files.filter(file => /\.json$/i.test(file.name));
  if (manifests.length !== 1) throw new Error('Select exactly one JSON manifest');
  const manifest = readManifest(JSON.parse(await manifests[0].text()));
  const sources = new Map();
  const read = async name => {
    if (!sources.has(name)) {
      const bytes = await uniqueNamed(files, name, 'Selected file').arrayBuffer();
      readGlbJson(bytes);
      sources.set(name, await loader.parseAsync(bytes, ''));
    }
    return sources.get(name);
  };
  const actor = new THREE.Group();
  const root = new THREE.Group();
  let car;
  try {
    const { scene: character } = await read(manifest.character);
    const nodes = nodesByName(character);
    if (!nodes.some(node => node.isSkinnedMesh)) throw new Error('Character GLB must contain a skinned mesh');
    character.updateMatrixWorld(true);
    const mixer = new THREE.AnimationMixer(character);
    const actions = {};
    for (const [role, ref] of Object.entries(manifest.clips)) {
      const gltf = await read(ref.file);
      const matchingClip = uniqueNamed(gltf.animations, ref.name, `Clip ${role}`);
      const { clip, motion } = animation(gltf.scene, matchingClip, nodes, role);
      actions[role] = { player: mixer.clipAction(clip), motion };
    }
    actor.add(character);
    root.add(actor);
    if (manifest.vehicle) {
      const spec = manifest.vehicle;
      const { scene: body } = await read(spec.file);
      const chassis = new THREE.Group();
      chassis.add(body);
      root.add(chassis);
      chassis.position.set(0, 0, 4);
      const bodyNodes = nodesByName(body);
      const binding = name => {
        const node = uniqueNamed(bodyNodes, name, 'Vehicle node');
        return { node, rotation: node.quaternion.clone() };
      };
      car = {
        chassis, spec, entry: new THREE.Vector3(...spec.entry), seat: new THREE.Vector3(...spec.seat),
        door: spec.door ? binding(spec.door) : undefined,
        wheels: spec.wheels.map(wheel => ({ ...binding(wheel.node), steer: wheel.steer })),
      };
    }
    for (const [name, gltf] of sources) {
      if (name !== manifest.character && name !== manifest.vehicle?.file) dispose(gltf.scene);
    }
    return { root, actor, mixer, actions, car, speed: 0, steer: 0, wheelAngle: 0, stage: 'foot' };
  } catch (error) {
    for (const { scene: loaded } of sources.values()) dispose(loaded);
    throw error;
  }
}

let session;
function select(role, duration = .3) {
  const { actions, current, mixer } = session;
  const { player } = actions[role];
  if (role === current) return;
  const scheduled = player.isScheduled();
  const start = scheduled ? player.getEffectiveWeight() : current ? 0 : 1;
  if (role === 'enter' || role === 'exit') {
    player.setLoop(THREE.LoopOnce, 1);
    player.clampWhenFinished = true;
  }
  if (!scheduled || role === 'enter' || role === 'exit') player.reset();
  player.setEffectiveWeight(start).play();
  // Retain the current contributions even if the user interrupts a crossfade.
  const weights = Object.entries(actions).filter(([name, action]) => name !== role && action.player.isScheduled())
    .map(([, action]) => [action.player, action.player.getEffectiveWeight()]);
  session.fade = { elapsed: 0, duration, weights, start, speed: session.speed };
  session.current = role;
  if (!current) mixer.update(0);
}

function updatePose(dt) {
  const { actions, current, fade, mixer } = session;
  if (fade) {
    fade.elapsed += dt;
    const weight = smoothStep(fade.elapsed / fade.duration);
    actions[current].player.setEffectiveWeight(THREE.MathUtils.lerp(fade.start, 1, weight));
    for (const [player, start] of fade.weights) {
      if (weight === 1) player.stop();
      else player.setEffectiveWeight(start * (1 - weight));
    }
    if (weight === 1) session.fade = undefined;
  }
  mixer.update(dt);
}

function setStage(stage) {
  session.stage = stage;
  interactButton.textContent = stage === 'seated' ? 'Exit vehicle' : 'Enter vehicle';
  interactButton.disabled = !session.car || stage === 'enter' || stage === 'exit';
  status.textContent = {
    foot: session.car ? 'Walk to the driver door and press E.' : 'Character ready.',
    enter: 'Entering vehicle',
    seated: 'Driving controls ready.',
    exit: 'Exiting vehicle',
  }[stage];
}

function worldPoint(car, local, out) {
  return car.chassis.localToWorld(out.copy(local));
}

function interact() {
  if (!session?.car) return;
  const { actor, car, stage, speed } = session;
  if (stage === 'foot') {
    worldPoint(car, car.entry, drivePosition);
    if (actor.position.distanceTo(drivePosition) > 1.55) {
      status.textContent = 'Move closer to the driver door.';
      return;
    }
  } else if (stage !== 'seated' || speed > .3) {
    if (stage === 'seated') status.textContent = 'Stop the vehicle before exiting.';
    return;
  }
  const role = stage === 'foot' ? 'enter' : 'exit';
  session.interaction = { start: actor.position.clone(), yaw: actor.rotation.y };
  session.speed = 0;
  session.throttle = 0;
  setStage(role);
  select(role);
}

function updateInteraction() {
  const { actor, car, interaction, stage, actions } = session;
  const { player, motion } = actions[stage];
  const duration = player.getClip().duration;
  const time = player.time;
  const progress = time / duration;
  const blend = smoothStep(progress);
  const destination = stage === 'enter' ? car.seat : car.entry;
  worldPoint(car, destination, drivePosition);
  actor.position.lerpVectors(interaction.start, drivePosition, blend);
  const endYaw = car.chassis.rotation.y + radians(stage === 'enter' ? car.spec.seatYaw : car.spec.entryYaw);
  actor.rotation.y = interaction.yaw + angleDelta(interaction.yaw, endYaw) * blend;
  // Correct the endpoints to the user's anchors while keeping the authored motion between them.
  if (motion.position) {
    pathOffset.copy(motion.position(time))
      .sub(motion.position(0).lerp(motion.position(duration), blend))
      .applyQuaternion(car.chassis.quaternion);
    actor.position.add(pathOffset);
  }
  if (motion.yaw) {
    const start = motion.yaw(0);
    actor.rotation.y += angleDelta(start, motion.yaw(time)) - angleDelta(start, motion.yaw(duration)) * blend;
  }
  if (car.door) {
    const open = smoothStep(progress / .25) * (1 - smoothStep((progress - .66) / .34));
    car.door.node.quaternion.copy(car.door.rotation)
      .multiply(new THREE.Quaternion().setFromAxisAngle(up, radians(car.spec.doorAngle) * open));
  }
  if (time < duration) return;
  session.interaction = undefined;
  session.keepDoorView = stage === 'exit';
  setStage(stage === 'enter' ? 'seated' : 'foot');
  select(stage === 'enter' ? 'drive' : 'idle');
}

const damp = THREE.MathUtils.damp;
const boost = () => keys.has('ShiftLeft') || keys.has('ShiftRight');
function updateMovement(dt) {
  const { actor, car, stage } = session;
  const forward = keys.has('KeyW');
  const turn = Number(keys.has('KeyA')) - Number(keys.has('KeyD'));
  if (stage === 'foot') {
    const role = forward ? (boost() ? 'run' : 'walk') : 'idle';
    select(role, .42);
    const before = session.speed;
    const fade = session.fade;
    session.speed = fade
      ? THREE.MathUtils.lerp(fade.speed, gaitSpeeds[role], smoothStep((fade.elapsed + dt) / fade.duration))
      : gaitSpeeds[role];
    actor.rotation.y += (session.turnRate = damp(session.turnRate ?? 0, turn * 2.45, 9, dt)) * dt;
    actor.translateZ((before + session.speed) * .5 * dt);
  } else if (stage === 'seated') {
    const braking = keys.has('KeyS');
    const pace = braking || !forward ? 0 : boost() ? 16 : 10;
    const previousSpeed = session.speed;
    session.throttle = damp(session.throttle ?? 0, pace, braking ? 16 : 9, dt);
    session.speed = damp(session.speed, session.throttle, braking ? 9 : forward ? 1.6 : 2.8, dt);
    session.steeringInput = damp(session.steeringInput ?? 0, turn * .42, 17, dt);
    session.steer = damp(session.steer, session.steeringInput, 10, dt);
    const travel = (previousSpeed + session.speed) * .5 * dt;
    const yaw = travel * Math.tan(session.steer) / car.spec.wheelbase;
    car.chassis.rotation.y += yaw / 2;
    car.chassis.translateZ(travel);
    car.chassis.rotation.y += yaw / 2;
    session.wheelAngle -= travel / car.spec.wheelRadius;
    for (const wheel of car.wheels) {
      wheel.node.quaternion.copy(wheel.rotation)
        .multiply(new THREE.Quaternion().setFromAxisAngle(up, wheel.steer ? session.steer : 0))
        .multiply(new THREE.Quaternion().setFromAxisAngle(right, session.wheelAngle));
    }
    worldPoint(car, car.seat, actor.position);
    actor.rotation.y = car.chassis.rotation.y + radians(car.spec.seatYaw);
    speedLabel.value = `${Math.round(session.speed * 3.6)} km/h`;
  }
  if (stage === 'foot') speedLabel.value = '';
}

let following = true;
let orbit = .35;
let zoom = 1;
let cameraYaw = .35;
let yawTarget = .35;
let height = 2;
let distance = 5;
function updateCamera(dt, snap = false) {
  const { actor, car, stage } = session;
  const seated = stage === 'seated';
  const door = stage === 'enter' || stage === 'exit' ||
    session.keepDoorView && actor.position.distanceToSquared(car.chassis.position) < 16;
  if (!door) session.keepDoorView = false;
  target.copy(seated ? car.chassis.position : actor.position);
  if (door) target.lerp(car.chassis.position, .5);
  target.y += seated ? .9 : 1.15;
  const heading = seated ? car.chassis.rotation.y : actor.rotation.y;
  const side = door ? Math.sign(car.entry.x - car.seat.x) || 1 : 0;
  const shotYaw = door ? car.chassis.rotation.y + side * 1.3 : heading + Math.PI;
  const desired = orbit + (following ? shotYaw : 0);
  const amount = snap ? 1 : 1 - Math.exp(-5.4 * dt);
  const positionBlend = snap ? 1 : 1 - Math.exp(-10 * dt);
  aim.lerp(target, positionBlend);
  yawTarget += angleDelta(yawTarget, desired) * amount;
  cameraYaw += angleDelta(cameraYaw, yawTarget) * amount;
  distance = THREE.MathUtils.lerp(distance, (seated ? 6.2 : door ? 5.8 : 3.9) / zoom, amount);
  height = THREE.MathUtils.lerp(height, seated ? 1.85 : 1.05, amount);
  camera.position.set(aim.x + Math.sin(cameraYaw) * distance, aim.y + height,
    aim.z + Math.cos(cameraYaw) * distance);
  camera.lookAt(aim);
}

followButton.addEventListener('click', () => {
  following = !following;
  orbit = following ? .35 : cameraYaw;
  followButton.setAttribute('aria-pressed', String(following));
  followButton.textContent = following ? 'Follow camera' : 'Orbit camera';
});
const canvas = renderer.domElement;
canvas.addEventListener('pointerdown', event => canvas.setPointerCapture(event.pointerId));
canvas.addEventListener('pointermove', event => {
  if (canvas.hasPointerCapture(event.pointerId)) orbit -= event.movementX * .009;
});
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  zoom = THREE.MathUtils.clamp(zoom * Math.exp(-event.deltaY * .001), .6, 3);
}, { passive: false });

function displayKeys() {
  for (const badge of keyLabels) {
    badge.dataset.down = String(badge.dataset.code === 'ShiftLeft' ? boost() : keys.has(badge.dataset.code));
  }
}
const supported = new Set([...keyLabels.map(badge => badge.dataset.code), 'ShiftRight']);
window.addEventListener('keydown', event => {
  if (event.target.closest?.('input') || event.altKey || event.ctrlKey || event.metaKey || event.isComposing ||
      !session || !supported.has(event.code)) return;
  event.preventDefault();
  keys.add(event.code);
  displayKeys();
  if (event.repeat) return;
  if (event.code === 'KeyE') interact();
  if (event.code === 'KeyR') {
    orbit = .35;
    zoom = 1;
  }
});
window.addEventListener('keyup', event => {
  if (keys.delete(event.code)) displayKeys();
});
function clearKeys() {
  keys.clear();
  displayKeys();
}
window.addEventListener('blur', clearKeys);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearKeys();
});
interactButton.addEventListener('click', interact);

fileInput.addEventListener('change', async () => {
  if (!fileInput.files.length) return;
  fileInput.disabled = true;
  status.textContent = 'Reading local files…';
  try {
    const next = await prepare([...fileInput.files]);
    if (session) {
      session.mixer.stopAllAction();
      scene.remove(session.root);
      dispose(session.root);
    }
    session = next;
    scene.add(next.root);
    clearKeys();
    element('keys').hidden = false;
    setStage('foot');
    select('idle');
    updateCamera(0, true);
    fileInput.blur();
  } catch (error) {
    status.textContent = `Could not load selection · ${error.message}`;
  } finally {
    fileInput.disabled = false;
    fileInput.value = '';
  }
});

let previous;
renderer.setAnimationLoop(now => {
  const dt = previous === undefined ? 0 : Math.min((now - previous) / 1000, .05);
  previous = now;
  if (session) {
    updateMovement(dt);
    updatePose(dt);
    if (session.interaction) updateInteraction();
    updateCamera(dt);
    grid.position.set(Math.round(aim.x), 0, Math.round(aim.z));
  }
  renderer.render(scene, camera);
});
