const test = require('node:test');
const assert = require('node:assert/strict');

const modulePromise = import('../viewer/animation-data.mjs');

function manifest() {
  return {
    version: 1, character: 'actor.glb',
    clips: { idle: 'Stand', walk: { file: 'walk.glb', name: 'Move' }, run: 'Run' },
  };
}

function binary(document) {
  const json = new TextEncoder().encode(JSON.stringify(document));
  const buffer = new ArrayBuffer(20 + json.length);
  const header = new DataView(buffer);
  [0x46546c67, 2, buffer.byteLength, json.length, 0x4e4f534a].forEach((word, i) => {
    header.setUint32(i * 4, word, true);
  });
  new Uint8Array(buffer, 20).set(json);
  return buffer;
}

test('clip roles select exact names in one or several local files', async () => {
  const { readManifest, uniqueNamed } = await modulePromise;
  const input = manifest();
  const result = readManifest(input);
  assert.deepEqual(result.clips.idle, { name: 'Stand', file: 'actor.glb' });
  assert.deepEqual(result.clips.walk, { name: 'Move', file: 'walk.glb' });
  assert.equal(uniqueNamed([{ name: 'Walk' }], 'Walk', 'Clip').name, 'Walk');
  assert.throws(() => uniqueNamed([{ name: 'Walk' }, { name: 'Walk' }], 'Walk', 'Clip'), /found 2/);
  assert.throws(() => readManifest({ ...input, character: '../actor.glb' }), /local/);
  assert.throws(() => readManifest({ ...input, clips: { idle: 'Stand' } }), /walk/);
});

test('vehicle requires three animation roles and finite anchors', async () => {
  const { readManifest } = await modulePromise;
  const input = manifest();
  input.vehicle = { file: 'car.glb', entry: [-1, 0, 0], seat: [0, .4, 0] };
  assert.throws(() => readManifest(input), /enter/);
  Object.assign(input.clips, { enter: 'Enter', exit: 'Exit', drive: 'Drive' });
  assert.equal(readManifest(input).vehicle.wheelbase, 2.7);
  input.vehicle.seat[1] = Infinity;
  assert.throws(() => readManifest(input), /vehicle.seat/);
});

test('GLB validation permits embedded buffers and rejects external resources', async () => {
  const { readGlbJson } = await modulePromise;
  const input = { asset: { version: '2.0' }, buffers: [{ byteLength: 0 }] };
  assert.deepEqual(readGlbJson(binary(input)), input);
  assert.throws(() => readGlbJson(binary({ ...input, images: [{ uri: 'texture.png' }] })), /embedded/);
  assert.throws(() => readGlbJson(binary({ ...input, buffers: [{ uri: 'mesh.bin' }] })), /embedded/);
  assert.throws(() => readGlbJson(new ArrayBuffer(19)), /Truncated/);
  const corrupt = binary(input);
  new DataView(corrupt).setUint32(12, 0xffffffff, true);
  assert.throws(() => readGlbJson(corrupt), /Invalid GLB/);
});

test('transition easing clamps and camera takes the short path across the wrap', async () => {
  const { smoothStep, angleDelta } = await modulePromise;
  assert.equal(smoothStep(-1), 0);
  assert.equal(smoothStep(.5), .5);
  assert.equal(smoothStep(2), 1);
  assert(smoothStep(.001) < 1e-7);
  assert(Math.abs(angleDelta(3, -3) - (2 * Math.PI - 6)) < 1e-12);
});
