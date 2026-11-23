const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const source = readFileSync(join(__dirname, "../viewer/assets.js"), "utf8");
const { catalogEntries, checkLocalGLB } = vm.runInNewContext(
  source + "\n;({ catalogEntries, checkLocalGLB });",
  { document: { getElementById: () => ({}) }, TextDecoder },
);

function file(name, folder = "") {
  return { name, webkitRelativePath: folder ? folder + "/" + name : "" };
}

function glb(json) {
  let chunk = JSON.stringify({ asset: { version: "2.0" }, ...json });
  while (Buffer.byteLength(chunk) % 4) chunk += " ";
  const header = Buffer.alloc(20);
  header.write("glTF");
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(20 + Buffer.byteLength(chunk), 8);
  header.writeUInt32LE(Buffer.byteLength(chunk), 12);
  header.write("JSON", 16);
  return Uint8Array.from(Buffer.concat([header, Buffer.from(chunk)])).buffer;
}

test("folder names choose a category and loose GLBs use the selected category", () => {
  const entries = catalogEntries([
    file("Lamp.GLB", "Library/Props/Domestic"),
    file("Hand.glb", "Library/Ped parts/Left"),
    file("Buggy.glb", "Library/Cars"),
    file("Loose.glb"),
    file("notes.txt"),
  ], "vehicles");
  assert.deepEqual(Array.from(entries, entry => [entry.file.name, entry.category]), [
    ["Lamp.GLB", "props"], ["Hand.glb", "ped-parts"], ["Buggy.glb", "vehicles"], ["Loose.glb", "vehicles"],
  ]);
});

test("matching previews are scoped to the model's folder and file selection", () => {
  const preview = file("Bench.SHADER.PNG", "Library/Props/B");
  const loosePreview = file("Chair.shader.png");
  const entries = catalogEntries([
    file("Bench.glb", "Library/Props/A"), file("Bench.glb", "Library/Props/B"), preview,
    file("Chair.glb"), loosePreview, file("image.png"),
  ], "props");
  assert.deepEqual(Array.from(entries, entry => entry.preview), [undefined, preview, loosePreview]);
  assert.equal(catalogEntries([file("Chair.glb")], "props")[0].preview, undefined);
});

test("the binary glTF check accepts embedded buffers and images", () => {
  assert.doesNotThrow(() => checkLocalGLB(glb({
    buffers: [{ byteLength: 32 }, { uri: "data:application/octet-stream;base64,AA==" }],
    images: [{ bufferView: 0 }, { uri: "data:image/png;base64,AA==" }],
  })));
});

test("the binary glTF check refuses external buffers and images before loading", () => {
  for (const resource of ["buffers", "images"]) {
    for (const uri of ["local.bin", "/world/chunk.glb", "//example.test/file", "https://example.test/file",
      "blob:http://127.0.0.1/item", ["data:image/png;base64,AA=="]]) {
      assert.throws(() => checkLocalGLB(glb({ [resource]: [{ uri }] })), /self-contained GLB/);
    }
  }
});

test("the binary glTF check rejects incomplete headers and chunks", () => {
  const truncated = glb({}).slice(0, -1);
  const badChunk = glb({});
  new DataView(badChunk).setUint32(12, badChunk.byteLength, true);
  for (const bytes of [new ArrayBuffer(0), truncated, badChunk]) {
    assert.throws(() => checkLocalGLB(bytes), /valid binary glTF 2.0/);
  }
});
