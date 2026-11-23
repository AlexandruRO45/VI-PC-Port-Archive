const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const source = readFileSync(join(__dirname, "../viewer/assets.js"), "utf8");
function assetContext(globals = {}) {
  return vm.createContext({
    document: { getElementById: () => ({}) },
    window: { addEventListener() {} },
    TextDecoder, AbortController, AbortSignal, setTimeout, clearTimeout,
    ...globals,
  });
}

const { catalogEntries, checkLocalGLB, parseCatalog } = vm.runInContext(
  source + "\n;({ catalogEntries, checkLocalGLB, parseCatalog });", assetContext(),
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

test("catalog paths encode individual segments and retain the library status", () => {
  const status = { complete: false, processed: 2, total: 5 };
  const result = parseCatalog({ version: 1, status, assets: [
    { name: "Trolley", category: "props", model: "Props/Pad #2%.glb", preview: "Pics/Pad ?2.png" },
    { name: "Rover", category: "vehicles", model: "Vehicles/Rover.GLB" },
    { name: "Crate", category: "props", model: "props/data/Crate%20%2B%23.glb",
      preview: "props/data/Crate%20%2B%23.shader.png" },
  ] });
  assert.equal(result.status, status);
  assert.equal(result.entries[0].modelUrl, "/catalog/Props/Pad%20%232%25.glb");
  assert.equal(result.entries[0].previewUrl, "/catalog/Pics/Pad%20%3F2.png");
  assert.equal(result.entries[1].previewUrl, undefined);
  assert.equal(result.entries[2].modelUrl, "/catalog/props/data/Crate%20%2B%23.glb");
  assert.equal(result.entries[2].previewUrl, "/catalog/props/data/Crate%20%2B%23.shader.png");
  assert.equal(result.entries[2].previewName, "Crate +#.shader.png");
});

test("catalog entries reject absolute paths, unsafe segments, and unexpected types", () => {
  const parse = model => parseCatalog({
    version: 1, status: { complete: false },
    assets: [{ name: "Cart", category: "vehicles", model }],
  });
  for (const path of ["http://example.test/cart.glb", "/cart.glb", "C:/cart.glb", "../cart.glb",
    "cars/./cart.glb", "cars//cart.glb", "cars\\cart.glb", "cars/cart.obj", "cars/a\u0000b.glb",
    "%2E%2E/cart.glb", "cars%2Fcart.glb", "cars%5Ccart.glb", "cars/a%00b.glb"]) {
    assert.throws(() => parse(path), /Invalid catalog file path/);
  }
  assert.throws(() => parseCatalog({ version: 2, assets: [], status: { complete: true } }), /version 1/);
});

test("catalog refresh keeps the selected model and search while adding its new preview", async () => {
  const initial = { name: "Cart", category: "vehicles", model: "vehicles/cart.glb" };
  const button = { disabled: false };
  const status = {};
  const context = assetContext({
    document: { getElementById: id => ({ refresh: button, "catalog-status": status })[id] || {} },
    fetch: async (url, options) => {
      assert.equal(url, "/catalog/catalog.json");
      assert.equal(options.redirect, "error");
      return { ok: true, json: async () => ({
        version: 1, status: { complete: true },
        assets: [{ ...initial, preview: "vehicles/cart.png" }],
      }) };
    },
    initial,
  });
  vm.runInContext(source + `
    catalogAssets = parseCatalog({ version: 1, status: { complete: false }, assets: [initial] }).entries;
    selected = catalogAssets[0];
    category.value = "vehicles";
    search.value = "cart";
    let renders = 0;
    let loads = 0;
    renderList = () => renders++;
    selectAsset = () => loads++;
    imageStage.hidden = true;
  `, context);
  await vm.runInContext("refreshCatalog()", context);
  assert.equal(status.textContent, "Catalog ready.");
  assert.equal(vm.runInContext("selected.previewUrl", context), "/catalog/vehicles/cart.png");
  assert.equal(vm.runInContext("translated.disabled", context), false);
  assert.equal(vm.runInContext("category.value + ':' + search.value", context), "vehicles:cart");
  assert.equal(vm.runInContext("renders + ':' + loads", context), "1:0");
});

test("a catalog build polls until complete and reports waiting, partial, and empty states", async () => {
  const button = { disabled: false };
  const status = {};
  let pending;
  const payloads = [
    { status: 503 },
    { ok: true, json: async () => ({ version: 1, status: { complete: false, processed: 1, total: 3 },
      assets: [{ name: "Cart", category: "vehicles", model: "cart.glb" }] }) },
    { ok: true, json: async () => ({ version: 1, status: { complete: true }, assets: [] }) },
  ];
  const context = assetContext({
    document: { getElementById: id => ({ refresh: button, "catalog-status": status })[id] || {} },
    fetch: async () => payloads.shift(),
    setTimeout: (callback, delay) => { assert.equal(delay, 5000); pending = callback; },
    clearTimeout: () => { pending = undefined; },
  });
  vm.runInContext(source + "\napplyCatalog = entries => { catalogAssets = entries; };", context);
  await vm.runInContext("refreshCatalog()", context);
  assert.match(status.textContent, /Waiting for the generated catalog/);
  await pending();
  assert.match(status.textContent, /Building catalog \(1\/3\).*Partial results/);
  await pending();
  assert.equal(status.textContent, "The catalog is empty.");
  assert.equal(pending, undefined);
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
