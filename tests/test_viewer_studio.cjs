const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = readFileSync(join(__dirname, '..', 'viewer', 'studio.html'), 'utf8');
const script = html.match(/<script>\n([\s\S]*?)<\/script>/)[1];

function fakeElement() {
    return {
        children: [], style: {}, innerHTML: '', innerText: '', value: '',
        classList: { add() {}, remove() {}, toggle() {} },
        appendChild(child) { this.children.push(child); },
        remove() {},
    };
}

class Vector3 { set() { return this; } sub() { return this; } }
class Disposable {
    disposals = 0;
    get disposed() { return this.disposals > 0; }
    dispose() { this.disposals++; }
}
class FakeBitmap { closes = 0; close() { this.closes++; } }
const flushPromises = () => new Promise(setImmediate);

function loadStudio() {
    const elements = new Map(), requests = [], parses = [], alerts = [], timers = [];
    const document = {
        getElementById: id => elements.get(id) ?? elements.set(id, fakeElement()).get(id),
        createElement: fakeElement,
        querySelectorAll: () => [],
    };
    const context = vm.createContext({
        document, localStorage: { getItem: () => null }, window: {},
        setTimeout: cb => timers.push(cb),
        alert: message => alerts.push(message), console: { error() {} },
        AbortController, ImageBitmap: FakeBitmap,
        fetch: (url, { signal }) => new Promise((resolve, reject) => {
            signal.addEventListener('abort', () => reject(signal.reason), { once: true });
            requests.push({ url, signal, respond: (overrides = {}) => resolve({
                ok: true, status: 200, url: '', arrayBuffer: async () => new ArrayBuffer(8), ...overrides,
            }) });
        }),
        THREE: {
            MeshStandardMaterial: Disposable, MeshNormalMaterial: Disposable, Vector3,
            Box3: class { setFromObject() { return this; } getCenter(v) { return v; } getSize(v) { return v; } },
            LoaderUtils: { extractUrlBase: url => url.slice(0, url.lastIndexOf('/') + 1) },
            GLTFLoader: class { parse(bytes, path, onLoad, onError) { parses.push({ bytes, path, onLoad, onError }); } },
        },
    });
    vm.runInContext(script, context);
    vm.runInContext(`
        scene = { objects: new Set(), add(o) { this.objects.add(o); }, remove(o) { this.objects.delete(o); } };
        camera = { position: new THREE.Vector3() };
        controls = { target: new THREE.Vector3(), update() {} };`, context);
    const run = code => vm.runInContext(code, context);
    const loadModel = run('loadModel');
    return { context, elements, requests, parses, alerts, timers, run,
        select: id => loadModel(`/models/${id}.glb`, `${id}.glb`, '', 1) };
}

async function finishModel(studio, id, model = fakeModel()) {
    const selection = studio.select(id);
    studio.requests.at(-1).respond();
    await flushPromises();
    studio.parses.at(-1).onLoad({ scene: model });
    await selection;
}

function fakeModel() {
    const geometry = Object.assign(new Disposable(), { attributes: { position: { count: 3 } }, index: null });
    const texture = Object.assign(new Disposable(), { isTexture: true });
    const material = Object.assign(new Disposable(), { map: texture });
    const mesh = { isMesh: true, geometry, material, userData: {} };
    return { position: new Vector3(), userData: {}, parts: [geometry, material, texture], meshes: [mesh],
        traverse(visit) { visit(this); this.meshes.forEach(visit); } };
}

const HOSTILE = '"><img src=x onerror=alert(1)>';
const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" };
const tagShape = markup => markup.match(/<[^>]*>/g).map(tag => tag.replace(/"[^"]*"/g, '""'));
const visibleText = markup => markup.replace(/<[^>]*>/g, '').replace(/&(?:#(\d+)|#x([\da-f]+)|([a-z]+));/gi,
    (_, dec, hex, name) => dec ? String.fromCharCode(dec) : hex ? String.fromCharCode(parseInt(hex, 16)) : ENTITIES[name]);

function renderModelList(name) {
    const studio = loadStudio();
    studio.context.models = [{ name: name + '.glb', folder: name, category: name, size: 4, url: '/models/x.glb' }];
    studio.run('allModels = models; renderAccordionList();');
    const [group] = studio.elements.get('model-list').children;
    return group.children[0].innerHTML + group.children[1].children[0].innerHTML;
}

function renderTextureGrid(name) {
    const studio = loadStudio();
    studio.context.textures = Array.from({ length: 81 },
        () => ({ name: name + '.png', format: 'PNG Image', size: 4, preview: '/textures/x.png' }));
    studio.run('allTextures = textures; renderTexturesGrid();');
    const grid = studio.elements.get('textures-grid');
    grid.children.at(-1).onclick();
    const cards = grid.children.filter(child => child.className === 'tex-card');
    assert.equal(cards.length, 81);
    return cards[0].innerHTML + cards[80].innerHTML;
}

test('hostile model and texture file names render as text, not as elements or attributes', () => {
    for (const render of [renderModelList, renderTextureGrid]) {
        const hostile = render(HOSTILE);
        assert.deepEqual(tagShape(hostile), tagShape(render('plain name')));
        assert.ok(visibleText(hostile).includes(HOSTILE), `${render.name} must still show the name`);
    }
});

test('late parse success is disposed and late parse failure is ignored after a newer selection', async () => {
    const studio = loadStudio();
    const loadA = studio.select('a');
    studio.requests[0].respond();
    await flushPromises();
    const loadB = studio.select('b');
    studio.requests[1].respond();
    await flushPromises();
    const [a, b] = studio.parses;
    const modelC = fakeModel(), modelA = fakeModel();
    await finishModel(studio, 'c', modelC);
    b.onError(new Error('late failure'));
    a.onLoad({ scene: modelA });
    await Promise.all([loadA, loadB]);
    assert.equal(studio.run('currentMesh'), modelC);
    assert.deepEqual([...studio.run('scene.objects')], [modelC]);
    assert.equal(studio.elements.get('stat-name').innerText, 'c.glb');
    assert.deepEqual(studio.alerts, []);
    assert.deepEqual(modelA.parts.map(part => part.disposals), [1, 1, 1]);
});

test('a new selection aborts older requests while waiting for headers or reading the body', async () => {
    const studio = loadStudio();
    const headers = studio.select('headers');
    const body = studio.select('body');
    const [headerRequest, bodyRequest] = studio.requests;
    assert.equal(headerRequest.signal.aborted, true);

    let bodyStarted = false;
    bodyRequest.respond({ arrayBuffer: () => new Promise((resolve, reject) => {
        bodyStarted = true;
        bodyRequest.signal.addEventListener('abort', () => reject(bodyRequest.signal.reason), { once: true });
    }) });
    await flushPromises();
    assert.equal(bodyStarted, true);

    const model = fakeModel();
    await finishModel(studio, 'current', model);
    await Promise.all([headers, body]);
    assert.equal(bodyRequest.signal.aborted, true);
    assert.equal(studio.parses.length, 1);
    assert.equal(studio.run('currentMesh'), model);
    assert.deepEqual(studio.alerts, []);
});

test('the previous selection cannot hide the loading bar for a newer model', async () => {
    const studio = loadStudio();
    await finishModel(studio, 'a');
    const second = studio.select('b');
    studio.timers[0]();
    const bar = studio.elements.get('loading-bar');
    assert.equal(bar.style.transform, 'scaleX(0.4)');
    studio.requests.at(-1).respond();
    await flushPromises();
    studio.parses.at(-1).onLoad({ scene: fakeModel() });
    await second;
    assert.equal(bar.style.transform, 'scaleX(1)');
    studio.timers[1]();
    assert.equal(bar.style.transform, 'scaleX(0)');
});

test('a redirect preserves the model dependency directory and an HTTP error keeps the current model', async () => {
    const studio = loadStudio();
    const good = studio.select('redirected');
    studio.requests[0].respond({ url: 'http://localhost/models/other/sample.gltf?token=a/b' });
    await flushPromises();
    const [parse] = studio.parses;
    assert.equal(parse.path, 'http://localhost/models/other/');
    const model = fakeModel();
    parse.onLoad({ scene: model });
    await good;
    const missing = studio.select('missing');
    studio.requests[1].respond({ ok: false, status: 404 });
    await missing;
    assert.equal(studio.parses.length, 1);
    assert.equal(studio.run('currentMesh'), model);
    assert.equal(studio.alerts.length, 1);
    assert.equal(studio.elements.get('loading-bar').style.transform, 'scaleX(0)');
});

test('switching models releases shared resources once and retains the Studio shading materials', async () => {
    const studio = loadStudio();
    const [modelA, modelB] = [fakeModel(), fakeModel()];
    const [geometry, material, texture] = modelA.parts;
    const image = new FakeBitmap();
    texture.image = image;
    const otherTexture = Object.assign(new Disposable(), { isTexture: true, image });
    const otherMaterial = Object.assign(new Disposable(), { map: otherTexture, normalMap: texture });
    const skeleton = new Disposable();
    const materials = [material, otherMaterial];
    Object.assign(modelA.meshes[0], { material: materials, isSkinnedMesh: true, skeleton });
    modelA.meshes.push({ ...modelA.meshes[0], userData: {} });
    await finishModel(studio, 'a', modelA);
    studio.run('setShadingMode("clay"); setShadingMode("original"); toggleWireframe();');
    assert.equal(modelA.meshes[0].material, materials);
    assert.deepEqual(materials.map(m => m.wireframe), [true, true]);
    studio.run('setShadingMode("normals");');
    await finishModel(studio, 'b', modelB);
    const resources = [geometry, ...materials, texture, otherTexture, skeleton];
    assert.deepEqual(resources.map(part => part.disposals), [1, 1, 1, 1, 1, 1]);
    assert.equal(image.closes, 1);
    assert.deepEqual(modelB.parts.map(part => part.disposed), [false, false, false]);
    assert.equal(studio.run('clayMaterial.disposed'), false);
    assert.equal(studio.run('normalMaterial.disposed'), false);
});
