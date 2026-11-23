const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

// Execute the Studio's own load and reset code with synthetic models, no browser or game files.
const html = readFileSync(join(__dirname, '..', 'viewer', 'studio.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];

class Vector3 {
    constructor(x = 0, y = 0, z = 0) { this.set(x, y, z); }
    set(x, y, z) { Object.assign(this, { x, y, z }); return this; }
    copy(v) { return this.set(v.x, v.y, v.z); }
    clone() { return new Vector3(this.x, this.y, this.z); }
    sub(v) { return this.set(this.x - v.x, this.y - v.y, this.z - v.z); }
    toArray() { return [this.x, this.y, this.z]; }
}

function studioCamera() {
    const camera = { position: new Vector3(0, 2, 5), zoom: 1 };
    const controls = {
        target: new Vector3(),
        update() {},
        // OrbitControls r128 saves its constructor pose until saveState is called again.
        saveState() {
            this.position0 = camera.position.clone();
            this.target0 = this.target.clone();
            this.zoom0 = camera.zoom;
        },
        reset() {
            camera.position.copy(this.position0);
            this.target.copy(this.target0);
            camera.zoom = this.zoom0;
            this.update();
        },
    };
    controls.saveState();
    const pending = [];
    const elements = new Map();
    const context = vm.createContext({
        THREE: {
            Vector3,
            MeshStandardMaterial: class {},
            MeshNormalMaterial: class {},
            Box3: class {
                setFromObject(model) { this.model = model; return this; }
                getCenter(result) { return result.copy(this.model.center); }
                getSize(result) { return result.copy(this.model.size); }
            },
            GLTFLoader: class { load(url, onLoad) { pending.push(onLoad); } },
        },
        localStorage: { getItem() { return null; } },
        window: {},
        document: {
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, { style: {} });
                return elements.get(id);
            },
        },
        setTimeout() {},
        cameraMock: camera,
        controlsMock: controls,
        sceneMock: { add() {}, remove() {} },
    });
    vm.runInContext(source + '\ncamera = cameraMock; controls = controlsMock; scene = sceneMock;', context);
    return {
        camera, controls,
        load(size) {
            vm.runInContext('loadModel("synthetic.glb", "Synthetic", "Demo", 100)', context);
            assert.equal(pending.length, 1);
            pending.shift()({ scene: {
                position: new Vector3(), center: new Vector3(1, 2, 3), size: new Vector3(...size),
                traverse() {},
            } });
        },
        reset() { vm.runInContext('resetCamera()', context); },
    };
}

test('Reset Camera restores the fitted view of each selected Studio model', () => {
    const studio = studioCamera();
    for (const [size, fitted] of [[[10, 20, 30], [45, 36, 54]], [[40, 10, 20], [60, 48, 72]]]) {
        studio.load(size);
        assert.deepEqual(studio.camera.position.toArray(), fitted);
        assert.deepEqual(studio.controls.target.toArray(), [0, 0, 0]);
        studio.camera.position.set(3, 5, 8);
        studio.controls.target.set(1, 2, 3);
        studio.reset();
        assert.deepEqual(studio.camera.position.toArray(), fitted);
        assert.deepEqual(studio.controls.target.toArray(), [0, 0, 0]);
    }
});

test('Reset Camera before a model loads restores the original empty Studio view', () => {
    const studio = studioCamera();
    studio.camera.position.set(3, 5, 8);
    studio.controls.target.set(1, 2, 3);
    studio.reset();
    assert.deepEqual(studio.camera.position.toArray(), [0, 2, 5]);
    assert.deepEqual(studio.controls.target.toArray(), [0, 0, 0]);
});
