const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Run the actual Studio load callback, with synthetic bounds and no WebGL or game data.
const html = readFileSync(join(__dirname, '..', 'viewer', 'studio.html'), 'utf8');
const begin = html.indexOf('function loadModel(');
const end = html.indexOf('async function loadModels(', begin);
assert(begin >= 0 && end > begin, 'Studio loadModel must exist');
const source = html.slice(begin, end);

class Vector3 {
    constructor(x = 0, y = 0, z = 0) { this.set(x, y, z); }
    set(x, y, z) { Object.assign(this, { x, y, z }); return this; }
    sub(v) { return this.set(this.x - v.x, this.y - v.y, this.z - v.z); }
}

function fixture() {
    let nextModel;
    // Same initial perspective settings as initThree, in a landscape viewport.
    const camera = {
        position: new Vector3(0, 2, 5), near: 0.1, far: 1000, fov: 45, aspect: 4 / 3,
        projectionUpdates: 0,
        updateProjectionMatrix() {
            const scale = 1 / Math.tan(this.fov * Math.PI / 360);
            this.projection = [scale / this.aspect, scale,
                -(this.far + this.near) / (this.far - this.near),
                -2 * this.far * this.near / (this.far - this.near)];
            this.projectionUpdates++;
        },
    };
    camera.updateProjectionMatrix();
    const controls = { target: new Vector3(), update() {}, saveState() {} };
    const context = vm.createContext({
        camera, controls, currentMesh: null,
        scene: { add() {}, remove() {} },
        document: { getElementById() { return { style: {}, innerText: '' }; } },
        setTimeout() {}, applyMeshShading() {}, formatBytes() { return 'synthetic'; },
        THREE: {
            Vector3,
            GLTFLoader: class { load(url, complete) { complete({ scene: nextModel }); } },
            Box3: class {
                setFromObject(model) { this.model = model; return this; }
                getCenter(v) { return v.set(...this.model.center); }
                getSize(v) { return v.set(...this.model.size); }
            },
        },
        console, alert(message) { throw new Error(message); },
    });
    vm.runInContext(source, context);
    return {
        camera, controls,
        load(size, center = [0, 0, 0]) {
            nextModel = {
                size, center, position: new Vector3(),
                traverse(visit) {
                    visit({ isMesh: true, userData: {}, material: {},
                        geometry: { attributes: { position: { count: 8 } }, index: { count: 36 } } });
                },
            };
            context.loadModel('synthetic.glb', 'synthetic box', 'fixture', 1);
            return nextModel;
        },
    };
}

const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const unit = v => v.map(value => value / Math.hypot(...v));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function assertBoxVisible(f, box) {
    const p = [f.camera.position.x, f.camera.position.y, f.camera.position.z];
    const target = [f.controls.target.x, f.controls.target.y, f.controls.target.z];
    const backward = unit(p.map((value, i) => value - target[i]));
    const right = unit(cross([0, 1, 0], backward));
    const up = cross(backward, right);
    const offset = [box.position.x, box.position.y, box.position.z];
    const [sx, sy, sz, tz] = f.camera.projection;
    assert(f.camera.near > 0 && f.camera.far > f.camera.near);
    // Project all eight corners after the callback recenters the model. Test the actual clip volume,
    // including the refreshed projection, rather than an exact near/far formula.
    for (let corner = 0; corner < 8; corner++) {
        const relative = box.size.map((size, i) =>
            box.center[i] + offset[i] + ((corner & (1 << i)) ? 0.5 : -0.5) * size - p[i]);
        const x = dot(right, relative), y = dot(up, relative), z = dot(backward, relative);
        const w = -z;
        assert(w > 0, 'a fitted corner must be in front of the camera');
        for (const value of [sx * x, sy * y, sz * z + tz])
            assert(Math.abs(value) <= w + 1e-9,
                `corner ${corner} is clipped: near=${f.camera.near}, far=${f.camera.far}, clip=${value}, w=${w}`);
    }
}

for (const extent of [0.01, 10, 1000]) {
    test(`Studio fit keeps a ${extent} m model inside the camera frustum`, () => {
        const f = fixture();
        const before = f.camera.projectionUpdates;
        const model = f.load([extent, extent, extent]);
        assertBoxVisible(f, model);
        assert.equal(f.camera.projectionUpdates, before + 1, 'fitting must refresh the projection');
    });
}

test('Studio refreshes clipping when switching among model sizes', () => {
    const f = fixture();
    for (const extent of [1000, 0.01, 10, 0.01, 1000]) {
        const before = f.camera.projectionUpdates;
        const model = f.load([extent, extent / 2, extent / 4], [40, -20, 70]);
        assertBoxVisible(f, model);
        assert.equal(f.camera.projectionUpdates, before + 1);
    }
});
