const byId = id => document.getElementById(id);
const category = byId("category");
const search = byId("search");
const modelStage = byId("model-stage");
const imageStage = byId("image-stage");
const modelStatus = byId("model-status");
const modelError = byId("model-error");
const imageStatus = byId("image-status");
const interactive = byId("interactive");
const translated = byId("translated");
const fit = byId("fit");
const png = byId("png");

const categories = new Map([
  ["props", "props"],
  ["pedparts", "ped-parts"], ["peds", "ped-parts"],
  ["vehicles", "vehicles"], ["cars", "vehicles"],
]);
const assets = [];
const pageSize = 12;
let page = 0;
let selected;
let loadId = 0;
let imageId = 0;
let imageUrl;
let viewer;
let viewerPromise;

function catalogEntries(files, fallbackCategory) {
  const paths = Array.from(files, file => [file.webkitRelativePath || file.name, file]);
  // Match only within this selection. Loose Files have no directory path in browsers.
  const previews = new Map(paths.filter(([path]) => /\.shader\.png$/i.test(path))
    .map(([path, file]) => [path.toLowerCase(), file]));
  const added = [];
  for (const [path, file] of paths) {
    if (!/\.glb$/i.test(file.name)) continue;
    const folders = path.split("/").slice(0, -1).reverse();
    const folderCategory = folders.map(folder => categories.get(folder.toLowerCase().replace(/[ _-]/g, "")))
      .find(Boolean);
    const preview = previews.get(path.replace(/\.glb$/i, ".shader.png").toLowerCase());
    added.push({ path, file, preview, category: folderCategory || fallbackCategory });
  }
  return added;
}

function addFiles(files) {
  const added = catalogEntries(files, category.value);
  const status = byId("import-status");
  status.hidden = false;
  status.textContent = added.length ? `Added ${added.length} GLB ${added.length === 1 ? "file" : "files"}.` :
    "This selection contains no GLB files. Add a GLB to the library first.";
  if (!added.length) return;
  for (const asset of added) assets.push(asset);
  assets.sort((a, b) => a.file.name.localeCompare(b.file.name, undefined, { numeric: true }) ||
    a.path.localeCompare(b.path));
  if (!added.some(asset => asset.category === category.value)) category.value = added[0].category;
  search.value = "";
  page = 0;
  if (selected?.category !== category.value) selectAsset(added.find(asset => asset.category === category.value));
  else renderList();
}

function renderList() {
  const available = assets.filter(asset => asset.category === category.value);
  const query = search.value.trim().toLowerCase();
  const matches = available.filter(asset => asset.file.name.toLowerCase().includes(query));
  const pageCount = Math.max(1, Math.ceil(matches.length / pageSize));
  page = Math.min(page, pageCount - 1);
  byId("results").textContent = matches.length ? `${matches.length} ${matches.length === 1 ? "model" : "models"}` :
    available.length ? "No models match this search." : "No models in this category.";
  const list = byId("models");
  list.replaceChildren();
  for (const asset of matches.slice(page * pageSize, (page + 1) * pageSize)) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = asset.file.name;
    button.setAttribute("aria-pressed", String(asset === selected));
    if (asset.path !== asset.file.name) {
      const folder = document.createElement("span");
      folder.className = "path";
      folder.textContent = asset.path.slice(0, asset.path.lastIndexOf("/"));
      button.append(folder);
    }
    button.onclick = () => {
      if (asset === selected) return;
      list.querySelector('[aria-pressed="true"]')?.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-pressed", "true");
      selectAsset(asset, false);
    };
    item.append(button);
    list.append(item);
  }
  byId("pager").hidden = pageCount === 1;
  byId("previous").disabled = page === 0;
  byId("next").disabled = page + 1 === pageCount;
  byId("page").textContent = `Page ${page + 1} of ${pageCount}`;
}

function clearImage() {
  imageStage.querySelector("img")?.remove();
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = undefined;
}

async function showMode(mode) {
  const id = ++imageId;
  const showingImage = mode === "image";
  interactive.setAttribute("aria-pressed", String(!showingImage));
  translated.setAttribute("aria-pressed", String(showingImage));
  modelStage.hidden = showingImage;
  imageStage.hidden = !showingImage;
  byId("orbit-tools").hidden = showingImage;
  const capture = byId("capture");
  capture.hidden = !showingImage;
  clearImage();
  if (!showingImage) {
    viewer?.resize();
    return;
  }
  const asset = selected;
  const file = asset.preview;
  capture.textContent = `Offline capture · ${file.name}`;
  imageStatus.textContent = "Loading preview PNG…";
  const image = new Image();
  image.alt = `Offline translated Vulkan capture for ${asset.file.name}`;
  try {
    const signature = new Uint8Array(await file.slice(0, 8).arrayBuffer());
    if (id !== imageId) return;
    if (signature.join() !== "137,80,78,71,13,10,26,10") throw new Error("Invalid PNG signature");
    imageUrl = URL.createObjectURL(file);
    image.src = imageUrl;
    await image.decode();
    if (id !== imageId) return;
    imageStage.append(image);
    imageStatus.textContent = "";
  } catch {
    if (id === imageId) {
      clearImage();
      imageStatus.textContent = "Could not open this PNG. Choose another preview image.";
    }
  }
}

async function selectAsset(asset, updateList = true) {
  selected = asset;
  const id = ++loadId;
  viewer?.clear();
  fit.disabled = true;
  interactive.disabled = !asset;
  translated.disabled = !asset?.preview;
  byId("attachment").hidden = !asset;
  byId("name").textContent = asset?.file.name || "Asset preview";
  showMode("model");
  modelError.hidden = true;
  modelStatus.textContent = asset ? "Loading GLB…" : "Select a GLB to view its model and materials.";
  if (updateList) renderList();
  if (!asset) return;
  try {
    const bytes = await asset.file.arrayBuffer();
    if (id !== loadId) return;
    checkLocalGLB(bytes);
    viewerPromise ??= createViewer().catch(error => {
      viewerPromise = undefined;
      throw error;
    });
    viewer = await viewerPromise;
    if (id !== loadId) return;
    const gltf = await viewer.loader.parseAsync(bytes, "");
    if (id !== loadId) {
      disposeScenes(gltf.scenes);
      return;
    }
    viewer.show(gltf);
    modelStatus.textContent = "";
    fit.disabled = false;
  } catch (error) {
    if (id !== loadId) return;
    viewer?.clear();
    modelStatus.textContent = "";
    modelError.textContent = `Could not open GLB. ${error.message || "Check the selected file."}`;
    modelError.hidden = false;
  }
}

// GLTFLoader also supports JSON glTF and external images and buffers. Check the binary header and
// the JSON chunk before handing it the bytes so opening an asset never requests a sidecar.
function checkLocalGLB(bytes) {
  const view = new DataView(bytes);
  if (view.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67 ||
      view.getUint32(4, true) !== 2 || view.getUint32(8, true) !== view.byteLength ||
      view.getUint32(16, true) !== 0x4e4f534a || view.getUint32(12, true) > view.byteLength - 20)
    throw new Error("Choose a valid binary glTF 2.0 (.glb) file.");
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 20, view.getUint32(12, true))));
  for (const resource of [...(json.buffers || []), ...(json.images || [])]) {
    if (resource.uri !== undefined && (typeof resource.uri !== "string" || !resource.uri.startsWith("data:")))
      throw new Error("Export a self-contained GLB with embedded buffers and images.");
  }
}

function disposeScenes(scenes) {
  const resources = new Set();
  const bitmaps = new Set();
  for (const scene of scenes) scene.traverse(object => {
    if (object.geometry) resources.add(object.geometry);
    if (object.skeleton) resources.add(object.skeleton);
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (!material) continue;
      resources.add(material);
      for (const value of Object.values(material)) {
        if (!value?.isTexture) continue;
        resources.add(value);
        if (value.image?.close) bitmaps.add(value.image);
      }
    }
  });
  for (const resource of resources) resource.dispose();
  for (const bitmap of bitmaps) bitmap.close();
}

async function createViewer() {
  const [THREE, { GLTFLoader }, { OrbitControls }, { RoomEnvironment }] = await Promise.all([
    import("three"), import("three/addons/loaders/GLTFLoader.js"),
    import("three/addons/controls/OrbitControls.js"), import("three/addons/environments/RoomEnvironment.js"),
  ]).catch(() => { throw new Error("Three.js could not load. Check access to the viewer’s pinned CDN."); });
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true });
  } catch {
    throw new Error("WebGL is unavailable in this browser.");
  }
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.setAttribute("aria-label", "Interactive 3D model");
  canvas.setAttribute("aria-describedby", "orbit-help");
  canvas.hidden = true;
  modelStage.prepend(canvas);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x10151c);
  // A neutral room gives metallic glTF surfaces something to reflect without a downloaded environment.
  const room = new RoomEnvironment();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(room, 0.04).texture;
  room.dispose();
  pmrem.dispose();
  const key = new THREE.DirectionalLight(0xfff1df, 1.7);
  key.position.set(3, 5, 4);
  scene.add(key, new THREE.HemisphereLight(0xdde9ff, 0x6b6258, 0.75));
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
  const controls = new OrbitControls(camera, canvas);
  const manager = new THREE.LoadingManager();
  manager.setURLModifier(url => {
    if (/^(blob:|data:)/.test(url)) return url;
    throw new Error("Export a self-contained GLB with embedded buffers and images.");
  });
  const loader = new GLTFLoader(manager);
  let current;
  let bounds;

  function render() {
    if (current && !modelStage.hidden) renderer.render(scene, camera);
  }

  function resize() {
    const { clientWidth: width, clientHeight: height } = modelStage;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    render();
  }

  function fitModel() {
    if (!bounds) return;
    const { center, radius } = bounds;
    const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
    const halfAngle = Math.min(halfFov, Math.atan(Math.tan(halfFov) * camera.aspect));
    const distance = 1.12 * radius / Math.sin(halfAngle);
    const direction = new THREE.Vector3(1, 0.65, 1).normalize();
    camera.position.copy(center).addScaledVector(direction, distance);
    camera.near = radius / 500;
    camera.far = distance + radius * 100;
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.minDistance = radius / 10;
    controls.maxDistance = distance + radius * 10;
    controls.update();
    render();
  }

  canvas.addEventListener("keydown", event => {
    if (!current || event.altKey || event.ctrlKey || event.metaKey) return;
    const offset = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));
    switch (event.key) {
      case "ArrowLeft": offset.theta -= 0.15; break;
      case "ArrowRight": offset.theta += 0.15; break;
      case "ArrowUp": offset.phi -= 0.15; break;
      case "ArrowDown": offset.phi += 0.15; break;
      case "+": case "=": offset.radius *= 0.85; break;
      case "-": offset.radius /= 0.85; break;
      default: return;
    }
    event.preventDefault();
    camera.position.setFromSpherical(offset.makeSafe()).add(controls.target);
    controls.update();
  });
  controls.addEventListener("change", render);
  new ResizeObserver(resize).observe(modelStage);

  return {
    loader, resize, fitModel,
    show(gltf) {
      current = gltf;
      if (!gltf.scene) throw new Error("This GLB has no default scene.");
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const sphere = box.getBoundingSphere(new THREE.Sphere());
      if (box.isEmpty() || !Number.isFinite(sphere.radius) || sphere.radius <= 0)
        throw new Error("This GLB has no drawable bounds.");
      bounds = sphere;
      scene.add(gltf.scene);
      canvas.hidden = false;
      if (!modelStage.hidden) resize();
      fitModel();
    },
    clear() {
      if (current?.scene) scene.remove(current.scene);
      if (current) disposeScenes(current.scenes);
      current = bounds = undefined;
      canvas.hidden = true;
    },
  };
}

for (const input of [byId("files"), byId("folder")]) {
  input.onchange = () => {
    if (input.files.length) addFiles(input.files);
    input.value = "";
  };
}
category.onchange = () => {
  page = 0;
  search.value = "";
  selectAsset(undefined);
};
search.oninput = () => { page = 0; renderList(); };
byId("previous").onclick = () => { page--; renderList(); };
byId("next").onclick = () => { page++; renderList(); };
interactive.onclick = () => showMode("model");
translated.onclick = () => showMode("image");
fit.onclick = () => viewer?.fitModel();
png.onchange = () => {
  const file = png.files[0];
  png.value = "";
  if (!selected || !file) return;
  selected.preview = file;
  translated.disabled = false;
  showMode("image");
};
