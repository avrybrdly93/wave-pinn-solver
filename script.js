// ================================
//  Backend-Driven Wave Simulator (Three.js)
// ================================

//import * as THREE from './lib/three.module.js';
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@latest/build/three.module.js';  
import { OrbitControls } from './lib/OrbitControls.js';

const BACKENDS = {
  render: "https://wave-pinn-solver.onrender.com",
  local: "http://localhost:8000",
};

const isLocalHost =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";

function resolveInitialBackend() {
  const urlBackend = new URLSearchParams(window.location.search).get("backend");
  if (urlBackend && BACKENDS[urlBackend]) return urlBackend;

  const stored = localStorage.getItem("backendTarget");
  if (stored && BACKENDS[stored]) return stored;

  return isLocalHost ? "local" : "render";
}

let backendTarget = resolveInitialBackend();

function apiBase() {
  return BACKENDS[backendTarget];
}

function apiUrl(path) {
  return `${apiBase()}${path}`;
}

const Lx = 10.0;
const Ly = 10.0;
const HEIGHT_SCALE = 6.0; // exaggerate vertical displacement for visibility

let plotInitialized = false;
let Nx = 0;
let Ny = 0;

// Three.js scene elements
let scene;
let camera;
let renderer;
let controls;
let mesh;
let container;
let raycaster;
let positionAttribute;
let colorAttribute;

// ===========================================
//  API calls to backend FastAPI wave solver
// ===========================================

async function apiReset() {
  await fetch(apiUrl("/reset"), { method: "POST" });
}

async function apiDrop(x, y, A = 0.02, sigma = 0.05) {
  await fetch(apiUrl("/drop"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x, y, A, sigma }),
  });
}

async function apiStep(steps = 1) {
  await fetch(apiUrl(`/step?steps=${steps}`), {
    method: "POST",
  });
}

async function apiGetFrame() {
  const res = await fetch(apiUrl("/frame"));
  return await res.json(); // { time, u, x, y }
}

// =================================================
//  HELPERS
// =================================================

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function heightToColor(h) {
  // Map height -> simple blue->cyan->yellow palette
  const minH = -0.1;
  const maxH = 0.1;
  const t = clamp((h - minH) / (maxH - minH), 0, 1);

  const c1 = { r: 0.1, g: 0.2, b: 0.7 }; // low: deep blue
  const c2 = { r: 0.0, g: 0.8, b: 0.8 }; // mid: cyan
  const c3 = { r: 1.0, g: 0.9, b: 0.2 }; // high: yellow

  let r;
  let g;
  let b;
  if (t < 0.5) {
    const k = t * 2;
    r = c1.r + (c2.r - c1.r) * k;
    g = c1.g + (c2.g - c1.g) * k;
    b = c1.b + (c2.b - c1.b) * k;
  } else {
    const k = (t - 0.5) * 2;
    r = c2.r + (c3.r - c2.r) * k;
    g = c2.g + (c3.g - c2.g) * k;
    b = c2.b + (c3.b - c2.b) * k;
  }

  return { r, g, b };
}

function updateSurfaceHeights(zData) {
  if (!mesh) return;

  const positions = positionAttribute.array;
  const colors = colorAttribute.array;

  for (let i = 0; i < Nx; i++) {
    for (let j = 0; j < Ny; j++) {
      const idx = i * Ny + j;
      const posIndex = idx * 3 + 2; // z component

      const h = zData[i][j] * HEIGHT_SCALE;
      positions[posIndex] = h;

      const { r, g, b } = heightToColor(zData[i][j]);
      const colorIndex = idx * 3;
      colors[colorIndex] = r;
      colors[colorIndex + 1] = g;
      colors[colorIndex + 2] = b;
    }
  }

  positionAttribute.needsUpdate = true;
  colorAttribute.needsUpdate = true;
  mesh.geometry.computeVertexNormals();
}

function createScene(initialZ) {
  container = document.getElementById("plot");
  if (!container) {
    throw new Error("Missing #plot container");
  }

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1221);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 200);
  camera.position.set(Lx * 0.6, -Ly * 1.2, 12);
  camera.up.set(0, 0, 1); // z-up

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(Lx / 2, Ly / 2, 0);
  controls.enableDamping = true;
  controls.enablePan = false;

  raycaster = new THREE.Raycaster();

  const geometry = new THREE.PlaneGeometry(Lx, Ly, Nx - 1, Ny - 1);
  geometry.translate(Lx / 2, Ly / 2, 0); // shift to [0, Lx] x [0, Ly]

  const colors = new Float32Array(Nx * Ny * 3);
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    flatShading: false,
    side: THREE.DoubleSide,
  });

  mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  positionAttribute = mesh.geometry.attributes.position;
  colorAttribute = mesh.geometry.attributes.color;
  updateSurfaceHeights(initialZ);

  const hemiLight = new THREE.HemisphereLight(0xb0c4ff, 0x0b1221, 0.7);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
  dirLight.position.set(-Lx, -Ly, 20);
  scene.add(dirLight);

  const grid = new THREE.GridHelper(Math.max(Lx, Ly), 10, 0x1f2937, 0x1f2937);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(Lx / 2, Ly / 2, -0.01);
  scene.add(grid);

  renderer.render(scene, camera);

  renderer.domElement.addEventListener("pointerdown", (evt) => {
    handlePointer(evt);
  });

  window.addEventListener("resize", onResize);
}

function onResize() {
  if (!renderer || !camera || !container) return;
  const width = container.clientWidth || window.innerWidth;
  const height = container.clientHeight || window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function handlePointer(event) {
  if (!mesh || !container) return;

  const rect = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(mesh);
  if (hits.length === 0) return;

  const point = hits[0].point;
  apiDrop(point.x, point.y, 0.02, 0.05).catch((err) => {
    console.error("Error calling /drop:", err);
  });
}

// Initialize scene using the first frame
async function initializePlot() {
  console.log("Initializing backend simulation...");

  await apiReset();
  const frame = await apiGetFrame(); // { u, x, y }

  const zData = frame.u;
  Nx = zData.length;
  Ny = zData[0].length;

  createScene(zData);
  plotInitialized = true;
}

// Continuous animation loop
async function animateLoop() {
  if (controls) controls.update();

  if (plotInitialized) {
    try {
      await apiStep(1); // adjust steps for speed
      const frame = await apiGetFrame();
      updateSurfaceHeights(frame.u);
    } catch (err) {
      console.error("Error in animation loop:", err);
    }
  }

  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }

  requestAnimationFrame(animateLoop);
}

// =================================================
//  MAIN ENTRY
// =================================================

window.addEventListener("load", async () => {
  try {
    await initializePlot();
    requestAnimationFrame(animateLoop);
  } catch (err) {
    console.error("Failed to initialize Three.js viewer:", err);
  }
});

function setBackend(target) {
  if (!BACKENDS[target]) {
    console.warn(`Unknown backend: ${target}`);
    return;
  }

  if (backendTarget === target) return;

  backendTarget = target;
  localStorage.setItem("backendTarget", target);

  const select = document.getElementById("backendSelect");
  if (select) select.value = target;

  plotInitialized = false;

  initializePlot().catch((err) => {
    console.error(`Failed to initialize backend ${target}:`, err);
  });
}

document.getElementById("backendSelect")?.addEventListener("change", (evt) => {
  setBackend(evt.target.value);
});

const backendSelect = document.getElementById("backendSelect");
if (backendSelect) {
  backendSelect.value = backendTarget;
}
