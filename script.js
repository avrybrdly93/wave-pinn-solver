// ================================
//  Backend-Driven Wave Simulator
// ================================

const API_BASE = "https://wave-pinn-solver.onrender.com";

const Lx = 10.0;
const Ly = 10.0;

// Plot / grid state
let plotInitialized = false;
let Nx = 0;
let Ny = 0;
let xCoords = [];
let yCoords = [];

// ===========================================
//  API calls to backend FastAPI wave solver
// ===========================================

async function apiReset() {
  await fetch(`${API_BASE}/reset`, { method: "POST" });
}

async function apiDrop(x, y, A = 0.02, sigma = 0.05) {
  await fetch(`${API_BASE}/drop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x, y, A, sigma }),
  });
}

async function apiStep(steps = 1) {
  await fetch(`${API_BASE}/step?steps=${steps}`, {
    method: "POST",
  });
}

async function apiGetFrame() {
  const res = await fetch(`${API_BASE}/frame`);
  return await res.json(); // { time, u }
}

// =================================================
//  PLOTTING HELPERS
// =================================================

function getLayout() {
  return {
    title: "",
    autosize: true,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      xaxis: {
        title: "x (m)",
        backgroundcolor: "#020617",
        gridcolor: "#1f2933",
        range: [0, Lx],
      },
      yaxis: {
        title: "y (m)",
        backgroundcolor: "#020617",
        gridcolor: "#1f2933",
        range: [0, Ly],
      },
      zaxis: {
        title: "u",
        backgroundcolor: "#020617",
        gridcolor: "#1f2933",
        // tweak as needed depending on amplitude
        range: [-0.1, 0.1],
      },
      aspectmode: "cube",
      aspectratio: { x: 1, y: 1, z: 0.5 },
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
  };
}

// Initialize Plotly using the first frame from the backend
async function initializePlot() {
  console.log("Initializing backend simulation...");

  await apiReset();
  const frame = await apiGetFrame(); // { time, u }

  const zData = frame.u;
  Nx = zData.length;
  Ny = zData[0].length;

  // Physical coordinates to match backend domain [0, Lx], [0, Ly]
  xCoords = Array.from({ length: Nx }, (_, i) => (Lx * i) / (Nx - 1));
  yCoords = Array.from({ length: Ny }, (_, j) => (Ly * j) / (Ny - 1));

  const data = [
    {
      type: "surface",
      x: xCoords,
      y: yCoords,
      z: zData,
      colorscale: "Viridis",
      showscale: false,
      hovertemplate: "x: %{x:.2f}<br>y: %{y:.2f}<br>u: %{z:.3f}<extra></extra>",
    },
  ];

  const layout = getLayout();

  Plotly.newPlot("plot", data, layout, { responsive: true }).then((gd) => {
    plotInitialized = true;

    // Click → drop pebble on backend
    gd.on("plotly_click", async (evt) => {
      const pt = evt.points[0];
      const x0 = pt.x;
      const y0 = pt.y;

      console.log("Dropping pebble at:", x0, y0);
      try {
        await apiDrop(x0, y0, 0.02, 0.05);
      } catch (err) {
        console.error("Error calling /drop:", err);
      }
    });
  });
}

// Update plot with new z data from backend
function updatePlotWithBackend(zData) {
  const data = [
    {
      type: "surface",
      x: xCoords,
      y: yCoords,
      z: zData,
      colorscale: "Viridis",
      showscale: false,
    },
  ];

  Plotly.react("plot", data, getLayout(), { responsive: true });
}

// Continuous animation loop
async function animateLoop() {
  if (!plotInitialized) {
    requestAnimationFrame(animateLoop);
    return;
  }

  try {
    // Advance backend simulation
    await apiStep(1); // try 2 or 3 for faster waves

    // Get new frame and update surface
    const frame = await apiGetFrame();
    updatePlotWithBackend(frame.u);
  } catch (err) {
    console.error("Error in animation loop:", err);
  }

  requestAnimationFrame(animateLoop);
}

// =================================================
//  MAIN ENTRY
// =================================================

window.addEventListener("load", async () => {
  await initializePlot();
  requestAnimationFrame(animateLoop);
});
