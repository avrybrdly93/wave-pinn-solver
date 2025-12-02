// ================================
//  Backend-Driven Wave Simulator
// ================================

const API_BASE = "https://wave-pinn-solver.onrender.com";

// Plot state
let plotInitialized = false;
let Nx = 0;
let Ny = 0;

// ===========================================
// API calls to backend FastAPI wave solver
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

// Build Plotly layout for 3D surface
function getLayout() {
  return {
    title: "",
    autosize: true,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      xaxis: {
        title: "x",
        backgroundcolor: "#020617",
        gridcolor: "#1f2933",
      },
      yaxis: {
        title: "y",
        backgroundcolor: "#020617",
        gridcolor: "#1f2933",
      },
      zaxis: {
        title: "u",
        backgroundcolor: "#020617",
        gridcolor: "#1f2933",
        range: [-0.5, 0.5],
      },
      aspectmode: "cube",
      aspectratio: { x: 1, y: 1, z: 0.5 },
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
  };
}

// =================================================
//  INITIALIZE PLOTLY WITH FIRST FRAME
// =================================================

async function initializePlot() {
  console.log("Initializing backend simulation...");

  await apiReset();
  const frame = await apiGetFrame();

  // Convert frame.u to Plotly-friendly structure
  const zData = frame.u;

  Nx = zData.length;
  Ny = zData[0].length;

  const x = [...Array(Nx).keys()];
  const y = [...Array(Ny).keys()];

  const data = [
    {
      type: "surface",
      x: x,
      y: y,
      z: zData,
      colorscale: "Viridis",
      showscale: false,
      hovertemplate: "x: %{x}<br>y: %{y}<br>u: %{z:.3f}<extra></extra>",
    },
  ];

  const layout = getLayout();

  Plotly.newPlot("plot", data, layout, { responsive: true }).then((gd) => {
    plotInitialized = true;

    // ===========================
    //    Click → drop pebble
    // ===========================
    gd.on("plotly_click", async (evt) => {
      const pt = evt.points[0];
      const x0 = pt.x;
      const y0 = pt.y;

      console.log("Dropping pebble at:", x0, y0);

      await apiDrop(x0, y0, 0.02, 0.05);
    });
  });
}

// =================================================
//  LIVE ANIMATION LOOP
// =================================================

async function animateLoop() {
  if (!plotInitialized) {
    requestAnimationFrame(animateLoop);
    return;
  }

  // Step simulation on backend
  await apiStep(1);

  // Get updated frame
  const frame = await apiGetFrame();

  // Update surface
  Plotly.react(
    "plot",
    [
      {
        type: "surface",
        x: [...Array(Nx).keys()],
        y: [...Array(Ny).keys()],
        z: frame.u,
        colorscale: "Viridis",
        showscale: false,
      },
    ],
    getLayout(),
    { responsive: true }
  );

  requestAnimationFrame(animateLoop);
}

// =================================================
//  MAIN ENTRY
// =================================================

window.addEventListener("load", async () => {
  await initializePlot();
  animateLoop(); // continuous animation
});
