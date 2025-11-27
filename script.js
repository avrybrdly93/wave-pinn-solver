// script.js

let waveData = null;
let currentMode = "fd"; // "fd", "pinn", or "error"
let currentIndex = 0;
let isPlaying = false;
let playInterval = null;

let globalZMin = -1;
let globalZMax = 1;
let xMin = 0;
let xMax = 1;
let yMin = 0;
let yMax = 1;
const zScale = 0.3;  // 0.3 = 30% of the original height


const timeSlider   = document.getElementById("timeSlider");
const timeLabel    = document.getElementById("timeLabel");
const modeSelect   = document.getElementById("modeSelect");
const playButton   = document.getElementById("playButton");
const frameCountEl = document.getElementById("frameCount");

// Set footer year
const yearSpan = document.getElementById("year");
if (yearSpan) {
  yearSpan.textContent = new Date().getFullYear();
}

async function loadData() {
  try {
    const response = await fetch("data/wave_data.json");
    if (!response.ok) {
      throw new Error("Failed to load data/wave_data.json");
    }
    waveData = await response.json();

    updateDomainBounds();
    initializeControls();
    initializePlot();
  } catch (err) {
    console.error(err);
    alert("Failed to load wave_data.json. Check console for details.");
  }
}

function initializeControls() {
  const Nt = waveData.t.length;

  timeSlider.min = 0;
  timeSlider.max = Nt - 1;
  timeSlider.value = 0;
  currentIndex = 0;

  updateTimeLabel();

  if (frameCountEl) {
    frameCountEl.textContent = Nt.toString();
  }

  timeSlider.addEventListener("input", () => {
    currentIndex = parseInt(timeSlider.value, 10);
    updateTimeLabel();
    updatePlot();
  });

  modeSelect.addEventListener("change", () => {
    currentMode = modeSelect.value;
    updatePlot();
  });

  playButton.addEventListener("click", togglePlay);
}

function updateTimeLabel() {
  const tVal = waveData.t[currentIndex];
  timeLabel.textContent = `t = ${tVal.toFixed(3)}`;
}

function getZ(mode, idx) {
  const fd   = waveData.fd[idx];   // [Nx][Ny]
  const pinn = waveData.pinn[idx]; // [Nx][Ny]

  if (mode === "fd")   return fd;
  if (mode === "pinn") return pinn;

  // error field: pinn - fd
  const Nx = fd.length;
  const Ny = fd[0].length;
  const err = new Array(Nx);

  for (let i = 0; i < Nx; i++) {
    err[i] = new Array(Ny);
    for (let j = 0; j < Ny; j++) {
      err[i][j] = pinn[i][j] - fd[i][j];
    }
  }
  return err;
}

function initializePlot() {
  const z0 = getZ(currentMode, currentIndex);

  const data = [
    {
      type: "surface",
      x: waveData.x, // 1D arrays
      y: waveData.y,
      z: z0,
      colorscale: "Viridis",
      showscale: true,
      hovertemplate: "x: %{x:.2f} m<br>y: %{y:.2f} m<br>u: %{z:.3f}<extra></extra>",
    },
  ];

    Plotly.newPlot("plot", data, getLayout(), { responsive: true })
    .then(gd => {
      // gd is the graph div with Plotly’s event system wired up
      gd.on("plotly_click", function(event) {
        const pt = event.points[0];
        const x = pt.x;
        const y = pt.y;
        console.log("This was clicked! Nice:", x, y);

        dropPebble(x, y, 0.02, 0.05);
      });
    });


  const layout = getLayout();

  Plotly.newPlot("plot", data, layout, { responsive: true });
}

function updatePlot() {
  if (!waveData) return;

  const rawZ = getZ(currentMode, currentIndex);
  //const z = rawZ.map(row => row.map(v => v * zScale));
  const z = u;
  //z = z * zScale
  const data = [
    {
      type: "surface",
      x: waveData.x,
      y: waveData.y,
      z,
      colorscale: "Viridis",
      showscale: false,
      hovertemplate: "x: %{x:.2f} m<br>y: %{y:.2f} m<br>u: %{z:.3f}<extra></extra>",
    },
  ];

  const layout = getLayout();

  Plotly.react("plot", data, layout, { responsive: true });
}

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
        range: [xMin, xMax],
      },
      yaxis: {
        title: "y (m)",
        backgroundcolor: "#020617",
        gridcolor: "#1f2933",
        range: [yMin, yMax],
      },
      zaxis: {
        title: "u",
        backgroundcolor: "#020617",
        gridcolor: "#1f2933",
        range: [-1,1]
      },
      aspectmode: "cube",
      aspectratio: {
        x: 1,
        y: 1,
        z: 10,
      }
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
  };
}

function togglePlay() {
  if (!waveData) return;

  if (isPlaying) {
    stopPlay();
  } else {
    startPlay();
  }
}

function startPlay() {
  isPlaying = true;
  playButton.textContent = "⏸ Pause";

  const Nt = waveData.t.length;

  playInterval = setInterval(() => {
    currentIndex = (currentIndex + 1) % Nt;
    timeSlider.value = currentIndex;
    updateTimeLabel();
    updatePlot();
  }, 80); // ms per frame
}

function stopPlay() {
  isPlaying = false;
  playButton.textContent = "▶ Play";
  if (playInterval !== null) {
    clearInterval(playInterval);
    playInterval = null;
  }
}

function updateDomainBounds() {
  if (!waveData || !waveData.x || !waveData.y) return;
  xMin = waveData.x[0];
  xMax = waveData.x[waveData.x.length - 1];
  yMin = waveData.y[0];
  yMax = waveData.y[waveData.y.length - 1];
}

function computeGlobalZRange() {
  if (!waveData) return;

  // adjust keys depending on what you have: ["fd", "pinn"], etc.
  ["fd", "pinn"].forEach((key) => {
    if (!waveData[key]) return;
    waveData[key].forEach(frame => {
      frame.forEach(row => {
        row.forEach(value => {
          if (value < globalZMin) globalZMin = value;
          if (value > globalZMax) globalZMax = value;
        });
      });
    });
  });

  // Optional: pad slightly so surface isn't clipped
  const padding = 0.05 * (globalZMax - globalZMin || 1);
  globalZMin -= padding;
  globalZMax += padding;
}




//testing
// Utility to make a 2D array
// ---------------------------------------------------------------------
// Wave PDE solver + interactive ripples
// ---------------------------------------------------------------------

// Utility to make a 2D array
function zeros2D(nx, ny) {
  const a = new Array(nx);
  for (let i = 0; i < nx; i++) {
    a[i] = new Float32Array(ny);
  }
  return a;
}

/**
 * Create a 2D damped wave-equation solver.
 *
 * PDE: u_tt = c^2 (u_xx + u_yy) - gamma * u_t
 *
 * Params:
 *  nx, ny  : grid size
 *  Lx, Ly  : physical size of domain (meters)
 *  c       : wave speed (m/s)
 *  gamma   : damping coefficient
 *  lambda  : Courant number (c * dt / dx), <= 1/sqrt(2)
 */
function createWaveSolver({ nx, ny, Lx, Ly, c, gamma, lambda }) {
  const dx = Lx / (nx - 1);
  const dy = Ly / (ny - 1);
  const dt = lambda * dx / c;  // stability based on dx; assume dx ~ dy

  const x = new Array(nx);
  const y = new Array(ny);
  for (let i = 0; i < nx; i++) x[i] = -Lx / 2 + i * dx;
  for (let j = 0; j < ny; j++) y[j] = -Ly / 2 + j * dy;

  let uPrev = zeros2D(nx, ny);
  let u     = zeros2D(nx, ny);
  let uNext = zeros2D(nx, ny);

  const c2dt2_over_dx2 = (c * c * dt * dt) / (dx * dx);
  const c2dt2_over_dy2 = (c * c * dt * dt) / (dy * dy);
  const dampingTerm    = gamma * dt;

  /**
   * Set initial condition.
   * fn(x, y) should return u(x,y, t=0).
   * We start with u_t = 0, so uPrev = u at t=0.
   */
  function setInitialCondition(fn) {
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const val = fn(x[i], y[j]);
        u[i][j] = val;
        uPrev[i][j] = val;
      }
    }
  }

  /**
   * Advance the wave field by one time step.
   * Returns a reference to the current u field (2D array).
   */
  function step() {
    // interior points
    for (let i = 1; i < nx - 1; i++) {
      for (let j = 1; j < ny - 1; j++) {
        const laplacian =
          (u[i+1][j] - 2 * u[i][j] + u[i-1][j]) * c2dt2_over_dx2 +
          (u[i][j+1] - 2 * u[i][j] + u[i][j-1]) * c2dt2_over_dy2;

        const u_ij_prev = uPrev[i][j];
        const u_ij      = u[i][j];

        uNext[i][j] =
          (2 - dampingTerm) * u_ij
          - (1 - dampingTerm) * u_ij_prev
          + laplacian;
      }
    }

    // fixed boundaries: u = 0 at edges
    for (let i = 0; i < nx; i++) {
      uNext[i][0]      = 0;
      uNext[i][ny - 1] = 0;
    }
    for (let j = 0; j < ny; j++) {
      uNext[0][j]      = 0;
      uNext[nx - 1][j] = 0;
    }

    // rotate buffers
    const tmp = uPrev;
    uPrev = u;
    u = uNext;
    uNext = tmp;

    return u;
  }

  // Expose x, y, current u, and uPrev via getters
  return {
    x, y, dt,
    get u() { return u; },
    get uPrev() { return uPrev; },
    setInitialCondition,
    step,
  };
}

// 1. Create the solver
const solver = createWaveSolver({
  nx: 101,
  ny: 101,
  Lx: 1.0,
  Ly: 1.0,
  c: Math.sqrt(9.81 * 0.05),  // shallow water: h = 5 cm
  gamma: 0.5,
  lambda: 0.4                // Courant number (<= ~0.7 in 2D)
});

// 2. Gaussian "drop" initial condition
const A0 = 0.01;   // amplitude
const sigma0 = 0.05;

solver.setInitialCondition((x, y) => {
  const r2 = x*x + y*y;
  return A0 * Math.exp(-r2 / (sigma0 * sigma0));
});

// 3. Layout and plotting for the solver
function getSolverLayout() {
  return {
    scene: {
      xaxis: { title: "x (m)" },
      yaxis: { title: "y (m)" },
      zaxis: { title: "u (m)", range: [-A0, A0] }
    },
    margin: { l: 0, r: 0, t: 0, b: 0 }
  };
}

function updateSolverPlot() {
  const data = [{
    type: "surface",
    x: solver.x,
    y: solver.y,
    z: solver.u,
    colorscale: "Viridis",
    showscale: false
  }];

  Plotly.react("plot", data, getSolverLayout(), { responsive: true });
}

// 4. Pebble drop: use solver.x, solver.y, solver.u, solver.uPrev
function dropPebble(x0, y0, A = 0.02, sigma = 0.05) {
  const xArr   = solver.x;
  const yArr   = solver.y;
  const u      = solver.u;
  const uPrev  = solver.uPrev;

  const Nx = xArr.length;
  const Ny = yArr.length;

  for (let i = 0; i < Nx; i++) {
    const dx = xArr[i] - x0;
    for (let j = 0; j < Ny; j++) {
      const dy = yArr[j] - y0;
      u[i][j] += A * Math.exp(-(dx*dx + dy*dy) / (sigma * sigma));
      // keep previous field in sync so this is a displacement, not a huge velocity
      uPrev[i][j] = u[i][j];
    }
  }
}

// 5. Initialize Plotly once, attach click, then animate
function initSolverPlot() {
  const z0 = transpose2D(solver.u);  // <--- same idea
  const data = [{
    type: "surface",
    x: solver.x,
    y: solver.y,
    z: z0,
    colorscale: "Viridis",
    showscale: false
  }];

  Plotly.newPlot("plot", data, getSolverLayout(), { responsive: true })
    .then(gd => {
      gd.on("plotly_click", evt => {
        const pt = evt.points[0];
        const x0 = pt.x;
        const y0 = pt.y;
        console.log("This was clicked! Nice:", x0, y0);
        dropPebble(x0, y0, 0.02, 0.05);
      });

      // Start animation loop after the first render
      function animate() {
        solver.step();
        updateSolverPlot();
        requestAnimationFrame(animate);
      }
      animate();
    });
}

function transpose2D(a) {
  const Nx = a.length;
  const Ny = a[0].length;
  const out = new Array(Ny);
  for (let j = 0; j < Ny; j++) {
    out[j] = new Array(Nx);
    for (let i = 0; i < Nx; i++) {
      out[j][i] = a[i][j];
    }
  }
  return out;
}


// Run the solver visualization once the page is loaded.
// (You can comment out the old loadData if you want while testing.)
window.addEventListener("load", () => {
  initSolverPlot();
  // If you don’t want the JSON viewer right now, you can comment out:
  // loadData();
});
