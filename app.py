# =============================================================================
#  🤖 ROBOT VACUUM CLEANER — ZERO FLICKER EDITION
#  Teknik: Flask + MJPEG Stream
#  Gambar dikirim sebagai video stream (multipart/x-mixed-replace)
#  persis seperti live CCTV — TIDAK PERNAH hilang timbul, smooth 30fps
#  Jalankan: python robot_vacuum_flask.py
#  Buka: http://localhost:5000
# =============================================================================

import io
import time
import threading
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from flask import Flask, Response, render_template_string

matplotlib.use("Agg")

# =============================================================================
#  KONFIGURASI
# =============================================================================
ROOM_W       = 20
ROOM_H       = 15
GRID_RES     = 0.25

ROBOT_R      = 0.45
ROBOT_SPEED  = 0.12
ACCEL        = 0.22
ANGLE_LRP    = 0.18
SENSOR_R     = 2.0
CLEAN_R      = 0.65

SUBSTEPS     = 4
TRAIL_MAX    = 300

TARGET_FPS   = 30         # frame per detik stream
SIM_STEPS    = 3          # langkah simulasi per frame

C_BG         = "#080c14"
C_DIRTY      = "#111827"
C_CLEAN      = "#0e4d6e"
C_OBS_BASE   = "#1f2937"
C_ROBOT      = "#f43f5e"
C_GLOW       = "#fb7185"
C_TRAIL      = "#22d3ee"
C_SENSOR     = "#fbbf24"
C_GRID       = "#1e2a3a"

RENDER_DPI   = 90
FIG_W        = 12
FIG_H        = 7.5


# =============================================================================
#  OBSTACLE
# =============================================================================
class Obstacle:
    def __init__(self, x, y, w, h, label="", color="#533483"):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.label     = label
        self.color     = color
        self._mg       = ROBOT_R + 0.18

    @property
    def ex0(self): return self.x - self._mg
    @property
    def ey0(self): return self.y - self._mg
    @property
    def ex1(self): return self.x + self.w + self._mg
    @property
    def ey1(self): return self.y + self.h + self._mg

    def contains(self, px, py):
        return self.ex0 <= px <= self.ex1 and self.ey0 <= py <= self.ey1

    def dist_to(self, px, py):
        cx = max(self.x, min(px, self.x + self.w))
        cy = max(self.y, min(py, self.y + self.h))
        return np.hypot(px - cx, py - cy)

    def repulsion(self, px, py):
        d = self.dist_to(px, py)
        if d >= SENSOR_R:
            return 0.0, 0.0
        ocx = self.x + self.w / 2
        ocy = self.y + self.h / 2
        dx, dy  = px - ocx, py - ocy
        norm    = np.hypot(dx, dy) + 1e-9
        strength = ((SENSOR_R - d) / SENSOR_R) ** 2 * 2.2
        return (dx / norm) * strength, (dy / norm) * strength


# =============================================================================
#  ROBOT
# =============================================================================
class CleaningRobot:
    def __init__(self, x, y, obstacles, grid_cols, grid_rows):
        self.x  = float(x)
        self.y  = float(y)
        init_a  = np.random.uniform(0, 2 * np.pi)
        self.vx = ROBOT_SPEED * np.cos(init_a)
        self.vy = ROBOT_SPEED * np.sin(init_a)
        self.angle = init_a

        self.obstacles   = obstacles
        self.grid_cols   = grid_cols
        self.grid_rows   = grid_rows
        self.step_count  = 0
        self.stuck_timer = 0

        self.trail_x: list = []
        self.trail_y: list = []

        self._wander_angle = np.random.uniform(0, 2 * np.pi)
        self._wander_timer = 0

    def _desired_vel(self):
        self._wander_timer += 1
        if self._wander_timer >= 40:
            self._wander_angle += np.random.uniform(-0.6, 0.6)
            self._wander_timer  = 0

        wx = np.cos(self._wander_angle)
        wy = np.sin(self._wander_angle)

        rep_x, rep_y = 0.0, 0.0
        for obs in self.obstacles:
            rx, ry  = obs.repulsion(self.x, self.y)
            rep_x  += rx
            rep_y  += ry

        margin = ROBOT_R + 0.6
        def wf(d): return ((margin - d) / margin) ** 2 * 2.5 if d < margin else 0.0

        wall_x = wf(self.x) - wf(ROOM_W - self.x)
        wall_y = wf(self.y) - wf(ROOM_H - self.y)

        tx  = wx + rep_x + wall_x
        ty  = wy + rep_y + wall_y
        mag = np.hypot(tx, ty) + 1e-9
        return (tx / mag) * ROBOT_SPEED, (ty / mag) * ROBOT_SPEED

    def _substep(self, dvx, dvy, clean_grid):
        self.vx += (dvx - self.vx) * ACCEL
        self.vy += (dvy - self.vy) * ACCEL

        scale = 1.0 / SUBSTEPS
        nx    = self.x + self.vx * scale
        ny    = self.y + self.vy * scale

        r  = ROBOT_R
        nx = np.clip(nx, r, ROOM_W - r)
        ny = np.clip(ny, r, ROOM_H - r)

        hit_x = any(obs.contains(nx, self.y) for obs in self.obstacles)
        hit_y = any(obs.contains(self.x, ny) for obs in self.obstacles)

        if not hit_x:
            self.x = nx
        else:
            self.vx *= -0.4
            self._wander_angle += np.pi / 2 + np.random.uniform(-0.5, 0.5)
            self.stuck_timer   += 1

        if not hit_y:
            self.y = ny
        else:
            self.vy *= -0.4
            self._wander_angle += np.pi / 2 + np.random.uniform(-0.5, 0.5)
            self.stuck_timer   += 1

        if abs(self.vx) > 1e-4 or abs(self.vy) > 1e-4:
            target = np.arctan2(self.vy, self.vx)
            delta  = (target - self.angle + np.pi) % (2 * np.pi) - np.pi
            self.angle += delta * ANGLE_LRP

        self._mark_clean(clean_grid)

    def step(self, clean_grid):
        dvx, dvy = self._desired_vel()
        for _ in range(SUBSTEPS):
            self._substep(dvx, dvy, clean_grid)

        self.step_count += 1
        self.trail_x.append(self.x)
        self.trail_y.append(self.y)
        if len(self.trail_x) > TRAIL_MAX:
            self.trail_x.pop(0)
            self.trail_y.pop(0)

        if self.stuck_timer > 12:
            self._wander_angle = np.random.uniform(0, 2 * np.pi)
            self.vx            = np.cos(self._wander_angle) * ROBOT_SPEED
            self.vy            = np.sin(self._wander_angle) * ROBOT_SPEED
            self.stuck_timer   = 0

    def _mark_clean(self, clean_grid):
        cx   = int(self.x / GRID_RES)
        cy   = int(self.y / GRID_RES)
        cr   = int(CLEAN_R / GRID_RES) + 1
        rows, cols = clean_grid.shape
        for dr in range(-cr, cr + 1):
            for dc in range(-cr, cr + 1):
                gr, gc = cy + dr, cx + dc
                if 0 <= gr < rows and 0 <= gc < cols:
                    cell_x = gc * GRID_RES + GRID_RES / 2
                    cell_y = gr * GRID_RES + GRID_RES / 2
                    if (np.hypot(cell_x - self.x, cell_y - self.y) <= CLEAN_R
                            and clean_grid[gr, gc] == 0):
                        clean_grid[gr, gc] = 1


# =============================================================================
#  SIMULASI
# =============================================================================
class RoomCleaningSimulation:
    def __init__(self):
        self.grid_cols  = int(ROOM_W / GRID_RES)
        self.grid_rows  = int(ROOM_H / GRID_RES)
        self.clean_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        self.obstacles  = self._create_obstacles()
        self._mark_obstacle_cells()
        self.robot = CleaningRobot(2.0, 2.0, self.obstacles,
                                   self.grid_cols, self.grid_rows)
        self.total_cleanable = int(np.sum(self.clean_grid == 0))
        self._cmap = ListedColormap([C_DIRTY, C_CLEAN, C_OBS_BASE])
        # Figure permanen — dibuat SEKALI seumur hidup proses
        self._fig, self._ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=RENDER_DPI)
        self._fig.patch.set_facecolor(C_BG)

    def _create_obstacles(self):
        return [
            Obstacle(3.5,  5.0,  3.5, 2.2, "Meja Kerja",  "#e94560"),
            Obstacle(13.0, 10.5, 4.5, 2.8, "Sofa",        "#7c3aed"),
            Obstacle(0.3,  11.5, 2.5, 3.2, "Lemari",      "#1d4ed8"),
            Obstacle(14.5, 5.5,  2.2, 2.2, "Kursi",       "#d97706"),
            Obstacle(10.5, 1.0,  3.0, 2.0, "Meja Kecil",  "#059669"),
            Obstacle(17.5, 1.0,  2.2, 5.5, "Rak Buku",    "#7c3aed"),
            Obstacle(8.5,  6.5,  1.5, 1.5, "Tanaman",     "#10b981"),
        ]

    def _mark_obstacle_cells(self):
        for obs in self.obstacles:
            c0 = max(0, int(obs.x / GRID_RES))
            c1 = min(self.grid_cols, int((obs.x + obs.w) / GRID_RES) + 1)
            r0 = max(0, int(obs.y / GRID_RES))
            r1 = min(self.grid_rows, int((obs.y + obs.h) / GRID_RES) + 1)
            self.clean_grid[r0:r1, c0:c1] = 2

    def update(self, steps=1):
        for _ in range(steps):
            self.robot.step(self.clean_grid)

    @property
    def pct_clean(self):
        return (int(np.sum(self.clean_grid == 1)) /
                max(self.total_cleanable, 1)) * 100

    @property
    def is_done(self):
        return int(np.sum(self.clean_grid == 0)) == 0

    def reset(self):
        plt.close(self._fig)
        self.__init__()

    def render_jpeg(self, quality: int = 88) -> bytes:
        """Render ke JPEG bytes — lebih kecil dari PNG, cocok untuk stream."""
        fig = self._fig
        fig.clf()
        ax  = fig.add_subplot(111)
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor(C_DIRTY)

        # Grid latar
        for gx in np.arange(0, ROOM_W + 1, 2.0):
            ax.axvline(gx, color=C_GRID, lw=0.3, alpha=0.4)
        for gy in np.arange(0, ROOM_H + 1, 2.0):
            ax.axhline(gy, color=C_GRID, lw=0.3, alpha=0.4)

        # Peta kebersihan
        ax.imshow(
            self.clean_grid,
            origin='lower',
            extent=[0, ROOM_W, 0, ROOM_H],
            cmap=self._cmap, vmin=0, vmax=2,
            interpolation='bilinear',
            alpha=0.92, zorder=1,
        )

        # Obstacle
        for obs in self.obstacles:
            ax.add_patch(patches.FancyBboxPatch(
                (obs.x + 0.08, obs.y - 0.08), obs.w, obs.h,
                boxstyle="round,pad=0.1",
                facecolor="black", alpha=0.35, linewidth=0, zorder=2))
            ax.add_patch(patches.FancyBboxPatch(
                (obs.x, obs.y), obs.w, obs.h,
                boxstyle="round,pad=0.1",
                linewidth=1.2, edgecolor="white",
                facecolor=obs.color, alpha=0.92, zorder=3))
            ax.text(
                obs.x + obs.w / 2, obs.y + obs.h / 2, obs.label,
                ha='center', va='center',
                fontsize=7, color='white', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='black')],
                zorder=4)

        # Trail robot
        tx_a = np.array(self.robot.trail_x)
        ty_a = np.array(self.robot.trail_y)
        n    = len(tx_a)
        if n > 2:
            for i in range(1, n):
                alpha = (i / n) * 0.65 + 0.05
                lw    = (i / n) * 1.6  + 0.3
                ax.plot(tx_a[i-1:i+1], ty_a[i-1:i+1],
                        color=C_TRAIL, linewidth=lw,
                        alpha=alpha, solid_capstyle='round', zorder=5)

        # Sensor & clean radius
        ax.add_patch(plt.Circle(
            (self.robot.x, self.robot.y), SENSOR_R,
            fill=False, linestyle='--', linewidth=0.7,
            color=C_SENSOR, alpha=0.20, zorder=6))
        ax.add_patch(plt.Circle(
            (self.robot.x, self.robot.y), CLEAN_R,
            color=C_CLEAN, alpha=0.10, zorder=6))

        # Glow
        for r_g, a_g in [(ROBOT_R+0.5, 0.05),(ROBOT_R+0.28,0.12),(ROBOT_R+0.12,0.18)]:
            ax.add_patch(plt.Circle(
                (self.robot.x, self.robot.y), r_g,
                color=C_GLOW, alpha=a_g, zorder=7))

        # Bodi
        ax.add_patch(plt.Circle(
            (self.robot.x, self.robot.y), ROBOT_R,
            color=C_ROBOT, zorder=8))
        ax.add_patch(plt.Circle(
            (self.robot.x - ROBOT_R*0.2, self.robot.y + ROBOT_R*0.2),
            ROBOT_R * 0.32,
            color='white', alpha=0.20, zorder=9))

        # Panah arah
        ex = self.robot.x + np.cos(self.robot.angle) * (ROBOT_R + 0.38)
        ey = self.robot.y + np.sin(self.robot.angle) * (ROBOT_R + 0.38)
        ax.annotate(
            "", xy=(ex, ey), xytext=(self.robot.x, self.robot.y),
            arrowprops=dict(arrowstyle="-|>", color="white",
                            lw=1.8, mutation_scale=12),
            zorder=10)

        # Banner selesai
        if self.is_done:
            ax.text(
                ROOM_W/2, ROOM_H/2, "✅  RUANGAN BERSIH!",
                ha='center', va='center',
                fontsize=24, color='#4ade80', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=6, foreground='black')],
                zorder=20)

        for sp in ax.spines.values():
            sp.set_edgecolor("#334155"); sp.set_linewidth(1.5)
        ax.set_xlim(0, ROOM_W)
        ax.set_ylim(0, ROOM_H)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        fig.tight_layout(pad=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", facecolor=C_BG,
                    bbox_inches='tight', pad_inches=0.05,
                    pil_kwargs={"quality": quality, "optimize": True})
        buf.seek(0)
        return buf.getvalue()


# =============================================================================
#  STATE GLOBAL (dilindungi dengan Lock karena multi-thread)
# =============================================================================
_lock   = threading.Lock()
_sim    = RoomCleaningSimulation()
_running = False
_speed   = 3   # langkah per frame


# =============================================================================
#  MJPEG GENERATOR — inti anti-kedip
#  Browser menerima satu koneksi HTTP yang tidak pernah ditutup.
#  Setiap frame dikirim sebagai "boundary" multipart.
#  Browser memperbarui <img> di tempat → ZERO flicker.
# =============================================================================
def generate_mjpeg():
    frame_interval = 1.0 / TARGET_FPS
    while True:
        t0 = time.perf_counter()

        with _lock:
            if _running and not _sim.is_done:
                _sim.update(steps=_speed)
            frame_bytes = _sim.render_jpeg(quality=85)

        # Format MJPEG multipart
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
            b"\r\n" + frame_bytes + b"\r\n"
        )

        elapsed = time.perf_counter() - t0
        sleep   = frame_interval - elapsed
        if sleep > 0:
            time.sleep(sleep)


# =============================================================================
#  FLASK APP
# =============================================================================
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🤖 Robot Vacuum Cleaner</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #080c14;
    color: #f1f5f9;
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  header {
    width: 100%;
    padding: 14px 24px;
    background: #0d1117;
    border-bottom: 1px solid #1e293b;
    text-align: center;
    font-family: monospace;
    font-size: 22px;
    font-weight: bold;
    letter-spacing: 1px;
    color: #f8fafc;
  }

  #status-bar {
    width: 100%;
    max-width: 1100px;
    margin: 10px auto 0;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: bold;
    text-align: center;
    transition: background 0.3s, color 0.3s;
  }
  #status-bar.running {
    background: #052e16; border: 1px solid #16a34a; color: #4ade80;
  }
  #status-bar.paused {
    background: #1c1917; border: 1px solid #44403c; color: #a8a29e;
  }
  #status-bar.done {
    background: #052e16; border: 2px solid #4ade80; color: #4ade80; font-size: 16px;
  }

  #main {
    width: 100%;
    max-width: 1100px;
    padding: 10px 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    align-items: center;
  }

  /* ── GAMBAR STREAM ─────────────────────────────────────────────────
     img dengan src MJPEG tidak pernah dilepas browser — zero flicker.
     display: block menghilangkan gap bawah gambar (inline baseline).
  */
  #stream-img {
    width: 100%;
    max-width: 1060px;
    height: auto;
    border-radius: 10px;
    border: 1px solid #1e293b;
    display: block;
    background: #080c14;   /* warna fallback saat pertama load */
  }

  /* Metrik */
  #metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    width: 100%;
    max-width: 1060px;
  }
  .metric {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
  }
  .metric .label {
    font-size: 11px;
    color: #64748b;
    margin-bottom: 4px;
  }
  .metric .value {
    font-size: 20px;
    font-weight: bold;
    color: #f1f5f9;
  }

  /* Progress bar */
  #prog-wrap {
    width: 100%;
    max-width: 1060px;
    background: #1e293b;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
  }
  #prog-bar {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #0ea5e9, #22d3ee);
    border-radius: 6px;
    transition: width 0.3s ease;
  }

  /* Kontrol */
  #controls {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
    width: 100%;
    max-width: 1060px;
  }
  button {
    padding: 9px 22px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: bold;
    cursor: pointer;
    border: 1px solid #334155;
    background: #0f172a;
    color: #f1f5f9;
    transition: background 0.15s, transform 0.1s;
    min-width: 120px;
  }
  button:hover { background: #1e293b; }
  button:active { transform: scale(0.97); }
  #btn-start { background: #15803d; border-color: #16a34a; color: #fff; }
  #btn-start:hover { background: #166534; }

  /* Slider kecepatan */
  #speed-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: #94a3b8;
  }
  input[type=range] { accent-color: #22d3ee; width: 140px; }

  /* Legenda */
  #legend {
    display: flex;
    gap: 18px;
    font-size: 12px;
    color: #94a3b8;
    flex-wrap: wrap;
    justify-content: center;
  }
  .leg { display: flex; align-items: center; gap: 5px; }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
</style>
</head>
<body>

<header>🤖 AI Robot Vacuum Cleaner Simulation</header>

<div id="status-bar" class="paused">⏸ Robot Pause — Tekan ▶ START</div>

<div id="main">

  <!--
    Kunci ZERO FLICKER:
    <img> dengan src /stream (MJPEG) adalah satu koneksi HTTP permanen.
    Browser terus-menerus memperbarui piksel gambar di tempat yang sama
    tanpa pernah menghapus elemen ini dari DOM.
    Tidak ada rerun, tidak ada rebuild, tidak ada hilang-timbul.
  -->
  <img id="stream-img" src="/stream" alt="Robot Vacuum Simulation">

  <div id="metrics">
    <div class="metric">
      <div class="label">🧹 Area Bersih</div>
      <div class="value" id="m-pct">0.0%</div>
    </div>
    <div class="metric">
      <div class="label">👣 Total Langkah</div>
      <div class="value" id="m-steps">0</div>
    </div>
    <div class="metric">
      <div class="label">📍 Posisi Robot</div>
      <div class="value" id="m-pos">(2.0, 2.0)</div>
    </div>
    <div class="metric">
      <div class="label">🗺️ Sel Bersih</div>
      <div class="value" id="m-cells">0</div>
    </div>
  </div>

  <div id="prog-wrap"><div id="prog-bar"></div></div>

  <div id="controls">
    <button id="btn-start" onclick="ctrlStart()">▶ START</button>
    <button id="btn-pause" onclick="ctrlPause()">⏸ PAUSE</button>
    <button id="btn-reset" onclick="ctrlReset()">↺ RESET</button>
    <div id="speed-wrap">
      <span>Kecepatan:</span>
      <input type="range" id="sl-speed" min="1" max="8" value="3"
             oninput="ctrlSpeed(this.value)">
      <span id="sl-val">3x</span>
    </div>
  </div>

  <div id="legend">
    <div class="leg"><div class="dot" style="background:#0e4d6e"></div> Area Bersih</div>
    <div class="leg"><div class="dot" style="background:#111827;border:1px solid #334155"></div> Area Kotor</div>
    <div class="leg"><div class="dot" style="background:#f43f5e"></div> Robot</div>
    <div class="leg"><div class="dot" style="background:#fbbf24"></div> Sensor Range</div>
    <div class="leg"><div class="dot" style="background:#22d3ee"></div> Jejak Robot</div>
  </div>

</div>

<script>
  // Poll status dari server setiap 300ms untuk update metrik & UI
  // (TERPISAH dari stream gambar — stream tidak pernah diinterupsi)
  let statusInterval = null;

  function startPolling() {
    if (statusInterval) return;
    statusInterval = setInterval(async () => {
      try {
        const r = await fetch('/status');
        const d = await r.json();

        document.getElementById('m-pct').textContent   = d.pct.toFixed(1) + '%';
        document.getElementById('m-steps').textContent = d.steps.toLocaleString();
        document.getElementById('m-pos').textContent   =
          '(' + d.x.toFixed(1) + ', ' + d.y.toFixed(1) + ')';
        document.getElementById('m-cells').textContent = d.cleaned.toLocaleString();
        document.getElementById('prog-bar').style.width = Math.min(d.pct, 100) + '%';

        const sb = document.getElementById('status-bar');
        if (d.done) {
          sb.className = 'done';
          sb.textContent = '✅ Selesai! Seluruh ruangan sudah bersih. Tekan ↺ RESET untuk ulang.';
          document.getElementById('btn-start').disabled = true;
        } else if (d.running) {
          sb.className = 'running';
          sb.textContent = '🟢 Robot Aktif — Sedang Membersihkan';
        } else {
          sb.className = 'paused';
          sb.textContent = '⏸ Robot Pause — Tekan ▶ START';
        }
      } catch(e) {}
    }, 300);
  }

  async function ctrlStart() {
    await fetch('/control/start', { method: 'POST' });
    document.getElementById('btn-start').disabled = true;
    document.getElementById('status-bar').className = 'running';
    document.getElementById('status-bar').textContent = '🟢 Robot Aktif — Sedang Membersihkan';
  }

  async function ctrlPause() {
    await fetch('/control/pause', { method: 'POST' });
    document.getElementById('btn-start').disabled = false;
    document.getElementById('status-bar').className = 'paused';
    document.getElementById('status-bar').textContent = '⏸ Robot Pause — Tekan ▶ START';
  }

  async function ctrlReset() {
    await fetch('/control/reset', { method: 'POST' });
    document.getElementById('btn-start').disabled = false;
    document.getElementById('m-pct').textContent   = '0.0%';
    document.getElementById('m-steps').textContent = '0';
    document.getElementById('m-pos').textContent   = '(2.0, 2.0)';
    document.getElementById('m-cells').textContent = '0';
    document.getElementById('prog-bar').style.width = '0%';
    document.getElementById('status-bar').className = 'paused';
    document.getElementById('status-bar').textContent = '⏸ Robot Pause — Tekan ▶ START';
  }

  function ctrlSpeed(v) {
    document.getElementById('sl-val').textContent = v + 'x';
    fetch('/control/speed/' + v, { method: 'POST' });
  }

  // Mulai polling saat halaman siap
  startPolling();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/stream')
def stream():
    """MJPEG stream — koneksi ini TIDAK PERNAH ditutup selama browser terbuka."""
    return Response(
        generate_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/status')
def status():
    with _lock:
        return {
            "pct":     round(_sim.pct_clean, 2),
            "steps":   _sim.robot.step_count,
            "x":       round(_sim.robot.x, 2),
            "y":       round(_sim.robot.y, 2),
            "cleaned": int(np.sum(_sim.clean_grid == 1)),
            "running": _running,
            "done":    _sim.is_done,
        }


@app.route('/control/start', methods=['POST'])
def ctrl_start():
    global _running
    with _lock:
        _running = True
    return {'ok': True}


@app.route('/control/pause', methods=['POST'])
def ctrl_pause():
    global _running
    with _lock:
        _running = False
    return {'ok': True}


@app.route('/control/reset', methods=['POST'])
def ctrl_reset():
    global _running, _sim
    with _lock:
        _running = False
        plt.close(_sim._fig)
        _sim = RoomCleaningSimulation()
    return {'ok': True}


@app.route('/control/speed/<int:v>', methods=['POST'])
def ctrl_speed(v):
    global _speed
    with _lock:
        _speed = max(1, min(8, v))
    return {'ok': True}


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  🤖 Robot Vacuum Cleaner — Zero Flicker Edition")
    print("  Buka browser: http://localhost:5000")
    print("=" * 60)
    # use_reloader=False wajib agar tidak ada double-thread yang bentrok
    app.run(host='0.0.0.0', port=5000, debug=False,
            threaded=True, use_reloader=False)