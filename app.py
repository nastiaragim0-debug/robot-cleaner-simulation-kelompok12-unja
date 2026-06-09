# =============================================================================
#  🤖 SIMULASI ROBOT PEMBERSIH — SMART SKIP v4
#  - Waypoint di-cek real-time: kalau area sudah bersih → skip langsung
#  - Robot HANYA bergerak ke tempat yang benar-benar masih kotor
#  - Setelah semua waypoint selesai, cari sisa sel kotor tersembunyi
#  - Gerakan smooth: EMA velocity + 8 substeps + angle lerp
# =============================================================================

import io, base64, time
import numpy as np
import streamlit as st
import matplotlib, matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from collections import deque

matplotlib.use("Agg")

# =============================================================================
#  KONFIGURASI
# =============================================================================
ROOM_W, ROOM_H = 20, 15
GRID_RES       = 0.25
ROBOT_R        = 0.45
CLEAN_R        = 0.65
SENSOR_R       = 1.8
ROBOT_SPEED    = 0.15
ACCEL          = 0.11
ANGLE_LRP      = 0.11
SUBSTEPS       = 8
TRAIL_MAX      = 500

# Jarak antar strip sapuan — sedikit lebih kecil dari CLEAN_R*2 agar overlap
SWEEP_STEP     = CLEAN_R * 1.55   # ~1.0075 unit

# Dianggap sampai kalau dalam radius ini dari waypoint
WP_REACH       = CLEAN_R * 0.85

# Cek "apakah area di waypoint sudah bersih?" dalam radius ini
CLEAN_CHECK_R  = CLEAN_R * 0.9

# Berapa sel kotor yang boleh ada di sekitar waypoint agar dianggap "perlu dikunjungi"
DIRTY_THRESHOLD = 3   # kurang dari ini → skip waypoint itu

C_BG    = "#080c14";  C_DIRTY = "#111827"; C_CLEAN = "#0e4d6e"
C_OBS   = "#1f2937";  C_ROBOT = "#f43f5e"; C_GLOW  = "#fb7185"
C_TRAIL = "#22d3ee";  C_GRID  = "#1e2a3a"; C_WP    = "#a3e635"
C_PATH  = "#1e3a2f"

RENDER_DPI = 80; FIG_W, FIG_H = 11, 7


# =============================================================================
#  OBSTACLE
# =============================================================================
class Obstacle:
    def __init__(self, x, y, w, h, label="", color="#533483"):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label, self.color = label, color
        self._mg = ROBOT_R + 0.22

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
        if d >= SENSOR_R: return 0., 0.
        ocx, ocy = self.x + self.w/2, self.y + self.h/2
        dx, dy   = px - ocx, py - ocy
        norm     = np.hypot(dx, dy) + 1e-9
        s        = ((SENSOR_R - d) / SENSOR_R) ** 2 * 3.2
        return dx/norm*s, dy/norm*s


# =============================================================================
#  HELPER: cek berapa sel kotor di sekitar titik (wx, wy)
# =============================================================================
def count_dirty_cells(wx, wy, clean_grid):
    """Hitung jumlah sel kotor dalam radius CLEAN_CHECK_R dari (wx,wy)."""
    cx = int(wx / GRID_RES)
    cy = int(wy / GRID_RES)
    cr = int(CLEAN_CHECK_R / GRID_RES) + 1
    rows, cols = clean_grid.shape
    count = 0
    for dr in range(-cr, cr+1):
        for dc in range(-cr, cr+1):
            gr, gc = cy+dr, cx+dc
            if 0 <= gr < rows and 0 <= gc < cols:
                px = gc*GRID_RES + GRID_RES/2
                py = gr*GRID_RES + GRID_RES/2
                if np.hypot(px-wx, py-wy) <= CLEAN_CHECK_R:
                    if clean_grid[gr, gc] == 0:
                        count += 1
    return count


# =============================================================================
#  PATH PLANNER: boustrophedon + grid halus untuk corner pass
# =============================================================================
def build_sweep_waypoints(obstacles):
    """Jalur lawn-mower utama."""
    margin = ROBOT_R + 0.08
    wps = []
    row = 0
    y = margin
    while y <= ROOM_H - margin:
        xs = np.arange(margin, ROOM_W - margin + 0.01, SWEEP_STEP)
        if row % 2 == 1:
            xs = xs[::-1]
        for x in xs:
            if not any(o.contains(x, y) for o in obstacles):
                wps.append((float(x), float(y)))
        y += SWEEP_STEP
        row += 1
    return wps


def build_corner_waypoints(obstacles):
    """Grid halus untuk sisa sudut setelah fase utama."""
    margin = ROBOT_R + 0.05
    step   = SWEEP_STEP * 0.6   # lebih rapat
    wps = []
    row = 0
    y = margin
    while y <= ROOM_H - margin:
        xs = np.arange(margin, ROOM_W - margin + 0.01, step)
        if row % 2 == 1:
            xs = xs[::-1]
        for x in xs:
            if not any(o.contains(x, y) for o in obstacles):
                wps.append((float(x), float(y)))
        y += step
        row += 1
    return wps


# =============================================================================
#  ROBOT
# =============================================================================
class CleaningRobot:
    def __init__(self, sx, sy, obstacles, sweep_wps, corner_wps):
        self.x, self.y   = float(sx), float(sy)
        self.vx = self.vy = 0.0
        self.angle        = 0.0
        self.obstacles    = obstacles
        self.step_count   = 0

        self.sweep_wps    = deque(sweep_wps)
        self.corner_wps   = deque(corner_wps)
        self.phase        = "sweep"   # "sweep" | "corner" | "done"
        self.current_wp   = None
        self.wp_skipped   = 0        # statistik berapa wp di-skip

        self.trail_x: list = []
        self.trail_y: list = []

        self._svx = self._svy = 0.0  # smooth velocity
        self.stuck_timer  = 0
        self.stuck_x, self.stuck_y = sx, sy
        self.stuck_check  = 0
        self.escape_mode  = False
        self.escape_timer = 0
        self.escape_angle = 0.0

        self._next_wp()

    # ------------------------------------------------------------------ internal
    def _is_area_clean(self, wx, wy, clean_grid):
        """True jika area di sekitar waypoint sudah bersih (tidak perlu dikunjungi)."""
        return count_dirty_cells(wx, wy, clean_grid) < DIRTY_THRESHOLD

    def _next_wp(self, clean_grid=None):
        """
        Ambil waypoint berikutnya.
        Kalau clean_grid tersedia, skip waypoint yang areanya sudah bersih.
        """
        queue = self.sweep_wps if self.phase == "sweep" else self.corner_wps

        while queue:
            wx, wy = queue.popleft()

            # Skip kalau di dalam obstacle
            if any(o.contains(wx, wy) for o in self.obstacles):
                self.wp_skipped += 1
                continue

            # Skip kalau area sudah bersih (INTI PERBAIKAN)
            if clean_grid is not None and self._is_area_clean(wx, wy, clean_grid):
                self.wp_skipped += 1
                continue

            self.current_wp = (wx, wy)
            return

        # Queue habis → pindah fase atau selesai
        self.current_wp = None
        if self.phase == "sweep":
            self.phase = "corner"
            self._next_wp(clean_grid)
        else:
            self.phase = "done"

    # ------------------------------------------------------------------ desired velocity
    def _desired_vel(self, clean_grid):
        if self.escape_mode:
            return (np.cos(self.escape_angle) * ROBOT_SPEED,
                    np.sin(self.escape_angle) * ROBOT_SPEED)

        if self.current_wp is None:
            return 0., 0.

        wx, wy = self.current_wp
        dx, dy = wx - self.x, wy - self.y
        dist   = np.hypot(dx, dy) + 1e-9

        # Sudah sampai di waypoint?
        if dist < WP_REACH:
            self._next_wp(clean_grid)
            if self.current_wp is None:
                return 0., 0.
            wx, wy = self.current_wp
            dx, dy = wx - self.x, wy - self.y
            dist   = np.hypot(dx, dy) + 1e-9
        else:
            # Cek real-time: kalau area waypoint ini sudah bersih saat robot menuju ke sana,
            # langsung skip ke waypoint berikutnya tanpa perlu sampai dulu
            if self._is_area_clean(wx, wy, clean_grid):
                self.wp_skipped += 1
                self._next_wp(clean_grid)
                if self.current_wp is None:
                    return 0., 0.
                wx, wy = self.current_wp
                dx, dy = wx - self.x, wy - self.y
                dist   = np.hypot(dx, dy) + 1e-9

        speed  = min(1.0, dist / 1.5) * ROBOT_SPEED
        goal_x = dx / dist * speed
        goal_y = dy / dist * speed

        rep_x = rep_y = 0.
        for o in self.obstacles:
            rx, ry = o.repulsion(self.x, self.y)
            rep_x += rx; rep_y += ry

        mg = ROBOT_R + 0.7
        def wf(d): return ((mg-d)/mg)**2 * 2.5 if d < mg else 0.
        wx2 = wf(self.x) - wf(ROOM_W - self.x)
        wy2 = wf(self.y) - wf(ROOM_H - self.y)

        tx = goal_x*2.2 + rep_x + wx2
        ty = goal_y*2.2 + rep_y + wy2
        mag = np.hypot(tx, ty) + 1e-9
        return tx/mag*ROBOT_SPEED, ty/mag*ROBOT_SPEED

    # ------------------------------------------------------------------ substep
    def _substep(self, dvx, dvy, clean_grid):
        if self.escape_mode:
            self.escape_timer -= 1
            if self.escape_timer <= 0:
                self.escape_mode = False
                self.stuck_timer = 0
                self._next_wp(clean_grid)

        self._svx = self._svx*(1-ACCEL) + dvx*ACCEL
        self._svy = self._svy*(1-ACCEL) + dvy*ACCEL

        sc = 1.0 / SUBSTEPS
        nx = np.clip(self.x + self._svx*sc, ROBOT_R, ROOM_W - ROBOT_R)
        ny = np.clip(self.y + self._svy*sc, ROBOT_R, ROOM_H - ROBOT_R)

        hx = any(o.contains(nx, self.y) for o in self.obstacles)
        hy = any(o.contains(self.x, ny) for o in self.obstacles)

        if not hx: self.x = nx
        else: self._svx *= -0.3; self.stuck_timer += 2

        if not hy: self.y = ny
        else: self._svy *= -0.3; self.stuck_timer += 2

        spd = np.hypot(self._svx, self._svy)
        if spd > 1e-4:
            ta = np.arctan2(self._svy, self._svx)
            d  = (ta - self.angle + np.pi) % (2*np.pi) - np.pi
            self.angle += d * ANGLE_LRP

        self._mark_clean(clean_grid)

    # ------------------------------------------------------------------ step
    def step(self, clean_grid):
        dvx, dvy = self._desired_vel(clean_grid)
        for _ in range(SUBSTEPS):
            self._substep(dvx, dvy, clean_grid)

        self.step_count += 1
        self.trail_x.append(self.x); self.trail_y.append(self.y)
        if len(self.trail_x) > TRAIL_MAX:
            self.trail_x.pop(0); self.trail_y.pop(0)

        # Anti-stuck
        self.stuck_check += 1
        if self.stuck_check >= 25:
            moved = np.hypot(self.x - self.stuck_x, self.y - self.stuck_y)
            if moved < 0.22: self.stuck_timer += 12
            self.stuck_x, self.stuck_y = self.x, self.y
            self.stuck_check = 0

        if self.stuck_timer > 20 and not self.escape_mode:
            self.escape_mode  = True
            self.escape_timer = SUBSTEPS * 18
            self.escape_angle = self.angle + np.pi + np.random.uniform(-0.5, 0.5)
            self.stuck_timer  = 0

    # ------------------------------------------------------------------ mark clean
    def _mark_clean(self, clean_grid):
        cx, cy = int(self.x/GRID_RES), int(self.y/GRID_RES)
        cr = int(CLEAN_R/GRID_RES) + 1
        rows, cols = clean_grid.shape
        for dr in range(-cr, cr+1):
            for dc in range(-cr, cr+1):
                gr, gc = cy+dr, cx+dc
                if 0 <= gr < rows and 0 <= gc < cols:
                    px = gc*GRID_RES + GRID_RES/2
                    py = gr*GRID_RES + GRID_RES/2
                    if np.hypot(px-self.x, py-self.y) <= CLEAN_R:
                        if clean_grid[gr, gc] == 0:
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

        sweep  = build_sweep_waypoints(self.obstacles)
        corner = build_corner_waypoints(self.obstacles)
        sx, sy = sweep[0]
        self.robot = CleaningRobot(sx, sy, self.obstacles, sweep, corner)
        self.total_cleanable  = int(np.sum(self.clean_grid == 0))
        self.total_sweep_wps  = len(sweep)
        self.total_corner_wps = len(corner)

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
            c0 = max(0, int(obs.x/GRID_RES))
            c1 = min(self.grid_cols, int((obs.x+obs.w)/GRID_RES)+1)
            r0 = max(0, int(obs.y/GRID_RES))
            r1 = min(self.grid_rows, int((obs.y+obs.h)/GRID_RES)+1)
            self.clean_grid[r0:r1, c0:c1] = 2

    def update(self):
        if self.robot.phase != "done":
            self.robot.step(self.clean_grid)

    @property
    def pct_clean(self):
        return int(np.sum(self.clean_grid == 1)) / max(self.total_cleanable, 1) * 100

    # ------------------------------------------------------------------
    def render_to_base64(self) -> str:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=RENDER_DPI)
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor(C_DIRTY)

        for gx in np.arange(0, ROOM_W+1, 2.):
            ax.axvline(gx, color=C_GRID, lw=0.3, alpha=0.4)
        for gy in np.arange(0, ROOM_H+1, 2.):
            ax.axhline(gy, color=C_GRID, lw=0.3, alpha=0.4)

        cmap = ListedColormap([C_DIRTY, C_CLEAN, C_OBS])
        ax.imshow(self.clean_grid, origin='lower',
                  extent=[0, ROOM_W, 0, ROOM_H],
                  cmap=cmap, vmin=0, vmax=2,
                  interpolation='bilinear', alpha=0.93, zorder=1)

        # Tampilkan waypoint tersisa yang MASIH KOTOR (bukan semua waypoint)
        queue = (self.robot.sweep_wps if self.robot.phase == "sweep"
                 else self.robot.corner_wps)
        future = [(wx, wy) for wx, wy in list(queue)
                  if count_dirty_cells(wx, wy, self.clean_grid) >= DIRTY_THRESHOLD]
        if future:
            fxs = [w[0] for w in future]
            fys = [w[1] for w in future]
            ax.scatter(fxs, fys, s=3, c=C_PATH, alpha=0.5, zorder=2, linewidths=0)

        # Obstacles
        for obs in self.obstacles:
            ax.add_patch(patches.FancyBboxPatch(
                (obs.x+0.08, obs.y-0.08), obs.w, obs.h,
                boxstyle="round,pad=0.1", facecolor="black",
                alpha=0.35, linewidth=0, zorder=3))
            ax.add_patch(patches.FancyBboxPatch(
                (obs.x, obs.y), obs.w, obs.h,
                boxstyle="round,pad=0.1", linewidth=1.2,
                edgecolor="white", facecolor=obs.color, alpha=0.92, zorder=4))
            ax.text(obs.x+obs.w/2, obs.y+obs.h/2, obs.label,
                    ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')],
                    zorder=5)

        # Trail
        tx, ty = self.robot.trail_x, self.robot.trail_y
        if len(tx) > 2:
            n = len(tx)
            for i in range(1, n):
                t = i/n
                ax.plot(tx[i-1:i+1], ty[i-1:i+1],
                        color=C_TRAIL, linewidth=t*2.0+0.3,
                        alpha=t*0.70+0.05, solid_capstyle='round', zorder=6)

        # Garis ke target aktif
        if self.robot.current_wp and self.robot.phase != "done":
            wx, wy = self.robot.current_wp
            ax.plot([self.robot.x, wx], [self.robot.y, wy],
                    color=C_WP, lw=0.9, ls=':', alpha=0.6, zorder=7)
            ax.add_patch(plt.Circle((wx, wy), 0.22,
                                    color=C_WP, alpha=0.85, zorder=7))
            ax.add_patch(plt.Circle((wx, wy), 0.48, fill=False,
                                    edgecolor=C_WP, lw=0.8, alpha=0.35, zorder=7))

        # Sensor
        ax.add_patch(plt.Circle((self.robot.x, self.robot.y), SENSOR_R,
                                fill=False, ls='--', lw=0.7,
                                color="#fbbf24", alpha=0.18, zorder=8))

        # Glow + body
        for rg, ag in [(ROBOT_R+0.5, 0.05), (ROBOT_R+0.28, 0.12), (ROBOT_R+0.12, 0.20)]:
            ax.add_patch(plt.Circle((self.robot.x, self.robot.y), rg,
                                    color=C_GLOW, alpha=ag, zorder=9))
        ax.add_patch(plt.Circle((self.robot.x, self.robot.y), ROBOT_R,
                                color=C_ROBOT, zorder=10))
        ax.add_patch(plt.Circle((self.robot.x - ROBOT_R*.2, self.robot.y + ROBOT_R*.2),
                                ROBOT_R*.32, color='white', alpha=.20, zorder=11))

        # Panah arah
        ex = self.robot.x + np.cos(self.robot.angle)*(ROBOT_R+0.38)
        ey = self.robot.y + np.sin(self.robot.angle)*(ROBOT_R+0.38)
        ax.annotate("", xy=(ex, ey), xytext=(self.robot.x, self.robot.y),
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                   lw=1.8, mutation_scale=12), zorder=12)

        # Label fase
        phase_txt = {"sweep": "🧹 Fase 1 — Sapuan Baris Sistematis",
                     "corner": "🔍 Fase 2 — Bersihkan Sisa Sudut",
                     "done": "✅ Selesai!"}
        ax.text(0.5, 14.4, phase_txt.get(self.robot.phase, ""),
                ha='left', va='top', fontsize=8.5, color='white', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='black')], zorder=13)

        if self.robot.escape_mode:
            ax.text(self.robot.x, self.robot.y+ROBOT_R+0.55, "⚠",
                    ha='center', va='bottom', fontsize=10,
                    color='#facc15', zorder=14)

        # Banner selesai
        if self.robot.phase == "done":
            ax.add_patch(patches.FancyBboxPatch((3, 5.5), 14, 3.8,
                         boxstyle="round,pad=0.3",
                         facecolor="#052e16", edgecolor="#4ade80",
                         linewidth=2.5, alpha=0.95, zorder=15))
            ax.text(10, 7.5, "✅ RUANGAN 100% BERSIH!",
                    ha='center', va='center', fontsize=16,
                    color='#4ade80', fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground='#001a0a')],
                    zorder=16)
            ax.text(10, 6.5, f"Langkah: {self.robot.step_count:,}  |  "
                             f"Waypoint di-skip: {self.robot.wp_skipped:,}",
                    ha='center', va='center', fontsize=9,
                    color='#86efac', zorder=16)

        for sp in ax.spines.values():
            sp.set_edgecolor("#334155"); sp.set_linewidth(1.5)
        ax.set_xlim(0, ROOM_W); ax.set_ylim(0, ROOM_H)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        fig.tight_layout(pad=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=C_BG,
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================================
#  STREAMLIT APP
# =============================================================================
st.set_page_config(page_title="Robot Vacuum Cleaner", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #080c14; }
    .block-container { padding-top: 1.2rem !important; }
    h1 { color: #f8fafc !important; letter-spacing: 1px; }
    .stButton > button {
        width:100%; border-radius:8px; font-weight:bold;
        font-size:14px; padding:0.45rem 1rem; transition:all 0.2s ease;
    }
    [data-testid="metric-container"] {
        background:#0f172a; border:1px solid #1e293b;
        border-radius:10px; padding:10px 14px;
    }
    [data-testid="metric-container"] label { color:#64748b !important; font-size:11px !important; }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color:#f1f5f9 !important; font-size:20px !important; font-weight:bold !important;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg,#0ea5e9,#22d3ee) !important;
    }
    [data-testid="stSidebar"] {
        background-color:#0d1117 !important; border-right:1px solid #1e293b;
    }
    .sim-frame-container {
        width:100%; min-height:420px; background:#080c14;
        border-radius:10px; overflow:hidden;
        border:1px solid #1e293b; line-height:0;
    }
    .sim-frame-container img { width:100%; height:auto; display:block; }
</style>
""", unsafe_allow_html=True)

# Session state
if "sim"             not in st.session_state: st.session_state.sim             = RoomCleaningSimulation()
if "running"         not in st.session_state: st.session_state.running         = False
if "steps_per_frame" not in st.session_state: st.session_state.steps_per_frame = 4
if "refresh_ms"      not in st.session_state: st.session_state.refresh_ms      = 50

sim = st.session_state.sim

# SIDEBAR
with st.sidebar:
    st.markdown("## 🤖 Robot Vacuum")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ START", type="primary", use_container_width=True):
            st.session_state.running = True
    with c2:
        if st.button("⏸ PAUSE", use_container_width=True):
            st.session_state.running = False
    if st.button("↺ RESET", use_container_width=True):
        st.session_state.sim     = RoomCleaningSimulation()
        st.session_state.running = False
        sim = st.session_state.sim

    st.markdown("---")
    st.markdown("### ⚙️ Pengaturan")
    st.session_state.steps_per_frame = st.slider(
        "Langkah per Frame", 1, 10, st.session_state.steps_per_frame)
    st.session_state.refresh_ms = st.slider(
        "Refresh (ms)", 20, 200, st.session_state.refresh_ms, 5)

    st.markdown("---")
    st.markdown("### 📊 Statistik")
    q = sim.robot.sweep_wps if sim.robot.phase == "sweep" else sim.robot.corner_wps
    total_q = (sim.total_sweep_wps if sim.robot.phase == "sweep"
               else sim.total_corner_wps)
    done_q  = total_q - len(q)
    st.markdown(f"**Fase:** `{sim.robot.phase.upper()}`")
    st.markdown(f"**Waypoint selesai:** {done_q} / {total_q}")
    st.markdown(f"**Di-skip (sudah bersih):** `{sim.robot.wp_skipped}`")
    st.progress(min(done_q / max(total_q, 1), 1.0))

    st.markdown("---")
    st.markdown("### 🗺️ Legenda")
    st.markdown("""
<div style='font-size:13px;line-height:2.2;'>
🔵 &nbsp; Area Sudah Bersih<br>
⬛ &nbsp; Area Kotor<br>
🔴 &nbsp; Robot<br>
🟢 &nbsp; Target Waypoint<br>
🩵 &nbsp; Jejak Robot<br>
⬤ &nbsp; Rencana Jalur (kotor)<br>
⚠️ &nbsp; Escape Mode
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='color:#475569;font-size:11px;line-height:1.9'>"
        "✔ Skip real-time: cek kotor sebelum jalan<br>"
        "✔ Skip on-the-fly: kalau bersih saat menuju → langsung skip<br>"
        "✔ Boustrophedon + corner pass fase 2<br>"
        "✔ EMA velocity + 8 substeps smooth<br>"
        "✔ Anti-flicker base64 rendering</div>", unsafe_allow_html=True)

# HEADER
st.markdown("<h1 style='text-align:center;font-family:monospace'>"
            "🤖 AI Robot Vacuum Cleaner Simulation</h1>", unsafe_allow_html=True)

phase_map = {
    "sweep":  ("🟢 Fase 1 — Menyapu Baris Sistematis",  "#052e16","#16a34a","#4ade80"),
    "corner": ("🔍 Fase 2 — Membersihkan Sisa Sudut",   "#0f2240","#3b82f6","#93c5fd"),
    "done":   ("✅ SELESAI — Seluruh Ruangan Bersih!",  "#052e16","#4ade80","#4ade80"),
}
label, bg, bd, tx = phase_map.get(sim.robot.phase, phase_map["sweep"])
if not st.session_state.running and sim.robot.phase != "done":
    label, bg, bd, tx = "⏸ Pause — Tekan ▶ START di sidebar", "#1c1917","#44403c","#a8a29e"

st.markdown(
    f"<div style='text-align:center;background:{bg};border:1px solid {bd};"
    f"border-radius:8px;padding:7px;color:{tx};font-weight:bold;"
    f"font-size:14px;margin-bottom:10px'>{label}</div>", unsafe_allow_html=True)

# UPDATE
if st.session_state.running and sim.robot.phase != "done":
    for _ in range(st.session_state.steps_per_frame):
        sim.update()

# METRIK
pct           = sim.pct_clean
cleaned_cells = int(np.sum(sim.clean_grid == 1))
wp_remain     = len(sim.robot.sweep_wps if sim.robot.phase == "sweep"
                    else sim.robot.corner_wps)

m1, m2, m3, m4 = st.columns(4)
m1.metric("🧹 Area Bersih",       f"{pct:.1f}%")
m2.metric("👣 Total Langkah",     f"{sim.robot.step_count:,}")
m3.metric("⏭️ Waypoint Di-skip",  f"{sim.robot.wp_skipped:,}")
m4.metric("🗺️ Sel Dibersihkan",   f"{cleaned_cells:,}")
st.progress(min(pct/100, 1.0))

# RENDER
img_b64 = sim.render_to_base64()
st.markdown(
    f'<div class="sim-frame-container">'
    f'<img src="data:image/png;base64,{img_b64}" alt="Simulation"/>'
    f'</div>', unsafe_allow_html=True)

# AUTO REFRESH
if st.session_state.running and sim.robot.phase != "done":
    time.sleep(st.session_state.refresh_ms / 1000)
    st.rerun()