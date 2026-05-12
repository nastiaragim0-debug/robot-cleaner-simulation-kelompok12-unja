# =============================================================================
#  🤖 SIMULASI ROBOT PEMBERSIH RUANGAN — SMOOTH EDITION
#  Streamlit App · Anti-Flicker · Ultra-Smooth Movement
#
#  Perbaikan utama vs versi sebelumnya:
#  1. ANTI-FLICKER  → render ke BytesIO buffer → st.image() bukan st.pyplot()
#                     Komponen <img> di-update src-nya di tempat, tidak flicker.
#  2. ROBOT SMOOTH  → velocity inertia, sub-step movement, angle lerp
#  3. AVOIDANCE     → potential field kuadratik yang lebih halus
#  4. UI MODERN     → layout sidebar + metrics + tema gelap konsisten
#
#  Cara deploy ke Streamlit Cloud:
#    requirements.txt berisi: streamlit, numpy, matplotlib
# =============================================================================

import io
import time
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap

# Wajib Agg agar tidak ada thread-safety issue di server
matplotlib.use("Agg")

# =============================================================================
#  KONFIGURASI
# =============================================================================
ROOM_W      = 20
ROOM_H      = 15
GRID_RES    = 0.25       # Resolusi grid kebersihan

ROBOT_R     = 0.45       # Radius robot (unit ruangan)
ROBOT_SPEED = 0.12       # Kecepatan maksimum per sub-step
ACCEL       = 0.22       # Kelembutan akselerasi [0-1] — makin kecil makin lembut
ANGLE_LRP   = 0.18       # Kelembutan rotasi [0-1]
SENSOR_R    = 2.0        # Radius sensor obstacle
CLEAN_R     = 0.65       # Radius area yang dibersihkan per langkah

SUBSTEPS    = 4          # Sub-langkah per frame (smooth tanpa nambah FPS)
TRAIL_MAX   = 300        # Panjang maksimum jejak

# Warna
C_BG        = "#080c14"
C_DIRTY     = "#111827"
C_CLEAN     = "#0e4d6e"
C_OBS_BASE  = "#1f2937"
C_ROBOT     = "#f43f5e"
C_GLOW      = "#fb7185"
C_TRAIL     = "#22d3ee"
C_SENSOR    = "#fbbf24"
C_GRID      = "#1e2a3a"

# Render
RENDER_DPI  = 80
FIG_W, FIG_H = 11, 7


# =============================================================================
#  OBSTACLE
# =============================================================================
class Obstacle:
    """Obstacle persegi panjang dengan potential field repulsion."""

    def __init__(self, x, y, w, h, label="", color="#533483"):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.label     = label
        self.color     = color
        self._mg       = ROBOT_R + 0.18   # margin collision

    # Batas expanded untuk collision detection
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
        """Jarak Euclidean dari titik ke sisi terdekat obstacle."""
        cx = max(self.x, min(px, self.x + self.w))
        cy = max(self.y, min(py, self.y + self.h))
        return np.hypot(px - cx, py - cy)

    def repulsion(self, px, py):
        """
        Vektor tolakan potential field (kuadratik).
        Semakin dekat → tolakan semakin kuat.
        """
        d = self.dist_to(px, py)
        if d >= SENSOR_R:
            return 0.0, 0.0
        # Arah: dari pusat obstacle ke robot
        ocx = self.x + self.w / 2
        ocy = self.y + self.h / 2
        dx, dy = px - ocx, py - ocy
        norm   = np.hypot(dx, dy) + 1e-9
        strength = ((SENSOR_R - d) / SENSOR_R) ** 2 * 2.2
        return (dx / norm) * strength, (dy / norm) * strength


# =============================================================================
#  ROBOT
# =============================================================================
class CleaningRobot:
    """
    Robot pembersih dengan gerakan ultra-smooth:
    - Velocity inertia  : percepatan/perlambatan bertahap
    - Sub-step movement : SUBSTEPS mikro-langkah per frame
    - Angle lerp        : rotasi halus, tidak tiba-tiba
    - Potential field   : hindari obstacle secara natural
    - Auto-unstuck      : escape jika tersangkut
    """

    def __init__(self, x, y, obstacles, grid_cols, grid_rows):
        self.x  = float(x)
        self.y  = float(y)
        init_a  = np.random.uniform(0, 2*np.pi)
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

        self._wander_angle = np.random.uniform(0, 2*np.pi)
        self._wander_timer = 0

    # ── Hitung kecepatan yang diinginkan ──────────────────────────────────
    def _desired_vel(self):
        """
        Gabungkan tiga gaya:
        1. Wandering (arah jelajah yang berubah lambat)
        2. Tolakan obstacle (potential field)
        3. Tolakan dinding
        """
        # Wandering
        self._wander_timer += 1
        if self._wander_timer >= 40:
            self._wander_angle += np.random.uniform(-0.6, 0.6)
            self._wander_timer  = 0
        wx = np.cos(self._wander_angle)
        wy = np.sin(self._wander_angle)

        # Tolakan obstacle
        rep_x, rep_y = 0.0, 0.0
        for obs in self.obstacles:
            rx, ry  = obs.repulsion(self.x, self.y)
            rep_x  += rx
            rep_y  += ry

        # Tolakan dinding (kuadratik)
        margin = ROBOT_R + 0.6
        wall_x, wall_y = 0.0, 0.0
        def wf(d):
            return ((margin - d) / margin) ** 2 * 2.5 if d < margin else 0.0
        wall_x += wf(self.x)               # kiri → dorong ke kanan
        wall_x -= wf(ROOM_W - self.x)      # kanan → dorong ke kiri
        wall_y += wf(self.y)               # bawah → dorong ke atas
        wall_y -= wf(ROOM_H - self.y)      # atas → dorong ke bawah

        # Normalisasi → skala ke ROBOT_SPEED
        tx = wx + rep_x + wall_x
        ty = wy + rep_y + wall_y
        mag = np.hypot(tx, ty) + 1e-9
        return (tx / mag) * ROBOT_SPEED, (ty / mag) * ROBOT_SPEED

    # ── Sub-step: satu mikro-langkah ─────────────────────────────────────
    def _substep(self, dvx, dvy, clean_grid):
        # Lerp velocity (inertia / smooth acceleration)
        self.vx += (dvx - self.vx) * ACCEL
        self.vy += (dvy - self.vy) * ACCEL

        scale = 1.0 / SUBSTEPS
        nx    = self.x + self.vx * scale
        ny    = self.y + self.vy * scale

        # Klem ke batas ruangan
        r  = ROBOT_R
        nx = np.clip(nx, r, ROOM_W - r)
        ny = np.clip(ny, r, ROOM_H - r)

        # Collision per sumbu (sliding response)
        hit_x = any(obs.contains(nx, self.y) for obs in self.obstacles)
        hit_y = any(obs.contains(self.x, ny) for obs in self.obstacles)

        if not hit_x:
            self.x = nx
        else:
            self.vx *= -0.4
            self._wander_angle += np.pi/2 + np.random.uniform(-0.5, 0.5)
            self.stuck_timer += 1

        if not hit_y:
            self.y = ny
        else:
            self.vy *= -0.4
            self._wander_angle += np.pi/2 + np.random.uniform(-0.5, 0.5)
            self.stuck_timer += 1

        # Update angle dari velocity nyata (lerp agar mulus)
        if abs(self.vx) > 1e-4 or abs(self.vy) > 1e-4:
            target = np.arctan2(self.vy, self.vx)
            delta  = (target - self.angle + np.pi) % (2*np.pi) - np.pi
            self.angle += delta * ANGLE_LRP

        self._mark_clean(clean_grid)

    # ── Step utama (dipanggil tiap frame) ────────────────────────────────
    def step(self, clean_grid):
        dvx, dvy = self._desired_vel()
        for _ in range(SUBSTEPS):
            self._substep(dvx, dvy, clean_grid)

        self.step_count += 1

        # Rekam trail
        self.trail_x.append(self.x)
        self.trail_y.append(self.y)
        if len(self.trail_x) > TRAIL_MAX:
            self.trail_x.pop(0)
            self.trail_y.pop(0)

        # Auto-unstuck
        if self.stuck_timer > 12:
            self._wander_angle = np.random.uniform(0, 2*np.pi)
            self.vx = np.cos(self._wander_angle) * ROBOT_SPEED
            self.vy = np.sin(self._wander_angle) * ROBOT_SPEED
            self.stuck_timer = 0

    # ── Tandai area bersih ────────────────────────────────────────────────
    def _mark_clean(self, clean_grid):
        cx = int(self.x / GRID_RES)
        cy = int(self.y / GRID_RES)
        cr = int(CLEAN_R / GRID_RES) + 1
        rows, cols = clean_grid.shape
        for dr in range(-cr, cr+1):
            for dc in range(-cr, cr+1):
                gr, gc = cy+dr, cx+dc
                if 0 <= gr < rows and 0 <= gc < cols:
                    cell_x = gc*GRID_RES + GRID_RES/2
                    cell_y = gr*GRID_RES + GRID_RES/2
                    if np.hypot(cell_x-self.x, cell_y-self.y) <= CLEAN_R:
                        if clean_grid[gr, gc] == 0:
                            clean_grid[gr, gc] = 1


# =============================================================================
#  SIMULASI
# =============================================================================
class RoomCleaningSimulation:

    def __init__(self):
        self.grid_cols  = int(ROOM_W / GRID_RES)
        self.grid_rows  = int(ROOM_H / GRID_RES)
        self.clean_grid = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=np.uint8
        )
        self.obstacles  = self._create_obstacles()
        self._mark_obstacle_cells()
        self.robot = CleaningRobot(
            2.0, 2.0, self.obstacles,
            self.grid_cols, self.grid_rows
        )
        self.total_cleanable = int(np.sum(self.clean_grid == 0))

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
            c1 = min(self.grid_cols, int((obs.x+obs.w)/GRID_RES)+1)
            r0 = max(0, int(obs.y / GRID_RES))
            r1 = min(self.grid_rows, int((obs.y+obs.h)/GRID_RES)+1)
            self.clean_grid[r0:r1, c0:c1] = 2

    def update(self):
        self.robot.step(self.clean_grid)

    @property
    def pct_clean(self):
        return (int(np.sum(self.clean_grid == 1)) /
                max(self.total_cleanable, 1)) * 100

    # ------------------------------------------------------------------
    def render_to_bytes(self) -> bytes:
        """
        ★ KUNCI ANTI-FLICKER ★
        Render matplotlib figure ke PNG bytes (BytesIO).
        Dipanggil dengan st.image(bytes) → hanya src tag yang diupdate,
        komponen tidak di-unmount → TIDAK ADA KEDIP.
        """
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=RENDER_DPI)
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor(C_DIRTY)

        # Grid dekoratif
        for gx in np.arange(0, ROOM_W+1, 2.0):
            ax.axvline(gx, color=C_GRID, lw=0.3, alpha=0.4)
        for gy in np.arange(0, ROOM_H+1, 2.0):
            ax.axhline(gy, color=C_GRID, lw=0.3, alpha=0.4)

        # Area bersih & obstacle (imshow)
        cmap = ListedColormap([C_DIRTY, C_CLEAN, C_OBS_BASE])
        ax.imshow(
            self.clean_grid,
            origin='lower',
            extent=[0, ROOM_W, 0, ROOM_H],
            cmap=cmap, vmin=0, vmax=2,
            interpolation='bilinear',
            alpha=0.92, zorder=1,
        )

        # Obstacle (kotak dengan label)
        for obs in self.obstacles:
            # Shadow
            ax.add_patch(patches.FancyBboxPatch(
                (obs.x+0.08, obs.y-0.08), obs.w, obs.h,
                boxstyle="round,pad=0.1",
                facecolor="black", alpha=0.35, linewidth=0, zorder=2
            ))
            # Body
            ax.add_patch(patches.FancyBboxPatch(
                (obs.x, obs.y), obs.w, obs.h,
                boxstyle="round,pad=0.1",
                linewidth=1.2, edgecolor="white",
                facecolor=obs.color, alpha=0.92, zorder=3
            ))
            ax.text(
                obs.x+obs.w/2, obs.y+obs.h/2, obs.label,
                ha='center', va='center',
                fontsize=7, color='white', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='black')],
                zorder=4
            )

        # Trail gradien (opacity meningkat ke ujung)
        tx, ty = self.robot.trail_x, self.robot.trail_y
        if len(tx) > 2:
            n = len(tx)
            for i in range(1, n):
                alpha = (i / n) * 0.65 + 0.05
                lw    = (i / n) * 1.6  + 0.3
                ax.plot(
                    tx[i-1:i+1], ty[i-1:i+1],
                    color=C_TRAIL, linewidth=lw,
                    alpha=alpha, solid_capstyle='round', zorder=5
                )

        # Sensor range
        ax.add_patch(plt.Circle(
            (self.robot.x, self.robot.y), SENSOR_R,
            fill=False, linestyle='--', linewidth=0.7,
            color=C_SENSOR, alpha=0.20, zorder=6
        ))

        # Cleaning radius
        ax.add_patch(plt.Circle(
            (self.robot.x, self.robot.y), CLEAN_R,
            color=C_CLEAN, alpha=0.10, zorder=6
        ))

        # Glow berlapis
        for r_g, a_g in [(ROBOT_R+0.5, 0.05), (ROBOT_R+0.28, 0.12),
                         (ROBOT_R+0.12, 0.18)]:
            ax.add_patch(plt.Circle(
                (self.robot.x, self.robot.y), r_g,
                color=C_GLOW, alpha=a_g, zorder=7
            ))

        # Robot body
        ax.add_patch(plt.Circle(
            (self.robot.x, self.robot.y), ROBOT_R,
            color=C_ROBOT, zorder=8
        ))
        # Highlight (efek bola)
        ax.add_patch(plt.Circle(
            (self.robot.x - ROBOT_R*0.2,
             self.robot.y + ROBOT_R*0.2),
            ROBOT_R * 0.32,
            color='white', alpha=0.20, zorder=9
        ))

        # Arah (panah)
        ex = self.robot.x + np.cos(self.robot.angle) * (ROBOT_R + 0.38)
        ey = self.robot.y + np.sin(self.robot.angle) * (ROBOT_R + 0.38)
        ax.annotate(
            "", xy=(ex, ey), xytext=(self.robot.x, self.robot.y),
            arrowprops=dict(
                arrowstyle="-|>", color="white",
                lw=1.8, mutation_scale=12
            ), zorder=10
        )

        # Styling
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155"); sp.set_linewidth(1.5)
        ax.set_xlim(0, ROOM_W)
        ax.set_ylim(0, ROOM_H)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        fig.tight_layout(pad=0.3)

        # ── Render ke bytes ──────────────────────────────────────────
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=C_BG,
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


# =============================================================================
#  STREAMLIT APP
# =============================================================================
st.set_page_config(
    page_title="Robot Vacuum Cleaner",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #080c14; }
    .block-container { padding-top: 1.2rem !important; }
    h1 { color: #f8fafc !important; letter-spacing: 1px; }

    /* Tombol */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        font-size: 14px;
        padding: 0.45rem 1rem;
        transition: all 0.2s ease;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 10px 14px;
    }
    [data-testid="metric-container"] label {
        color: #64748b !important; font-size: 11px !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #f1f5f9 !important;
        font-size: 20px !important;
        font-weight: bold !important;
    }

    /* Progress */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0ea5e9, #22d3ee) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #1e293b;
    }

    /* Hapus padding atas sidebar */
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  SESSION STATE
# =============================================================================
if "sim"              not in st.session_state:
    st.session_state.sim             = RoomCleaningSimulation()
if "running"          not in st.session_state:
    st.session_state.running         = False
if "steps_per_frame"  not in st.session_state:
    st.session_state.steps_per_frame = 3
if "refresh_ms"       not in st.session_state:
    st.session_state.refresh_ms      = 60

sim = st.session_state.sim


# =============================================================================
#  SIDEBAR
# =============================================================================
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

    steps = st.slider(
        "Langkah per Frame", 1, 8, 3,
        help="Lebih tinggi = simulasi lebih cepat (tapi lebih berat CPU)"
    )
    st.session_state.steps_per_frame = steps

    ref_ms = st.slider(
        "Refresh (ms)", 30, 300, 60, 10,
        help="Delay antar frame — lebih rendah = lebih fluid"
    )
    st.session_state.refresh_ms = ref_ms

    st.markdown("---")
    st.markdown("### 🗺️ Legenda")
    st.markdown("""
<div style='font-size:13px;line-height:2.0;'>
🔵 &nbsp; Area Bersih<br>
⬛ &nbsp; Area Kotor<br>
🔴 &nbsp; Robot<br>
🟡 &nbsp; Sensor Range<br>
🩵 &nbsp; Jejak Robot
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='color:#475569;font-size:11px;line-height:1.6'>"
        "✔ Anti-flicker via BytesIO<br>"
        "✔ Velocity inertia + angle lerp<br>"
        "✔ Potential field avoidance<br>"
        "✔ Sub-step movement</div>",
        unsafe_allow_html=True
    )


# =============================================================================
#  HEADER
# =============================================================================
st.markdown(
    "<h1 style='text-align:center;font-family:monospace'>"
    "🤖 AI Robot Vacuum Cleaner Simulation</h1>",
    unsafe_allow_html=True
)

# Status banner
if st.session_state.running:
    st.markdown(
        "<div style='text-align:center;background:#052e16;"
        "border:1px solid #16a34a;border-radius:8px;padding:7px;"
        "color:#4ade80;font-weight:bold;font-size:14px;margin-bottom:10px'>"
        "🟢 Robot Aktif — Sedang Membersihkan</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div style='text-align:center;background:#1c1917;"
        "border:1px solid #44403c;border-radius:8px;padding:7px;"
        "color:#a8a29e;font-weight:bold;font-size:14px;margin-bottom:10px'>"
        "⏸ Robot Pause — Tekan ▶ START di sidebar</div>",
        unsafe_allow_html=True
    )


# =============================================================================
#  UPDATE SIMULASI
# =============================================================================
if st.session_state.running:
    for _ in range(st.session_state.steps_per_frame):
        sim.update()


# =============================================================================
#  METRIK
# =============================================================================
pct          = sim.pct_clean
cleaned_cells = int(np.sum(sim.clean_grid == 1))

m1, m2, m3, m4 = st.columns(4)
m1.metric("🧹 Area Bersih",    f"{pct:.1f}%")
m2.metric("👣 Total Langkah",  f"{sim.robot.step_count:,}")
m3.metric("📍 Posisi Robot",   f"({sim.robot.x:.1f}, {sim.robot.y:.1f})")
m4.metric("🗺️ Sel Dibersihkan", f"{cleaned_cells:,}")

st.progress(min(int(pct), 100))


# =============================================================================
#  VISUALISASI — ANTI-FLICKER
#
#  Teknik kunci:
#  • Render ke BytesIO BUKAN plt.show() / st.pyplot()
#  • st.empty() dibuat SEKALI di session_state
#  • Setiap frame hanya memanggil placeholder.image(bytes)
#  • Streamlit hanya update <img src=...> tanpa unmount komponen
#  • Hasil: ZERO FLICKER
# =============================================================================
if "img_ph" not in st.session_state:
    st.session_state.img_ph = st.empty()

st.session_state.img_ph.image(
    sim.render_to_bytes(),
    use_container_width=True,
    output_format="PNG"
)

# Detail teknis
with st.expander("ℹ️ Detail Teknis"):
    st.markdown(f"""
    | Parameter | Nilai | Keterangan |
    |-----------|-------|------------|
    | Render DPI | `{RENDER_DPI}` | Resolusi render per inci |
    | Sub-steps/frame | `{SUBSTEPS}` | Mikro-langkah per update |
    | Steps/render | `{st.session_state.steps_per_frame}` | Simulasi per frame |
    | Akselerasi | `{ACCEL}` | Lerp velocity [0-1] |
    | Angle lerp | `{ANGLE_LRP}` | Kelembutan rotasi [0-1] |
    | Anti-flicker | `st.image(BytesIO)` | Tidak re-mount komponen |
    | Avoidance | Potential Field kuadratik | Hindari obstacle secara halus |
    """)


# =============================================================================
#  AUTO-REFRESH
# =============================================================================
if st.session_state.running:
    time.sleep(st.session_state.refresh_ms / 1000)
    st.rerun()
