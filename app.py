# =========================================================
# SIMULASI ROBOT PEMBERSIH RUANGAN - STREAMLIT VERSION
# =========================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
import time

# =========================================================
# KONFIGURASI
# =========================================================
ROOM_WIDTH = 20
ROOM_HEIGHT = 15

GRID_RES = 0.25

ROBOT_RADIUS = 0.5
ROBOT_SPEED = 0.18

SENSOR_RANGE = 1.8
CLEAN_RADIUS = 0.7

TRAIL_MAXLEN = 180

# warna
COLOR_FLOOR_DIRTY = "#1a1a2e"
COLOR_OBSTACLE_BG = "#533483"
COLOR_ROBOT = "#e94560"
COLOR_ROBOT_GLOW = "#ff6b6b"
COLOR_TRAIL = "#4ecdc4"
COLOR_SENSOR = "#f7d794"
COLOR_BG = "#0d0d1a"

# =========================================================
# CLASS OBSTACLE
# =========================================================
class Obstacle:

    def __init__(self, x, y, w, h, label="", color="#533483"):

        self.x = x
        self.y = y

        self.w = w
        self.h = h

        self.label = label
        self.color = color

        self.margin = ROBOT_RADIUS + 0.15

    def get_expanded_bounds(self):

        return (
            self.x - self.margin,
            self.y - self.margin,
            self.x + self.w + self.margin,
            self.y + self.h + self.margin
        )

    def contains_point(self, px, py):

        x0, y0, x1, y1 = self.get_expanded_bounds()

        return x0 <= px <= x1 and y0 <= py <= y1

    def distance_to_point(self, px, py):

        cx = max(self.x, min(px, self.x + self.w))
        cy = max(self.y, min(py, self.y + self.h))

        return np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

# =========================================================
# CLASS ROBOT
# =========================================================
class CleaningRobot:

    def __init__(self, x, y, obstacles, grid_cols, grid_rows):

        self.x = float(x)
        self.y = float(y)

        self.angle = np.random.uniform(0, 2 * np.pi)

        self.obstacles = obstacles

        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

        self.step_count = 0
        self.stuck_count = 0

        self.trail_x = []
        self.trail_y = []

    def sense_obstacles(self):

        detected = []

        for obs in self.obstacles:

            dist = obs.distance_to_point(self.x, self.y)

            if dist < SENSOR_RANGE:
                detected.append((dist, obs))

        return detected

    def compute_avoidance_vector(self, detected):

        avoid_x = 0.0
        avoid_y = 0.0

        for dist, obs in detected:

            if dist < 0.01:
                dist = 0.01

            cx = obs.x + obs.w / 2
            cy = obs.y + obs.h / 2

            dx = self.x - cx
            dy = self.y - cy

            norm = np.sqrt(dx**2 + dy**2) + 1e-6

            force = (SENSOR_RANGE - dist) / SENSOR_RANGE

            avoid_x += (dx / norm) * force * 2.5
            avoid_y += (dy / norm) * force * 2.5

        return avoid_x, avoid_y

    def compute_wall_avoidance(self):

        wall_margin = ROBOT_RADIUS + 0.3

        wx = 0.0
        wy = 0.0

        if self.x < wall_margin:
            wx += (wall_margin - self.x) / wall_margin * 3

        if self.x > ROOM_WIDTH - wall_margin:
            wx -= (self.x - (ROOM_WIDTH - wall_margin)) / wall_margin * 3

        if self.y < wall_margin:
            wy += (wall_margin - self.y) / wall_margin * 3

        if self.y > ROOM_HEIGHT - wall_margin:
            wy -= (self.y - (ROOM_HEIGHT - wall_margin)) / wall_margin * 3

        return wx, wy

    def step(self, clean_grid):

        self.step_count += 1

        detected = self.sense_obstacles()

        avoid_x, avoid_y = self.compute_avoidance_vector(detected)
        wall_x, wall_y = self.compute_wall_avoidance()

        move_x = np.cos(self.angle)
        move_y = np.sin(self.angle)

        total_x = move_x + avoid_x * 0.8 + wall_x
        total_y = move_y + avoid_y * 0.8 + wall_y

        noise = 0.12

        total_x += np.random.uniform(-noise, noise)
        total_y += np.random.uniform(-noise, noise)

        mag = np.sqrt(total_x**2 + total_y**2) + 1e-6

        total_x /= mag
        total_y /= mag

        self.angle = np.arctan2(total_y, total_x)

        new_x = self.x + total_x * ROBOT_SPEED
        new_y = self.y + total_y * ROBOT_SPEED

        collision = False

        for obs in self.obstacles:

            if obs.contains_point(new_x, new_y):
                collision = True
                break

        r = ROBOT_RADIUS

        new_x = max(r, min(ROOM_WIDTH - r, new_x))
        new_y = max(r, min(ROOM_HEIGHT - r, new_y))

        if not collision:

            self.x = new_x
            self.y = new_y

            self.stuck_count = 0

        else:

            self.angle += np.pi + np.random.uniform(-0.5, 0.5)

            self.stuck_count += 1

        if self.stuck_count > 25:

            self.angle = np.random.uniform(0, 2 * np.pi)

            self.stuck_count = 0

        self.trail_x.append(self.x)
        self.trail_y.append(self.y)

        if len(self.trail_x) > TRAIL_MAXLEN:

            self.trail_x.pop(0)
            self.trail_y.pop(0)

        self.mark_clean(clean_grid)

    def mark_clean(self, clean_grid):

        cx = int(self.x / GRID_RES)
        cy = int(self.y / GRID_RES)

        cr = int(CLEAN_RADIUS / GRID_RES) + 1

        for dr in range(-cr, cr + 1):
            for dc in range(-cr, cr + 1):

                gr = cy + dr
                gc = cx + dc

                if 0 <= gr < self.grid_rows and 0 <= gc < self.grid_cols:

                    cell_x = gc * GRID_RES + GRID_RES / 2
                    cell_y = gr * GRID_RES + GRID_RES / 2

                    dist = np.sqrt(
                        (cell_x - self.x) ** 2 +
                        (cell_y - self.y) ** 2
                    )

                    if dist <= CLEAN_RADIUS:
                        clean_grid[gr, gc] = 1

# =========================================================
# CLASS SIMULATION
# =========================================================
class RoomCleaningSimulation:

    def __init__(self):

        self.grid_cols = int(ROOM_WIDTH / GRID_RES)
        self.grid_rows = int(ROOM_HEIGHT / GRID_RES)

        self.clean_grid = np.zeros(
            (self.grid_rows, self.grid_cols),
            dtype=np.uint8
        )

        self.obstacles = self.create_obstacles()

        self.mark_obstacle_cells()

        self.robot = CleaningRobot(
            2.0,
            2.0,
            self.obstacles,
            self.grid_cols,
            self.grid_rows
        )

        self.total_cleanable = int(np.sum(self.clean_grid == 0))

    def create_obstacles(self):

        return [

            Obstacle(3.5, 5.0, 3.5, 2.2, "Meja", "#e94560"),
            Obstacle(13.0, 10.5, 4.5, 2.8, "Sofa", "#533483"),
            Obstacle(0.3, 11.5, 2.5, 3.2, "Lemari", "#0f3460"),
            Obstacle(14.5, 5.5, 2.2, 2.2, "Kursi", "#f5a623"),
            Obstacle(10.5, 1.0, 3.0, 2.0, "Meja Kecil", "#27ae60"),
            Obstacle(17.5, 1.0, 2.2, 5.5, "Rak Buku", "#8e44ad"),
            Obstacle(8.5, 6.5, 1.5, 1.5, "Tanaman", "#16a085"),
        ]

    def mark_obstacle_cells(self):

        for obs in self.obstacles:

            c0 = max(0, int(obs.x / GRID_RES))
            c1 = min(self.grid_cols, int((obs.x + obs.w) / GRID_RES) + 1)

            r0 = max(0, int(obs.y / GRID_RES))
            r1 = min(self.grid_rows, int((obs.y + obs.h) / GRID_RES) + 1)

            self.clean_grid[r0:r1, c0:c1] = 2

    def update(self):

        for _ in range(6):
            self.robot.step(self.clean_grid)

    def draw(self):

        fig, ax = plt.subplots(figsize=(10, 6), dpi=85)

        fig.patch.set_facecolor(COLOR_BG)

        ax.set_facecolor(COLOR_FLOOR_DIRTY)

        cmap_clean = ListedColormap([
            COLOR_FLOOR_DIRTY,
            "#1e5f74",
            COLOR_OBSTACLE_BG,
        ])

        ax.imshow(
            self.clean_grid,
            origin='lower',
            extent=[0, ROOM_WIDTH, 0, ROOM_HEIGHT],
            cmap=cmap_clean,
            vmin=0,
            vmax=2,
            interpolation='nearest'
        )

        # obstacle
        for obs in self.obstacles:

            rect = patches.FancyBboxPatch(
                (obs.x, obs.y),
                obs.w,
                obs.h,
                boxstyle="round,pad=0.08",
                linewidth=1.2,
                edgecolor="white",
                facecolor=obs.color,
                alpha=0.9
            )

            ax.add_patch(rect)

            ax.text(
                obs.x + obs.w / 2,
                obs.y + obs.h / 2,
                obs.label,
                ha='center',
                va='center',
                fontsize=7,
                color='white',
                fontweight='bold',
                path_effects=[
                    pe.withStroke(linewidth=2, foreground='black')
                ]
            )

        # trail
        ax.plot(
            self.robot.trail_x,
            self.robot.trail_y,
            color=COLOR_TRAIL,
            linewidth=1,
            alpha=0.6
        )

        # sensor
        sensor = plt.Circle(
            (self.robot.x, self.robot.y),
            SENSOR_RANGE,
            fill=False,
            linestyle='--',
            linewidth=0.8,
            color=COLOR_SENSOR,
            alpha=0.3
        )

        ax.add_patch(sensor)

        # glow
        glow = plt.Circle(
            (self.robot.x, self.robot.y),
            ROBOT_RADIUS + 0.25,
            color=COLOR_ROBOT_GLOW,
            alpha=0.2
        )

        ax.add_patch(glow)

        # robot
        robot_circle = plt.Circle(
            (self.robot.x, self.robot.y),
            ROBOT_RADIUS,
            color=COLOR_ROBOT
        )

        ax.add_patch(robot_circle)

        # arah robot
        ex = self.robot.x + np.cos(self.robot.angle) * 0.8
        ey = self.robot.y + np.sin(self.robot.angle) * 0.8

        ax.plot(
            [self.robot.x, ex],
            [self.robot.y, ey],
            color='white',
            linewidth=2
        )

        ax.set_xlim(0, ROOM_WIDTH)
        ax.set_ylim(0, ROOM_HEIGHT)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect('equal')

        ax.set_title(
            "🤖 AI Robot Vacuum Cleaner Simulation",
            fontsize=16,
            color='white',
            fontweight='bold'
        )

        return fig

# =========================================================
# STREAMLIT APP
# =========================================================
st.set_page_config(
    page_title="Robot Cleaner",
    layout="wide"
)

st.title("🤖 Simulasi Robot Pembersih Ruangan")

# =========================================================
# SESSION STATE
# =========================================================
if "sim" not in st.session_state:
    st.session_state.sim = RoomCleaningSimulation()

if "running" not in st.session_state:
    st.session_state.running = False

sim = st.session_state.sim

# =========================================================
# SIDEBAR CONTROL
# =========================================================
st.sidebar.title("🎮 Kontrol")

if st.sidebar.button("▶ START"):
    st.session_state.running = True

if st.sidebar.button("⏸ PAUSE"):
    st.session_state.running = False

if st.sidebar.button("↺ RESET"):

    st.session_state.sim = RoomCleaningSimulation()
    st.session_state.running = False

    sim = st.session_state.sim

# =========================================================
# PLACEHOLDER
# =========================================================
progress_placeholder = st.empty()
metric_placeholder = st.empty()
plot_placeholder = st.empty()
status_placeholder = st.empty()

# =========================================================
# UPDATE REALTIME
# =========================================================
cleaned = int(np.sum(sim.clean_grid == 1))

pct = (
    cleaned /
    max(sim.total_cleanable, 1)
) * 100

# progress
progress_placeholder.progress(min(int(pct), 100))

# metrics
col1, col2, col3 = metric_placeholder.columns(3)

col1.metric(
    "Persentase Bersih",
    f"{pct:.1f}%"
)

col2.metric(
    "Langkah Robot",
    f"{sim.robot.step_count}"
)

col3.metric(
    "Posisi Robot",
    f"({sim.robot.x:.2f}, {sim.robot.y:.2f})"
)

# status
if st.session_state.running:
    status_placeholder.success("🟢 Robot Sedang Membersihkan")
else:
    status_placeholder.warning("⏸ Robot Pause")

# =========================================================
# CONTAINER ANIMASI
# =========================================================
animation_container = st.container()

# =========================================================
# UPDATE ROBOT
# =========================================================
if st.session_state.running:
    sim.update()

# =========================================================
# TAMPILKAN GAMBAR
# =========================================================
with animation_container:

    fig = sim.draw()

    st.pyplot(
        fig,
        clear_figure=True,
        use_container_width=True
    )

    plt.close(fig)

# =========================================================
# REFRESH HALUS
# =========================================================
if st.session_state.running:

    time.sleep(0.05)

    st.rerun()