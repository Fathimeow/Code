"""
F1 RACE PREDICTOR
-----------------
1. Physics model  -> calculates each car's speed at every point of the track
2. Lap simulator  -> turns speed into lap time
3. Monte Carlo    -> simulates the race thousands of times to estimate win probability
4. Animation      -> races the 3 cars live, with speedometers and win-probability bars

Run:  python f1_race_predictor.py      (or: py f1_race_predictor.py)
Needs: pip install matplotlib numpy
"""
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# =====================================================================
# SETTINGS  (change these to experiment!)
# =====================================================================
RACE_LAPS = 5            # laps in the race
N_SIMULATIONS = 5000     # Monte Carlo races used for win probability
ANIMATION_SPEEDUP = 1.0  # raise to watch the race faster
RANDOM_SEED = 42

G = 9.81      # gravity (m/s^2)
RHO = 1.225   # air density (kg/m^3)


# =====================================================================
# 1. CARS
# =====================================================================
@dataclass
class Car:
    name: str
    color: str
    power_kw: float      # engine power
    mass: float          # car + driver (kg)
    cd_a: float          # drag area  (lower = faster on straights)
    cl_a: float          # downforce area (higher = faster in corners)
    mu: float            # tyre grip coefficient
    consistency: float   # driver lap-to-lap variation (lower = steadier)
    mistake_prob: float  # chance of a mistake per lap


CARS = [
    Car("Red Falcon",   "#e10600", 740, 800, 1.10, 3.7, 1.70, 0.0060, 0.020),  # balanced
    Car("Blue Arrow",   "#2f80ff", 750, 800, 1.00, 3.3, 1.68, 0.0080, 0.030),  # low drag, fast on straights
    Car("Green Phantom", "#22c55e", 740, 800, 1.18, 4.0, 1.72, 0.0050, 0.015), # high downforce, fast in corners
]


# =====================================================================
# 2. TRACK
# =====================================================================
def build_track(n_points=700):
    """Create a closed circuit and return x, y, distance step and curvature."""
    th = np.linspace(0, 2 * np.pi, 6000, endpoint=False)
    r = 1 + 0.30 * np.cos(2 * th) + 0.13 * np.sin(3 * th + 0.5) + 0.06 * np.sin(5 * th)
    x, y = 480 * r * np.cos(th), 320 * r * np.sin(th)

    # resample so points are evenly spaced along the track
    seg = np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))
    s_dense = np.r_[0, np.cumsum(seg)]
    length = s_dense[-1]
    s = np.linspace(0, length, n_points, endpoint=False)
    xs = np.interp(s, s_dense, np.r_[x, x[0]])
    ys = np.interp(s, s_dense, np.r_[y, y[0]])
    ds = length / n_points

    # curvature = how fast the heading changes per metre
    heading = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    kappa = np.abs(np.gradient(heading, ds))
    k = 15  # smooth (wraps around the lap)
    pad = np.r_[kappa[-k:], kappa, kappa[:k]]
    kappa = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    return xs, ys, s, ds, kappa, length


# =====================================================================
# 3. PHYSICS: SPEED AT EVERY POINT OF THE TRACK
# =====================================================================
def speed_profile(car, kappa, ds):
    P = car.power_kw * 1000
    v_top = (P / (0.5 * RHO * car.cd_a)) ** (1 / 3)          # power = drag * speed

    # (a) Cornering limit:  m v^2 k = mu (m g + 0.5 rho ClA v^2)
    denom = car.mass * kappa - 0.5 * RHO * car.cl_a * car.mu
    with np.errstate(divide="ignore", invalid="ignore"):
        v_corner = np.where(denom > 1e-9,
                            np.sqrt(car.mu * car.mass * G / denom), v_top)
    v = np.minimum(v_corner, v_top)
    n = len(v)

    def brake_decel(speed):
        grip = car.mu * (car.mass * G + 0.5 * RHO * car.cl_a * speed ** 2)
        drag = 0.5 * RHO * car.cd_a * speed ** 2
        return min((grip + drag) / car.mass, 5.5 * G)

    def accel(speed):
        drive = P / max(speed, 5.0) - 0.5 * RHO * car.cd_a * speed ** 2
        traction = car.mu * (car.mass * G + 0.5 * RHO * car.cl_a * speed ** 2)
        return min(drive, traction) / car.mass

    # (b) Braking pass (backwards) and (c) acceleration pass (forwards),
    #     repeated twice so the loop closes smoothly
    for _ in range(2):
        for i in range(2 * n - 1, 0, -1):
            a, b = i % n, (i - 1) % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * brake_decel(v[a]) * ds))
        for i in range(2 * n - 1):
            a, b = i % n, (i + 1) % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * max(accel(v[a]), 0) * ds))
    return v


def lap_time_from_profile(v, ds):
    return np.sum(ds / v)


# =====================================================================
# 4. MONTE CARLO: WIN PROBABILITY
# =====================================================================
def simulate_race_times(base_laps, cars, rng, n_laps, n_sims):
    """Return total race time for every car in every simulation: (n_sims, n_cars)."""
    totals = np.zeros((n_sims, len(cars)))
    for j, car in enumerate(cars):
        noise = rng.normal(0, car.consistency, (n_sims, n_laps))
        mistakes = (rng.random((n_sims, n_laps)) < car.mistake_prob) * rng.uniform(2.0, 4.0, (n_sims, n_laps))
        totals[:, j] = np.sum(base_laps[j] * (1 + noise) + mistakes, axis=1)
    return totals


def win_probabilities(totals):
    winners = np.argmin(totals, axis=1)
    return np.bincount(winners, minlength=totals.shape[1]) / len(totals)


# =====================================================================
# RUN THE ANALYSIS
# =====================================================================
rng = np.random.default_rng(RANDOM_SEED)
tx, ty, ts, DS, kappa, TRACK_LEN = build_track()
profiles = [speed_profile(c, kappa, DS) for c in CARS]
base_laps = np.array([lap_time_from_profile(v, DS) for v in profiles])

totals = simulate_race_times(base_laps, CARS, rng, RACE_LAPS, N_SIMULATIONS)
probs = win_probabilities(totals)

print(f"\nTrack length: {TRACK_LEN:.0f} m   |   Race: {RACE_LAPS} laps   |   Simulations: {N_SIMULATIONS}\n")
print(f"{'Car':<15}{'Top speed':>11}{'Avg speed':>11}{'Min speed':>11}{'Lap time':>10}{'Win %':>8}")
for c, v, lt, p in zip(CARS, profiles, base_laps, probs):
    print(f"{c.name:<15}{v.max()*3.6:>8.0f} km/h{(TRACK_LEN/lt)*3.6:>7.0f} km/h"
          f"{v.min()*3.6:>7.0f} km/h{lt:>8.2f} s{p*100:>7.1f}%")
print()

# One sample race for the animation (a single random draw of the Monte Carlo)
anim_rng = np.random.default_rng(RANDOM_SEED + 1)
lap_mult = []
for car in CARS:
    noise = anim_rng.normal(0, car.consistency, RACE_LAPS)
    lap_mult.append(1 + noise)

cum_base = [np.r_[0, np.cumsum(ds_i)] for ds_i in [DS / v for v in profiles]]  # time at each track point


def build_timeline(j):
    """Arrays (time, distance) for car j over the whole race."""
    times, dists, t0 = [], [], 0.0
    for lap in range(RACE_LAPS):
        times.append(t0 + cum_base[j][:-1] * lap_mult[j][lap])
        dists.append(lap * TRACK_LEN + ts)
        t0 += base_laps[j] * lap_mult[j][lap]
    times.append([t0]); dists.append([RACE_LAPS * TRACK_LEN])
    return np.concatenate(times), np.concatenate(dists), t0


timelines = [build_timeline(j) for j in range(len(CARS))]
finish_times = [t[2] for t in timelines]
RACE_DURATION = max(finish_times) + 3

# =====================================================================
# 5. ANIMATION
# =====================================================================
plt.style.use("dark_background")
fig = plt.figure(figsize=(13, 7))
fig.patch.set_facecolor("#101018")
gs = GridSpec(2, 2, width_ratios=[1.6, 1], figure=fig, hspace=0.45, wspace=0.12)
ax_track = fig.add_subplot(gs[:, 0])
ax_speed = fig.add_subplot(gs[0, 1])
ax_prob = fig.add_subplot(gs[1, 1])

# --- track panel
ax_track.set_aspect("equal"); ax_track.axis("off")
ax_track.plot(np.r_[tx, tx[0]], np.r_[ty, ty[0]], color="#3b3b46", lw=22, solid_capstyle="round", zorder=1)
ax_track.plot(np.r_[tx, tx[0]], np.r_[ty, ty[0]], color="#55555f", lw=1, ls=(0, (6, 6)), zorder=2)
ax_track.plot([tx[0]], [ty[0]], marker="s", color="white", ms=9, zorder=3)  # start/finish
ax_track.set_title("F1 RACE PREDICTOR", color="white", fontsize=15, weight="bold")
dots = [ax_track.plot([], [], "o", color=c.color, ms=13, mec="white", mew=1.5, zorder=5)[0] for c in CARS]
tags = [ax_track.text(0, 0, c.name.split()[0], color=c.color, fontsize=9, weight="bold", zorder=6) for c in CARS]
clock = ax_track.text(0.02, 0.02, "", transform=ax_track.transAxes, color="white", family="monospace")
board = ax_track.text(0.02, 0.97, "", transform=ax_track.transAxes, color="white",
                      family="monospace", va="top", fontsize=10)

# --- live speedometer
ax_speed.set_title("Live speed (km/h)", color="white", fontsize=11)
top = max(v.max() for v in profiles) * 3.6
ax_speed.set_xlim(0, top * 1.1); ax_speed.set_yticks(range(3))
ax_speed.set_yticklabels([c.name for c in CARS]); ax_speed.invert_yaxis()
speed_bars = ax_speed.barh(range(3), [0] * 3, color=[c.color for c in CARS])
speed_txt = [ax_speed.text(0, i, "", va="center", color="white", fontsize=9) for i in range(3)]

# --- win probability
ax_prob.set_title(f"Win probability ({N_SIMULATIONS} simulated races)", color="white", fontsize=11)
ax_prob.set_ylim(0, max(probs) * 130)
ax_prob.set_xticks(range(3)); ax_prob.set_xticklabels([c.name.split()[0] for c in CARS])
ax_prob.bar(range(3), probs * 100, color=[c.color for c in CARS])
for i, p in enumerate(probs):
    ax_prob.text(i, p * 100 + 1, f"{p*100:.1f}%", ha="center", color="white", weight="bold")
ax_prob.set_ylabel("%")
for a in (ax_speed, ax_prob):
    a.set_facecolor("#16161f")
    for sp in a.spines.values():
        sp.set_color("#333344")

FPS = 30
TOTAL_FRAMES = int(RACE_DURATION * FPS / ANIMATION_SPEEDUP)


def car_state(j, t):
    times, dists, tf = timelines[j]
    d = np.interp(t, times, dists)
    finished = d >= RACE_LAPS * TRACK_LEN
    pos = d % TRACK_LEN
    x = np.interp(pos, ts, tx, period=TRACK_LEN)
    y = np.interp(pos, ts, ty, period=TRACK_LEN)
    v = 0.0 if finished else np.interp(pos, ts, profiles[j], period=TRACK_LEN)
    v = v / lap_mult[j][min(int(d // TRACK_LEN), RACE_LAPS - 1)]
    return x, y, v, d, finished


def update(frame):
    t = frame / FPS * ANIMATION_SPEEDUP
    states = [car_state(j, t) for j in range(3)]
    order = sorted(range(3), key=lambda j: (-states[j][3], finish_times[j]))
    lines = ["POS  CAR            LAP"]
    for rank, j in enumerate(order, 1):
        x, y, v, d, fin = states[j]
        dots[j].set_data([x], [y]); tags[j].set_position((x + 12, y + 12))
        speed_bars[j].set_width(v * 3.6)
        speed_txt[j].set_position((v * 3.6 + 4, j)); speed_txt[j].set_text(f"{v*3.6:.0f}")
        lap = min(int(d // TRACK_LEN) + 1, RACE_LAPS)
        status = "FINISH" if fin else f"{lap}/{RACE_LAPS}"
        lines.append(f"P{rank}   {CARS[j].name:<14} {status}")
    board.set_text("\n".join(lines))
    clock.set_text(f"Race time {t:6.1f} s")
    return dots + tags + list(speed_bars) + speed_txt + [board, clock]


ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000 / FPS, blit=False, repeat=False)

# To save the race as a GIF, uncomment this line (takes a minute):
# ani.save("f1_race.gif", writer="pillow", fps=20)

if __name__ == "__main__":
    plt.show()
