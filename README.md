# Schwarzschild Black Hole Ray Tracer

Relativistic ray tracer for a Schwarzschild black hole with accretion disk, inspired by **Gargantua** from *Interstellar* (Kip Thorne, Double Negative, 2014).

Traces null geodesics through curved spacetime using the equation:

$$\frac{d^2 u}{d\phi^2} + u = \frac{3}{2}\, r_s\, u^2$$

where $u = 1/r$ and $r_s = 2GM/c^2$ is the Schwarzschild radius.

![Schwarzschild Black Hole](outputs/frames/frame_phase7.png)

![Accretion Disk Animation](outputs/animation.gif)

## Features

- **Backward ray tracing** through Schwarzschild spacetime with RK4 integration
- **Accretion disk** with Novikov-Thorne temperature profile and zero-torque ISCO boundary
- **Relativistic effects**: gravitational redshift, Doppler beaming, relativistic aberration
- **Gravitational lensing**: background star field distorted by spacetime curvature
- **Disk turbulence**: procedural FBM noise with Keplerian differential rotation
- **Limb darkening**: electron scattering model for emission angle dependence
- **Bloom post-processing**: multi-pass Gaussian blur for photon ring glow
- **Blackbody colormap**: temperature to RGB via Tanner Helland approximation
- **Dual pipeline**: Numba (CPU, parallel) and GLSL (GPU, real-time)
- **Interactive viewer**: real-time GPU ray tracer at 60+ FPS with mouse/keyboard controls
- **Screenshot & video capture**: F12 for PNG, F9 for MP4/GIF recording

## Quick Start

### Requirements

- Python 3.11+
- GPU with OpenGL 3.3+ support (for interactive mode)

### Installation

```bash
git clone https://github.com/your-username/interstellar.git
cd interstellar
pip install -r requirements.txt
```

### Usage

Launch the GUI:

```bash
python main.py
```

Or run individual modes from CLI:

```bash
python main.py geodesics      # Plot photon trajectories
python main.py frame           # Render single frame (Numba CPU)
python main.py animation       # 10s cinematic animation (240 frames)
python main.py interactive     # Real-time GPU viewer
```

## Controls (Interactive Mode)

| Key | Action |
|-----|--------|
| Mouse drag | Orbit camera (theta, phi) |
| Scroll | Zoom in/out |
| W/A/S/D | Orbit camera |
| R | Reset camera position |
| +/- | Adjust RK4 quality |
| F12 | Save screenshot (PNG) |
| F9 | Start/stop video recording (MP4 + GIF) |
| ESC | Exit |

## Project Structure

```
interstellar/
├── main.py                      # GUI launcher + CLI entry point
├── src/
│   ├── constants.py             # G=c=M=1, rs=2, ISCO, disk bounds
│   ├── geodesics.py             # ODE solver (SciPy RK45)
│   ├── disk.py                  # Disk crossing detection
│   ├── camera.py                # Pixel-to-ray mapping
│   ├── raytracer.py             # CPU ray tracing loop
│   ├── numba_kernels.py         # Numba @njit parallel kernels
│   ├── effects.py               # Redshift, Doppler, beaming
│   └── colormap.py              # Blackbody temperature to RGB
├── shaders/
│   ├── blackhole.vert           # Fullscreen quad vertex shader
│   ├── blackhole.frag           # GPU ray tracer (complete pipeline)
│   └── bloom.frag               # Bloom post-processing (extract/blur/compose)
├── scripts/
│   ├── plot_geodesics.py        # Geodesic trajectory plots
│   ├── render_frame.py          # Single frame renderer
│   ├── render_animation.py      # Cinematic animation generator
│   └── interactive.py           # Real-time OpenGL viewer
└── outputs/
    ├── frames/                  # Rendered frames
    ├── geodesic_plots/          # Trajectory plots
    ├── screenshots/             # F12 captures
    └── recordings/              # F9 video recordings
```

## Physics

### Schwarzschild Metric

Exact vacuum solution of Einstein's field equations for a spherically symmetric, non-rotating mass:

$$ds^2 = -\left(1 - \frac{r_s}{r}\right) c^2\, dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1} dr^2 + r^2\, d\Omega^2$$

### Key Radii

| Radius | Value | Meaning |
|--------|-------|---------|
| $r_s$ | $2M$ | Event horizon |
| $r_{ph}$ | $3M$ | Photon sphere (unstable circular orbits) |
| $r_{ISCO}$ | $6M$ | Innermost stable circular orbit |
| $b_{crit}$ | $3\sqrt{3}\,M$ | Critical impact parameter (shadow boundary) |

### Accretion Disk Model

Novikov-Thorne thin disk with zero-torque boundary at ISCO:

$$T(r) = T_{max} \cdot \left(\frac{r_{ISCO}}{r}\right)^{3/4} \cdot \left(1 - \sqrt{\frac{r_{ISCO}}{r}}\right)^{1/4}$$

Turbulence modeled via FBM noise with Keplerian differential rotation $\Omega(r) = \sqrt{M/r^3}$.

### Geodesic Trajectories

![Geodesic Trajectories](outputs/geodesic_plots/phase1_geodesics.png)

20 photon trajectories with varying impact parameters, showing deflection, orbiting, and capture by the black hole.

## Performance

Measured on AMD RX 6600 XT + Ryzen 5 5600X:

| Mode | Resolution | Performance |
|------|-----------|-------------|
| Interactive (GPU) | 960x540 | 800-1600 FPS |
| Single frame (Numba) | 640x360 | ~2s |
| Animation (Numba) | 640x360, 240 frames | ~8 min |

## References

- James, O., von Tunzelmann, E., Franklin, P., Thorne, K. (2015). *Gravitational Lensing by Spinning Black Holes in Astrophysics, and in the Movie Interstellar*. Class. Quantum Grav. 32, 065001.
- Novikov, I.D. & Thorne, K.S. (1973). *Astrophysics of Black Holes*. In: Black Holes.
- Page, D.N. & Thorne, K.S. (1974). *Disk-Accretion onto a Black Hole*. ApJ 191, 499.
- Riazuelo, A. (2018). *Seeing relativity*. arXiv:1511.06025.
- Misner, C.W., Thorne, K.S., Wheeler, J.A. (1973). *Gravitation*. W.H. Freeman.

## License

MIT
