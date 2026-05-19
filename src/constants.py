import numpy as np

# Fundamentales (unidades geometrizadas)
G: float = 1.0
C: float = 1.0
M: float = 1.0

# Radio de Schwarzschild (horizonte de eventos)
RS: float = 2.0 * G * M / C**2  # = 2M

# Radio de la esfera de fotones (orbita circular inestable de fotones)
R_PHOTON: float = 1.5 * RS  # = 3M

# Parametro de impacto critico (frontera de la sombra)
B_CRIT: float = 3.0 * np.sqrt(3.0) * M  # = 3*sqrt(3) M ~ 5.196

# Orbita circular estable mas interna (para particulas masivas / borde interno del disco)
R_ISCO: float = 3.0 * RS  # = 6M

# Radio exterior del disco por defecto
R_DISK_OUTER: float = 10.0 * RS  # = 20M

# Distancia de la camara por defecto
R_CAMERA: float = 30.0 * M
