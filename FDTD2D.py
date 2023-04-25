import numpy as np
import matplotlib as plt

grid_row = 200
grid_col = 200

ia = 10
ib = grid_row - ia - 1
ja = 10
jb = grid_col - ja - 1

ez = np.zeros((grid_row, grid_col))
dz = np.zeros((grid_row, grid_col))
hx = np.zeros((grid_row, grid_col))
hy = np.zeros((grid_row, grid_col))
iz = np.zeros((grid_row, grid_col))
ihx = np.zeros((grid_row, grid_col))
ihy = np.zeros((grid_row, grid_col))

ez_inc = np.zeros(grid_col)
hx_inc = np.zeros(grid_col)

delta_x = 0.01                          # Cell size
delta_t = delta_x / 6e8                 # Time step size

gaz = np.ones((grid_row, grid_col))
gbz = np.zeros((grid_row, grid_col))

# create the dielectric cylinder
epsr = 30
sigma = 0.3
radius = 5
centers = [(50, 50), (50, 100), (50, 150),
           (100, 50), (100, 100), (100, 150),
           (150, 50), (150, 100), (150, 150)]
epsz = 8.854e-12
for n in range(len(centers)):
    x, y = centers[n]
    for j in range(ja, jb):
        for i in range(ia, ib):
            xdist = x - i
            ydist = y - j
            dist = np.sqrt(xdist ** 2 + ydist ** 2)
            if dist <= radius:
                gaz[i, j] = 1 / (epsr + (sigma * delta_t / epsz))
                gbz[i, j] = (sigma * delta_t / epsz)


# Calculate the PML parameters
boundary_low = [0, 0]
boundary_high = [0, 0]
gi2 = np.ones(grid_row)
gi3 = np.ones(grid_row)
fi1 = np.zeros(grid_row)
fi2 = np.ones(grid_row)
fi3 = np.ones(grid_row)
gj2 = np.ones(grid_col)
gj3 = np.ones(grid_col)
fj1 = np.zeros(grid_col)
fj2 = np.ones(grid_col)
fj3 = np.ones(grid_col)

# Create the PML
npml = 10
for n in range(npml):
    xnum = npml - n
    xxn = xnum / npml
    xn = 0.33 * xxn ** 3
    gi2[n] = 1 / (1 + xn)
    gi2[grid_row - 1 - n] = 1 / (1 + xn)
    gi3[n] = (1 - xn) / (1 + xn)
    gi3[grid_row - 1 - n] = (1 - xn) / (1 + xn)
    gj2[n] = 1 / (1 + xn)
    gj2[grid_col - 1 - n] = 1 / (1 + xn)
    gj3[n] = (1 - xn) / (1 + xn)
    gj3[grid_col - 1 - n] = (1 - xn) / (1 + xn)
    xxn = (xnum - 0.5) / npml
    xn = 0.33 * xxn ** 3
    fi1[n] = xn
    fi1[grid_row - 2 - n] = xn
    fi2[n] = 1 / (1 + xn)
    fi2[grid_row - 2 - n] = 1 / (1 + xn)
    fi3[n] = (1 - xn) / (1 + xn)
    fi3[grid_row - 2 - n] = (1 - xn) / (1 + xn)
    fj1[n] = xn
    fj1[grid_col - 2 - n] = xn
    fj2[n] = 1 / (1 + xn)
    fj2[grid_col - 2 - n] = 1 / (1 + xn)
    fj3[n] = (1 - xn) / (1 + xn)
    fj3[grid_col - 2 - n] = (1 - xn) / (1 + xn)

# Pulse Parameters
t0 = 20
spread = 8

nsteps = 1000

# Main FDTD Loop
for time_step in range(1, nsteps + 1):
    # Incident Ez values
    for j in range(1, grid_col):
        ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j - 1] - hx_inc[j])


    # Absorbing Boundary Conditions
    ez_inc[0] = boundary_low.pop(0)
    boundary_low.append(ez_inc[1])

    ez_inc[grid_col - 1] = boundary_high.pop(0)
    boundary_high.append(ez_inc[grid_col - 2])

    # Calculate the Dz field
    for j in range(1, grid_col):
        for i in range(1, grid_row):
            dz[i, j] = gi3[i] * gj3[j] * dz[i, j] + gi2[i] * gj2[j] * 0.5 * \
                        (hy[i, j] - hy[i - 1, j] - hx[i, j] + hx[i, j - 1])
        
    # Source
    pulse = np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)
    ez_inc[3] = pulse

    # Incident Dz values
    for i in range(ia, ib + 1):
        dz[i, ja] = dz[i, ja] + 0.5 * hx_inc[ja - 1]
        dz[i, jb] = dz[i, jb] - 0.5 * hx_inc[jb]

    # Calculate the Ez field
    for j in range(0, grid_col):
        for i in range(0, grid_row):
            ez[i, j] = gaz[i, j] * (dz[i, j] - iz[i, j])
            iz[i, j] = iz[i, j] + gbz[i, j] * ez[i, j]

    # Calculate the Incident Hx
    for j in range(0, grid_col - 1):
        hx_inc[j] = hx_inc[j] + 0.5 * (ez_inc[j] - ez_inc[j + 1])

    # Calculate the Hx field
    for j in range(0, grid_col - 1):
        for i in range(0, grid_row):
            curl_e = ez[i, j] - ez[i, j + 1]
            ihx[i, j] = ihx[i, j] + curl_e
            hx[i, j] = fj3[j] * hx[i, j] + fj2[j] * \
                        (0.5 * curl_e + fi1[i] * ihx[i, j])

    # Incident Hx values
    for i in range(ia, ib + 1):
        hx[i, ja - 1] = hx[i, ja - 1] + 0.5 * ez_inc[ja]
        hx[i, jb] = hx[i, jb] - 0.5 * ez_inc[jb]

    # Calculate the Hy field
    for j in range(0, grid_col):
        for i in range(0, grid_row - 1):
            curl_e = ez[i, j] - ez[i + 1, j]
            ihy[i, j] = ihy[i, j] + curl_e
            hy[i, j] = fi3[i] * hy[i, j] - fi2[i] * \
                        (0.5 * curl_e + fj1[j] * ihy[i, j])

    # Incident Hy value
    for j in range(ja, jb + 1):
        hy[ia - 1, j] = hy[ia - 1, j] - 0.5 * ez_inc[j]
        hy[ib, j] = hy[ib, j] + 0.5 * ez_inc[j]

    with open("data_py.txt", "a") as f:
        np.savetxt(f, ez, fmt="%.2f", delimiter=" ")


    if time_step%20 == 0:
        print(f"{round((time_step/nsteps)*100)}% completed\n")

