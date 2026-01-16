import numpy as np
import math

# Configuration
FILENAME = "nema_iec_128.vox"
NVOX = [128, 128, 128]
DVOX = [0.25, 0.25, 0.25] # cm

# Materials and Activities (Bq)
MAT_AIR = 1
DENS_AIR = 0.0012
ACT_AIR = 0.0

MAT_WATER = 2
DENS_WATER = 1.0
ACT_BKG = 50.0
ACT_HOT = 200.0  # 4:1 ratio is typical, let's go 4:1 (200 vs 50)
ACT_COLD = 0.0

# Geometry Definitions (in cm)
# Center of the grid
CX = (NVOX[0] * DVOX[0]) / 2.0
CY = (NVOX[1] * DVOX[1]) / 2.0
CZ = (NVOX[2] * DVOX[2]) / 2.0

PHANTOM_RADIUS = 10.0 # 20cm diameter
SPHERE_RING_RADIUS = 5.72 

# Sphere Diameters (mm) -> Radius (cm)
# 10, 13, 17, 22, 28, 37 mm
SPHERE_RADII = [
    1.0/2.0, 1.3/2.0, 1.7/2.0, 
    2.2/2.0, 2.8/2.0, 3.7/2.0
]
# Let's make the smallest 2 cold, largest 4 hot? Or typical NEMA is all hot vs background?
# Actually usually all spheres are fillable. 
# Typical setup: The 4 largest are hot, 2 smallest are cold? No, NEMA image quality usually has all hot or cold/hot mix.
# Let's do: 4 largest HOT, 2 smallest COLD.
SPHERE_TYPES = ['COLD', 'COLD', 'HOT', 'HOT', 'HOT', 'HOT']

def generate_phantom():
    print(f"Generating {FILENAME} ({NVOX} voxels)...")
    
    with open(FILENAME, 'w') as f:
        # Header
        f.write('[SECTION VOXELS HEADER v.2008-04-13]\n')
        f.write(f'{NVOX[0]} {NVOX[1]} {NVOX[2]}   No. OF VOXELS IN X,Y,Z\n')
        f.write(f'{DVOX[0]} {DVOX[1]} {DVOX[2]}   VOXEL SIZE (cm) ALONG X,Y,Z\n')
        f.write(' 1                  COLUMN NUMBER WHERE MATERIAL ID IS LOCATED\n')
        f.write(' 2                  COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED\n')
        f.write(' 1                  BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)\n')
        f.write('[END OF VXH SECTION]  # MCGPU-PET voxel format: Material  Density  Activity\n')

        # Sphere Centers
        sphere_centers = []
        for i in range(6):
            angle = i * (360.0 / 6.0) * (math.pi / 180.0)
            sx = CX + SPHERE_RING_RADIUS * math.cos(angle)
            sy = CY + SPHERE_RING_RADIUS * math.sin(angle)
            sz = CZ # Centered in Z
            sphere_centers.append((sx, sy, sz))
            print(f"Sphere {i+1}: Radius={SPHERE_RADII[i]:.2f}cm, Type={SPHERE_TYPES[i]}, Pos=({sx:.1f}, {sy:.1f}, {sz:.1f})")

        # 7. EXTRA LARGE HOT INSERT (User Request)
        # Radius 2.5cm, Positioned clearly off-center
        extra_rad = 2.5
        extra_pos = (CX + 6.0, CY + 0.0, CZ)
        sphere_centers.append(extra_pos)
        SPHERE_RADII.append(extra_rad)
        SPHERE_TYPES.append('HOT')
        print(f"Sphere 7 (Extra): Radius={extra_rad:.2f}cm, Type=HOT, Pos={extra_pos}")

        # Generate Voxels (Z, Y, X loops as per format)
        for k in range(NVOX[2]):
            z = (k + 0.5) * DVOX[2]
            for j in range(NVOX[1]):
                y = (j + 0.5) * DVOX[1]
                for i in range(NVOX[0]):
                    x = (i + 0.5) * DVOX[0]
                    
                    # Logic
                    dist_to_center_sq = (x - CX)**2 + (y - CY)**2
                    
                    # Default to Air
                    mat = MAT_AIR
                    dens = DENS_AIR
                    act = ACT_AIR
                    
                    # Inside Phantom Cylinder? (Infinite cylinder in Z for simplicity, or bounded?)
                    # Let's bound Z to be within some margin
                    if dist_to_center_sq <= PHANTOM_RADIUS**2 and 2.0 < z < (NVOX[2]*DVOX[2] - 2.0):
                        mat = MAT_WATER
                        dens = DENS_WATER
                        act = ACT_BKG
                        
                        # Check Spheres
                        for s_idx, (sx, sy, sz) in enumerate(sphere_centers):
                            dist_to_sphere_sq = (x - sx)**2 + (y - sy)**2 + (z - sz)**2
                            if dist_to_sphere_sq <= SPHERE_RADII[s_idx]**2:
                                if SPHERE_TYPES[s_idx] == 'HOT':
                                    act = ACT_HOT
                                elif SPHERE_TYPES[s_idx] == 'COLD':
                                    act = ACT_COLD
                                break # Found a sphere, stop checking others
                    
                    f.write(f"{mat} {dens} {act}\n")
                f.write('\n') # End of X loop
            f.write('\n') # End of Y loop

    print("Done.")

if __name__ == "__main__":
    generate_phantom()
