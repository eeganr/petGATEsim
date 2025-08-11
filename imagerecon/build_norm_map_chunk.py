import numpy as np
import parallelproj
import nibabel as nib
import os

input_lm_file = '/Users/grantpark/Downloads/Norm_15files.lm'
output_name = "short_norm_test.npy"
radius_mm = 140
shape = (310, 310, 310)
voxel_size = (1, 1, 1) #mm

def make_circular_mask(shape, voxel_size, radius_mm):
    Nx, Ny, Nz = shape
    dx, dy, dz = voxel_size

    # Create grid of physical coordinates centered at 0
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dy
    z = (np.arange(Nz) - Nz // 2) * dz

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Compute radial distance from center (only xy-plane for cylindrical mask)
    r = np.sqrt(xx**2 + yy**2)

    # Create binary mask: 1 inside detector, 0 outside
    mask = (r <= radius_mm).astype(np.float32)

    return mask

mask = make_circular_mask(shape, voxel_size, radius_mm)

print("Loading and processing listmode file...")

chunk_size = int(1e8)  # number of events per chunk (adjust based on available RAM)
event_size_bytes = 10 * 4  # 10 float32s per event
file_size_bytes = os.path.getsize(input_lm_file)
num_events_total = file_size_bytes // event_size_bytes
print(f"Total events: {num_events_total}")

norm_image = np.zeros(shape, dtype=np.float32)  # initialize final result
TOR_FWHM = 2
res_model = parallelproj.GaussianFilterOperator(
    shape, sigma=TOR_FWHM / (2.35 * np.array(voxel_size))
)

with open(input_lm_file, "rb") as f:
    for chunk_start in range(0, num_events_total, chunk_size):
        num_to_read = min(chunk_size, num_events_total - chunk_start)
        data = np.fromfile(f, dtype=np.float32, count=num_to_read * 10).reshape((-1, 10))
        start_coord = data[:, [0, 1, 2]]
        end_coord = data[:, [5, 6, 7]]
        listmode_data = np.ones(len(start_coord), dtype=np.float32)

        projector = parallelproj.ListmodePETProjector(start_coord, end_coord, shape, voxel_size)
        norm_proj = parallelproj.CompositeLinearOperator((projector, res_model))
        partial_norm = norm_proj.adjoint(listmode_data)
        norm_image += partial_norm  # accumulate partial result

        print(f"Processed events {chunk_start} to {chunk_start + num_to_read}")

# Post-process and save
epsilon = 1e-6
norm_image *= mask
norm_image[np.isnan(norm_image)] = epsilon
norm_image[np.isinf(norm_image)] = epsilon
norm_image = np.clip(norm_image, a_min=epsilon, a_max=None)
np.save(output_name, norm_image)
nib.save(nib.Nifti1Image(norm_image, affine=np.eye(4)), "short_norm_test.nii.gz")

