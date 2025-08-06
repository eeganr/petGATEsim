import numpy as np
import parallelproj
import nibabel as nib
import os
import cupy as cp
xp = cp

#diagonal length: ~350mm
input_lm_file = r"G:\My Drive\PET_Insert_Gen_II\Example_Data\Gen II Study_and_Dose\Phantom_06032025\LM\Resolution_Large.lm"
# NORMALIZATION
normalization_weights = xp.asarray(np.load(r"E:\LM\res_norm_att_WATER_corrected_registered2.npy"))
num_iterations = 2
num_subsets = 16
image_shape = (512, 512, 397)
voxel_size = (0.48606, 0.48606, 0.48606) #mm
cont_magnitude = 1e-5
num_TOF_bins = 351
TOF_bin_width = 1
sigma_TOF = 31.85
radius_mm = 140
use_tof = False
use_att = False
if use_att:
    att_nifti = nib.load(r"E:\LM\registered_att_img_water.nii").get_fdata().astype(np.float32)
    att_img = xp.asarray(att_nifti)
ref_affine = nib.load(r"E:\LM\res_attenuation_map_ct.nii").affine
#efficiency_per_event = np.memmap(r"E:\LM\event_efficiency.npy", dtype=np.float32, mode='r')
# scatter_per_event = np.load(r"E:\LM\hoffman_scatter_per_event.npy").astype(np.float32)

print("Loading and processing listmode file...")

# GAUSS BLUR
TOR_FWHM = 2
res_model = parallelproj.GaussianFilterOperator(
    image_shape, sigma=TOR_FWHM / (2.35 * np.array(voxel_size))
)

img = xp.ones(image_shape, dtype=np.float32)  # Use ones for stability

def make_circular_mask(shape, voxel_size, radius_mm):
    Nx, Ny, Nz = shape
    dx, dy, dz = voxel_size
    x = (xp.arange(Nx) - Nx // 2) * dx
    y = (xp.arange(Ny) - Ny // 2) * dy
    z = (xp.arange(Nz) - Nz // 2) * dz
    xx, yy, zz = xp.meshgrid(x, y, z, indexing='ij')

    # Compute radial distance from center (only xy-plane for cylindrical mask)
    r = xp.sqrt(xx**2 + yy**2)

    # Create binary mask: 1 inside detector, 0 outside
    mask = (r <= radius_mm).astype(xp.float32)
    return mask

mask = make_circular_mask(image_shape, voxel_size, radius_mm)
img = mask * img

def lm_em_update(x_cur, op, s):
    ybar = op(x_cur) + s
    return x_cur * op.adjoint(1 / ybar) / normalization_weights

# === CHUNKED PROCESSING ===
chunk_size = int(1e8)
num_floats_per_event = 10
float_size = 4  # bytes
event_size_bytes = num_floats_per_event * float_size
file_size = os.path.getsize(input_lm_file)
num_events_total = file_size // event_size_bytes
events_per_subset = num_events_total // num_subsets

print(f"Total events: {num_events_total:,}")
print(f"Events per subset (chunk): {events_per_subset:,}")

with open(input_lm_file, 'rb') as f:
    for iter in range(num_iterations):
        print(f"\n=== Iteration {iter+1}/{num_iterations} ===")
        f.seek(0)
        for subset_idx in range(num_subsets):
            print(f"Subset {subset_idx+1}/{num_subsets}")
            buffer = np.fromfile(f, dtype=np.float32, count=events_per_subset * num_floats_per_event)
            if buffer.size == 0:
                break
            buffer = buffer.reshape(-1, num_floats_per_event)
            start_idx = subset_idx * events_per_subset
            end_idx = start_idx + len(buffer)
            # eff_subset = xp.asarray(efficiency_per_event[start_idx:end_idx])
            start_coord = xp.asarray(buffer[:, [0, 1, 2]])
            end_coord   = xp.asarray(buffer[:, [5, 6, 7]])
            delta_distance_mm = xp.asarray(buffer[:, 3])
            c_mm_per_ps = 0.299792  # mm/ps
            # print(f"Delta Distance (mm) - Min: {xp.min(delta_distance_mm)}, Max: {xp.max(delta_distance_mm)}")
            # print(f"Delta Distance (mm) - Mean: {xp.mean(delta_distance_mm)}, Std Dev: {xp.std(delta_distance_mm)}")
            # delta_distance_mm = 0.5 * c_mm_per_ps * delta_time  # convert ps → mm
            half_bins = num_TOF_bins // 2
            tof_bin_indices = xp.round(delta_distance_mm / TOF_bin_width).astype(np.int32)
            tof_bin_indices = xp.clip(tof_bin_indices, -half_bins, half_bins)
            tof_bin_indices = xp.asarray(tof_bin_indices)  # Move to GPU
            # print("Shape:", tof_bin_indices.shape)
            # print("Min:", tof_bin_indices.min())
            # print("Max:", tof_bin_indiceconss.max())
            # print("Mean:", tof_bin_indices.mean())
            # print("Std Dev:", tof_bin_indices.std())
            # scatter_subset = xp.asarray(scatter_per_event[start_idx:end_idx])
            # contamination_list = scatter_subset
            contamination_list = xp.full(start_coord.shape[0], cont_magnitude, dtype=np.float32)

            proj = parallelproj.ListmodePETProjector(
                start_coord, end_coord, image_shape, voxel_size
            )
            # TOF
            if use_tof:
                proj.tof_parameters = parallelproj.TOFParameters(
                    num_tofbins=num_TOF_bins,
                    tofbin_width=TOF_bin_width,
                    sigma_tof=sigma_TOF
                )
                proj.event_tofbins = tof_bin_indices
                proj.tof = True
            # attenuation
            # subset_eff_op = parallelproj.ElementwiseMultiplicationOperator(eff_subset)
            if use_att:
                subset_att_list = xp.exp(-proj(att_img))
                subset_lm_att_op = parallelproj.ElementwiseMultiplicationOperator(subset_att_list)
                # op = parallelproj.CompositeLinearOperator((subset_lm_att_op, subset_eff_op, proj, res_model))
                op = parallelproj.CompositeLinearOperator((subset_lm_att_op, proj, res_model))
            else:
                op = parallelproj.CompositeLinearOperator((proj, res_model)) #subset_lm_att_op, subset_eff_op, 
            img = lm_em_update(img, op, contamination_list)
        print(f"  Saving image at iteration {iter+1}")
        img_np = img.get()
        # img_np = np.flip(img_np, axis=1)  # Flip y-axis
        # img_np = np.flip(img_np, axis=2)  # Flip z-axis
        nib.save(
            nib.Nifti1Image(img_np, affine=ref_affine), f"E:\\LM\\res_ATT_registered_{iter+1:03d}.nii"
        )

print("OSEM Reconstruction complete!")