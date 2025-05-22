import numpy as np
import nibabel as nib
import parallelproj
import uproot

# === CONFIG ===
input_root_file = "simoutput.root"
cont_magnitude = 1e-5
num_TOF_bins = 9
TOF_bin_width = 29
sigma_TOF = 60
num_iterations = 2
num_subsets = 8
image_shape = (200, 200, 700)  # (x, y, z) voxels # originally (310, 310, 310)
voxel_size = (0.1, 0.1, 0.1)  #mm #originally (1,1,1)
radius_mm = 10 # orinally 130 (mm)
use_tof = True # originally False

# === FORMAT ROOT FILE ===
print("Loading and processing ROOT file...")
file = uproot.open(input_root_file)
tree = file["Coincidences;1"]
branches = [
    "globalPosX1", "globalPosY1", "globalPosZ1",
    "globalPosX2", "globalPosY2", "globalPosZ2",
    "energy1", "energy2",
    "time1", "time2"
]
data = tree.arrays(branches, library="np")
array = np.column_stack([data[branch].astype(np.float32) for branch in branches])

delta_time = array[:, 9] - array[:, 8] # deltaT
c_mm_per_ps = 0.299792  # mm/ps
delta_distance_mm = 0.5 * c_mm_per_ps * delta_time  # convert ps → mm
tof_bin_indices = np.round(delta_distance_mm / TOF_bin_width).astype(np.int32)
tof_bin_indices = np.clip(tof_bin_indices, -num_TOF_bins // 2, num_TOF_bins // 2)
start_coord = array[:, [0, 1, 2]]  # x1, y1, z1
end_coord   = array[:, [3, 4, 5]]  # x2, y2, z2

print(f"🎉 Done! Processed {len(start_coord)} events.")

# GAUSS BLUR
TOR_FWHM = 3
## image based resolution model
res_model = parallelproj.GaussianFilterOperator(
    image_shape, sigma=TOR_FWHM / (2.35 * np.array(voxel_size))
)

parallelproj.__file__

img = np.ones(image_shape, dtype=np.float32)  # Use ones for stability

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

# mask = make_circular_mask(image_shape, voxel_size, radius_mm)
# img = mask * img

# NORMALIZATION

# normalization_weights = np.load("normalization_weights_test_masked.npy")
# normalization_weights = np.clip(normalization_weights, a_min=1, a_max=None)

def lm_em_update(x_cur, op, s):
    ybar = op(x_cur) + s
    return x_cur * op.adjoint(1 / ybar) # / normalization_weights

subset_ops = []
subset_slices = [slice(i, None, num_subsets) for i in range(num_subsets)]

for i, sl in enumerate(subset_slices):
    subset_lm_proj = parallelproj.ListmodePETProjector(
        start_coord[sl, :],
        end_coord[sl, :],
        image_shape,
        voxel_size,
    )

    ### attenuation correction as in Grant's code
    # att_img not given here
    #subset_att_list = np.exp(-subset_lm_proj(att_img))
    #subset_lm_att_op = parallelproj.ElementwiseMultiplicationOperator(subset_att_list)

    # TOF
    if use_tof:
        subset_lm_proj.tof_parameters = parallelproj.TOFParameters(num_tofbins=num_TOF_bins, tofbin_width=TOF_bin_width, sigma_tof=sigma_TOF)
        subset_lm_proj.event_tofbins = 1 * tof_bin_indices[sl]
        subset_lm_proj.tof = True
    ### if using Grant's attenuation correction, place subset_lm_att_op before subset_lm_proj
    subset_ops.append(parallelproj.CompositeLinearOperator((subset_lm_proj, res_model)))

subset_ops = parallelproj.LinearOperatorSequence(subset_ops)
contamination_list = np.full(
    start_coord.shape[0],
    float(cont_magnitude),
#    device=dev,
    dtype=np.float32,
)

for iter in range(num_iterations):
    print(f"Iteration {iter+1}/{num_iterations}")
    for k, (proj_k, sl) in enumerate(zip(subset_ops, subset_slices)):
        print(f"  Processing subset {k+1}/{num_subsets}")

        img = lm_em_update(
            img,
            subset_ops[k],
            contamination_list[sl]
        )  

print("OSEM Reconstruction complete!")

# Image Generation
img_np = np.asarray(img)
# img_np = np.flip(img_np, axis=0)  # Flip along the x-axis, xflip is this commented out
img_np = np.flip(img_np, axis=1)  # Flip along the y-axis
img_np = np.flip(img_np, axis=2)  # Flip along the z-axis
nib.save(nib.Nifti1Image(img_np, affine=np.eye(4)), "new3testcylinder3_norm_test.nii.gz")