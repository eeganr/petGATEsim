import randoms

# Define input and output file paths
inpath = "/path/to/input/file.lm"
outpath = "/path/to/output/file_half.lm"

# Split the listmode file in half
randoms.split_lm_half(inpath, outpath)

print(f"File split successfully. Output written to: {outpath}")