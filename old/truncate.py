import os

file_path = '/scratch/groups/cslevin/eeganr/gen2annulus2/annulus_delay.lm'

# Get the current size of the file
file_size = os.path.getsize(file_path)
half_size = file_size // 2  # Use integer division

# Open file in read/write binary mode and truncate it
with open(file_path, 'rb+') as f:
    f.truncate(half_size)

print(f"Truncated file to {half_size} bytes.")