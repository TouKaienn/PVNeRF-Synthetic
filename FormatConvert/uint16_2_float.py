import numpy as np

# File paths
input_file = '/home/kuangshiai/Documents/Datasets/chameleon_1024x1024x1080_uint16.raw'
output_file = '/home/kuangshiai/Documents/Datasets/chameleon_1024x1024x1080_float.raw'

# Volume dimensions and data type
volume_shape = (1024, 1024, 1080)

# Load the .raw file
data_uint16 = np.fromfile(input_file, dtype=np.uint16)
# Reshape the data to the specified shape
data_uint16 = data_uint16.reshape(volume_shape)

# Convert the data to float
data_float = data_uint16.astype(np.float32)

# Save the data in float format
data_float.tofile(output_file)

print(f"Data converted to float and saved to {output_file}")