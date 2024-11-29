import os
import numpy as np
import vtk
from vtk.util import numpy_support
from paraview.simple import *
# This script must be run with pvpython, not python
# for example: resources/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython FormatConvert/vti2npy_raw.py

def process_vti_to_npy_and_raw(input_path, output_npy_dir, output_raw_dir, slice_height):
    """
    Processes all .vti files in the input directory, restores the voxel-wise labels,
    and saves them as both .npy and .raw files in the respective output directories.

    Args:
        input_path (str): Path to the directory containing .vti files.
        output_npy_dir (str): Path to the directory where .npy files will be saved.
        output_raw_dir (str): Path to the directory where .raw files will be saved.
        slice_height (int): Number of times each slice is repeated along the z-axis.
    """
    # Ensure output directories exist
    os.makedirs(output_npy_dir, exist_ok=True)
    os.makedirs(output_raw_dir, exist_ok=True)

    # Iterate through all .vti files in the input directory
    for file_name in os.listdir(input_path):
        if file_name.endswith(".vti"):
            # Full path to the input file
            file_path = os.path.join(input_path, file_name)
            print(f"Processing: {file_path}")

            # Load the .vti file using ParaView
            data = OpenDataFile(file_path)
            UpdatePipeline()

            # Fetch the data and metadata
            data_array = servermanager.Fetch(data)
            dims = data_array.GetDimensions()  # (X, Y, Z)
            vtk_data = data_array.GetPointData().GetScalars()  # Assuming scalar data
            print(f"Data Type in VTK: {vtk_data.GetDataTypeAsString()}")
            print(f"Original Dimensions: {dims}")

            # Convert VTK data to a NumPy array
            np_data = numpy_support.vtk_to_numpy(vtk_data)

            # Reshape to ParaView's expected raw format (Z, Y, X)
            np_data = np_data.reshape(dims[::-1], order='C')  # Reverse dimensions for proper alignment

            # Debugging: Check data range and shape
            print(f"Data Range: {np.min(np_data)} to {np.max(np_data)}")
            print(f"Reshaped Dimensions: {np_data.shape}")

            # Expand each slice along the z-axis by repeating it `slice_height` times
            expanded_data = np.repeat(np_data, slice_height, axis=0)

            # Debugging: Check new shape
            print(f"Expanded Dimensions: {expanded_data.shape}")

            # Generate the output file paths
            output_npy_file_name = file_name.replace(".vti", ".npy")
            output_npy_file_path = os.path.join(output_npy_dir, output_npy_file_name)

            output_raw_file_name = file_name.replace(".vti", ".raw")
            output_raw_file_path = os.path.join(output_raw_dir, output_raw_file_name)

            # Save the expanded data as a .npy file
            np.save(output_npy_file_path, expanded_data)
            print(f"Saved: {output_npy_file_path}")

            # Save the expanded data as a .raw file
            with open(output_raw_file_path, 'wb') as f:
                f.write(expanded_data.tobytes())
            print(f"Saved: {output_raw_file_path}")

# Example usage
input_dir = "/home/kuangshiai/Documents/Datasets/FLARE22_LabeledCase50/images_vti"  # Replace with your input directory
output_npy_dir = "/home/kuangshiai/Documents/Datasets/FLARE22_LabeledCase50/images_npy"  # Replace with your .npy output directory
output_raw_dir = "/home/kuangshiai/Documents/Datasets/FLARE22_LabeledCase50/images_raw"  # Replace with your .raw output directory
slice_height = 5  # Replace with the desired slice height

process_vti_to_npy_and_raw(input_dir, output_npy_dir, output_raw_dir, slice_height)
