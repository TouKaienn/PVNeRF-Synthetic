import os
import nibabel as nib
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np
np.bool = np.bool_

def convert_nii_to_vti(input_dir, output_dir):
    """
    Convert all .nii.gz files in the input directory to .vti format and save them to the output directory.

    Args:
        input_dir (str): Path to the directory containing .nii.gz files.
        output_dir (str): Path to the directory where .vti files will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".nii.gz"):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_name = file_name.replace(".nii.gz", ".vti")
            output_file_path = os.path.join(output_dir, output_file_name)

            # Load the .nii.gz file
            print(f"Processing: {input_file_path}")
            img = nib.load(input_file_path)
            data = img.get_fdata()

            # Convert to VTK format
            vtk_image = vtk.vtkImageData()
            vtk_image.SetDimensions(data.shape)
            vtk_image.SetOrigin(0, 0, 0)
            vtk_image.SetSpacing(1.0, 1.0, 5.0)

            vtk_data = numpy_to_vtk(data.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT)
            vtk_image.GetPointData().SetScalars(vtk_data)

            # Write to .vti file
            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(output_file_path)
            writer.SetInputData(vtk_image)
            writer.Write()

            print(f"Saved: {output_file_path}")

# Example usage
input_path = "/home/kuangshiai/Documents/Datasets/FLARE22_LabeledCase50/labels"  # Replace with your input directory path
output_path = "/home/kuangshiai/Documents/Datasets/FLARE22_LabeledCase50/labels_vti"  # Replace with your output directory path
convert_nii_to_vti(input_path, output_path)