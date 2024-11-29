import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import argparse
import math
import time
import numpy as np
from Render.ParaviewRender import ParaviewRender
from Render.TFutils import getTFs, write_TF_json
from Render.filesUtils import ensure_dirs #type: ignore
from icecream import ic
# from Render.tools import write_palette_txt

#* Note: Paraview TFs json file format:
#* GetOpacityTransferFunction("ImageFile").Points:  get the opacity term of transfer function
#* Points:  a flattened list of (data-value, opacity, mid-point, sharpness), mid-point usually set to 0.5, sharpness usually set to 0.0

#* GetColorTransferFunction("ImageFile").RGBPoints:  get the color term of transfer function
#* RGBPoints: The value is a flattened list of tuples, (data-value, red, green, blue). The color components, must be in the range [0.0, 1.0]

#* these two attributes can also be set manually with an array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Volume Render in ParaView for one volume dataset and output json file contain camera pose\n',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('d', type=str, help='the dataset name of datasets variable in dataset config file.')
    parser.add_argument('outImgDir', type=str, help='The Dir path for rendered images')
    parser.add_argument('noTransform', action='store_false', help='Do not output the transform json file that contain camera pose')
    parser.add_argument('-m','--mode', type=str, default='train', choices=['train', 'test', 'val', 'custom'], help='Train mode, test mode (circle render path), valid mode (nu=2)')
    parser.add_argument('--bins', type=int, default=10, help='number of bins')
    parser.add_argument('--nu', type=int, default=2, help='(Only work when test not toggled) The subdivision frequency of icosphere that decide the number of multi-view images, larger than 1 to make a change,nu->vertices: 2->42, 3->92, 4->162, 5->252')
    parser.add_argument('--H', type=int, default=800, help='The height of rendered image')
    parser.add_argument('--W', type=int, default=800, help='The width of rendered image')
    parser.add_argument('--c', type=str, default='RGBA', help='RGBA or RGB')
    parser.add_argument('--l', type=str, default='Headlight', help='Types of light, Headlight or Orbital')
    parser.add_argument('--absPath', action='store_true', help='Use absolute path for data path')
    parser.add_argument('--timeEval', action='store_true', help='Time evaluation, not saving images')
    
    args = parser.parse_args()

    #* create root dir for all TFs
    ensure_dirs(args.outImgDir)

    render = ParaviewRender(args)

    sphere_sample_method = 'icosphere' if args.mode in ['train', 'val'] else 'circle'
    all_cam_theta_phi = render.load_camera_views(method=sphere_sample_method)
    all_light_theta_phi = render.load_light_views(method=sphere_sample_method)
    
    timesteps = render.get_sampled_timesteps

    
    for t in timesteps: #* here we render all timesteps
        renderContext = render.create_paraview_render_context(t=t)
        volPath = render.get_dataPath if '.raw' in render.get_dataPath else f'{render.get_dataPath}{t:04d}.raw'
        allTFs = getTFs(volPath,args.bins)
        
        for TF_count, TF in enumerate(allTFs):
            #* setup RootImgDir for each TF
            TFRootDir = args.outImgDir + f'/TF{TF_count+1:02d}'
            render.setup(TFRootDir)
            write_TF_json(TF, f'{TFRootDir}/TF{TF_count+1:02d}.json', TF_name=f'{args.d}_TF{TF_count+1:02d}')
            
            json_data = {'camera_angle_x': math.pi * 30 / 180, 'frames': []}
            render.set_TFsColor(TF['RGBPoints'])
            render.set_TFsOpacity(TF['Points'])
            img_idx = 0
            
            #* render each view
            for idx, (cam_theta_phi, light_theta_phi) in enumerate(zip(all_cam_theta_phi, all_light_theta_phi)):
                render.set_camera(cam_theta_phi)
                render.set_light(light_theta_phi)
                saveImgName = f'r_{img_idx:04d}'

                frame_info = render.render_view(saveImgName,**renderContext)
                
                json_data['frames'].append(frame_info)
                img_idx += 1
            
            if not args.noTransform:
                render.save_transformJsonFile(json_data)