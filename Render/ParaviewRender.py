from paraview.simple import * #type:ignore
import paraview.web.venv #type:ignore
paraview.simple._DisableFirstRenderCameraReset()

#--include customized utils--#
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .filesUtils import ensure_dirs, delFilesInDir, json_loader 
from CameraHelper.camera_utils import get_ray_directions
from CameraHelper.icosphere import icosphere
from CameraHelper.fibonacci import fibonacci_sphere
from .tools import xyz2ThetaPhi, ThetaPhi2xyz, findXMLReaderID, create_cam2world_matrix,normalize_vecs#, sample_theta_phi_icosphere
#--include other libs--#
import math
import torch
import numpy as np
from icecream import ic
import json
import shutil
from pyquaternion import Quaternion

NVPLUGIN_PATH = os.path.abspath("./resources/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/lib/paraview-5.11/plugins/pvNVIDIAIndeX/pvNVIDIAIndeX.so") # has to be absolute path
DATACONFIG_PATH = './ConfigFiles/DataConfig.json'
CUSTOM_CAM_PATH = './ConfigFiles/CustomCam.json'
LoadPlugin(NVPLUGIN_PATH,False,globals()) #type:ignore

class ParaviewRender():
    def __init__(self, args):
        dataset_configs = json_loader(DATACONFIG_PATH)
        self.dataset = dataset_configs[args.d]
        # self.outImgDir = args.outImgDir
        self.isTimeEval = args.timeEval
        self.mode = args.mode
        self.nu = args.nu if self.mode in ['train', 'test'] else 2 if self.mode=='val' else ValueError(f"mode {self.mode} not valid")  # nu=2 for valid mode
        self.imgH, self.imgW = args.H, args.W
        self.channel = args.c # RGBA or RGB
        self.lightType = args.l # Headlight or Orbital
        self.useAbsPath = args.absPath
        
        self.state_loaded = False
        
        #* Paraview Rendering Settings
        self.volDim = self.dataset['dim']
        self.radius = self.dataset['r']
        
        #* Default Rendering Settings, Should not change the following variables easily when rendering
        self.__stateFilePath = self.dataset['state_path']
        self.__dataPath = self.dataset['data_path']
        self.__sampled_timesteps = [i+self.dataset['timestep_offset'] for i in range(1, self.dataset['total_timesteps']+1, 1)]
    
    @property
    def get_camera(self):
        return self.cam_position
    
    @property
    def get_light(self):
        return self.light_theta_phi

    @property
    def get_stateFilePath(self):
        return self.__stateFilePath
    
    @property
    def get_dataPath(self):
        return self.__dataPath
    
    @property
    def get_sampled_timesteps(self):
        return self.__sampled_timesteps
    
    def set_rootOutDir(self, outImgDir):
        self.outImgDir = outImgDir
        ensure_dirs(os.path.join(self.outImgDir, f"{self.mode}"))
        delFilesInDir(os.path.join(self.outImgDir, f"{self.mode}"))
    
    def set_camera(self, cam_position):
        self.cam_position = cam_position
    
    def set_light(self, light_theta_phi):
        self.light_theta_phi = light_theta_phi
    
    def set_TFsColor(self, RGBPoints):
        #* we keep array format for RGBPoints as it is easier to read when output in terminal
        if self.state_loaded:
            ColorTransferFunction = GetColorTransferFunction('ImageFile') # type: ignore
            ColorTransferFunction.RGBPoints = RGBPoints#.flatten().tolist()
        else:
            raise AttributeError("State not loaded, please load state first by calling create_paraview_render_context")
    
    def set_TFsOpacity(self, OpacityPoints):
        #* we keep array format for OpacityPoints as it is easier to read when output in terminal
        if self.state_loaded:
            OpacityTransferFunction = GetOpacityTransferFunction('ImageFile') # type: ignore
            OpacityTransferFunction.Points = OpacityPoints#.flatten().tolist()
        else:
            raise AttributeError("State not loaded, please load state first by calling create_paraview_render_context")
        
    def setup(self, rootOutDir):
        #*Setup the paraview render variables or states
        self.set_rootOutDir(rootOutDir)
        
        self.lookat_point = [0,0,0]
        self.camera_pivot = torch.tensor(self.lookat_point)
        self.focal_length = (1/2) / math.tan(15/180*math.pi) # focal_length = sensor_size / tan(fov/2) # default fov = 30
        self.intrinsic_matrix = torch.tensor([[self.focal_length, 0, 0.5],
                                              [0, self.focal_length, 0.5], 
                                              [0, 0, 1]])
        ray_dirs = get_ray_directions(1, 1, [self.focal_length, self.focal_length])
        self.ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        self.focal_point = [v/2 for v in self.dataset['dim']] if 'focal_point' not in self.dataset else self.dataset['focal_point']
        self.swap_YZrow = torch.FloatTensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        self.mask = torch.FloatTensor([[-1,1,1,-1],[1,-1,-1,1],[1,-1,-1,1],[1,1,1,1]])
        self.cam_position = [0, 0]
        self.light_theta_phi = [0, 0]
        # self.blender2opencv = torch.FloatTensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    def load_camera_views(self, method='circle'):
        if method == 'circle': 
            view_theta_phi = self.circle_sphere_sample()
        elif method == 'icosphere':
            view_theta_phi = self.icosphere_sphere_sample(self.nu)
        else:
            raise ValueError(f"method args {method} not valid")
        return view_theta_phi
                
    def load_light_views(self, method='circle'):
        if method == 'circle':
            light_theta_phi = self.circle_sphere_sample()
        elif method == 'icosphere':
            light_theta_phi = self.icosphere_sphere_sample(self.nu)
        else:
            raise ValueError(f"method args {method} not valid")
        return light_theta_phi
    
    def create_paraview_render_context(self, t=-1):
        #* Render create_paraview_render_context at timestep t
        # Load state
        try:
            ReaderID = findXMLReaderID(self.__stateFilePath)
            if ".raw" in self.__dataPath:
                ic("Data come from PublicSciVis dir, use data_path directly.")
                LoadState(self.__stateFilePath, # type: ignore
                    filenames=[{"id": ReaderID, "FileNames": f"{self.__dataPath}"}])
            else:
                LoadState(self.__stateFilePath, # type: ignore
                        filenames=[{"id": ReaderID, "FileNames": f"{self.__dataPath}{t:04d}.raw"}]) 
        except ValueError:
            print(f'LoadState failed, check:\n (1) "id" in state file is {ReaderID} \n (2) "FileNames" in state file is {self.dataset["data_path"]}')
            exit()
        self.state_loaded = True
        renderView = FindViewOrCreate('RenderView1', viewtype='RenderView') # type: ignore
        SetActiveView(renderView) # type: ignore
        layout = GetLayout() # type: ignore
        
        displayPropertyProxy = next(iter(GetSources().values())) # type: ignore # we assume there is only one object in the scene
        displayProperties = GetDisplayProperties(displayPropertyProxy, view=renderView) # type: ignore
        renderContext={'renderView':renderView, 'layout':layout, 'displayProperties':displayProperties}
        return renderContext
    
    
    def render_view(self, saveImgName, renderView, layout, displayProperties, cam_pos=None, cam_up_vec=None):
        #* Render the view with camera and light theta phi
        layout.SetSize(self.imgW, self.imgH)
        if cam_pos is None:
            if len(self.cam_position) == 2:
                cam_theta, cam_phi = self.cam_position
                sphere_x, sphere_y, sphere_z = ThetaPhi2xyz(cam_theta, cam_phi)
                cam_x = sphere_x * self.radius + self.focal_point[0]
                cam_y = sphere_y * self.radius + self.focal_point[1]
                cam_z = sphere_z * self.radius + self.focal_point[2]
            else:
                cam_x, cam_y, cam_z = self.cam_position
        else:
            cam_x, cam_y, cam_z = cam_pos
        
        print("focal_point: ",self.focal_point)
        print("cx,cy,cz: ",cam_x,cam_y,cam_z)
        
        renderView.CameraPosition = [cam_x,cam_y,cam_z]
        renderView.CameraFocalPoint = self.focal_point
        
        cam_xyzPos = np.array([cam_x,cam_y,cam_z])
        objcenter_xyzPos = np.array(self.focal_point)
        cam2obj_vec = objcenter_xyzPos - cam_xyzPos
        cam2obj_vec = cam2obj_vec / math.sqrt(cam2obj_vec[0]**2 + cam2obj_vec[1]**2 + cam2obj_vec[2]**2)
        
        if cam_up_vec is None:
            up_vec = np.array([0,0,1]) # we use z-up
            cam_right_vec = np.cross(cam2obj_vec, up_vec)
            cam_up_vec = np.cross(cam_right_vec, cam2obj_vec)
        
        light_theta, light_phi = self.light_theta_phi
        
        # Please comment the following lines if you want to render isosurfaces
        displayProperties.light_type = self.lightType
        displayProperties.light_angle = light_theta#+180
        displayProperties.light_elevation = light_phi
        
        renderView.CameraViewUp = cam_up_vec.tolist() if not isinstance(cam_up_vec, list) else cam_up_vec
        renderView.CameraParallelScale = 109.9852262806237
        saveImgPath = os.path.join(self.outImgDir, f'{self.mode}', f"{saveImgName}")
        
        if self.isTimeEval:
            RenderAllViews()#type:ignore
        else:
            if self.channel == 'RGB':
                SaveScreenshot(saveImgPath+'.png', renderView, ImageResolution=[self.imgW, self.imgH]) #type:ignore
            elif self.channel == 'RGBA':
                SaveScreenshot(saveImgPath+'.png', renderView, ImageResolution=[self.imgW, self.imgH], TransparentBackground=1)#type:ignore

        camera_origins = torch.FloatTensor([sphere_x,sphere_y,sphere_z]).view(1,3) * 4.0311
        up_vec = torch.FloatTensor([0,0,1]) if cam_up_vec is None else torch.FloatTensor(cam_up_vec) #* if you use customize view, need also to change this
        c2w = create_cam2world_matrix(normalize_vecs(self.camera_pivot + 1e-8 - camera_origins), camera_origins, up_vector=up_vec)[0]
        c2w = self.swap_YZrow @ c2w
        c2w = c2w * self.mask
        c2w += 1e-8
        
        frame_info = {}
        frame_info['file_path'] = f"./{self.mode}/{saveImgName}" if not self.useAbsPath else saveImgPath
        #* The theta and phi in this script is different from common settings, so a transform step is necessary for compatibility
        #* Theta: -180, 180. start rotation from minux x-axis  Theta_common: 0, 360.  rotation start from positive x-axis
        #* Phi: -90, 90. start rotation from x-y plane  Phi_common: 0, 180.  rotation start from positive z-axis toward x-y plane
        frame_info['rotation'] = 0.0
        if self.lightType == 'Orbital':
            # Theta_common = - light_theta + 180
            # Phi_common = 90 - light_phi
            # frame_info['light_angle'] = [Theta_common, Phi_common]
            frame_info['light_angle'] = [light_theta, light_phi]
        frame_info['transform_matrix'] = c2w.tolist()

        return frame_info
    

    def save_transformJsonFile(self, json_data):
        json_obj = json.dumps(json_data, indent=4)
        saveJsonPath = os.path.join(self.outImgDir, f'transforms_{self.mode}.json')
        with open(saveJsonPath, 'w') as f:
            f.write(json_obj)
        if self.mode == 'train':
            evalJsonPath = os.path.join(self.outImgDir, f'transforms_val.json')
            shutil.copy(saveJsonPath, evalJsonPath)
    
    @staticmethod
    def icosphere_sphere_sample(nu):
        #* Sample views on sphere with icosphere
        # if (self.view_theta_phi != []) or (self.light_theta_phi != []):
        #     raise ValueError('The view_theta_phi or light_theta_phi is not empty, please use setup() to reset the render environment')
        views_theta_phi = []
        camera_xyzPos, _ = icosphere(nu)
        for xyz in camera_xyzPos:
            if xyz[0]==0 and xyz[1] in [1,-1] and xyz[2] == 0:
                xyz[2] = 0.0001
            theta, phi = xyz2ThetaPhi(xyz)
            if phi == 90:
                phi = 89.99
            if phi == -90:
                phi = -89.99
            views_theta_phi.append([theta, phi])
        return views_theta_phi

    @staticmethod
    def circle_sphere_sample():
        views_theta_phi = []
        for phi in np.arange(-90,90.5,1):
            theta = phi*2
            if phi == 90:
                phi = 89.99
            if phi == -90:
                phi = -89.99
            views_theta_phi.append([theta, phi])
        return views_theta_phi
    
    @staticmethod
    def fibonacci_sphere_sample(num_points):
        views_theta_phi = []
        camera_xyzPos, _ = fibonacci_sphere(num_points)
        for xyz in camera_xyzPos:
            if xyz[0]==0 and xyz[1] in [1,-1] and xyz[2] == 0:
                xyz[2] = 0.0001
            theta, phi = xyz2ThetaPhi(xyz)
            if phi == 90:
                phi = 89.99
            if phi == -90:
                phi = -89.99
            views_theta_phi.append([theta, phi])
        return views_theta_phi
    
    @staticmethod
    def get_custom_cameras():
        custom_cams = json_loader(CUSTOM_CAM_PATH)
        return custom_cams


def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c + a*d), 2*(b*d - a*c)],
                     [2*(b*c - a*d), a*a + c*c - b*b - d*d, 2*(c*d + a*b)],
                     [2*(b*d + a*c), 2*(c*d - a*b), a*a + d*d - b*b - c*c]])

def rotate_camera(theta, phi, axis, angle):
    camera_pos = ThetaPhi2xyz(theta, phi)
    rot_matrix = rotation_matrix(axis, angle)
    rotated_pos = np.dot(rot_matrix, camera_pos)
    new_theta, new_phi = xyz2ThetaPhi(rotated_pos)
    return new_theta, new_phi