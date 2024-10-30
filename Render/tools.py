import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from math import sin, cos, pi, sqrt, atan2
import numpy as np

import torch
import re

import json
from CameraHelper import *
import shutil
# from icecream import ic
# exit()
def camera_rotate_along_axis(axis,offset_theta=0,offset_phi=0):
    assert axis in ['x', 'y', 'z']
    view_theta_phi = []
    if axis == 'x':
        theta = np.array([0])+offset_theta
        phi = np.linspace(-180, 180, 182)
    elif axis == 'y':
        raise ValueError('y axis has problem')
        theta = np.array([0])+offset_theta
        phi = np.linspace(-180, 180, 182//2)
    elif axis == 'z': 
        raise ValueError('z axis has problem')
        theta = np.linspace(-90, 90, 182//2) + np.linspace(90, -90, 182//2) # concat 
        phi = np.array([-180,180])+offset_phi
    theta,phi = np.meshgrid(theta, phi, indexing='ij')
    view_theta_phi = np.stack((theta.flatten(), phi.flatten()), axis=-1)[:-1].tolist()
    # view_theta_phi = np.stack((phi.flatten(), theta.flatten()), axis=-1)[:-1].tolist()
    return view_theta_phi



def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_cam2world_matrix(forward_vector, origin, up_vector=None):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    Modified by Kaiyuan, it is now support customized up_vector
    """
    
    forward_vector = normalize_vecs(forward_vector)
    if up_vector is None:
        up_vector = torch.FloatTensor([0, 1, 0]).expand_as(forward_vector)
    else:
        up_vector = normalize_vecs(up_vector).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


def xyz2ThetaPhi(xyz):
    x, y, z = xyz
    theta = atan2(-y, -x)
    dxy = sqrt(x**2 + y**2)
    phi = atan2(z, dxy)
    theta = theta / pi * 180
    phi = phi / pi * 180
    return theta, phi

def ThetaPhi2xyz(theta, phi):
    theta_rad = (theta + 180) / 180 * math.pi
    phi_rad = phi / 180 * math.pi
    x = cos(theta_rad)*cos(phi_rad)
    y = sin(theta_rad)*cos(phi_rad)
    z = sin(phi_rad)
    return [x, y, z]

def findXMLReaderID(stateFilePath,search_item="FileNames"):
    # when LoadState(), it need the reader ID to change the FileNames. This is based on re module
    # Note that Python has built-in module xml for parsing, but I implemented this using simple way
    all_patterns ={
        "FileNames":r"<Proxy group=\"sources\" type=\"ImageReader\" id=\"(\d+)\"",
        "light_angle":r"<Property name=\"light_angle\" id=\"(\d+)"
    }
    pattern = all_patterns[search_item]
    print(pattern)
    with open(stateFilePath, 'r') as f:
        for line in f.readlines():
            match = re.search(pattern, line)
            if match:
                return int(match.group(1))
    print(f'findXMLReaderID: Did not find the reader ID for "{search_item}" in state file based on the given pattern, replace a new pattern or implement with xml module.')
    
if __name__ == "__main__":
    from icecream import ic
    stateFilePath = "/home/dullpigeon/Softwares/PythonToolBox/ParaviewState/pyNvidiaIndex/vortex_orbital.pvsm"
    ic(findXMLReaderID(stateFilePath,"light_angle"))