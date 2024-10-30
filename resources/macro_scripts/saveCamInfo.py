# trace generated using paraview version 5.11.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
import json
import os
#### disable automatic camera reset on 'Show'
SAVE_JSON_PATH = "/path/to/save/camera_info.json"


def json_loader(file_path):
    content = None
    with open(file_path,'r') as f:
        content = json.load(f)
    return content

paraview.simple._DisableFirstRenderCameraReset()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

#-----------------------------------
# saving camera placements for views
cam = GetActiveCamera()
camPosition = cam.GetPosition()
camViewUp = cam.GetViewUp()
if os.path.exists(SAVE_JSON_PATH):
    content = json_loader(SAVE_JSON_PATH)
    view_values = {"cam_pos": camPosition, "cam_up_vec": camViewUp}
    content[f"cam{len(content.keys())+1:04d}"]=view_values
    
    
else:
    content = {"cam0001":{"cam_pos": camPosition, "cam_up_vec": camViewUp}}
    
with open(SAVE_JSON_PATH, "w") as f:
    json.dump(content, f, indent=4)

print(f"Camera position and view up vector saved to {SAVE_JSON_PATH}")
