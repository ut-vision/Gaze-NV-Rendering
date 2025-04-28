
import numpy as np
import torch
# from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
from pytorch3d.renderer import (
	PerspectiveCameras, 
	# FoVPerspectiveCameras,
	# PointLights, 
	DirectionalLights, 
	# Materials, 
	# RasterizationSettings, 
	MeshRenderer, 
	MeshRasterizer,  
	# SoftPhongShader,
	# TexturesUV,
	# TexturesVertex,
)


from lib.label_transform import rotation_matrix
from utils.pytorch3d_helper import SimpleShader, hard_rgb_blend_with_background, BlendParams, mySoftPhongShader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ======================================== Setting rendering camera parameters ==================================================
# Normalization parameters
focal_norm = 960 # focal length of normalized camera
distance_norm = 300 # normalized distance between eye and camera
roi_size = (448,448) # size of cropped eye image

camera_R = rotation_matrix(0,np.pi,0) # The camera should face to the -y axis in OpenGL coordinate
cameras = PerspectiveCameras(
	focal_length=((960.0, 960.0),),  # (fx_screen, fy_screen)
	principal_point=((roi_size[0]/2., roi_size[1]/2.),),  # (px_screen, py_screen)
	image_size=((roi_size[0], roi_size[1]),),  # (imwidth, imheight)
	R=[camera_R],
	T=[np.zeros(3,)],
	device=device
)


# set the background color
blend_params = BlendParams()
renderer1 = MeshRenderer(
	rasterizer=MeshRasterizer(
		cameras=cameras, 
		raster_settings=RasterizationSettings(image_size=roi_size[0], blur_radius=0.0, faces_per_pixel=1,)
	),
	shader=SimpleShader(
			device=device,
			blend_params=blend_params
		)
)
renderer2 = MeshRenderer(
	rasterizer=MeshRasterizer(
		cameras=cameras, 
		raster_settings=RasterizationSettings(image_size=roi_size[0], blur_radius=0.0, faces_per_pixel=1,)
	),
	shader=mySoftPhongShader(
		device=device,
		cameras=cameras,
		lights=DirectionalLights(device=device, direction=[[0.0, 0.0, 1.0]]), 
		blend_params=blend_params
		)
)

def run_render(mesh, random_color, bg_img):
    """ render 
    Args:
        mesh: Meshes
        random_color: (3,) tensor
        bg_img: (H,W,3) tensor
    Returns:
        images0: background is black
        images: background is random color
        images_bg: background is bg_img
        images_dark: same as images0 but darker
        images_cl_dark: same as images but darker
        images_bg_dark: same as images_bg but darker
    """
    # ------------------ 1. for black background output---------------------------
    images0 = renderer1(mesh,zfar=2000)
    images0 = images0.cpu().data.numpy()[:,:,:,:3]  # batch * H * W * 3

    # ------------------ 2. for random color output--------------------
    renderer1.shader.blend_params =  BlendParams(background_color=random_color)
    images = renderer1(mesh,zfar=2000)
    images = images.cpu().data.numpy()[:,:,:,:3]  # batch * H * W * 3

    # ------------------ 3. for background image output---------------------------
    renderer1.shader.blend_params =  BlendParams(background_image=bg_img)
    images_bg = renderer1(mesh,zfar=2000)
    images_bg = images_bg.cpu().data.numpy()[:,:,:,:3]  # batch * H * W * 3
    # ------------------ 4. for darker output-------------------------------------
    ac = 0.25 + 0.5 * np.random.rand()
    dc = 0.0
    sc = 0
    light = DirectionalLights(
        device=device, 
        ambient_color=((ac, ac, ac), ), 
        diffuse_color=((dc, dc, dc), ), 
        specular_color=((sc, sc, sc),), 
        direction=[[0.0, 0.0,1.0]]
        )
    renderer2.shader.blend_params =  BlendParams() # background_color=random_color, background_image=bg_img)
    renderer2.shader.lights= light
    images_dark = renderer2(mesh,zfar=2000)
    images_dark = images_dark.cpu().data.numpy()[:,:,:,:3]  # batch * H * W * 3
    
    # ------------------ 5. for darker color output-------------------------------------
    renderer2.shader.blend_params =  BlendParams( background_color=random_color)
    renderer2.shader.lights= light
    images_cl_dark = renderer2(mesh,zfar=2000)
    images_cl_dark = images_cl_dark.cpu().data.numpy()[:,:,:,:3]  # batch * H * W * 3

    # ------------------ 6. for darker background image output-------------------------------------
    renderer2.shader.blend_params =  BlendParams( background_image=bg_img)
    renderer2.shader.lights= light
    images_bg_dark = renderer2(mesh,zfar=2000)
    images_bg_dark = images_bg_dark.cpu().data.numpy()[:,:,:,:3]  # batch * H * W * 3
    # masked		


    return images0,images,images_bg, images_dark,images_cl_dark,images_bg_dark
    # return { "image0":images0, 
    #         "image":images, 
    #         "image_bg":images_bg, 
    #         "image_dark":images_dark, 
    #         "image_cl_dark":images_cl_dark, 
    #         "image_bg_dark":images_bg_dark, 
    #         "image_masked":images_masked}