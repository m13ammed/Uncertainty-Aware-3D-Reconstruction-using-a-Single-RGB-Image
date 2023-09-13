import blenderproc as bproc
import bpy
from blenderproc.python.utility.Initializer import _Initializer

import argparse
import os
import numpy as np
import debugpy
from blenderproc.python.renderer import RendererUtility
import random
#debugpy.listen(5678)
#debugpy.wait_for_client()
room_id = "6fc83660-65e7-4380-9d0e-f74a2c7ecec7"
def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed", "cabinet",  "unit", "light", "lamp", "stool", "nightstand", "desk", "wardrobe"]:  
        if category_name in name.lower():
            return True
    return False

def load_and_filter_file_list(file_list_path: os.PathLike):
    with open(file_list_path) as f:
        content = f.readlines()

    images = [line.strip() for line in content]
    scenes = {}
    for image in images:
      room_id = image.split("/")[0]
      image_id = image.split("/")[1]
      if room_id not in scenes.keys():
          scenes.update({
              room_id:[image_id]
          })
      else:
          scenes[room_id].append(image_id)
    return scenes
parser = argparse.ArgumentParser()
parser.add_argument("--front", help="Path to the 3D front file", default='/mnt/hdd/3D-FRONT/6fc83660-65e7-4380-9d0e-f74a2c7ecec7.json')
parser.add_argument("--future_folder", help="Path to the 3D Future Model folder.", default='/mnt/hdd/3D-FUTURE-model/')
parser.add_argument("--front_3D_texture_path", help="Path to the 3D FRONT texture folder.", default='/mnt/hdd/3D-FUTURE-model/')
parser.add_argument("--output_dir", help="Path to where the data should be saved", default='/mnt/hdd/BOP_new2')
args = parser.parse_args()
samples =  load_and_filter_file_list('/home/rashed/repos/panoptic-reconstruction/resources/front3d/train_list_3d.txt')
bproc.init()
#bproc.renderer.enable_depth_output(activate_antialiasing= False)
bproc.renderer.set_output_format('PNG')
bproc.renderer.enable_depth_output(activate_antialiasing= False)
RendererUtility.set_render_devices(desired_gpu_device_type="OPTIX")
start=False
total = 0
proximity_checks = {"min": 1.0,  "no_background": True}
while total <10000:
#for j, room_id in enumerate(samples.keys()):
  room_id = random.choice(list(samples.keys()))
  #if room_id !='ff507d6c-aab0-476e-9f58-1f765cb8956e' and not start: continue

    #RendererUtility.set_render_devices()

    # Set default parameters
  _Initializer.set_default_parameters()
  RendererUtility.set_render_devices(desired_gpu_device_type="OPTIX")
  RendererUtility.set_max_amount_of_samples(256)
  front = os.path.join('/mnt/hdd/3D-FRONT/',room_id+'.json')
  if not os.path.exists(front) or not os.path.exists(args.future_folder):
      raise Exception("One of the two folders does not exist!")

  mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
  mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

  # set the light bounces
  #bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
  #                                  transmission_bounces=200, transparent_max_bounces=200)
  #bproc.renderer.set_light_bounces(diffuse_bounces=1, glossy_bounces=1, max_bounces=1,
  #                                  transmission_bounces=1, transparent_max_bounces=1)

  # load the front 3D objects
  try:
    loaded_objects, furniture = bproc.loader.load_front3d(
        json_path=front,
        future_model_path=args.future_folder,
        front_3D_texture_path=args.front_3D_texture_path,
        label_mapping=mapping
    )
  except:
    bproc.clean_up(clean_up_camera=True)
    continue


  #vertices = np.empty((len(loaded_objects[100].blender_obj.to_mesh().vertices), 3), 'f')
  #loaded_objects[100].blender_obj.to_mesh().vertices.foreach_get("co", np.reshape(vertices, len(loaded_objects[100].blender_obj.to_mesh().vertices) * 3))
  #mesh.loops.foreach_set("vertex_index", faces), mesh.vertices.foreach_set("normal", normal)

  point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

  # Init bvh tree containing all mesh objects
  #bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

  poses = 0
  tries = 0

  # filter some objects from the loaded objects, which are later used in calculating an interesting score
  special_objects = [obj.get_cp("category_id") for obj in furniture if check_name(obj.get_name())]
  special_objects_ = [obj for obj in furniture if check_name(obj.get_name())]
  #if len (special_objects_) == 0: continue
  bproc.camera.set_intrinsics_from_K_matrix(
  np.array([[277.12811989,   0.        , 160.        ],
        [  0.        , 311.76912635, 120.        ],
        [  0.        ,   0.        ,   1.        ]]),  image_width=320, image_height=240)

  l = len(samples[room_id])
  random.shuffle(samples[room_id])
  added_cam = False
  for im_idx in range(0, min(5,l)):#min(5,l)):
    im = samples[room_id][im_idx]
    pose = np.load(os.path.join('/mnt/hdd/front3d' , room_id , f"campose_{im}.npz"))['blender_matrix']

    #if bproc.camera.scene_coverage_score(pose, special_objects, special_objects_weight=10.0) > 0.5:
    #  bproc.camera.add_camera_pose (pose)
    added_cam = True
    bproc.camera.add_camera_pose(pose)
    total +=1
  del samples[room_id][:min(5,l)]
  if not added_cam: continue


  # Also render normals
  #bproc.renderer.enable_normals_output()
  # render the whole pipeline
  bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance_id"], default_values={'instance_id': 255})
  
  #bproc.postprocessing.dist2depth and bproc.postprocessing.depth2dist
  data = bproc.renderer.render()
  # write the data to a .hdf5 container
  #bproc.writer.write_hdf5(args.output_dir, data)
    #  bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance_id"], default_values={'instance_id': 255})

  bproc.writer.write_bop(args.output_dir,
                            target_objects = special_objects_,
                            dataset = 'front3D_train',
                            depth_scale = 0.1,
                            depths = data["depth"],
                            colors = data["colors"], 
                            color_file_format = "JPEG",
                            ignore_dist_thres = 10,
                            room_id = room_id,
                            instances = data['instance_id_segmaps']
                            )
  bproc.clean_up(clean_up_camera=True)
#import bpy

#bpy.ops.export_scene.obj(filepath="myscene.obj")
'''

poses = [
    np.array([[ 5.82145214e-01,  5.20149206e-06, -8.13084841e-01,
         1.60530758e+00],
       [-8.13084841e-01,  3.71786268e-06, -5.82145214e-01,
         3.78537178e-01],
       [-5.08586595e-09,  1.00000000e+00,  6.39359041e-06,
         7.50000000e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]),
np.array([[ 8.29721153e-01,  3.43806028e-06, -5.58178067e-01,
         2.83762884e+00],
       [-5.58178067e-01,  5.10547625e-06, -8.29721153e-01,
         4.53512976e-03],
       [-2.86648572e-09,  1.00000000e+00,  6.15517183e-06,
         7.50000000e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]),

         np.array([[ 1.21046834e-01,  6.10925463e-06, -9.92646813e-01,
         1.49339676e+00],
       [-9.92646813e-01,  7.50450738e-07, -1.21046834e-01,
         1.16505361e+00],
       [ 5.42658851e-09,  1.00000000e+00,  6.15517183e-06,
         7.50000000e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]] ),
    np.array([[ 4.70671415e-01, -5.53948212e-06,  8.82308602e-01,
            3.05310464e+00],
          [ 8.82308602e-01,  2.94653250e-06, -4.70671415e-01,
            1.71876097e+00],
          [ 7.52489271e-09,  1.00000000e+00,  6.27438112e-06,
            7.50000000e-01],
          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]] )

]


'''
