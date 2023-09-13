from datasets import Front3D
from pathlib import Path
from PIL import Image 
import numpy as np
from utils import geometry 
import open3d as o3d

PATH_TO_DEPTHS = '/mnt/hdd/tmp/outputs/depth/'
PATH_TO_DEPTHS = Path(PATH_TO_DEPTHS)

#for item, depth_path in zip(dataset, sorted(PATH_TO_DEPTHS.glob('*.png'))):
    #model = depth_model(dataset.output_size).float().cuda()


val_dataset = Front3D(split='val', file_list_path='/home/rashed/repos/panoptic-reconstruction/resources/front3d/validation_list_3d.txt', num_samples = -1)

train_dataset = Front3D(split='train', file_list_path='/home/rashed/repos/panoptic-reconstruction/resources/front3d/train_list_3d.txt', num_samples = -1, shuffle=True)

#Confirm backprojection quality
i = 0

pc_ = o3d.geometry.PointCloud()
for item in val_dataset:
    gt = item['depth_gt'].numpy()
    K = item['intrinsic']
    #pc = geometry.dpt_2_pcld(np.swapaxes(gt, 1,2),1.0, K)
    pc = geometry.dpt_2_pcld(gt,1.0, K)

    color = item[('color',0,0)]#.float()#.cuda()
    scene_id = item["scene_id"]
    if i == 4: 
        scene__ = scene_id
    elif i<4:
        i +=1
        continue
    if scene__!=scene_id: break
    T_cam2world = item['camera2world']#['blender_matrix']#['camera2world']
    w2c = np.linalg.inv(T_cam2world)
    pc_flatten = pc.reshape(-1,3)@ w2c[:3,:3]- w2c[:3,:3].T@w2c[:3,-1]#@ np.array([[1,0,0],[0,0,-1],[0,-1,0]]).T# @ w2c[:3,:3]#.T + w2c[:3,-1]#@ np.array([[-1,0,0],[0,0,-1],[0,1,0]]).T #@T_cam2world[:3,:3].T  #+ T_cam2world[:3,-1]#@ np.array([[1,0,0],[0,0,-1],[0,-1,0]]).T #@ np.array([
    pc_flatten = pc_flatten @ np.array([[1,0,0],[0,0,1],[0,1,0]]).T
    #[1,0,0],
    #[0,0,1],
    #[0,1,0]
    #]).T @T_cam2world[:3,:3].T  + T_cam2world[:3,-1]
    pc_export = o3d.geometry.PointCloud()
    pc_export.points = o3d.utility.Vector3dVector(pc_flatten) 

    scene_mesh = geometry.create_pcd('/mnt/hdd/3D-FRONT/'+scene_id+'.json')
    pt = np.array(scene_mesh.points)
    pt = pt  #@ T_cam2world[:3,:3].T  #+ T_cam2world[:3,-1]
    #pt = pt @ np.array([[1,0,0],[0,0,1],[0,-1,0]])
    scene_mesh.points = o3d.utility.Vector3dVector(pt)

    pc_ = pc_export + pc_
    o3d.io.write_point_cloud('trial.ply',pc_)
    o3d.io.write_point_cloud('mesh.ply', scene_mesh)
    #if i ==3: break
    i += 1  

