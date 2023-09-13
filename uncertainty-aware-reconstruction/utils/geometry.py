import numpy as np
import math
import json
import open3d as o3d
import os 
from copy import deepcopy
#import igl 

import torch

def batched_rotation_matrix(axis, angle):
    """
    Compute batched rotation matrices from batched axis-angle representations.
    
    Args:
    axis (torch.Tensor): Batched axis vectors of shape (batch_size, 3).
    angle (torch.Tensor): Batched rotation angles in radians of shape (batch_size,).
    
    Returns:
    torch.Tensor: Batched rotation matrices of shape (batch_size, 3, 3).
    """
    # Ensure that axis is normalized
    axis = axis / torch.norm(axis, dim=1, keepdim=True)
    
    # Expand angle tensor to match the dimensions of the axis tensor
    angle = angle.unsqueeze(1)
    
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    one_minus_cos_theta = 1 - cos_theta
    
    # Compute skew-symmetric matrix for the cross product with each axis
    skew_symmetric = torch.zeros((axis.shape[0],9))
    skew_symmetric[:, 0] = 0
    skew_symmetric[:, 1] = -axis[:, 2]
    skew_symmetric[:, 2] = axis[:, 1]
    skew_symmetric[:, 3] = axis[:, 2]
    skew_symmetric[:, 4] = 0
    skew_symmetric[:, 5] = -axis[:, 0]
    skew_symmetric[:, 6] = -axis[:, 1]
    skew_symmetric[:, 7] = axis[:, 0]
    skew_symmetric[:, 8] = 0
    
    skew_symmetric = skew_symmetric.view(-1, 3, 3)
    
    # Compute rotation matrix using Rodrigues' formula
    rotation_matrix = (
        cos_theta.unsqueeze(1) * torch.eye(3, device=axis.device).unsqueeze(0) +
        one_minus_cos_theta.unsqueeze(1) * axis.unsqueeze(2) @ axis.unsqueeze(1) +
        sin_theta.unsqueeze(1) * skew_symmetric
    )
    
    return rotation_matrix
def dpt_2_pcld(dpt, cam_scale, K):
    if len(dpt.shape) > 2:
        dpt = dpt[0,:, :]
    
    idx = np.indices(dpt.shape[:2])
    xmap = idx[0].astype(float) #+ idx[0].max() / 2
    ymap = idx[1].astype(float) #+ idx[1].max() / 2

    dpt = dpt.astype(np.float32) / cam_scale
    row = (ymap - K[0,2]) * dpt / K[0,0]
    col = (xmap - K[1,2]) * dpt / K[1,1]

    dpt_3d = np.concatenate(
        (row[..., None], col[..., None] ,  dpt[..., None]), axis=2
    )
    return dpt_3d


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

'''
def create_mesh(scene_json_path = '/mnt/hdd/3D-FRONT/66352dcb-04af-421d-b2d9-ec7958de8f7e.json', future_folder = '/mnt/hdd/3D-FUTURE-model/'):
    with open(scene_json_path) as file:
        scene_json = json.load(file)

    structures = ['wall', 'ceil', 'floor', 'other']
    
    vert_dict = {}
    faces_dict = {}
    normals_dict = {}
    meshes_ = {}

    for mesh in scene_json['mesh']:

        name  = mesh['type'].lower()
        idx = -1
        for i, type_ in enumerate(structures[:-1]):
            if type_ in name:
                idx = i
                break
        
        vert = np.asarray(mesh['xyz']).reshape(-1,3)
        faces = np.asarray(mesh['faces']).reshape(-1,3)
        normals = np.asarray(mesh['normal']).reshape(-1,3)
        colors = np.ones_like(vert)*0

        vert = o3d.utility.Vector3dVector(vert)
        faces = o3d.utility.Vector3iVector(faces)
        normals = o3d.utility.Vector3dVector(normals)
        colors = o3d.utility.Vector3dVector(colors)
        mesh_ = o3d.geometry.TriangleMesh(vert, faces)
        mesh_.vertex_normals = normals
        mesh_.vertex_colors = colors
        if structures[idx] not in meshes_.keys():
            meshes_.update({structures[idx]: mesh_})
        else:
            meshes_[structures[idx]] = meshes_[structures[idx]] + mesh_ 

    whole_scene_mesh = None
    for idx in structures:
        if whole_scene_mesh is None:
            whole_scene_mesh = meshes_[idx]
        else:
            whole_scene_mesh = whole_scene_mesh + meshes_[idx]


    furniture_dict = {}

    for furniture in scene_json['furniture']:

        model_path = os.path.join(future_folder, furniture['jid'], 'raw_model.obj')
        if not os.path.exists(model_path): continue
        v, _, n, faces, _, _ = igl.read_obj(model_path)
        
        model = [v, n, faces]
        furniture_dict.update({furniture['uid']: model})


    for room in scene_json['scene']['room']:
        for c in room['children']:

            if c['ref'] in furniture_dict.keys():
                v, n, faces= deepcopy(furniture_dict[c['ref']])

                pos = c['pos']
                rot = c['rot']
                scale = c['scale']
                v = v.astype(np.float64) * scale
                ref = [0,0,1]
                axis = np.cross(ref, rot[1:])
                theta = np.arccos(np.dot(ref, rot[1:]))*2
                if np.sum(axis) != 0 and not math.isnan(theta):
                    R = rotation_matrix(axis, theta)
                    v = np.transpose(v)
                    v = np.matmul(R, v)
                    v = np.transpose(v)
                v = v + pos

                v = o3d.utility.Vector3dVector(v)
                faces = o3d.utility.Vector3iVector(faces)
                #n = o3d.utility.Vector3dVector(n)
                mesh_ = o3d.geometry.TriangleMesh(v, faces)
                #mesh_.vertex_normals = n
                colors = np.ones_like(v)*125
                colors = o3d.utility.Vector3dVector(colors)
                mesh_.vertex_colors = colors
                
                whole_scene_mesh = whole_scene_mesh + mesh_
    return whole_scene_mesh#.transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def create_pcd(scene_json_path = '/mnt/hdd/3D-FRONT/66352dcb-04af-421d-b2d9-ec7958de8f7e.json', future_folder = '/mnt/hdd/3D-FUTURE-model/',
               number_points = 2048*4):
    with open(scene_json_path) as file:
        scene_json = json.load(file)


    whole_scene = o3d.geometry.PointCloud()
    furniture_dict = {}

    for furniture in scene_json['furniture']:

        model_path = os.path.join(future_folder, furniture['jid'], 'raw_model.obj')
        if not os.path.exists(model_path): continue
        v, _, n, faces, _, _ = igl.read_obj(model_path)
        mesh = o3d.io.read_triangle_mesh(model_path)
        v = np.asarray(mesh.vertices)
        n = np.asarray(mesh.vertex_normals) 
        faces = np.asarray(mesh.triangles) 
        model = [v, n, faces]
        furniture_dict.update({furniture['uid']: model})
    from math import radians

    import mathutils
    i = 0
    for room in scene_json['scene']['room']:
        for c in room['children']:
            if "furniture" not in c["instanceid"]: continue
            if c['ref'] in furniture_dict.keys():
                v, n, faces= deepcopy(furniture_dict[c['ref']])

                pos = c['pos']
                rot = c['rot']
                scale = c['scale']
                v = v.astype(np.float64) * scale
                #v[:, 1], v[:, 2] = v[:, 2], v[:, 1].copy()
                ref = [0,0,1]
                axis = np.cross(ref, rot[1:])
                theta = np.arccos(np.dot(ref, rot[1:]))*2
                if np.sum(axis) != 0 and not math.isnan(theta):
                    #R = rotation_matrix(axis, theta)
                    blender_rot_mat = mathutils.Matrix.Rotation(radians(-90), 4, 'X')
                    R = mathutils.Quaternion(c["rot"]).to_euler().to_matrix().to_4x4()
                    R = np.array(blender_rot_mat @ R)[:3,:3]
                    v = v @ R.T
                v = v + np.asarray(mathutils.Vector(pos).xzy)

                v = o3d.utility.Vector3dVector(v)
                faces = o3d.utility.Vector3iVector(faces)
                #n = o3d.utility.Vector3dVector(n)
                mesh_ = o3d.geometry.TriangleMesh(v, faces)
                #mesh_.vertex_normals = n
                colors = np.ones_like(v)*125
                colors = o3d.utility.Vector3dVector(colors)
                mesh_.vertex_colors = colors
                pcd = mesh_.sample_points_uniformly(int(number_points))
                whole_scene = whole_scene + pcd
                if i ==0: break
                i = i+1
    return whole_scene#.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
'''