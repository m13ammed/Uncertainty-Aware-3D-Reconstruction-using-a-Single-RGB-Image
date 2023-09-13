import torch.utils.data as data
from pathlib import Path
import os

from typing import List,Any, Dict
from tqdm import tqdm 
import json
import numpy as np
from PIL import Image
import open3d as o3d

import torch
import point_cloud_utils as pcu
def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed", "cabinet",  "unit"]: #"lamp",
        if category_name in name.lower():
            return True
    return False

mapping_id_name = {
1:'smartcustomizedceiling',
2:"cabinet/lightband",
3:"tea table",
4:"cornice",
5:"sewerpipe",
6:"children cabinet",
7:"hole",
8:"ceiling lamp",
9:"chaise longue sofa",
10:"lazy sofa",
11:"appliance",
12:"round end table",
13:"build element",
14:"dining chair",
15:"others",
16:"armchair",
17:"bed",
18:"two-seat sofa",
19:"lighting",
20:"kids bed",
21:"pocket",
22:"storage unit",
23:"media unit",
24:"slabside",
25:"footstool / sofastool / bed end stool / stool",
26:"on top of others",
27:"customizedplatform",
28:"sideboard / side cabinet / console",
29:"plants",
30:"ceiling",
31:"slabtop",
32:"pendant lamp",
33:"lightband",
34:"electric",
35:"pier/stool",
36:"table",
37:"extrusioncustomizedceilingmodel",
38:"baseboard",
39:"front",
40:"wallinner",
41:"basin",
42:"bath",
43:"customizedpersonalizedmodel",
44:"baywindow",
45:"customizedfurniture",
46:"sofa",
47:"kitchen cabinet",
48:"cabinet",
49:"walltop",
50:"chair",
51:"floor",
52:"customizedceiling",
53:"attach to ceiling",
54:"customizedbackgroundmodel",
55:"drawer chest / corner cabinet",
56:"tv stand",
57:"attach to wall",
58:"window",
59:"art",
60:"back",
61:"accessory",
62:"200 - on the floor",
63:"beam",
64:"stair",
65:"wine cooler",
66:"outdoor furniture",
67:"double bed",
68:"dining table",
69:"cabinet/shelf/desk",
70:"single bed",
71:"classic chinese chair",
72:"corner/side table",
73:"flue",
74:"shelf",
75:"customizedfeaturewall",
76:"nightstand",
77:"recreation",
78:"lounge chair / book-chair / computer chair",
79:"slabbottom",
80:"dressing table",
81:"desk",
82:"column",
83:"dressing chair",
84:"wardrobe",
85:"extrusioncustomizedbackgroundwall",
86:"electronics",
87:"bunk bed",
88:"bed frame",
89:"three-seat / multi-person sofa",
90:"customizedfixedfurniture",
91:"bookcase / jewelry armoire",
92:"mirror",
93:"wallbottom",
94:"barstool",
95:"wallouter",
96:"l-shaped sofa",
97:"customized_wainscot",
98:"door",
}
class Front3D_Scene(data.Dataset):

    def __init__(self, dataset_root_path: os.PathLike = '/mnt/hdd/BOP_new2', future_data_path: os.PathLike = '/mnt/hdd/3D-FUTURE-model/', fields: List[str] =["color", "depth"],
                 num_samples: int = -1, shuffle: bool = False, split = 'train', return_paths_only = False, with_pred_depth =True) -> None:
        super().__init__()

        self.dataset_root_path = Path(dataset_root_path)
        self.future_data_path = Path(future_data_path)
        self.fields = fields
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.split = split
        self.return_paths_only = return_paths_only
        self.with_pred_depth = with_pred_depth
        self.gather_data()
    

    def gather_data(self):
        
        path = self.dataset_root_path / ("front3D_"+self.split)

        gt_scene_gen = path.glob("*/scene_gt.json") 
        sorted_gen = sorted(list(gt_scene_gen))
        self.samples = []
        for j, gt_json_path in tqdm(enumerate(sorted_gen)):
            if len(self.samples) >= self.num_samples: break
            json_gt = json.loads(gt_json_path.read_bytes())

            for key, value in json_gt.items():
                if  value == {}: continue
                else:
                    self.samples.append((gt_json_path.parent, key)) 
                    if len(self.samples) >= self.num_samples: break


    def __len__(self):
        
        if self.num_samples == -1: return len(self.samples)
        else: return min(self.num_samples, len(self.samples))

    def __getitem__(self, index) -> Dict:
        main_path, img_num = self.samples[index] #f"{img_num:06d}.jpg"

        json_gt = json.loads((main_path/"scene_gt.json").read_bytes())[str(img_num)]
        json_cam = json.loads((main_path/"scene_camera.json").read_bytes())[str(img_num)]
        transforms_list = [] 
        model_paths_list = []
        obj_ids_list = []
        instance_ids_list = []
        visib_fract_list = []
        scale_list = []
        for key, val in json_gt.items():
    
            transform = np.eye(4)
            model_path =  self.future_data_path / val['jid'] / 'raw_model.obj'
            
            transform[:3,:3] =   np.array(val["cam_R_m2c"]).reshape((3,3))  #np.array([[-1,0,0],[0,1,0],[0,0,-1]])  @
            transform[:3,-1] = [v/1000.0 for v in val["cam_t_m2c"]] #


            obj_id = val["obj_id"]
            instance_id = val["instance_id"]

            model_paths_list.append(model_path)
            transforms_list.append(transform)
            obj_ids_list.append(obj_id)
            instance_ids_list.append(instance_id)
            #visib_fract_list.append(val["visib_frac"])
            scale_list.append(val["scale"])
        return_dic = {

            "T_m2c": transforms_list,
            "obj_id" : obj_ids_list,
            "instance_id": instance_ids_list,
            "model_path" : model_paths_list,
            "scene_path" : main_path,
            "img_num": img_num,
            "cam_K" : np.array(json_cam["cam_K"]).reshape((3,3)),
            "depth_scale": json_cam["depth_scale"],
            "obj_scale": scale_list,
            #"visib_fract": visib_fract_list,

        }
        img_num = int(img_num)
        depth_path = main_path / "depth" / f"{img_num:06d}.png"
        rgb_path = main_path / "rgb" / f"{img_num:06d}.jpg"
        mask_path = main_path / "mask" / f"{img_num:06d}.png"
        pred_depth = main_path / "depth_pred" / f"{img_num:06d}.npy"
        uncer = main_path / "uncer" / f"{img_num:06d}.npy"

        if self.return_paths_only:

            return_dic.update({
                "depth_path":depth_path,
                "rgb_path":rgb_path,
                "mask_path":mask_path,

            })
            return return_dic
        else:
            left, top, right, bottom = 0,0,320,240
            if self.with_pred_depth:
                left, top, right, bottom = 8,16,224+8,288+16
            depth_image = np.asarray(Image.open(depth_path).crop((left, top, right, bottom)), dtype=np.uint16)
            rgb_image = np.asarray(Image.open(rgb_path).crop((left, top, right, bottom)), dtype=np.uint8)
            mask_image = np.asarray(Image.open(mask_path).crop((left, top, right, bottom)), dtype=np.uint8)
            masks = [np.argwhere(mask_image == int(idx)) for idx in instance_ids_list]
            pixel_nm = np.asarray([len(mask) for mask in masks])
            return_dic.update({
                "depth_image":depth_image,
                "rgb_image":rgb_image,
                "mask_image":mask_image,
                "obj_mask": masks,
                "pixel_nm":pixel_nm
            })
            if self.with_pred_depth:
                depth_pred_image = np.load(pred_depth).T
                uncer_image = np.load(uncer).T.astype(float)/10000
                return_dic.update({
                "depth_pred_image":depth_pred_image,
                "uncer_image":uncer_image,
                    })

            return return_dic



class Front3D_Obj(data.Dataset):

    def __init__(self, CAD_num_points = 1024*5, **kwargs) -> None:
        super().__init__()
        self.scenes = Front3D_Scene(return_paths_only=False, **kwargs)
        self.CAD_num_points = CAD_num_points
        self.gather_obj()
        self.mean = np.array([ 0.0100, -0.0103,  2.0816])
        self.std = np.array([0.9517, 2.7281, 0.9368])

        self.mean = np.array([0,0,0])
        self.std = np.array([1,1,1])

    def gather_obj(self):
        self.samples = []
        for i, scene in tqdm(enumerate(self.scenes)):
            js = len(scene["instance_id"])
            indices = np.arange(js)
            mask = scene["pixel_nm"]>1024
            indices = indices[mask]
            i_indices = np.full_like(indices, i)
            
            self.samples.append(np.column_stack((i_indices, indices)))
            
        self.samples = np.concatenate(self.samples, axis=0)

        
    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, index) -> Any:
        i,j = self.samples[index]
        scene = self.scenes[i]
        models_path = scene["scene_path"] / "models"
        models_path.mkdir(exist_ok = True)

        cache_model_path = models_path / (f"{self.CAD_num_points}_{scene['instance_id'][j]}.ply")
        depth_img = scene["depth_image"]
        mask = scene['obj_mask'][j]
        rgb_img = scene['rgb_image']
        K = scene["cam_K"]
        add = (16,8) if self.scenes.with_pred_depth else (0,0)
        if "depth_pred_image" in scene.keys():
            depth_pred_image = scene["depth_pred_image"]
            uncer_image = scene["uncer_image"]
            pred_pc = self.dpt_2_pcld(depth_pred_image, 1000.0/scene["depth_scale"], K, add)
        
        
        gt_pc = self.dpt_2_pcld(depth_img, 1000.0/scene["depth_scale"], K, add)
        T_m2c = scene["T_m2c"][j]#np.linalg.inv()
        gt_pc = gt_pc[mask[:,0], mask[:,1]]
        gt_pc = gt_pc.reshape(-1,3) #@ np.array([[1,0,0],[0,-1,0],[0,0,-1]]).T
        model_path = scene["model_path"][j]
        obj_scale = scene["obj_scale"][j]
        if cache_model_path.exists():
            #CAD = o3d.io.read_point_cloud(str(cache_model_path))#.transform(T_m2c)
            CAD = pcu.load_mesh_v(str(cache_model_path))
        else:
            #return {
            #                }
            v, f = pcu.load_mesh_vf(str(model_path))
            f_i, bc = pcu.sample_mesh_poisson_disk(v, f, 5400)
            CAD = pcu.interpolate_barycentric_coords(f, f_i, bc, v)
            CAD = CAD[:5024]
            #CAD = o3d.io.read_triangle_mesh(str(model_path))
            #try:
            #    CAD = CAD.sample_points_poisson_disk(self.CAD_num_points)
            #except:
            #    return {
            #                }
            pcu.save_triangle_mesh(str(cache_model_path), v=CAD)
            #o3d.io.write_point_cloud(str(cache_model_path), CAD)
            #CAD = CAD.transform(T_m2c)
        
            

        #o3d.io.write_point_cloud("cad.ply", CAD)
        #x = o3d.geometry.PointCloud()
        #x.points = o3d.utility.Vector3dVector(gt_pc)
        #o3d.io.write_point_cloud("depth.ply", x)
        
        CAD =  np.array(obj_scale).reshape(1,3)*  CAD  @  T_m2c[:3,:3].T + T_m2c[:3,-1]
        #if min((CAD.max(0) - gt_pc.max(0)).min(), (gt_pc.min(0) - CAD.min(0)).min()) < -0.01 : return {}
        #el
        #if not check_name(mapping_id_name[scene["obj_id"][j]]):   return {}
        ret_dict =  {
            "PC_from_CAD": (CAD-self.mean)/self.std,
            "PC_from_gt_depth":(gt_pc-self.mean)/self.std ,
            "mask": mask,
            #"PC_from_pred_depth":None,
            "rgb": rgb_img,
            "obj_id" : scene["obj_id"][j],
            "instance_id": scene["instance_id"][j],
            "scene" : scene["scene_path"].name,
            "img_num": scene["img_num"],
        }
        if  "depth_pred_image" in scene.keys():
            pred_pc = pred_pc[mask[:,0], mask[:,1]]
            pred_pc = pred_pc.reshape(-1,3) #@ np.array([[1,0,0],[0,-1,0],[0,0,-1]]).T

            uncer_image = uncer_image[mask[:,0], mask[:,1]]
            uncer_image = uncer_image.reshape(-1,1)
            ret_dict.update({
                "uncer": uncer_image,
                "pred_pc": pred_pc
                
            })
        return ret_dict

    @staticmethod
    def dpt_2_pcld(dpt, cam_scale, K, add = (16,8)):
        if len(dpt.shape) > 2:
            dpt = dpt[0,:, :]
        
        idx = np.indices(dpt.shape[:2])
        xmap = idx[0].astype(float) + add[0] #+ idx[0].max() / 2
        ymap = idx[1].astype(float) +  add[1] #+ idx[1].max() / 2

        
            
        dpt = dpt.astype(np.float32) / cam_scale
        row = (ymap - K[0,2]) * dpt / K[0,0]
        col = (xmap - K[1,2]) * dpt / K[1,1]

        dpt_3d = np.concatenate(
            (row[..., None], col[..., None] ,  dpt[..., None]), axis=2
        )
        return dpt_3d


def pseudo_collate(data):

    return 0
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    import time

    #starting_time = time.time()
    #x = Front3D_Scene(return_paths_only=False, num_samples = 10)
    #init_time = time.time()
    #for i in x:
    #    pass
    #looping_time = time.time()
    #print("time for init:", init_time - starting_time, "time for loop", looping_time - init_time, "total time:", looping_time - starting_time)

    starting_time = time.time()
    y = Front3D_Obj(split = 'train', num_samples = 10000)
    
    loader = data.DataLoader(y, num_workers= 8, batch_size= 8, collate_fn=pseudo_collate)
    #print(len(y), y[0])
    init_time = time.time()
    for i in tqdm(loader):
        pass
    looping_time = time.time()

    print("time for init:", init_time - starting_time, "time for loop", looping_time - init_time, "total time:", looping_time - starting_time)



