from datasets import   Front3D_Obj
from modeling.modules import Model
import torch
import torch.utils.data as data
from utils.mm3d_pn2 import furthest_point_sample, gather_points
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
from tensorboard_plugin_geometry import add_geometry
log_dir = "/mnt/hdd/tmp_new/logs"  # Choose the log directory path
writer = SummaryWriter(log_dir=log_dir)


def repeat_tensors_to_same_length(tensor_list):
    # Find the maximum length among all tensors
    max_length = max(tensor.size(0) for tensor in tensor_list)

    # Repeat each tensor along the first dimension to match the maximum length
    repeated_tensors = [tensor.repeat(max_length // tensor.size(0) + 1, 1)[:max_length]
                        for tensor in tensor_list]

    # Stack the tensors to create a new tensor of shape (num_tensors, max_length, ...)
    stacked_tensors = torch.stack(repeated_tensors)

    return stacked_tensors
def collate_function(data):
    PC_from_CAD = []
    PC_from_gt_depth = []
    mask = [] 
    rgb = []
    obj_id = []
    instance_id = []
    scene = []
    img_num = []
    for  value in data:
        if not value or value["PC_from_gt_depth"].shape[0]<256: continue
        PC_from_CAD.append(torch.Tensor(value["PC_from_CAD"]))
        PC_from_gt_depth.append(torch.Tensor(value["PC_from_gt_depth"]))
        mask.append(value["mask"])
        rgb.append(torch.Tensor(value["rgb"]))
        obj_id.append(value["obj_id"])
        instance_id.append(value["instance_id"])
        scene.append(value["scene"])
        img_num.append(value["img_num"])

    return {
        "PC_from_CAD": torch.stack(PC_from_CAD) ,
        "PC_from_gt_depth":  repeat_tensors_to_same_length(PC_from_gt_depth) ,
        "mask": mask ,
        "rgb": torch.stack(rgb) ,
        "obj_id": obj_id ,
        "instance_id": instance_id ,
        "scene": scene ,
        "img_num": img_num ,
    }

train_dataset = Front3D_Obj(num_samples = 10)
train_dataloader = data.DataLoader(train_dataset, batch_size= 4, collate_fn=collate_function, shuffle= False)
model = Model(args = {}).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(5000):
    for i,batch in enumerate(train_dataloader):

        inputs = (batch["PC_from_gt_depth"].transpose(2,1).contiguous().cuda() -10)/10

        idx_0 = furthest_point_sample(inputs.transpose(1, 2).contiguous(), 2048)
        inputs = gather_points(inputs.contiguous(), idx_0)
        gt = (batch["PC_from_CAD"].transpose(2,1).contiguous().cuda() -10)/10
        idx_0 = furthest_point_sample(gt.transpose(1, 2).contiguous(), 2048)
        gt = gather_points(gt.contiguous(), idx_0).transpose(1, 2).contiguous()
        (fine1, fine, coarse), (gt,gt_fine1, gt_coarse), (loss3, loss2, loss1), total_train_loss = model(inputs,gt,  is_training=True)
        
        optimizer.zero_grad()
        total_train_loss.backward()
        torch.cuda.empty_cache()
        optimizer.step()
        # Visualizing point clouds every x iterations (e.g., x = 100)
        if i % 100 == 0:

            writer.add_geometry('prediction fine', fine1[:1] , global_step=epoch)
            writer.add_geometry('gt fine', gt[:1] , global_step=epoch)

        print(total_train_loss, loss2, epoch, i)
        
    

