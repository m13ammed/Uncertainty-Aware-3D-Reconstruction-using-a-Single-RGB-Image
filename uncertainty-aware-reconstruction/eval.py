from datasets import   Front3D_Obj
from modeling.modules import Model
import torch
import torch.utils.data as data
from utils.mm3d_pn2 import furthest_point_sample, gather_points
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
#from tensorboard_plugin_geometry import add_geometry
#from tensorboard.plugins.mesh import summary as mesh_summary
import numpy as np
#SummaryWriter.add_geometry = add_geometry
from datetime import datetime
from pytorch3d.loss import chamfer_distance  as calc_cd
from torch.optim.lr_scheduler import StepLR
import os

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
        #if not value or value["PC_from_gt_depth"].shape[0]<256: continue
        PC_from_CAD.append(torch.Tensor(value["PC_from_CAD"]))
        PC_from_gt_depth.append(torch.Tensor(value["PC_from_gt_depth"]))
        mask.append(value["mask"])
        rgb.append(torch.Tensor(np.array(value["rgb"])))
        obj_id.append(value["obj_id"])
        instance_id.append(value["instance_id"])
        scene.append(value["scene"])
        img_num.append(value["img_num"])
    #if len(PC_from_CAD) == 0: return None
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

val_dataset = Front3D_Obj(num_samples = 3000, split = 'val')
val_dataloader = data.DataLoader(val_dataset, batch_size= 2, collate_fn=collate_function, shuffle= False, drop_last= False, num_workers= 0)

model = Model(args = {}).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
log_every = 100
comment = "overtfit_l1_1_lr_3e-6_10k"
log_dir = "/mnt/hdd/tmp_new/logs_new2"  # Choose the log directory path
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = log_dir[:-1]if log_dir[-1] == '/' else log_dir
log_dir = log_dir + "/" + current_time + "_"+comment
writer = SummaryWriter(log_dir="/mnt/hdd/tmp_new/logs_new2/Aug15_05-57-42_overtfit_l1_1_lr_3e-6_10k_val/")

start_epoch = 40
for epoch in range(start_epoch,5000):
    checkpoint_path = f"/mnt/hdd/tmp_new/logs_new2/Aug15_05-57-42_overtfit_l1_1_lr_3e-6_10k/checkpoint_{40}.pt"  # Provide the path to the saved checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #start_epoch = checkpoint['epoch'] + 1  # Start training from the next epoch
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else: 
        continue
    loss_fine = []
    loss_mid = []
    loss_coarse = []
    loss_total =[]
    count = 0
    model.eval()

    if epoch%5 == 0:
        loss_fine = []
        loss_mid = []
        loss_coarse = []
        loss_total =[]
        count = 0
        model.eval()
        for i,batch in enumerate(val_dataloader):
            
            if batch is None :continue
            inputs = (batch["PC_from_gt_depth"].transpose(2,1).contiguous().cuda())

            idx_0 = furthest_point_sample(inputs.transpose(1, 2).contiguous(), 1024)
            inputs = gather_points(inputs.contiguous(), idx_0)
            gt = (batch["PC_from_CAD"].contiguous().cuda()) 

            cd,_ = calc_cd(inputs.transpose(1,2),gt, single_directional=True, norm=1, batch_reduction = None)
            #cd,_ = calc_cd(inputs,gt, single_directional=True, norm=1, batch_reduction = None)
            max_ = gt.max(1)[0]
            min_ = gt.min(1)[0]
            
            D = ((max_ - min_)**2).sum()**0.5
            mask = (cd/D*100 < 10).type(torch.int8)
            if mask.sum() ==0 :continue
            median = ((gt.max(1)[0]+gt.min(1)[0])/2).unsqueeze(1)#torch.median(gt,1)[0].unsqueeze(1)
            #median = ((inputs.max(2)[0]+inputs.min(2)[0])/2).unsqueeze(2).transpose(1,2)#torch.median(gt,1)[0].unsqueeze(1)

            gt = gt - median
            inputs = inputs - median.transpose(1,2)
            inputs
            gt
            if i == 294 or i ==951:continue
            try:
                (fine1, fine, coarse), (gt,gt_fine1, gt_coarse), (loss3, loss2, loss1), total_val_loss = model(inputs,gt,  is_training=True, mask = mask)
            except:
                continue
            b = gt.shape[0]
            count += b
            loss_fine.append(loss3*b)
            loss_mid.append(loss2*b)
            loss_coarse.append(loss1*b)
            loss_total.append(total_val_loss.item()*b)
            
            # Visualizing point clouds every x iterations (e.g., x = 100)
            if i % log_every == 0: #total_train_loss>2.0: #i % 2 == 0:
                rgb = batch["rgb"].type(torch.uint8)
                inputs = inputs.detach().cpu()
                for j, (inputs_, rgb_, fine1_, gt_, coarse_, gt_coarse_, fine_, gt_fine1_) in enumerate(zip(inputs, rgb, fine1, gt, coarse, gt_coarse, fine, gt_fine1)):
                    writer.add_images(f"val_iteration_{i}_sample_{j}/rgb input", rgb_.unsqueeze(0), global_step=epoch, dataformats = "NHWC")

                    writer.add_mesh(f'val_iteration_{i}_sample_{j}_input', inputs_.unsqueeze(0).transpose(1,2)*1000, colors=np.ones_like(inputs, dtype = np.uint8)*125, global_step=epoch)
                    writer.add_mesh(f'val_iteration_{i}_sample_{j}_fine/prediction', fine1_.unsqueeze(0)*1000, colors=np.ones_like(fine1, dtype = np.uint8)*125, global_step=epoch)
                    writer.add_mesh(f'val_iteration_{i}_sample_{j}_fine/gt ', gt_.unsqueeze(0)*1000, colors=np.ones_like(gt, dtype = np.uint8)*125, global_step=epoch)
                    writer.add_mesh(f'val_iteration_{i}_sample_{j}_coarse/prediction ', coarse_.unsqueeze(0)*1000, colors=np.ones_like(fine1, dtype = np.uint8)*125, global_step=epoch)
                    writer.add_mesh(f'val_iteration_{i}_sample_{j}_coarse/gt', gt_coarse_.unsqueeze(0)*1000, colors=np.ones_like(gt, dtype = np.uint8)*125, global_step=epoch)
                    writer.add_mesh(f'val_iteration_{i}_sample_{j}_mid/prediction ', fine_.unsqueeze(0)*1000, colors=np.ones_like(fine1, dtype = np.uint8)*125, global_step=epoch)
                    writer.add_mesh(f'val_iteration_{i}_sample_{j}_mid/gt', gt_fine1_.unsqueeze(0)*1000, colors=np.ones_like(gt, dtype = np.uint8)*125, global_step=epoch)
               
                #writer.add_geometry('gt fine', gt , global_step=epoch)

            print(total_val_loss, loss3, loss2, loss1, epoch, i)
            
        writer.add_scalar("val/loss_fine", np.array(loss_fine)[~np.isnan(np.array(loss_fine))].sum() / (count - np.isnan(np.array(loss_fine)).sum()), global_step=epoch )
        writer.add_scalar("val/loss_mid", np.array(loss_mid)[~np.isnan(np.array(loss_mid))].sum() / (count - np.isnan(np.array(loss_mid)).sum()), global_step=epoch )
        writer.add_scalar("val/loss_coarse", np.array(loss_coarse)[~np.isnan(np.array(loss_coarse))].sum() / (count - np.isnan(np.array(loss_coarse)).sum()), global_step=epoch )
        writer.add_scalar("val/loss_total", np.array(loss_total)[~np.isnan(np.array(loss_total))].sum() / (count - np.isnan(np.array(loss_total)).sum()), global_step=epoch )



