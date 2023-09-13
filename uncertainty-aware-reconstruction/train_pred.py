from datasets import   Front3D_Obj
from modeling.modules import Model
import torch
import torch.utils.data as data
#from utils.mm3d_pn2 import furthest_point_sample, gather_points
#from mmcv.ops import furthest_point_sample, gather_points
from pytorch3d.ops import sample_farthest_points
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
#from tensorboard_plugin_geometry import add_geometry
#from tensorboard.plugins.mesh import summary as mesh_summary
import numpy as np
#SummaryWriter.add_geometry = add_geometry
from datetime import datetime
from pytorch3d.loss import chamfer_distance  as calc_cd
from torch.optim.lr_scheduler import StepLR
import os
from utils.geometry import batched_rotation_matrix

comment = "2k_Rot_noNorm_qxfeat_1to1_chamfer_pred_depth"
log_dir = "/mnt/hdd/tmp/logs"  # Choose the log directory path
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = log_dir[:-1]if log_dir[-1] == '/' else log_dir
log_dir = log_dir + "/" + current_time + "_"+comment
#log_dir = "/home/guests/mohammad_rashed/reports/logs/Aug22_14-11-18_train_l1_1_lr_3e-6_10k/"
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
    PC_from_depth = []

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
        PC_from_depth.append(torch.Tensor(value["pred_pc"]))

        mask.append(value["mask"])
        rgb.append(torch.Tensor(np.array(value["rgb"])))
        obj_id.append(value["obj_id"])
        instance_id.append(value["instance_id"])
        scene.append(value["scene"])
        img_num.append(value["img_num"])
    #if len(PC_from_CAD) == 0: return None
    return {
        "PC_from_CAD": torch.stack(PC_from_CAD) ,
        "PC_from_depth": repeat_tensors_to_same_length(PC_from_depth) ,
        "PC_from_gt_depth":  repeat_tensors_to_same_length(PC_from_gt_depth) ,
        "mask": mask ,
        "rgb": torch.stack(rgb) ,
        "obj_id": obj_id ,
        "instance_id": instance_id ,
        "scene": scene ,
        "img_num": img_num ,
    }

train_dataset = Front3D_Obj(num_samples = 10000)
train_dataloader = data.DataLoader(train_dataset, batch_size = 4, collate_fn=collate_function, shuffle= True, drop_last= True, num_workers= 12)
val_dataset = Front3D_Obj(num_samples = 1000, split = 'val')
val_dataloader = data.DataLoader(val_dataset, batch_size= 4, collate_fn=collate_function, shuffle= False, drop_last= False, num_workers= 12)

model = Model(args = {}).cuda()
mean = torch.Tensor(train_dataset.mean)
std = torch.Tensor(train_dataset.std)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00001)
log_every = 2400

start_epoch = 0
checkpoint_path = "checkpoint_249.pt"  # /mnt/hdd/tmp/logs/Provide the path to the saved checkpoint
#checkpoint_path = "/home/guests/mohammad_rashed/reports/logs/Aug22_14-11-18_train_l1_1_lr_3e-6_10k/checkpoint_5.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #start_epoch = checkpoint['epoch'] + 1  # Start training from the next epoch
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
#scheduler = StepLR(optimizer, step_size=60, gamma=0.1)
for epoch in range(start_epoch,250):
    loss_fine = []
    loss_mid = []
    loss_coarse = []
    loss_total =[]
    count = 0
    model.train()
    for i,batch in enumerate(train_dataloader):
        if batch is None :continue
        inputs = batch["PC_from_depth"].cuda()

        inputs, _ = sample_farthest_points(inputs, K = 1024*2)
        inputs2 = batch["PC_from_gt_depth"].cuda()

        inputs2, _ = sample_farthest_points(inputs2, K = 1024*2)

        #idx = torch.randperm(inputs.shape[1], device = "cuda")
        #inputs = inputs[:,idx[:2048], :]
        gt = batch["PC_from_CAD"].contiguous().cuda()
        cat = torch.nn.functional.one_hot((torch.Tensor(batch["obj_id"]) -1).type(torch.long), num_classes= 98).float().cuda()

        #torch.cuda.synchronize()
        #print(inputs.shape, gt.shape)

        cd,_ = calc_cd(inputs2, gt, single_directional=True, norm=1, batch_reduction = None)
        inputs2 = inputs2.cpu()

        #torch.cuda.synchronize()
        max_ = gt.max(1)[0]
        min_ = gt.min(1)[0]
        
        D = ((max_ - min_)**2).sum()**0.5
        mask = (cd < 0.1).type(torch.int8)
        if mask.sum() ==0 :continue


        angles = torch.rand((gt.shape[0])) *3.14
        axis = torch.Tensor([0,1,0])[None,:].expand(gt.shape[0],-1)
        Rs = batched_rotation_matrix(axis, angles).cuda().transpose(1,2)

        inputs = torch.bmm(inputs, Rs )
        gt = torch.bmm(gt, Rs )
        median = ((gt.max(1)[0]+gt.min(1)[0])/2).unsqueeze(1)#torch.median(gt,1)[0].unsqueeze(1)

        inputs = inputs.transpose(1,2).contiguous()
        gt = gt - median
        inputs = inputs - median.transpose(1,2)

        if torch.isnan(gt).sum() or torch.isnan(inputs).sum():
            print("skipped nan")
        #torch.cuda.synchronize()
        (fine1, fine, coarse), (gt,gt_fine1, gt_coarse), (loss3, loss2, loss1), total_train_loss = model(inputs,gt,  is_training=True, mask = mask, cat=cat)
        #torch.cuda.synchronize()
        print(total_train_loss.item(), loss3, loss2, loss1, epoch, i)
        #if total_train_loss.item() >2.5: continue
        optimizer.zero_grad()
        total_train_loss.backward()
        #torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        #torch.cuda.synchronize()
        optimizer.step()
        b = gt.shape[0]
        count += b
        loss_fine.append(loss3*b)
        loss_mid.append(loss2*b)
        loss_coarse.append(loss1*b)
        loss_total.append(total_train_loss.item()*b)
        
        # Visualizing point clouds every x iterations (e.g., x = 100)
        '''
        if i % log_every == 0: #total_train_loss>2.0: #i % 2 == 0:
            rgb = batch["rgb"].type(torch.uint8)
            inputs = inputs.detach().cpu()
            for j, (inputs_, rgb_, fine1_, gt_, coarse_, gt_coarse_, fine_, gt_fine1_) in enumerate(zip(inputs, rgb, fine1, gt, coarse, gt_coarse, fine, gt_fine1)):
                writer.add_images(f"train_iteration_{i}_sample_{j}/rgb input", rgb_.unsqueeze(0), global_step=epoch, dataformats = "NHWC")

                writer.add_mesh(f'train_iteration_{i}_sample_{j}_input', inputs_.unsqueeze(0).transpose(1,2)*1000, colors=np.ones_like(inputs, dtype = np.uint8)*125, global_step=epoch)
                writer.add_mesh(f'train_iteration_{i}_sample_{j}_fine/prediction', fine1_.unsqueeze(0)*1000, colors=np.ones_like(fine1, dtype = np.uint8)*125, global_step=epoch)
                writer.add_mesh(f'train_iteration_{i}_sample_{j}_fine/gt ', gt_.unsqueeze(0)*1000, colors=np.ones_like(gt, dtype = np.uint8)*125, global_step=epoch)
                writer.add_mesh(f'train_iteration_{i}_sample_{j}_coarse/prediction ', coarse_.unsqueeze(0)*1000, colors=np.ones_like(fine1, dtype = np.uint8)*125, global_step=epoch)
                writer.add_mesh(f'train_iteration_{i}_sample_{j}_coarse/gt', gt_coarse_.unsqueeze(0)*1000, colors=np.ones_like(gt, dtype = np.uint8)*125, global_step=epoch)
                writer.add_mesh(f'train_iteration_{i}_sample_{j}_mid/prediction ', fine_.unsqueeze(0)*1000, colors=np.ones_like(fine1, dtype = np.uint8)*125, global_step=epoch)
                writer.add_mesh(f'train_iteration_{i}_sample_{j}_mid/gt', gt_fine1_.unsqueeze(0)*1000, colors=np.ones_like(gt, dtype = np.uint8)*125, global_step=epoch)
                if j==4: break
            '''
            #writer.add_geometry('gt fine', gt , global_step=epoch)
        torch.cuda.empty_cache()

        
        
    writer.add_scalar("train/loss_fine", np.array(loss_fine).sum()/count, global_step=epoch )
    writer.add_scalar("train/loss_mid", np.array(loss_mid).sum()/count, global_step=epoch )
    writer.add_scalar("train/loss_coarse", np.array(loss_coarse).sum()/count, global_step=epoch )
    writer.add_scalar("train/loss_total", np.array(loss_total).sum()/count, global_step=epoch )
    inputs= inputs.detach().cpu()
    gt = gt.detach().cpu()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(log_dir, f"checkpoint_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    if epoch%5 == 0:
        loss_fine = []
        loss_mid = []
        loss_coarse = []
        loss_total =[]
        count = 0
        model.eval()
        for i,batch in enumerate(val_dataloader):
            
            if batch is None :continue
            inputs = batch["PC_from_depth"].cuda()

            inputs, _ = sample_farthest_points(inputs, K = 1024*2)
            inputs2 = batch["PC_from_gt_depth"].cuda()

            inputs2, _ = sample_farthest_points(inputs2, K = 1024*2)

            #idx = torch.randperm(inputs.shape[1], device = "cuda")
            #inputs = inputs[:,idx[:2048], :]
            gt = batch["PC_from_CAD"].contiguous().cuda()
            cat = torch.nn.functional.one_hot((torch.Tensor(batch["obj_id"]) -1).type(torch.long), num_classes= 98).float().cuda()

            #torch.cuda.synchronize()
            #print(inputs.shape, gt.shape)

            cd,_ = calc_cd(inputs2, gt, single_directional=True, norm=1, batch_reduction = None)
            inputs2 = inputs2.cpu()
            #torch.cuda.synchronize()
            max_ = gt.max(1)[0]
            min_ = gt.min(1)[0]
            
            D = ((max_ - min_)**2).sum()**0.5
            mask = (cd < 0.1).type(torch.int8)
            if mask.sum() ==0 :continue

            median = ((gt.max(1)[0]+gt.min(1)[0])/2).unsqueeze(1)#torch.median(gt,1)[0].unsqueeze(1)

            inputs = inputs.transpose(1,2).contiguous()
            gt = gt - median
            inputs = inputs - median.transpose(1,2)
            (fine1, fine, coarse), (gt,gt_fine1, gt_coarse), (loss3, loss2, loss1), total_val_loss = model(inputs,gt,  is_training=True, mask = mask, cat= cat)
            
            b = gt.shape[0]
            count += b
            loss_fine.append(loss3*b)
            loss_mid.append(loss2*b)
            loss_coarse.append(loss1*b)
            loss_total.append(total_val_loss.item()*b)
            ''' 
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
                    if j==4: break
                #writer.add_geometry('gt fine', gt , global_step=epoch)
            '''
            #print(total_train_loss, loss3, loss2, loss1, epoch, i)
            
        writer.add_scalar("val/loss_fine", np.array(loss_fine).sum()/count, global_step=epoch )
        writer.add_scalar("val/loss_mid", np.array(loss_mid).sum()/count, global_step=epoch )
        writer.add_scalar("val/loss_coarse", np.array(loss_coarse).sum()/count, global_step=epoch )
        writer.add_scalar("val/loss_total", np.array(loss_total).sum()/count, global_step=epoch )




