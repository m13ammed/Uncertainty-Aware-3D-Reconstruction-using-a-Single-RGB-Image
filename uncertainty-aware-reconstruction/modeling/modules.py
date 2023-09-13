import torch
from torch import nn
from mmcv.ops import furthest_point_sample, gather_points
#from pytorch3d.ops import sample_farthest_points, gather_points

from pytorch3d.ops import sample_farthest_points as fps
from pytorch3d.ops.utils import masked_gather

import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance  as calc_cd
#from utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
#from utils.ChamferDistancePytorch.fscore import fscore
#def calc_cd(output, gt, calc_f1=False):
#    cham_loss = dist_chamfer_3D.chamfer_3DDist()
#    dist1, dist2, _, _ = cham_loss(gt, output)
#    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
#    cd_t = (dist1.mean(1) + dist2.mean(1))
#    if calc_f1:
#        f1, _, _ = fscore(dist1, dist2)
#        return cd_p, cd_t, f1
#    else:
#        return cd_p, cd_t

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.reset_parameters()

    def forward(self, latents):
        # Compute L2 distances between latents and embedding weights
        dist = torch.linalg.vector_norm(latents.movedim(1, -1).unsqueeze(-2) - self.embedding.weight, dim=-1)
        encoding_inds = torch.argmin(dist, dim=-1)        # Get the number of the nearest codebook vector
        quantized_latents = self.quantize(encoding_inds)  # Quantize the latents

        # Compute the VQ Losses
        codebook_loss = F.mse_loss(latents, quantized_latents) #.detach()
        commitment_loss = F.mse_loss(latents, quantized_latents.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Make the gradient with respect to latents be equal to the gradient with respect to quantized latents 
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents, vq_loss
    
    def quantize(self, encoding_indices):
        z = self.embedding(encoding_indices)
        z = z.movedim(-1, 1) # Move channels back
        return z
    
    def reset_parameters(self):
        nn.init.uniform_(self.embedding.weight, -1 / self.num_embeddings, 1 / self.num_embeddings)

class sliding_window(nn.Module): #does not include pos encoding

    def __init__(self, patch_Size=16, overlap = 0, projection_size = None):
        super(sliding_window, self).__init__()
        self.patch_Size = patch_Size
        projection_size = 3*patch_Size if projection_size is None else projection_size
        self.input_process = torch.nn.Conv2d(
                in_channels=3,
                out_channels=projection_size,
                kernel_size=patch_Size,
                stride=patch_Size,
                padding=0,
            )
        self.norm = nn.LayerNorm(projection_size)

    def forward(self, inputs):
        assert inputs.si
        processed = self.input_process(inputs)
        processed = self.norm(processed)

        return processed


class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]


        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        src1 = src1.permute(1, 2, 0)

        return src1
    


class GDP_Block(nn.Module): 
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0, N_div = 4): #ADD FPS
        super(GDP_Block , self).__init__()
        self.N_div = N_div
        self.attention = cross_transformer(d_model, d_model_out, nhead, dim_feedforward, dropout)
    def forward(self, points, x0):
        batch_size, D0, N = points.size()
        batch_size2, D, N2 = x0.size()
        #print(x0.size(), points.size())
        #idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // self.N_div)
        points = points.transpose(1, 2).contiguous()
        points, idx_0 = fps(points, K = N // self.N_div)
        points = points.transpose(1, 2).contiguous()
        #x_g0 = masked_gather(x0.transpose(1, 2), idx_0).transpose(1, 2) #gather_points(x0, idx_0).contiguous()
        idx_expanded = idx_0[:,None,:].expand(-1, D, -1)
        #print(idx_expanded)
        x_g0 = x0.gather(dim=-1, index=idx_expanded)
        #points = gather_points(points, idx_0)
        x1 = self.attention(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        return x1, points
    
class SFA_Block(nn.Module):     
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0): #somehow add upsampling
        super(SFA_Block , self).__init__()
        self.attention = cross_transformer(d_model, d_model_out, nhead, dim_feedforward, dropout)

    def forward(self, input):
        out = self.attention(input, input)
        return out

class PCT_refine(nn.Module):
    def __init__(self, channel=128,ratio=1):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = SFA_Block(channel*2,512)
        self.sa2 = SFA_Block(512,512)
        self.sa3 = SFA_Block(512,channel*ratio)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)


    def forward(self, x, coarse,feat_g):
        batch_size, _, N = coarse.size()

        y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        y0 = torch.cat([y,feat_g.repeat(1,1,y.shape[-1])],dim=1)

        y1 = self.sa1(y0)
        y2 = self.sa2(y1)
        y3 = self.sa3(y2)
        y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio)

        y_up = y.repeat(1,1,self.ratio)
        y_cat = torch.cat([y3,y_up],dim=1)
        y4 = self.conv_delta(y_cat)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio)

        return x, y3
    
class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.gdp_1 = GDP_Block(channel,channel, N_div=2)
        self.sfa_1 = SFA_Block(channel*2,channel*2)

        self.gdp_2 = GDP_Block(channel*2,channel*2, N_div=4)
        self.sfa_2 = SFA_Block(channel*4,channel*4)

        self.gdp_3 = GDP_Block(channel*4,channel*4, N_div=8)
        self.sfa_3 = SFA_Block(channel*8,channel*8)


        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)

        self.sa0_d = SFA_Block(channel*8,channel*8)
        self.sa1_d = SFA_Block(channel*8,channel*8)
        self.sa2_d = SFA_Block(channel*8,channel*8)

        self.relu = nn.GELU()

    def forward(self, points):
        batch_size, _, N = points.size()
        #torch.cuda.synchronize(device=None)
        points = points.contiguous()
        x = self.relu(self.conv1(points))  # B, D, N
        #torch.cuda.synchronize(device=None)
        x0 = self.conv2(x).contiguous()
        #torch.cuda.synchronize(device=None)
        x1, pt1 = self.gdp_1(points, x0)
        #torch.cuda.synchronize(device=None)
        x1 = self.sfa_1(x1).contiguous()
        #torch.cuda.synchronize(device=None)
        x2, pt2 = self.gdp_2(pt1, x1)
        #torch.cuda.synchronize(device=None)
        x2 = self.sfa_2(x2).contiguous()
        #torch.cuda.synchronize(device=None)
        x3, pt3 = self.gdp_3(pt2, x2)
        #torch.cuda.synchronize(device=None)
        x3 = self.sfa_3(x3)
        #torch.cuda.synchronize(device=None)
        x_g = F.adaptive_max_pool1d(x3, 1)#.view(batch_size, -1)#.unsqueeze(-1)
        #torch.cuda.synchronize(device=None)
        x = self.relu(self.ps_adj(x_g))
        #torch.cuda.synchronize(device=None)
        x = self.relu(self.ps(x))
        #torch.cuda.synchronize(device=None)
        x = self.relu(self.ps_refuse(x))
        #torch.cuda.synchronize(device=None)
        x0_d = (self.sa0_d(x))
        #torch.cuda.synchronize(device=None)
        x1_d = (self.sa1_d(x0_d))
        #torch.cuda.synchronize(device=None)
        x2_d = (self.sa2_d(x1_d)).reshape(batch_size,self.channel,-1) #N//8
        #torch.cuda.synchronize(device=None)
        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
        #torch.cuda.synchronize(device=None)
        return x_g, fine

class PCT_encoder_vq(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder_vq, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.gdp_1 = GDP_Block(channel,channel, N_div=2)
        self.sfa_1 = SFA_Block(channel*2,channel*2)

        self.gdp_2 = GDP_Block(channel*2,channel*2, N_div=4)
        self.sfa_2 = SFA_Block(channel*4,channel*4)

        self.gdp_3 = GDP_Block(channel*4,channel*4, N_div=8)
        self.sfa_3 = SFA_Block(channel*8,channel*8)


        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)

        self.sa0_d = SFA_Block(channel*8,channel*8)
        self.sa1_d = SFA_Block(channel*8,channel*8)
        self.sa2_d = SFA_Block(channel*8,channel*8)

        self.relu = nn.GELU()

        self.VQ = VectorQuantizer(90, channel*8)

    def forward(self, points):
        batch_size, _, N = points.size()
        #torch.cuda.synchronize(device=None)
        points = points.contiguous()
        x = self.relu(self.conv1(points))  # B, D, N
        #torch.cuda.synchronize(device=None)
        x0 = self.conv2(x).contiguous()
        #torch.cuda.synchronize(device=None)
        x1, pt1 = self.gdp_1(points, x0)
        #torch.cuda.synchronize(device=None)
        x1 = self.sfa_1(x1).contiguous()
        #torch.cuda.synchronize(device=None)
        x2, pt2 = self.gdp_2(pt1, x1)
        #torch.cuda.synchronize(device=None)
        x2 = self.sfa_2(x2).contiguous()
        #torch.cuda.synchronize(device=None)
        x3, pt3 = self.gdp_3(pt2, x2)
        #torch.cuda.synchronize(device=None)
        x3 = self.sfa_3(x3)
        #torch.cuda.synchronize(device=None)
        x_g, self.vq_loss = self.VQ(x3)
        x_g = F.adaptive_max_pool1d(x_g, 1)#.view(batch_size, -1)#.unsqueeze(-1)
        #torch.cuda.synchronize(device=None)
        x = self.relu(self.ps_adj(x_g))
        #torch.cuda.synchronize(device=None)
        x = self.relu(self.ps(x))
        #torch.cuda.synchronize(device=None)
        x = self.relu(self.ps_refuse(x))
        #torch.cuda.synchronize(device=None)
        x0_d = (self.sa0_d(x))
        #torch.cuda.synchronize(device=None)
        x1_d = (self.sa1_d(x0_d))
        #torch.cuda.synchronize(device=None)
        x2_d = (self.sa2_d(x1_d)).reshape(batch_size,self.channel,-1) #N//8
        #torch.cuda.synchronize(device=None)
        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
        #torch.cuda.synchronize(device=None)
        return x_g, fine
class PCT_encoder_cat(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder_cat, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.gdp_1 = GDP_Block(channel,channel, N_div=2)
        self.sfa_1 = SFA_Block(channel*2,channel*2)

        self.gdp_2 = GDP_Block(channel*2,channel*2, N_div=4)
        self.sfa_2 = SFA_Block(channel*4,channel*4)

        self.gdp_3 = GDP_Block(channel*4,channel*4, N_div=8)
        self.sfa_3 = SFA_Block(channel*8,channel*8)


        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)

        self.sa0_d = SFA_Block(channel*8,channel*8)
        self.sa1_d = SFA_Block(channel*8,channel*8)
        self.sa2_d = SFA_Block(channel*8,channel*8)

        self.relu = nn.GELU()

        self.cat_cond2 = cross_transformer(channel, channel)
        self.cat_cond1 = nn.Conv1d(98, channel, kernel_size=1)

        #self.cat_prior2 = cross_transformer(channel*8, channel*8)
        #self.cat_prior1 = nn.Conv1d(98, channel*8, kernel_size=1)

    def forward(self, points, cat):
        batch_size, _, N = points.size()
        #torch.cuda.synchronize(device=None)
        points = points.contiguous()
        x = self.relu(self.conv1(points))  # B, D, N
        #torch.cuda.synchronize(device=None)
        x0 = self.conv2(x).contiguous()

        #cat = self.cat_cond1(cat.unsqueeze(-1)).expand(-1,-1, x0.shape[-1])
        #x0 = self.cat_cond2(cat, x0)

        cat_ = self.cat_cond1(cat.unsqueeze(-1))
        x0 = self.cat_cond2(x0, cat_)

        #torch.cuda.synchronize(device=None)
        x1, pt1 = self.gdp_1(points, x0)
        #torch.cuda.synchronize(device=None)
        x1 = self.sfa_1(x1).contiguous()
        #torch.cuda.synchronize(device=None)
        x2, pt2 = self.gdp_2(pt1, x1)
        #torch.cuda.synchronize(device=None)
        x2 = self.sfa_2(x2).contiguous()
        #torch.cuda.synchronize(device=None)
        x3, pt3 = self.gdp_3(pt2, x2)
        #torch.cuda.synchronize(device=None)
        x3 = self.sfa_3(x3)
        #torch.cuda.synchronize(device=None)
        x_g = F.adaptive_max_pool1d(x3, 1)#.view(batch_size, -1)#.unsqueeze(-1)

        #cat_ = self.cat_prior1(cat.unsqueeze(-1))
        #x_g  = self.cat_prior2(cat_, x3)
        #torch.cuda.synchronize(device=None)
        x = self.relu(self.ps_adj(x_g))
        #torch.cuda.synchronize(device=None)
        x = self.relu(self.ps(x))
        #torch.cuda.synchronize(device=None)
        x = self.relu(self.ps_refuse(x))
        #torch.cuda.synchronize(device=None)
        x0_d = (self.sa0_d(x))
        #torch.cuda.synchronize(device=None)
        x1_d = (self.sa1_d(x0_d))
        #torch.cuda.synchronize(device=None)
        x2_d = (self.sa2_d(x1_d)).reshape(batch_size,self.channel,-1) #N//8
        #torch.cuda.synchronize(device=None)
        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
        #torch.cuda.synchronize(device=None)
        return x_g, fine

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        step1 = 3
        step2 = 3
        
        #self.encoder = PCT_encoder_vq(channel=64)
        self.encoder = PCT_encoder_cat(channel=64)

        self.refine = PCT_refine(ratio=step1)
        
        self.refine1 = PCT_refine(ratio=step2)


    def forward(self, x, gt=None, is_training=True, mask = None, cat = None, chamfer = True, ret_fet = False):
        #torch.cuda.synchronize(device=None)
        feat_g, coarse = self.encoder(x, cat)
        #torch.cuda.synchronize(device=None)
        new_x = torch.cat([x,coarse],dim=2) #the sizes are sus
        #torch.cuda.synchronize(device=None)
        new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))

        #coarse = gather_points(coarse, furthest_point_sample(coarse.transpose(1, 2).contiguous(), 512))
        #torch.cuda.synchronize(device=None)
        fine, feat_fine = self.refine(None, new_x, feat_g)
        #torch.cuda.synchronize()
        fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)
        #torch.cuda.synchronize(device=None)
        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()
        #torch.cuda.synchronize(device=None)
         
        if is_training and chamfer:
            #torch.cuda.synchronize() ,single_directional = True
            loss3, _ = calc_cd(gt, fine1 ,single_directional = True , norm=1, batch_reduction = None)   #calc_cd(fine1, gt)
            loss3_, _ =  calc_cd(fine1, gt ,single_directional = True, norm=1, batch_reduction = None) 
            loss3 = loss3 + 1.0 * loss3_
            loss3 = (loss3 * mask).sum()/ mask.sum()
            gt_fine1,_ = fps(gt, K = fine.shape[1])
            #gt_fine1 = gather_points(gt.transpose(1, 2).contiguous(), furthest_point_sample(gt, fine.shape[1])).transpose(1, 2).contiguous()
            #torch.cuda.synchronize()
            loss2, _ = calc_cd(gt_fine1, fine ,single_directional = True, norm=1, batch_reduction = None) #calc_cd(fine, gt_fine1)
            loss2_, _ =  calc_cd(fine, gt_fine1, single_directional = True, norm=1, batch_reduction = None)
            loss2 = loss2 + 1.0 * loss2_
            loss2 = (loss2 * mask).sum()/ mask.sum()
            #loss2 = 0
            #torch.cuda.synchronize()
            gt_coarse,_ = fps(gt_fine1, K = coarse.shape[1])
            #gt_coarse = gather_points(gt_fine1.transpose(1, 2).contiguous(), furthest_point_sample(gt_fine1, coarse.shape[1])).transpose(1, 2).contiguous()
            #torch.cuda.synchronize()
            loss1, _ = calc_cd(gt_coarse, coarse ,single_directional = True, norm=1, batch_reduction = None)#calc_cd(fine, gt_fine1)
            loss1_, _ =  calc_cd(coarse, gt_coarse, single_directional = True, norm=1, batch_reduction = None)
            loss1 = loss1 + 1.0 * loss1_
            loss1 = (loss1 * mask).sum()/ mask.sum()
            #loss1 = 0
            #torch.cuda.synchronize()
            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean() #+ self.encoder.vq_loss.mean()*0.1
            #torch.cuda.synchronize()
            if ret_fet:
                return (fine1, fine, coarse), (gt.detach().cpu(),gt_fine1.detach().cpu(), gt_coarse.detach().cpu()), (loss3.item(), loss2.item(), loss1.item()), total_train_loss, feat_g, feat_fine, feat_fine1
            return (fine1.detach().cpu(), fine.detach().cpu(), coarse.detach().cpu()), (gt.detach().cpu(),gt_fine1.detach().cpu(), gt_coarse.detach().cpu()), (loss3.item(), loss2.item(), loss1.item()), total_train_loss
        elif ret_fet:
            return (fine1, fine, coarse), feat_g, feat_fine, feat_fine1
        elif is_training:
            #gt, _ = fps(gt, K = fine1.shape[1])
            loss3 = euc_loss(gt, fine1)
            gt_fine1,_ = fps(gt, K = fine.shape[1])
            loss2 = euc_loss(gt_fine1, fine)
            gt_coarse,_ = fps(gt_fine1, K = coarse.shape[1])
            loss1 = euc_loss(gt_coarse, coarse)
            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()
            return (fine1.detach().cpu(), fine.detach().cpu(), coarse.detach().cpu()), (gt.detach().cpu(),gt_fine1.detach().cpu(), gt_coarse.detach().cpu()), (loss3.item(), loss2.item(), loss1.item()), total_train_loss

        else:
            cd_p, cd_t = calc_cd(fine1, gt)
            cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt)

            return {'out1': coarse, 'out2': fine1, 'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse, 'cd_p': cd_p, 'cd_t': cd_t}
def euc_loss(gt, fine):
    return torch.abs(gt - fine).sum(-1).mean() #torch.square
