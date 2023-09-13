from datasets import Front3D
from pathlib import Path
from PIL import Image 
import numpy as np

PATH_TO_DEPTHS = '/mnt/hdd/tmp/outputs/depth/'
PATH_TO_DEPTHS = Path(PATH_TO_DEPTHS)
import torch
from torch import nn

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x
class Slice(torch.nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]

class depth_model(torch.nn.Module):

    def __init__(self, input_size):
        super(depth_model, self).__init__()
        features=[24, 48,96, 192]#[48,96, 192, 384]# [96, 192, 384, 768]
        hooks = [2, 5, 8, 11]
        vit_features = 384
        dpt_features = 64
        size = input_size
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.backbone.blocks[hooks[0]].register_forward_hook(get_activation("1"))
        self.backbone.blocks[hooks[1]].register_forward_hook(get_activation("2"))
        self.backbone.blocks[hooks[2]].register_forward_hook(get_activation("3"))
        self.backbone.blocks[hooks[3]].register_forward_hook(get_activation("4"))
        self.fix_backbone()

        self.backbone.act_postprocess1 = torch.nn.Sequential(
            Slice(),
            Transpose(1, 2),
            torch.nn.Unflatten(2, torch.Size([size[0] // 14, size[1] // 14])),
            torch.nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=8,
                stride=8,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
            torch.nn.Conv2d(
                features[0],
                dpt_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=1,
            )

        )

        self.backbone.act_postprocess2 = torch.nn.Sequential(
            Slice(),
            Transpose(1, 2),
            torch.nn.Unflatten(2, torch.Size([size[0] // 14, size[1] // 14])),
            torch.nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
            torch.nn.Conv2d(
                features[1],
                dpt_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=1,
            )
        )

        self.backbone.act_postprocess3 = torch.nn.Sequential(
            Slice(),
            Transpose(1, 2),
            torch.nn.Unflatten(2, torch.Size([size[0] // 14, size[1] // 14])),
            torch.nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.ConvTranspose2d(
                in_channels=features[2],
                out_channels=features[2],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
            torch.nn.Conv2d(
                features[2],
                dpt_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=1,
            )
        )

        self.backbone.act_postprocess4 = torch.nn.Sequential(
            Slice(),
            Transpose(1, 2),
            torch.nn.Unflatten(2, torch.Size([size[0] // 14, size[1] // 14])),
            torch.nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.Conv2d(
                features[3],
                dpt_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=1,
            )
        )
        self.fusion_1 = _make_fusion_block(dpt_features, False)
        self.fusion_2 = _make_fusion_block(dpt_features, False)
        self.fusion_3 = _make_fusion_block(dpt_features, False)
        self.fusion_4 = _make_fusion_block(dpt_features, False)

        self.head = nn.Sequential(
            nn.Conv2d(dpt_features, dpt_features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=0.875, mode="bilinear", align_corners=True),
            nn.Conv2d(dpt_features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity() )
    def forward(self, inputs): #N,C,H,W

        self.backbone(inputs)
        res_1 = self.backbone.act_postprocess1(activations['1'])
        res_2 = self.backbone.act_postprocess2(activations['2'])
        res_3 = self.backbone.act_postprocess3(activations['3'])
        res_4 = self.backbone.act_postprocess4(activations['4'])

        path_4  = self.fusion_4(res_4)
        path_3 = self.fusion_3(path_4, res_3)
        path_2 = self.fusion_2(path_3, res_2)
        path_1 = self.fusion_1(path_2, res_1)

        outputs = self.head(path_1)
        return outputs

    def fix_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
val_dataset = Front3D(split='val', file_list_path='/home/rashed/repos/panoptic-reconstruction/resources/front3d/validation_list_3d.txt', num_samples = -1)

train_dataset = Front3D(split='train', file_list_path='/home/rashed/repos/panoptic-reconstruction/resources/front3d/train_list_3d.txt', num_samples = -1, shuffle=True)
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, 32, shuffle=True, num_workers=12, pin_memory=True,
                        drop_last=True)
val_dataloader = DataLoader(val_dataset, 8, shuffle=False, num_workers=0, pin_memory=True,
                        drop_last=False)

def mse(pred,gt):

    return ((pred.cpu().numpy()-gt.cpu().numpy())**2).sum()


model = depth_model(val_dataset.output_size).float().cuda()
import torch.optim as optim
# loss function and optimizer
loss_fn_2 =  torch.nn.MSELoss() # mean square error
def uncertainty_loss(pred, gt, uct):
    abs_diff = torch.abs(pred - gt)
    #loss = (abs_diff / torch.exp(uct)) + uct
    loss = (2 ** 0.5)* (abs_diff/(uct+1e-8)) + torch.log(uct+1e-8)
    loss = torch.mean(loss, dim=[1, 2, 3])
    loss = torch.mean(loss)
    return loss
loss_fn = uncertainty_loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, 5, 0.1)
num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(num_params)
#train_dataset[0]
for epoch in range(200):
    loss_total = 0 
    loss_total_2 = 0 
    model = model.train()
    model_lr_scheduler.step()
    for iter, batch in enumerate(train_dataloader):
        color = batch[('color',0,0)].float().cuda()
        gt = batch['depth_gt'].float().cuda()

        outputs = model(color)

        loss = loss_fn(outputs[:,0], gt,outputs[:,1])
        loss_2 = loss_fn_2(outputs[:,0], gt).cpu().detach()
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        loss_total += loss.item()
        loss_total_2 += loss_2
        if iter>0 and iter%1 ==0:
            print("train Step: ", loss_total/(iter+1), loss_total_2/((iter+1)))
    print("train epoch: ",loss_total/(iter+1))
    loss_total_val = 0 
    loss_total_val_2 = 0
    model = model.eval()
    for iter, batch in enumerate (val_dataloader):
        with torch.no_grad():
            color = batch[('color',0,0)].float().cuda()
            gt = batch['depth_gt'].float().cuda()

            outputs = model(color)

            loss_ = loss_fn(outputs[:,0], gt,outputs[:,1])
            loss_total_val += loss_.item()
            loss_2 = loss_fn_2(outputs[:,0], gt)
            loss_total_2 += loss_2.cpu().detach()
            if iter>0 and iter%1 ==0:
                print("val Step: ",loss_total_val/(iter+1), loss_total_2/((iter+1) ))
    print("val epoch: ",loss_total_val/(iter+1))

#for item, depth_path in zip(dataset, sorted(PATH_TO_DEPTHS.glob('*.png'))):
#    model = depth_model(dataset.output_size).float().cuda()
#    gt = item['depth_gt']
#    color = item[('color',0,0)].float().cuda()
#    model(color.permute(1,2,0).unsqueeze(0))
#    pred = np.asarray(Image.open(depth_path))