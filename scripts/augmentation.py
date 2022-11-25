import os
import numpy as np
import cv2
import argparse
import sys
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo
import datetime
from FALME import FLAME, FLAMETex
from glob import glob
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from box import Box

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_json

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):

    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan,
                          out_chan,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


class Resnet18(nn.Module):

    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return feat8, feat16, feat32

    def init_weight(self):

        state_dict = modelzoo.load_url(resnet18_url)

        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class ConvBNReLU(nn.Module):

    def __init__(self,
                 in_chan,
                 out_chan,
                 ks=3,
                 stride=1,
                 padding=1,
                 *args,
                 **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan,
                                  n_classes,
                                  kernel_size=1,
                                  bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):

    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan,
                                    out_chan,
                                    kernel_size=1,
                                    bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):

    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4,
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):

    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(
            x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W),
                                 mode='bilinear',
                                 align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W),
                                   mode='bilinear',
                                   align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W),
                                   mode='bilinear',
                                   align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(
                    child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class Augmentation:

    def __init__(self, config, is_cuda=False):
        self.config = config
        self.n_classes = 19
        self.anno_path = self.config.dataset.anno_path
        self.img_root = self.config.dataset.img_root
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.idxs = [1, 2, 3, 4, 5, 10, 11, 12, 13]
        self.seg_net = BiSeNet(n_classes=self.n_classes)
        self.is_cuda = is_cuda
        self.device = 'cpu'
        if self.is_cuda:
            self.device = 'cuda'
            self.seg_net.cuda()

        self.seg_net.load_state_dict(
            torch.load(self.config.segmentation.model_path,
                       map_location=self.device))
        self.seg_net.eval()
        self.flame = FLAME(self.config.photometric).to(self.device)
        self.flametex = FLAMETex(self.config.photometric).to(self.device)

    def __call__(self):
        annos = load_json(self.anno_path)

        for frame in annos["frame_list"]:
            name = frame["name"]
            img_path = os.path.join(self.img_root, name)
            img = cv2.imread(img_path)
            batch_imgs, batch_lnmks, batch_img_masks = [], [], []

            for lb in frame["labels"]:
                box2d = lb["box2d"]
                keypoints = lb["keypoints"]
                tl = np.asarray((box2d['x1'], box2d['y1'])) - 50
                tl[1] -= 60
                br = np.asarray((box2d['x2'], box2d['y2'])) + 50
                br[0] += 30
                cropped_imgs = img[tl[1]:br[1], tl[0]:br[0], :]
                resize_ratio = np.array(
                    (256, 256)) / np.asarray(cropped_imgs.shape[:2])

                cropped_imgs = cv2.resize(cropped_imgs, (512, 512))
                # photometric fitting methods
                lnmks = np.asarray([keypoints[k] for k in keypoints])
                lnmks = lnmks[:, ::-1] - tl
                lnmks *= resize_ratio[::-1]

                img_mask = self.run_seg(cropped_imgs)

                cropped_imgs = cropped_imgs.astype(np.float32)
                img_mask = img_mask.astype(np.float32)
                cropped_imgs = cv2.resize(cropped_imgs, (256, 256))
                img_mask = np.expand_dims(cv2.resize(img_mask, (256, 256)),
                                          axis=-1)
                cropped_imgs = cropped_imgs[:, :, [2, 1, 0]].transpose(2, 0, 1)
                img_mask = img_mask.transpose(2, 0, 1)
                img_mask_bn = np.zeros_like(img_mask)
                img_mask_bn[np.where(img_mask != 0)] = 1.

                # TODO: implement photometric fitting
                batch_imgs.append(
                    torch.from_numpy(cropped_imgs[None, :, :, :]).to(
                        self.device))
                batch_img_masks.append(
                    torch.from_numpy(img_mask_bn[None, :, :, :]).to(
                        self.device))

                lnmks[:, 0] = lnmks[:, 0] / float(cropped_imgs.shape[2]) * 2 - 1
                lnmks[:, 1] = lnmks[:, 1] / float(cropped_imgs.shape[1]) * 2 - 1
                batch_lnmks.append(
                    torch.from_numpy(lnmks)[None, :, :].float().to(self.device))

            batch_imgs = torch.cat(batch_imgs, dim=0)
            batch_img_masks = torch.cat(batch_img_masks, dim=0)
            batch_lnmks = torch.cat(batch_lnmks, dim=0)
            self.run_photometric(batch_imgs, batch_img_masks, batch_lnmks)
        return

    def run_seg(self, cropped_imgs):
        seg_imgs = cv2.resize(cropped_imgs, (512, 512))
        with torch.no_grad():
            seg_imgs = self.to_tensor(seg_imgs)
            seg_imgs = torch.unsqueeze(seg_imgs, 0)
            if self.is_cuda:
                seg_imgs = seg_imgs.cuda()
            out = self.seg_net(seg_imgs)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            vis_parsing_anno = cv2.resize(vis_parsing_anno,
                                          None,
                                          fx=1,
                                          fy=1,
                                          interpolation=cv2.INTER_NEAREST)
            img_mask = np.zeros(shape=(vis_parsing_anno.shape[0],
                                       vis_parsing_anno.shape[1], 1),
                                dtype=np.bool8)
            for idx in self.idxs:
                index = np.where(vis_parsing_anno == idx)
                img_mask[index[0], index[1], :1] = True
        return img_mask

    def weighted_l2_distance(self, verts1, verts2, weights=None):
        if weights is None:
            return torch.sqrt(((verts1 - verts2)**2).sum(2)).mean(1).mean()
        else:
            return torch.sqrt(
                ((verts1 - verts2)**2 * weights).sum(2)).mean(1).mean()

    def _axis_angle_rotation(self, axis: str,
                             angle: torch.Tensor) -> torch.Tensor:
        """
        Return the rotation matrices for one of the rotations about an axis
        of which Euler angles describe, for each value of the angle given.

        Args:
            axis: Axis label "X" or "Y or "Z".
            angle: any shape tensor of Euler angles in radians

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)

        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either X, Y or Z.")

        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    def euler_angles_to_matrix(self, euler_angles: torch.Tensor,
                               convention: str) -> torch.Tensor:

        matrices = [
            self._axis_angle_rotation(c, e)
            for c, e in zip(convention, torch.unbind(euler_angles, -1))
        ]

        # return functools.reduce(torch.matmul, matrices)
        return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

    def batch_orth_proj(self, X, camera):
        '''
            X is N x num_points x 3
        '''
        angles = camera[:, :3]
        R = self.euler_angles_to_matrix(angles, "XYZ")
        X = X @ R
        camera = camera[:, 3:].clone().view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
        shape = X_trans.shape
        # Xn = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
        Xn = (camera[:, :, 0:1] * X_trans)
        return Xn

    def prep_param(self, pose, exp, shape, cam):
        bsize = pose.size(0)
        shape = shape.expand(bsize, -1)
        return pose, exp, shape, cam

    def run_photometric(self, batch_imgs, batch_img_masks, batch_lnmks):
        b, c, h, w = batch_imgs.shape
        batch_imgs = F.interpolate(batch_imgs, [224, 224])
        batch_img_masks = F.interpolate(batch_img_masks, [224, 224])
        batch_img_masks = batch_img_masks.detach().cpu().numpy()
        batch_img_masks = np.transpose(batch_img_masks[0], (1, 2, 0))
        shape = nn.Parameter(torch.zeros(1, 100).float().to(self.device))
        shape = nn.Parameter(shape)
        tex = nn.Parameter(torch.zeros(b, 50).float().to(self.device))
        exp = nn.Parameter(
            torch.zeros(b,
                        self.config.photometric.expression_params).float().to(
                            self.device))
        pose = nn.Parameter(
            torch.zeros(b, self.config.photometric.pose_params).float().to(
                self.device))
        cam = torch.zeros(b, self.config.photometric.camera_params)
        cam[:, 3] = self.config.photometric.cam_prior
        cam = nn.Parameter(cam.float().to(self.device))
        lights = nn.Parameter(torch.zeros(b, 9, 3).float().to(self.device))
        e_opt = torch.optim.Adam([shape, exp, pose, cam, tex, lights],
                                 lr=self.config.photometric.e_lr,
                                 weight_decay=self.config.photometric.e_wd)
        e_opt_rigid = torch.optim.Adam(
            [cam],
            lr=self.config.photometric.e_lr,
            weight_decay=self.config.photometric.e_wd)

        batch_gt_landmark = batch_lnmks

        # TODO: others rigid
        print(batch_gt_landmark.shape)
        xxx
        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        for k in range(200):
            losses = {}
            p, e, s, c = self.prep_param(pose, exp, shape, cam)
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=s,
                                                            expression_params=e,
                                                            pose_params=p,
                                                            cam_params=c)

            trans_vertices = self.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = -trans_vertices[..., 1:]
            landmarks2d = self.batch_orth_proj(landmarks2d, cam)
            landmarks2d[..., 1:] = -landmarks2d[..., 1:]
            landmarks3d = self.batch_orth_proj(landmarks3d, cam)
            landmarks3d[..., 1:] = -landmarks3d[..., 1:]

            losses['landmark'] = self.weighted_l2_distance(
                landmarks2d[:, 17:, :2], batch_gt_landmark[:, 17:, :2],
                self.weights51) * self.config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()
            self.config.post_param(pose, exp, shape, cam)

            loss_info = '----iter: {}, time: {}\n'.format(
                k,
                datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + '{}: {}, '.format(
                    key, float(losses[key]))
            if k % 10 == 0:
                print(loss_info)

            if k % 50 == 0:
                grids = {}
                visind = range(bz)  # [0]
                grids['images'] = torchvision.utils.make_grid(
                    images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind],
                                              landmarks[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind],
                                              landmarks2d[visind]))
                grids['landmarks3d'] = torchvision.utils.make_grid(
                    util.tensor_vis_landmarks(images[visind],
                                              landmarks3d[visind]))

                grid = torch.cat(list(grids.values()), 1)
                grid_image = (grid.numpy().transpose(1, 2, 0).copy() *
                              255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(grid_image, 0),
                                        255).astype(np.uint8)
                print(savefolder)
                cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

        return


def parse_config():
    parser = argparse.ArgumentParser('Argparser for model image generate')
    parser.add_argument('--config', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()
    print('Result imgs generating')
    print(f"Use following config to produce tensorflow graph: {args.config}.")
    config = Box(load_json(args.config))
    aug = Augmentation(config)()
