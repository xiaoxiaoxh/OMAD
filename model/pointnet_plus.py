import torch
import torch.nn as nn

from model.pointnet_util import PointNetSetAbstraction, PointNetFeaturePropagation


class PointNetPlus(nn.Module):
    def __init__(self, in_channel=3):
        super(PointNetPlus, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + in_channel, mlp=[128, 128, 128])

        self.fc = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5))

    def forward(self, l0_xyz, l0_points=None):
        l0_xyz = l0_xyz.permute(0, 2, 1).contiguous()  # (B, 3, N)
        if l0_points is not None:
            l0_points = l0_points.permute(0, 2, 1).contiguous()  # (B, C, N)

        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        if l0_points is not None:
            l0_points = self.fp1(l0_xyz, l1_xyz,
                                 torch.cat([l0_xyz, l0_points], 1), l1_points)
        else:
            l0_points = self.fp1(l0_xyz, l1_xyz, l0_xyz, l1_points)

        out = self.fc(l0_points)

        return out


if __name__ == '__main__':
    net = PointNetPlus(3)

    batch = dict()
    batch['P'] = torch.randn(16, 1024, 3)
    out = net(batch['P'])

    print(out.shape)
