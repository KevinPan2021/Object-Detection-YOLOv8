# the following code is copied directly from
# https://github.com/romulus0914/YOLOv4-PyTorch/blob/master/YOLOv4.py
# not used in this project!


import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

ACTIVATIONS = {
    'mish': Mish(),
    'linear': nn.Identity()
}

class CSPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='mish'):
        super(CSPConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            ACTIVATIONS[activation]
        )

    def forward(self, x):
        return self.conv(x)

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, residual_activation='linear'):
        super(CSPBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.block = nn.Sequential(
            CSPConv(in_channels, hidden_channels, 1),
            CSPConv(hidden_channels, out_channels, 3)
        )

        self.activation = ACTIVATIONS[residual_activation]

    def forward(self, x):
        return self.activation(x+self.block(x))



class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()

        self.downsample_conv = CSPConv(in_channels, out_channels, 3, stride=2)

        self.split_conv0 = CSPConv(out_channels, out_channels, 1)
        self.split_conv1 = CSPConv(out_channels, out_channels, 1)

        self.blocks_conv = nn.Sequential(
            CSPBlock(out_channels, out_channels, in_channels),
            CSPConv(out_channels, out_channels, 1)
        )

        self.concat_conv = CSPConv(out_channels*2, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x

class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPStage, self).__init__()

        self.downsample_conv = CSPConv(in_channels, out_channels, 3, stride=2)

        self.split_conv0 = CSPConv(out_channels, out_channels//2, 1)
        self.split_conv1 = CSPConv(out_channels, out_channels//2, 1)

        self.blocks_conv = nn.Sequential(
            *[CSPBlock(out_channels//2, out_channels//2) for _ in range(num_blocks)],
            CSPConv(out_channels//2, out_channels//2, 1)
        )

        self.concat_conv = CSPConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x

class CSPDarknet53(nn.Module):
    def __init__(self, stem_channels=32, feature_channels=[64, 128, 256, 512, 1024], num_features=1):
        super(CSPDarknet53, self).__init__()

        self.stem_conv = CSPConv(3, stem_channels, 3)

        self.stages = nn.ModuleList([
            CSPFirstStage(stem_channels, feature_channels[0]),
            CSPStage(feature_channels[0], feature_channels[1], 2),
            CSPStage(feature_channels[1], feature_channels[2], 8),
            CSPStage(feature_channels[2], feature_channels[3], 8),
            CSPStage(feature_channels[3], feature_channels[4], 4)
        ])
 
        self.feature_channels = feature_channels
        self.num_features = num_features

    def forward(self, x):
        x = self.stem_conv(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features[-self.num_features:]


def _BuildCSPDarknet53(num_features=3):
    model = CSPDarknet53(num_features=num_features)

    return model, model.feature_channels[-num_features:]



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat([x]+features, dim=1)

        return features



class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale)
        )

    def forward(self, x):
        return self.upsample(x)



class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Downsample, self).__init__()

        self.downsample = Conv(in_channels, out_channels, 3, 2)

    def forward(self, x):
        return self.downsample(x)



class PANet(nn.Module):
    def __init__(self, feature_channels):
        super(PANet, self).__init__()

        self.feature_transform3 = Conv(feature_channels[0], feature_channels[0]//2, 1)
        self.feature_transform4 = Conv(feature_channels[1], feature_channels[1]//2, 1)
        
        self.resample5_4 = Upsample(feature_channels[2]//2, feature_channels[1]//2)
        self.resample4_3 = Upsample(feature_channels[1]//2, feature_channels[0]//2)
        self.resample3_4 = Downsample(feature_channels[0]//2, feature_channels[1]//2)
        self.resample4_5 = Downsample(feature_channels[1]//2, feature_channels[2]//2)

        self.downstream_conv5 = nn.Sequential(
            Conv(feature_channels[2]*2, feature_channels[2]//2, 1),
            Conv(feature_channels[2]//2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2]//2, 1)
        )
        self.downstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
        )
        self.downstream_conv3 = nn.Sequential(
            Conv(feature_channels[0], feature_channels[0]//2, 1),
            Conv(feature_channels[0]//2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0]//2, 1),
            Conv(feature_channels[0]//2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0]//2, 1),
        )

        self.upstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
        )
        self.upstream_conv5 = nn.Sequential(
            Conv(feature_channels[2], feature_channels[2]//2, 1),
            Conv(feature_channels[2]//2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2]//2, 1),
            Conv(feature_channels[2]//2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2]//2, 1)
        )

    def forward(self, features):
        features = [self.feature_transform3(features[0]), self.feature_transform4(features[1]), features[2]]

        downstream_feature5 = self.downstream_conv5(features[2])
        downstream_feature4 = self.downstream_conv4(torch.cat([features[1], self.resample5_4(downstream_feature5)], dim=1))
        downstream_feature3 = self.downstream_conv3(torch.cat([features[0], self.resample4_3(downstream_feature4)], dim=1))

        upstream_feature4 = self.upstream_conv4(torch.cat([self.resample3_4(downstream_feature3), downstream_feature4], dim=1))
        upstream_feature5 = self.upstream_conv5(torch.cat([self.resample4_5(upstream_feature4), downstream_feature5], dim=1))

        return [downstream_feature3, upstream_feature4, upstream_feature5]



class PredictNet(nn.Module):
    def __init__(self, feature_channels, target_channels=255):
        super(PredictNet, self).__init__()

        self.predict_conv = nn.ModuleList([
            nn.Sequential(
                Conv(feature_channels[i]//2, feature_channels[i], 3),
                nn.Conv2d(feature_channels[i], target_channels, 1)
            ) for i in range(len(feature_channels))
        ])

    def forward(self, features):
        predicts = [predict_conv(feature) for predict_conv, feature in zip(self.predict_conv, features)]

        return predicts



class YOLOv4(nn.Module):
    def __init__(self):
        super(YOLOv4, self).__init__()

        # CSPDarknet53 backbone
        self.backbone, feature_channels = _BuildCSPDarknet53()

        # head conv
        self.head_conv = nn.Sequential(
            Conv(feature_channels[-1], feature_channels[-1]//2, 1),
            Conv(feature_channels[-1]//2, feature_channels[-1], 3),
            Conv(feature_channels[-1], feature_channels[-1]//2, 1),
        )

        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling()

        # Path Aggregation Net
        self.panet = PANet(feature_channels)

        # predict
        self.predict_net = PredictNet(feature_channels)
        

    def forward(self, x):
        features = self.backbone(x)
        features[-1] = self.head_conv(features[-1])
        features[-1] = self.spp(features[-1])
        features = self.panet(features)
        predicts = self.predict_net(features)

        return predicts
    
    
    
    
    
    
def main():
    # Setting number of classes and image size 
    num_classes = 20
    IMAGE_SIZE = 416
  
    # Creating model and testing output shape
    model = YOLOv4()
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
    
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    predicts = model(x)
    print('predicts', len(predicts), predicts[0].shape)
    

if __name__ == "__main__": 
    main()
    
    
    
    
    