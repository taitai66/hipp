import torch
from torch.nn import BatchNorm2d, Softmax2d
from torchvision.models.resnet import ResNet
from typing import Tuple
import torch.nn as nn
from clustering import PartsResort
from feature.LSKblock import LSKblock
from feature.CrossAttention import CrossAttention1,CrossAttention2,CrossAttentionC
from feature.CBAM.MODELS.cbam import CBAM
from feature.CAFM import CAFMC
from feature.MCAM import MCAMC
from feature.CGA import CGAFusion
from feature.HRAMi import HRAMi
from feature.CMAB import CMA_Block
from feature.ParNetAttention import ParNetAttention

class IndividualLandmarkNet(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_landmarks: int = 4,
                 num_classes: int = 2000, landmark_dropout: float = 0.3) -> None:
        """
        Parameters
        ----------
        init_model: ResNet
            The pretrained ResNet model
        num_landmarks: int
            Number of landmarks to detect
        num_classes: int
            Number of classes for the classification
        landmark_dropout: float
            Probability of dropping out a given landmark
        """
        super().__init__()

        # The base model
        self.num_landmarks = num_landmarks
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        self.relu = init_model.relu
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4
        self.finalpool = torch.nn.AdaptiveAvgPool2d(1)

        # New part of the model
        self.softmax: Softmax2d = torch.nn.Softmax2d()  # notrain
        self.batchnorm = BatchNorm2d(11)  # notrain
        self.fc_landmarks1 = torch.nn.Conv2d(512, num_landmarks + 1, 1, bias=False)
        self.fc_class_landmarks1 = torch.nn.Linear(512, num_classes, bias=False)
        self.modulation1 = torch.nn.Parameter(torch.ones((1, 512, num_landmarks + 1)))

        self.fc_landmarks2 = torch.nn.Conv2d(1024, num_landmarks + 1, 1, bias=False)
        self.fc_class_landmarks2 = torch.nn.Linear(1024, num_classes, bias=False)
        self.modulation2 = torch.nn.Parameter(torch.ones((1, 1024, num_landmarks + 1)))

        self.fc_landmarks3 = torch.nn.Conv2d(2048, num_landmarks + 1, 1, bias=False)
        self.fc_class_landmarks3 = torch.nn.Linear(2048, num_classes, bias=False)
        self.modulation3 = torch.nn.Parameter(torch.ones((1, 2048, num_landmarks + 1)))

        self.dropout = torch.nn.Dropout(landmark_dropout)  # notrain
        self.dropout_full_landmarks = torch.nn.Dropout1d(landmark_dropout)  # notrain

        self.fc_class_landmarks_concat = torch.nn.Linear(512+1024+2048, num_classes, bias=False)
        self.modulation_concat = torch.nn.Parameter(torch.ones((1,512+1024+2048, num_landmarks + 1)))
        # fuse
        # self.fc_landmarks_concat = torch.nn.Conv2d(2048, num_landmarks + 1, 1, bias=False)

        # self.LSKblock1=LSKblock(512)
        # self.LSKblock2 = LSKblock(1024)
        # self.LSKblock3 = LSKblock(2048)
        # self.cbam1 = CBAM(gate_channels=512, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        # self.cbam2 = CBAM(gate_channels=1024, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        # self.cbam3 = CBAM(gate_channels=2048, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        # self.fusion = CBAM(gate_channels=512 + 1024 + 2048, reduction_ratio=16, pool_types=['avg', 'max'],no_spatial=False)
        # self.fusion1=CrossAttention1()
        # self.fusion2 = CrossAttention2()
        self.fusionc1 = CGAFusion()
        self.fusionc2 = CGAFusion()
        self.fusionc3 = CGAFusion()
        # self.augmentation1 = CBAM(gate_channels=512, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        # self.augmentation2 = CBAM(gate_channels=1024, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        # self.augmentation3 = CBAM(gate_channels=2048, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        # self.augmentation1 = ParNetAttention(channel=512)
        # self.augmentation2 = ParNetAttention(channel=1024)
        # self.augmentation3 = ParNetAttention(channel=2048)

        # # mlp for regularization
        self.reg_mlp1 = nn.Sequential(
            nn.Linear((1024) *(self.num_landmarks+1),1024),
            nn.ELU(inplace=True),
            nn.Linear((1024 ), (1024))
        )
        self.reg_mlp2 = nn.Sequential(
            nn.Linear((1024) * (self.num_landmarks+1), (1024)),
            nn.ELU(inplace=True),
            nn.Linear((1024), (1024 ))
        )
        self.reg_mlp3 = nn.Sequential(
            nn.Linear((1024) * (self.num_landmarks+1), (1024)),
            nn.ELU(inplace=True),
            nn.Linear((1024), (1024))
        )
        self.num_ftrs = 2048
        self.feature_size = 512
        self.part_ftrs=2048

        # self.PR1 = PartsResort(self.num_landmarks + 1, 512)
        # self.PR2 = PartsResort(self.num_landmarks + 1, self.part_ftrs//2)
        # self.PR3 = PartsResort(self.num_landmarks + 1, 2048)
        self.PR = PartsResort(self.num_landmarks + 1, self.part_ftrs//2)

        self.conv_block1_part = nn.Sequential(
            BasicConv1D(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv1D(512,1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block2_part = nn.Sequential(
            BasicConv1D(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv1D(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block3_part = nn.Sequential(
            BasicConv1D(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv1D(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )

        # 对resnet各阶段进行处理，得到分类结果
        # stage 1
        self.conv_block1 = nn.Sequential(  # 512->512->1024
            BasicConv(512, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(self.feature_size,self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier1 =  nn.Sequential(  # 1024->512->200
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, self.feature_size),
            nn.BatchNorm1d(self.feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, num_classes),
        )

        # stage 2
        self.conv_block2 = nn.Sequential(  # 1024->512->1024
            BasicConv(1024, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(self.feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier2 = nn.Sequential(  # 1024->512->200
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, self.feature_size),
            nn.BatchNorm1d(self.feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, num_classes),
        )

        # stage 3
        self.conv_block3 = nn.Sequential(  # 2048->512->1024
            BasicConv(2048, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(self.feature_size,  self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier3 =nn.Sequential(  # 1024->512->200
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, self.feature_size),
            nn.BatchNorm1d(self.feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, num_classes),
        )

        # concat features from different stages
        self.classifier_concat = nn.Sequential(  # 1024*3->512->200
            nn.BatchNorm1d(1024*3),
            nn.Linear(1024*3, self.feature_size),
            nn.BatchNorm1d(self.feature_size),
            nn.ELU(inplace=True),
            nn.Linear(self.feature_size, num_classes),
        )

        #attentionfusion
        self.classifierc1 = nn.Sequential(  # 1024*5->512*5->200
            nn.BatchNorm1d(self.num_ftrs // 2*5),
            nn.Linear(self.num_ftrs // 2*5, self.feature_size*5),
            nn.BatchNorm1d(self.feature_size*5),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size*5, num_classes),
        )
        self.classifierc2 = nn.Sequential(  # 1024->512->200
            nn.BatchNorm1d(self.num_ftrs // 2 * 5),
            nn.Linear(self.num_ftrs // 2 * 5, self.feature_size * 5),
            nn.BatchNorm1d(self.feature_size * 5),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size * 5, num_classes),
        )
        self.classifierc3 = nn.Sequential(  # 1024->512->200
            nn.BatchNorm1d(self.num_ftrs // 2 * 5),
            nn.Linear(self.num_ftrs // 2 * 5, self.feature_size * 5),
            nn.BatchNorm1d(self.feature_size * 5),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_size * 5, num_classes),
        )
        self.classifierc_concat = nn.Sequential(  # 1024*3->512->200
            nn.BatchNorm1d(1024 *3*5),
            nn.Linear(1024 * 3*5, self.feature_size*5),
            nn.BatchNorm1d(self.feature_size*5),
            nn.ELU(inplace=True),
            nn.Linear(self.feature_size*5, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        x: torch.Tensor
            Input image

        Returns
        -------
        all_features: torch.Tensor
            Features per landmark
        maps: torch.Tensor
            Attention maps per landmark
        scores: torch.Tensor
            Classification scores per landmark
        """
        # Pretrained ResNet part of the model
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)  # 512
        l2 = x
        l3 = self.layer3(x)  # 1024
        x = self.layer4(l3)  # 2048
        l4 = x
        #上采样：将较小的特征图放大
        # x = torch.nn.functional.upsample_bilinear(x, size=(l3.shape[-2], l3.shape[-1]))
        # x = torch.cat((x, l3), dim=1)


        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps resnet, a = convolution kernel
        batch_size = x.shape[0]

        # l2=self.augmentation1(fl2)
        ab = self.fc_landmarks1(l2)
        b_sq = l2.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1)
        a_sq = self.fc_landmarks1.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, l2.shape[-2], l2.shape[-1])
        a_sq = a_sq.permute(1, 0, 2, 3)
        maps1 = b_sq - 2 * ab + a_sq
        maps1 = -maps1
        # Softmax so that the attention maps for each pixel add up to 1
        maps1 = self.softmax(maps1)
        # Use maps to get weighted average features per landmark
        feature_tensor = l2
        all_features1 = ((maps1).unsqueeze(1) * feature_tensor.unsqueeze(2)).mean(-1).mean(-1)
        # Classification based on the landmarks
        all_features_modulated = all_features1 * self.modulation1
        all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)
        scores1 = self.fc_class_landmarks1(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)

        # l3=self.augmentation2(fl3)
        ab = self.fc_landmarks2(l3)
        b_sq = l3.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1)
        a_sq = self.fc_landmarks2.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, l3.shape[-2], l3.shape[-1])
        a_sq = a_sq.permute(1, 0, 2, 3)
        maps2 = b_sq - 2 * ab + a_sq
        maps2 = -maps2
        # Softmax so that the attention maps for each pixel add up to 1
        maps2 = self.softmax(maps2)
        # Use maps to get weighted average features per landmark
        feature_tensor = l3
        all_features2 = ((maps2).unsqueeze(1) * feature_tensor.unsqueeze(2)).mean(-1).mean(-1)
        # Classification based on the landmarks
        all_features_modulated = all_features2 * self.modulation2
        all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)
        scores2 = self.fc_class_landmarks2(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)

        # l4=self.augmentation3(fl4)
        ab = self.fc_landmarks3(l4)
        b_sq = l4.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1)
        a_sq = self.fc_landmarks3.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, l4.shape[-2], l4.shape[-1])
        a_sq = a_sq.permute(1, 0, 2, 3)
        maps3 = b_sq - 2 * ab + a_sq
        maps3 = -maps3
        # Softmax so that the attention maps for each pixel add up to 1
        maps3 = self.softmax(maps3)  # (16, 5, 14, 14)
        # Use maps to get weighted average features per landmark
        feature_tensor = l4
        all_features3 = ((maps3).unsqueeze(1) * feature_tensor.unsqueeze(2)).mean(-1).mean(-1)  # (16,2048,5)
        # Classification based on the landmarks
        all_features_modulated = all_features3 * self.modulation3
        all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)
        scores3 = self.fc_class_landmarks3(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)  # (16,200,5)

        # l2_up = torch.nn.functional.upsample_bilinear(l2, size=(l3.shape[-2], l3.shape[-1]))
        # l3_up = torch.nn.functional.upsample_bilinear(l3, size=(l2.shape[-2], l2.shape[-1]))
        # l4_up = torch.nn.functional.upsample_bilinear(l4, size=(l2.shape[-2], l2.shape[-1]))
        # l = torch.cat((l2, l3_up, l4_up), dim=1)
        # l_fuse = self.fusion(l)  # (16,512+1024+2048,28,28)
        # channels = [512, 1024, 2048]  # 通道分割的大小
        # l2, l3, l4 = torch.split(l, channels, dim=1)
        # l_fuse = self.fusion1(l2,l3,l4) # (16,2048,28,28)

        xl1 = self.conv_block1_part(all_features1)  # (16,1024,5)
        xl2 = self.conv_block2_part(all_features2)  # (16,1024,5)
        xl3 = self.conv_block3_part(all_features3)  # (16,1024,5)
        # all_features_concat = self.fusion1(xl1, xl2, xl3)  # (16,2048,28,28)

        # all_features_concat = torch.cat((xl1, xl2, xl3), dim=1)  # 在通道维度拼接，得到 (16, 3072, 5)  # (16,3072,5)
        all_features_concat = torch.cat((all_features1, all_features2, all_features3), dim=1)  # 在通道维度拼接，得到512+1024+2048
        all_features_modulated = all_features_concat * self.modulation_concat
        all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)
        scores_concat = self.fc_class_landmarks_concat(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1) # (16,200,5)

        # fuse
        # l4_up = torch.nn.functional.interpolate(l3, size=(l2.shape[-2], l2.shape[-1]), mode='bilinear', align_corners=True)
        # l2_down = torch.nn.functional.interpolate(l1, size=(l2.shape[-2], l2.shape[-1]), mode='bilinear', align_corners=True)
        # l_fuse = torch.cat((l2_down, l2,l4_up), dim=1)#(16,512+1024+2048,14,14)
        # ab = self.fc_landmarks_concat(l_fuse)
        # b_sq = l_fuse.pow(2).sum(1, keepdim=True)
        # b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1)
        # a_sq = self.fc_landmarks_concat.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, l_fuse.shape[-2], l_fuse.shape[-1])
        # a_sq = a_sq.permute(1, 0, 2, 3)
        # maps_fuse = b_sq - 2 * ab + a_sq
        # maps_fuse = -maps_fuse
        # # Softmax so that the attention maps for each pixel add up to 1
        # maps_fuse = self.softmax(maps_fuse)  # (16, 5, 14, 14)
        # # Use maps to get weighted average features per landmark
        # feature_tensor = l_fuse
        # all_features_fuse = ((maps_fuse).unsqueeze(1) * feature_tensor.unsqueeze(2)).mean(-1).mean(-1)  # (16,2048,5)
        # # Classification based on the landmarks
        # all_features_modulated = all_features_fuse * self.modulation_concat
        # all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)
        # scores_fuse = self.fc_class_landmarks_concat(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)  # (16,200,5)
        #
        #

        # resort parts只用于得到排序后的部分特征，分类用排序前
        feature_points = xl1.transpose(1, 2)  # (16,5,512)
        # feature_points = feature_points.view(batch, self.topn, -1)  # 将 f3_part 的形状调整为 [batch, topn, -1](16,4,1024)
        parts_order = self.PR.classify(feature_points.data.cpu().numpy(), True)  # (16,4)对显著部分的特征进行分类并得到排序顺序
        parts_order = torch.from_numpy(parts_order).long().to(x.device)  # (16,4)
        parts_order = parts_order.unsqueeze(2).expand(batch_size,self.num_landmarks+1, 1024)  # (16,4,1024)
        # 增加一个维度并扩展，使得其形状变为 [batch, topn, num_ftrs//2]。这样做是为了后续的特征收集操作
        # print('parts_order:'+str(parts_order.dtype))
        # print(parts_order.shape)
        f1_points = torch.gather(feature_points.view(batch_size, self.num_landmarks+1, -1), dim=1, index=parts_order)  # (16,5,512)
        # 将 f1_part 的形状调整为 (64,1024)-->[batch, topn, -1](16,4,1024),以便根据 parts_order 提取特定的特征
        # 根据 parts_order 提取排序后的特征。dim=1 指明要在第二维（即 topn 维度）上进行提取。
        # 即 torch.gather 根据 parts_order 中的索引，从 f1_part 张量中选择相应的特征
        xl1 = (f1_points.transpose(1, 2))  # (16,512,k+1)

        feature_points = xl2.transpose(1, 2)  # (16,5,1024)
        parts_order = self.PR.classify(feature_points.data.cpu().numpy(), True)  # (16,4)对显著部分的特征进行分类并得到排序顺序
        parts_order = torch.from_numpy(parts_order).long().to(x.device)  # (16,4)
        parts_order = parts_order.unsqueeze(2).expand(batch_size, self.num_landmarks+1, 1024)  # (16,4,1024)
        f2_points = torch.gather(feature_points.view(batch_size, self.num_landmarks+1, -1), dim=1, index=parts_order)  # (16,5,2048)
        xl2 = (f2_points.transpose(1, 2))  # (16,1024,k+1)

        feature_points = xl3.transpose(1, 2)  # (16,5,2048)
        parts_order = self.PR.classify(feature_points.data.cpu().numpy(), True)  # (16,4)对显著部分的特征进行分类并得到排序顺序
        parts_order = torch.from_numpy(parts_order).long().to(x.device)  # (16,4)
        parts_order = parts_order.unsqueeze(2).expand(batch_size, self.num_landmarks+1, 1024)  # (16,4,2048)
        f3_points = torch.gather(feature_points.view(batch_size, self.num_landmarks+1, -1), dim=1,index=parts_order)  # (16,5,2048)
        xl3 = (f3_points.transpose(1, 2))  # (16,2048,k+1)

        #part(16,1024,5)--(16,1024)  取前四 调制前后？
        f1_m=xl1.transpose(1, 2)#（16，5，512）
        f1_m=self.reg_mlp1( f1_m.reshape(batch_size, -1) )
        f2_m = xl2.transpose(1, 2)
        f2_m = self.reg_mlp2(f2_m.reshape(batch_size, -1) )
        f3_m=xl3.transpose(1, 2)
        f3_m=self.reg_mlp3(f3_m.reshape(batch_size, -1))

        # stage-wise classification
        f1 = self.conv_block1(l2).view(batch_size, -1)  # 特征图调整为形状为 (batch, -1)（16,1024）
        # conv_block1对这些特征图应用的卷积块，用于进一步处理和特征提取。
        f2 = self.conv_block2(l3).view(batch_size, -1)  # （16,1024）
        f3 = self.conv_block3(l4).view(batch_size, -1)  # （16,）

        # f_fuse=self.fusion2(f1,f2,f3) # （16,1024）

        y1 = self.classifier1(f1)  # （16,200）
        y2 = self.classifier2(f2)  # （16,200）
        y3 = self.classifier3(f3)  # （16,200）
        y4 = self.classifier_concat(torch.cat((f1, f2, f3), -1))#1024*3
        # y4 = self.classifier_concat(f_fuse)


        c1 = self.fusionc1(xl1,f1).view(batch_size, -1)# (16, 5* 1024)
        c2 = self.fusionc2(xl2, f2).view(batch_size, -1)
        c3 = self.fusionc3(xl3,f3).view(batch_size, -1)# (16, 5* 1024)
        yc1 = self.classifierc1(c1)  # （16,200）
        yc2 = self.classifierc2(c2)  # （16,200）
        yc3 = self.classifierc3(c3)  # （16,200）
        yc4 = self.classifierc_concat(torch.cat((c1, c2, c3), -1))  # 1024*3



        # return (all_features1, maps1, scores1, all_features2, maps2, scores2, all_features3, maps3, scores3,
        #         all_features_concat, scores_concat,\
        #        f1_m,f2_m,f3_m,f1,f2,f3,
        #         y1,y2,y3,y4,
        #         yc1,yc2,yc3,yc4)
        return (all_features1, maps1, scores1, all_features2, maps2, scores2, all_features3, maps3, scores3,
                all_features_concat, scores_concat,
                f1_m, f2_m, f3_m, f1, f2, f3,
                y1, y2, y3, y4,
                yc1, yc2, yc3, yc4,
                c1, c2, c3)  # 添加 c1, c2, c3


class BasicConv1D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True, bn=True,
                 bias=False):
        super(BasicConv1D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
