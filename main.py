import random

from torch.optim import Adam
from torchvision.datasets import ImageFolder

from lib import get_epoch
from datasets import *
# from nets import IndividualLandmarkNet
# from nets_p2p import IndividualLandmarkNet
# from nets_p2p_y4 import IndividualLandmarkNet
# from nets_detect2 import IndividualLandmarkNet
# from nets_detect2_p2py4 import IndividualLandmarkNet
# from nets_detect2_allval import IndividualLandmarkNet
# from nets_detect2_contact import IndividualLandmarkNet
# from nets_double import IndividualLandmarkNet
# from nets_3sizenk import IndividualLandmarkNet
# from nets_partpic import IndividualLandmarkNet
# from nets_3size import IndividualLandmarkNet
# from nets_3size_p2p1024 import IndividualLandmarkNet
from nets_3size_p2p1024_attentionconcat import IndividualLandmarkNet

import os
import argparse
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import Dataset
from torchvision.models import resnet101, ResNet101_Weights,resnet50 ,ResNet50_Weights,DenseNet,DenseNet121_Weights,resnext50_32x4d,ResNeXt50_32X4D_Weights
import json
from torch.utils.tensorboard import SummaryWriter
# from train import train, validation2fuseCGA1024
# from train_p2p import train, validation
# from train_p2p_y4 import train, validation
# from train_p2p_y4_jg345_loss import train, validation
# from train_detect2 import train, validation
# from train_detect2_p2py4 import train, validation
# from train_detect2_allval import train, validation
# from train_double import train, validation
# from train_doubleloss import train, validation
# from train_doubleloss import train, validation
# from train_3sizenk import train, validation
# from train_nloss_y4 import train, validation
from train_nloss_y4_attentionconcat import train, validation

import torchvision.transforms.v2 as transforms
# import torch.optim as optim
from early_stop import EarlyStopping,Focal_Loss,calculate_class_weights,FocalLossMultiClass
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#--model_name NXbirds_CGA_unequiv_PAC_num6 --data_root datasets --dataset NXbirds

def setup_seed(seed=1):
    random.seed(seed)  # 添加这一行
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='PDiscoNet')
    parser.add_argument('--model_name', help='Name under which the model will be saved',
                        default='wildfish73_p2ptrans')  # , required=True
    parser.add_argument('--data_root',
                        help='directory that contains the celeba, cub, or partimagenet folder',
                        default='datasets')  # , required=True
    parser.add_argument('--dataset', help='The dataset to use. Choose celeba, cub, or partimagenet.',
                        default='wildfish_73')  #, required=True
    parser.add_argument('--num_parts', help='number of parts to predict', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)#train
    parser.add_argument('--image_size', default=224, type=int) # 256 for celeba, 448 for cub,  224 for partimagenet
    parser.add_argument('--epochs', default=100, type=int) # 15 for celeba, 28 for cub, 20 for partimagenet
    parser.add_argument('--pretrained_model_path', default='', help='If you want to load a pretrained model,'
                        'specify the path to the model here.')
    parser.add_argument('--save_figures', default=False,
                        help='Whether to save the attention maps to png', action='store_true')
    parser.add_argument('--only_test', default=False, action='store_true', help='Whether to only test the model')
    args = parser.parse_args()

    setup_seed(3407)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(log_dir=f'{args.dataset}/{args.model_name}')
    writer.add_text('Dataset', args.dataset.lower())
    writer.add_text('Device', str(device))
    writer.add_text('Learning rate', str(args.lr))
    writer.add_text('Batch size', str(args.batch_size))
    writer.add_text('Epochs', str(args.epochs))
    writer.add_text('Number of parts', str(args.num_parts))

    with open(f'{args.dataset}/{args.model_name}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    np.random.seed(1)
    data_path = args.data_root + '/' + args.dataset#.lower()
    if args.dataset.lower() == 'celeba':
        dataset_train = CelebA(data_path, 'train', 0.3)
        dataset_val = CelebA(data_path, 'val', 0.3)
        num_cls = 10177
    elif args.dataset.lower() == 'cub':
        dataset_train = CUBDataset(data_path + '/CUB_200_2011', split=1.0, mode='train', image_size=args.image_size)
        dataset_val = CUBDataset(data_path + '/CUB_200_2011', mode='test',
                                 train_samples=dataset_train.trainsamples, image_size=args.image_size)
        num_cls = 200
    elif args.dataset.lower() == 'partimagenet':
        dataset_train = PartImageNetDataset(data_path, mode='train')
        dataset_val = PartImageNetDataset(data_path, mode='test')
        num_cls = 110
    elif args.dataset.lower() == 'nxbirds':
        dataset_train = NXBirdsataset(data_path , split=1.0, mode='train', image_size=args.image_size)
        dataset_val = NXBirdsataset(data_path , mode='test',
                                 train_samples=dataset_train.trainsamples, image_size=args.image_size)
        num_cls = 206
    elif args.dataset.lower() == 'butterfly200_73':
        train_transforms = transforms.Compose([
            transforms.Resize(size=256, antialias=True),  # 调整图像大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ColorJitter(0.1),  # 调整亮度、对比度、饱和度和色调
            transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # 随机仿射变换
            transforms.RandomCrop(args.image_size),  # 随机裁剪
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ToTensor(),  # 将图像转换为 Tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化，缩放到[-1, 1]范围
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(size=256, antialias=True),  # 调整图像大小
            transforms.CenterCrop(size=args.image_size),  # 中心裁剪
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ToTensor(),  # 将图像转换为 Tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])
        # p2p
        # train_transforms = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.RandomCrop(224, padding=8),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])
        # test_transforms = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])
        # p2p448
        # train_transforms = transforms.Compose([
        #     transforms.Resize((550, 550)),
        #     transforms.RandomCrop(448, padding=8),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])
        # test_transforms = transforms.Compose([
        #     transforms.Resize((550, 550)),
        #     transforms.CenterCrop(448),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])
        dataset_train = ImageFolder(os.path.join(data_path, 'train'), transform=train_transforms)
        dataset_val = ImageFolder(os.path.join(data_path, 'val'), transform=test_transforms)
        num_cls = len(dataset_train.classes)
    elif args.dataset.lower() == 'snake35' or args.dataset.lower() == 'wildfish_73':
        # train_transforms = transforms.Compose([
        #     transforms.Resize(size=256, antialias=True),  # 调整图像大小
        #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
        #     transforms.ColorJitter(0.1),  # 调整亮度、对比度、饱和度和色调
        #     transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # 随机仿射变换
        #     transforms.RandomCrop(args.image_size),  # 随机裁剪
        #     transforms.ToDtype(torch.float32, scale=True),
        #     transforms.ToTensor(),  # 将图像转换为 Tensor
        #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化，缩放到[-1, 1]范围
        # ])
        #
        # test_transforms = transforms.Compose([
        #     transforms.Resize(size=256, antialias=True),  # 调整图像大小
        #     transforms.CenterCrop(size=args.image_size),  # 中心裁剪
        #     transforms.ToDtype(torch.float32, scale=True),
        #     transforms.ToTensor(),  # 将图像转换为 Tensor
        #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        # ])
        # p2p
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        # p2p448
        # train_transforms = transforms.Compose([
        #     transforms.Resize((550, 550)),
        #     transforms.RandomCrop(448, padding=8),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])
        # test_transforms = transforms.Compose([
        #     transforms.Resize((550, 550)),
        #     transforms.CenterCrop(448),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])
        dataset_train = ImageFolder(os.path.join(data_path, 'train'), transform=train_transforms)
        dataset_val = ImageFolder(os.path.join(data_path, 'val'), transform=test_transforms)
        num_cls = len(dataset_train.classes)
    elif args.dataset.lower() == 'cub100':
        train_transforms = transforms.Compose([
            transforms.Resize(size=256, antialias=True),  # 调整图像大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ColorJitter(0.1),  # 调整亮度、对比度、饱和度和色调
            transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # 随机仿射变换
            transforms.RandomCrop(args.image_size),  # 随机裁剪
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ToTensor(),  # 将图像转换为 Tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化，缩放到[-1, 1]范围
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(size=256, antialias=True),  # 调整图像大小
            transforms.CenterCrop(size=args.image_size),  # 中心裁剪
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ToTensor(),  # 将图像转换为 Tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])
        dataset_train = ImageFolder(os.path.join(data_path, 'train'), transform=train_transforms)
        dataset_val = ImageFolder(os.path.join(data_path, 'val'), transform=test_transforms)
        num_cls = len(dataset_train.classes)

    elif args.dataset.lower() == '102flowers':
        train_transforms = transforms.Compose([
            transforms.Resize(size=args.image_size, antialias=True),  # Resize image
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip (equivalent to your flip parameter)
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Color variations
            transforms.RandomRotation(degrees=(-20, 20)),  # Rotation (similar to your theta parameter)
            transforms.RandomAffine(
                degrees=0,  # Rotation is handled separately
                translate=(0.1, 0.1),  # Translation (similar to your tx, ty parameters)
                scale=(0.9, 1.1)  # Scaling (similar to your scale parameter)
            ),
            transforms.RandomCrop(args.image_size),  # Random crop
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Optional normalization
        ])

        # Validation/Test Transformations
        test_transforms = transforms.Compose([
            transforms.Resize(size=args.image_size, antialias=True),  # Resize image
            transforms.CenterCrop(args.image_size),  # Center crop
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Optional normalization
        ])
        dataset_train = ImageFolder(os.path.join(data_path, 'train'), transform=train_transforms)
        dataset_val = ImageFolder(os.path.join(data_path, 'val'), transform=test_transforms)
        num_cls = len(dataset_train.classes)
    elif args.dataset.lower() == 'air':
        # dataset_train = PartImageNetDataset(data_path, mode='train')
        # dataset_val = PartImageNetDataset(data_path, mode='test')
        dataset_train = AIR(root=data_path, is_train=True, data_len=None)
        dataset_val = AIR(root=data_path, is_train=False, data_len=None)
        num_cls = 100
    elif args.dataset.lower() == 'car':
        dataset_train = CAR(root=data_path, is_train=True, data_len=None)
        dataset_val = CAR(root=data_path, is_train=False, data_len=None)
        num_cls = 196

    elif args.dataset.lower() == 'df20m':
        # 定义图片文件夹路径和 CSV 文件路径
        image_dir = "datasets/DF20M-images/DF20M"
        train_csv = "datasets/DF20M-images/DF20M-train_metadata_PROD.csv"
        val_csv = "datasets/DF20M-images/DF20M-public_test_metadata_PROD.csv"

        # 定义数据增强/预处理
        # train_transforms = transforms.Compose([
        #     transforms.Resize((224, 224)),  # 调整图片大小
        #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
        #     transforms.ToTensor(),  # 转为张量
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        # ])
        #
        # test_transforms = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        #part_transform
        train_transforms = transforms.Compose([
            transforms.Resize(size=256, antialias=True),  # 调整图像大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ColorJitter(0.1),  # 调整亮度、对比度、饱和度和色调
            transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # 随机仿射变换
            transforms.RandomCrop(args.image_size),  # 随机裁剪
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ToTensor(),  # 将图像转换为 Tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化，缩放到[-1, 1]范围
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(size=256, antialias=True),  # 调整图像大小
            transforms.CenterCrop(size=args.image_size),  # 中心裁剪
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ToTensor(),  # 将图像转换为 Tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])
        # p2p_train_transforms = transforms.Compose([
        #     transforms.Resize((225, 225)),
        #     transforms.RandomCrop((args.image_size,args.image_size), padding=8),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])
        # p2p_test_transforms = transforms.Compose([
        #     transforms.Resize((225, 225)),
        #     transforms.CenterCrop((args.image_size,args.image_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # ])

        # 创建数据集
        dataset_train = DF20M(csv_file=train_csv, image_dir=image_dir, transform=train_transforms)
        dataset_val = DF20M(csv_file=val_csv, image_dir=image_dir, transform=test_transforms)
        # 打印统计信息
        train_species = pd.read_csv(train_csv)['species']
        val_species = pd.read_csv(val_csv)['species']
        all_species = pd.concat([train_species, val_species])
        # print(f"训练图片数: {len(dataset_train)}")
        # print(f"验证图片数: {len(val_species)}")
        # print(f"类别数: {len(all_species.unique())}")
        num_cls = 182
    else:
        raise RuntimeError("Choose celeba, cub, or partimagenet as dataset")

    # train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
    #                                            num_workers=4)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=lambda worker_id: np.random.seed(1 + worker_id),
        drop_last = False
    )

    test_batch = 8
    # val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=test_batch, shuffle=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=test_batch,
        shuffle=False,
        num_workers=4,
        worker_init_fn=lambda worker_id: np.random.seed(1 + worker_id)  # 添加这一行
    )
    # weights = ResNet101_Weights.DEFAULT
    # basenet = resnet101(weights=weights)


    weights = ResNet50_Weights.DEFAULT
    basenet = resnet50(weights=weights)
    # basenet = resnet50(pretrained=False)
    #
    # weights = ResNeXt50_32X4D_Weights.DEFAULT
    # basenet = resnext50_32x4d(weights=weights)

    # for param in basenet.parameters():
    #     param.requires_grad = True
    net = IndividualLandmarkNet(basenet, args.num_parts, num_classes=num_cls,landmark_dropout=0.5)#0.3

    # net.add_module()

    if not os.path.exists(f'./results_{args.model_name}'):
        os.mkdir(f'./results_{args.model_name}')
    model_name=args.model_name

    if args.pretrained_model_path:
        if not os.path.exists(f'./results_{args.model_name}'):
            os.mkdir(f'./results_{args.model_name}')
        net.load_state_dict(torch.load(args.pretrained_model_path))
        print("pretrained")



    net.to(device)

    epoch_leftoff = 0

    if args.only_test:
        args.epochs = 1

    all_losses = []

    high_lr_layers = ["modulation","modulation1","modulation1", "modulation2", "modulation3", "modulation_concat"]
    med_lr_layers = ["fc_class_landmarks", "fc_class_landmarks1", "fc_class_landmarks2", "fc_class_landmarks3",
                     "fc_class_landmarks_concat"]

    # 定义要包含的关键字
    include_keywords = ["proposal_net", "edge_anchors",
                        "reg_mlp1", "reg_mlp2", "reg_mlp3",
                        "conv_block1_part", "conv_block2_part", "conv_block3_part",
                        "conv_block1", "classifier1", "conv_block2", "classifier2", "conv_block3", "classifier3" ,
                        "classifier_concat","PR","PR1","PR2","PR3"]

    # 筛选包含任何关键字的参数
    p2p_params = [name for name, para in net.named_parameters() if
                       any(keyword in name for keyword in include_keywords)]

    # First entry contains parameters with high lr, second with medium lr, third with low lr
    param_dict = [{'params': [], 'lr': args.lr * 100},#args.lr * 100
                  {'params': [], 'lr': args.lr * 10},#args.lr * 10
                  {'params' : [], 'lr': args.lr},#1e-4
                  {'params' : [], 'lr': 2e-3,'weight_decay': 5e-4}]#p2p
    for name, p in net.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name in high_lr_layers:
            param_dict[0]['params'].append(p)#modulation,1e-2
        elif layer_name in med_lr_layers:
            param_dict[1]['params'].append(p)#fc_class_landmarks,1e-3
        elif name in p2p_params:
            param_dict[3]['params'].append(p)##p2p 2e-3
        else:
            param_dict[2]['params'].append(p)#剩余1e-4,baseline,fc_landmarks
    optimizer = torch.optim.Adam(params=param_dict)
    # optimizer = torch.optim.AdamW(params=param_dict)
    # optimizer = optim.SGD(params=param_dict,momentum=0.9, weight_decay=5e-4)
    # optimizer = {
    #     'optimizer_100x': torch.optim.Adam(param_dict[0]['params'], lr=args.lr * 100),
    #     'optimizer_10x': torch.optim.Adam(param_dict[1]['params'], lr=args.lr * 10),
    #     'optimizer_1x': torch.optim.Adam(param_dict[2]['params'], lr=2e-4),
    #     # 'optimizer_p2p': torch.optim.SGD(param_dict[3]['params'], lr=2e-3, momentum=0.9, weight_decay=5e-4),
    #     'optimizer_p2p': torch.optim.Adam(param_dict[3]['params'], lr=2e-3)
    # }

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    # 计算权重
    # class_weights = calculate_class_weights(train_loader)
    # loss_fn = Focal_Loss(weight=class_weights, gamma=2)
    # loss_fn = FocalLossMultiClass(alpha=class_weights, gamma=2)

    loss_hyperparams = {'l_class':5, 'l_pres': 1, 'l_equiv':1, 'l_conc': 1000, 'l_orth': 1,'l_cont':1,'l_all':1,'l_fuse':1}
    # loss_weights = LossWeights(loss_hyperparams).to(device)
    # # 设置优化器，包含网络参数和损失权重
    # optimizer = torch.optim.Adam(
    #     [{'params': loss_weights.parameters(), 'lr': args.lr}] + param_dict
    # )

    if args.dataset.lower() == 'celeba':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)
    elif (args.dataset.lower() == 'cub'or args.dataset.lower() == 'nxbirds'or args.dataset.lower() == 'snake35'
          or args.dataset.lower() == 'cub100'):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0)
        # schedulers = {
        #     # 'scheduler_100x': torch.optim.lr_scheduler.StepLR(optimizer['optimizer_100x'], step_size=5, gamma=0.5),
        #     # 'scheduler_10x': torch.optim.lr_scheduler.StepLR(optimizer['optimizer_10x'], step_size=5, gamma=0.5),
        #     # 'scheduler_1x': torch.optim.lr_scheduler.StepLR(optimizer['optimizer_1x'], step_size=5, gamma=0.5),
        #     # 'scheduler_p2p': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer['optimizer_p2p'], T_max=args.epochs, eta_min=0),
        #     # 'scheduler_100x': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer['optimizer_100x'], T_max=32, eta_min=0),
        #     # 'scheduler_10x': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer['optimizer_10x'], T_max=32, eta_min=0),
        #     # 'scheduler_1x': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer['optimizer_1x'],T_max=32, eta_min=0),
        #     # 'scheduler_p2p': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer['optimizer_p2p'], T_max=32, eta_min=0),
        # self.batchnorm = BatchNorm2d(11)
        #
        # }
    elif args.dataset.lower() == 'partimagenet':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    elif (args.dataset.lower() == 'butterfly200_73' or args.dataset.lower() == 'mushrooms215_82'
          or args.dataset.lower() == 'wildfish_73'):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
        # schedulers = {
        #     'scheduler_100x': torch.optim.lr_scheduler.StepLR(optimizer['optimizer_100x'], step_size=5, gamma=0.5),
        #     'scheduler_10x': torch.optim.lr_scheduler.StepLR(optimizer['optimizer_10x'], step_size=5, gamma=0.5),
        #     'scheduler_1x': torch.optim.lr_scheduler.StepLR(optimizer['optimizer_1x'], step_size=5, gamma=0.5),
        #     'scheduler_p2p': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer['optimizer_p2p'], T_max=args.epochs, eta_min=0)
        # }
    elif args.dataset.lower() == '102flowers':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    elif args.dataset.lower() == 'air':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    elif args.dataset.lower() == 'car':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    early_stopping = EarlyStopping(patience=20, min_delta=0.0001)
    for epoch in range(epoch_leftoff, args.epochs):
        best_val_acc = 0
        if not args.only_test:
            # print('_________')
            print(f'epoch {epoch} train')
            # print(epoch_leftoff )
            if all_losses:
                # net, all_losses = train(net, optimizer, train_loader, device, epoch, 0, loss_fn,
                #                         loss_hyperparams, writer, all_losses)
                net, all_losses = train(net, optimizer, train_loader, device, epoch, 0, loss_fn,
                                        loss_hyperparams, writer,model_name=args.model_name,all_losses=all_losses)#p2py4
                # net, all_losses = train(net, optimizer, train_loader, device, epoch, 0, loss_fn,
                                        # loss_weights(), writer, all_losses)
            else:
                # net, all_losses = train(net, optimizer, train_loader, device, epoch, epoch_leftoff,
                #                                         loss_fn, loss_hyperparams, writer)
                net, all_losses = train(net, optimizer, train_loader, device, epoch, epoch_leftoff,
                                        loss_fn, loss_hyperparams, writer, model_name=args.model_name)  # p2py4
                # # net, all_losses = train(net, optimizer, train_loader, device, epoch, epoch_leftoff,
                                        # loss_fn, loss_weights(), writer)

            scheduler.step()
            # Update schedulers at the end of each epoch or iteration
            # for scheduler_name, scheduler1 in scheduler.items():
            #     scheduler1.step()
            # 打印当前的损失权重
            # print_loss_weights(loss_weights)
            print(f'Validation accuracy in epoch {epoch}:')
            fixed_indices = [10, 20,30,40]  # 需要可视化的固定图片索引
            # validation(device, net, val_loader, epoch, args.model_name, args.save_figures, writer)
            # val_acc=validation(device, net, val_loader, epoch, args.model_name, args.save_figures, writer, loss_fn)
            val_acc = validation(device, net, val_loader, epoch, args.model_name, args.save_figures, writer, loss_fn,fixed_indices)
            # Check early stopping
            # early_stopping(val_acc, net)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break


        # Validation
        else:
            print('Validation accuracy with saved network:')
            # validation(device, net, val_loader, epoch, args.model_name, args.save_figures, writer)
            validation(device, net, val_loader, epoch, args.model_name, args.save_figures, writer,loss_fn)
        torch.save(net.state_dict(), f'./{args.dataset}/{args.model_name}.pt')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), f'./{args.dataset}/{args.model_name}_best.pt')


    writer.close()

if __name__ == "__main__":
    main()


#python /home/nmjgq/PycharmProjects/part_detection-main/mainb.py --model_name cub_detect2 --data_root ./datasets --dataset cub --num_parts 4 --batch_size 16 --image_size 224
