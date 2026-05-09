"""
Contains functions used for training and testing
"""

# Import statements
import torch
import numpy as np
from lib import rigid_transform, landmark_coordinates, save_maps,save_mapsone
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


# Function definitions
def conc_loss(centroid_x: torch.Tensor, centroid_y: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor,
              maps: torch.Tensor) -> torch.Tensor:
    """
    Calculates the concentration loss, which is the weighted sum of the squared distance of the landmark
    Parameters
    ----------
    centroid_x: torch.Tensor
        The x coordinates of the map centroids
    centroid_y: torch.Tensor
        The y coordinates of the map centroids
    grid_x: torch.Tensor
        The x coordinates of the grid
    grid_y: torch.Tensor
        The y coordinates of the grid
    maps: torch.Tensor
        The attention maps

    Returns
    -------
    loss_conc: torch.Tensor
        The concentration loss
    """
    spatial_var_x = ((centroid_x.unsqueeze(-1).unsqueeze(-1) - grid_x) / grid_x.shape[-1]) ** 2
    spatial_var_y = ((centroid_y.unsqueeze(-1).unsqueeze(-1) - grid_y) / grid_y.shape[-2]) ** 2
    spatial_var_weighted = (spatial_var_x + spatial_var_y) * maps
    loss_conc = spatial_var_weighted[:, 0:-1, :, :].mean()
    return loss_conc


def orth_loss(num_parts: int, landmark_features: torch.Tensor, device) -> torch.Tensor:
    """
    Calculates the orthogonality loss, which is the mean of the cosine similarities between every pair of landmarks
    Parameters
    ----------
    num_parts: int
        The number of landmarks
    landmark_features: torch.Tensor, [batch_size, feature_dim, num_landmarks + 1 (background)]
        Tensor containing the feature vector for each part
    device: torch.device
        The device to use
    Returns
    -------
    loss_orth: torch.Tensor
        The orthogonality loss
    """
    normed_feature = torch.nn.functional.normalize(landmark_features, dim=1)
    similarity = torch.matmul(normed_feature.permute(0, 2, 1), normed_feature)
    similarity = torch.sub(similarity, torch.eye(num_parts + 1).to(device))
    loss_orth = torch.mean(torch.square(similarity))
    return loss_orth

def smooth_CE(logits, label, peak):
    # logits - [batch, num_cls]   unsoftmax
    # label - [batch]
    batch, num_cls = logits.shape
    label_logits = np.zeros(logits.shape, dtype=np.float32) + (1 - peak) / (num_cls - 1)
    ind = ([i for i in range(batch)], list(label.data.cpu().numpy()))
    label_logits[ind] = peak
    smooth_label = torch.from_numpy(label_logits).to(logits.device)

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label)
    loss = torch.mean(-torch.sum(ce, -1))  # batch average

    return loss

def equiv_loss1(X: torch.Tensor, maps: torch.Tensor, net: torch.nn.Module, device: torch.device, num_parts: int) \
        -> torch.Tensor:
    """
    Calculates the equivariance loss, which we calculate from the cosine similarity between the original attention map
    and the inversely transformed attention map of a transformed image.
    Parameters
    ----------
    X: torch.Tensor
        The input image
    maps: torch.Tensor
        The attention maps
    net: torch.nn.Module
        The model
    device: torch.device
        The device to use
    num_parts: int
        The number of landmarks

    Returns
    -------
    loss_equiv: torch.Tensor
        The equivariance loss
    """
    # Forward pass
    angle = np.random.rand() * 180 - 90
    translate = list(np.int32(np.floor(np.random.rand(2) * 100 - 50)))
    scale = np.random.rand() * 0.6 + 0.8
    transf_img = rigid_transform(X, angle, translate, scale, invert=False)
    _, equiv_map, _,_,_,_,_,_,_,_,_,\
        _,_,_,_,_,_,_,_,_,_,_,_,_,_= net(transf_img.to(device))

    # Compare to original attention map, and penalise high difference
    translate = [(t * maps.shape[-1] / X.shape[-1]) for t in translate]
    rot_back = rigid_transform(equiv_map, angle, translate, scale, invert=True)
    num_elements_per_map = maps.shape[-2] * maps.shape[-1]
    orig_attmap_vector = torch.reshape(maps[:, :-1, :, :], (-1, num_parts, num_elements_per_map))
    transf_attmap_vector = torch.reshape(rot_back[:, 0:-1, :, :], (-1, num_parts, num_elements_per_map))
    cos_sim_equiv = F.cosine_similarity(orig_attmap_vector, transf_attmap_vector, -1)
    loss_equiv = 1 - torch.mean(cos_sim_equiv)
    return loss_equiv

def equiv_loss2(X: torch.Tensor, maps: torch.Tensor, net: torch.nn.Module, device: torch.device, num_parts: int) \
        -> torch.Tensor:
    """
    Calculates the equivariance loss, which we calculate from the cosine similarity between the original attention map
    and the inversely transformed attention map of a transformed image.
    Parameters
    ----------
    X: torch.Tensor
        The input image
    maps: torch.Tensor
        The attention maps
    net: torch.nn.Module
        The model
    device: torch.device
        The device to use
    num_parts: int
        The number of landmarks

    Returns
    -------
    loss_equiv: torch.Tensor
        The equivariance loss
    """
    # Forward pass
    angle = np.random.rand() * 180 - 90
    translate = list(np.int32(np.floor(np.random.rand(2) * 100 - 50)))
    scale = np.random.rand() * 0.6 + 0.8
    transf_img = rigid_transform(X, angle, translate, scale, invert=False)
    (_,_,_,_, equiv_map, _ ,_,_,_,_,_,
     _,_,_,_, _, _, _, _, _, _, _, _, _, _)= net(transf_img.to(device))

    # Compare to original attention map, and penalise high difference
    translate = [(t * maps.shape[-1] / X.shape[-1]) for t in translate]
    rot_back = rigid_transform(equiv_map, angle, translate, scale, invert=True)
    num_elements_per_map = maps.shape[-2] * maps.shape[-1]
    orig_attmap_vector = torch.reshape(maps[:, :-1, :, :], (-1, num_parts, num_elements_per_map))
    transf_attmap_vector = torch.reshape(rot_back[:, 0:-1, :, :], (-1, num_parts, num_elements_per_map))
    cos_sim_equiv = F.cosine_similarity(orig_attmap_vector, transf_attmap_vector, -1)
    loss_equiv = 1 - torch.mean(cos_sim_equiv)
    return loss_equiv
def equiv_loss3(X: torch.Tensor, maps: torch.Tensor, net: torch.nn.Module, device: torch.device, num_parts: int) \
        -> torch.Tensor:
    """
    Calculates the equivariance loss, which we calculate from the cosine similarity between the original attention map
    and the inversely transformed attention map of a transformed image.
    Parameters
    ----------
    X: torch.Tensor
        The input image
    maps: torch.Tensor
        The attention maps
    net: torch.nn.Module
        The model
    device: torch.device
        The device to use
    num_parts: int
        The number of landmarks

    Returns
    -------
    loss_equiv: torch.Tensor
        The equivariance loss
    """
    # Forward pass
    angle = np.random.rand() * 180 - 90
    translate = list(np.int32(np.floor(np.random.rand(2) * 100 - 50)))
    scale = np.random.rand() * 0.6 + 0.8
    transf_img = rigid_transform(X, angle, translate, scale, invert=False)
    _,_,_,_, _, _ ,_,equiv_map,_,_,_,\
        _,_,_,_,_,_,_,_,_,_,_,_,_,_= net(transf_img.to(device))

    # Compare to original attention map, and penalise high difference
    translate = [(t * maps.shape[-1] / X.shape[-1]) for t in translate]
    rot_back = rigid_transform(equiv_map, angle, translate, scale, invert=True)
    num_elements_per_map = maps.shape[-2] * maps.shape[-1]
    orig_attmap_vector = torch.reshape(maps[:, :-1, :, :], (-1, num_parts, num_elements_per_map))
    transf_attmap_vector = torch.reshape(rot_back[:, 0:-1, :, :], (-1, num_parts, num_elements_per_map))
    cos_sim_equiv = F.cosine_similarity(orig_attmap_vector, transf_attmap_vector, -1)
    loss_equiv = 1 - torch.mean(cos_sim_equiv)
    return loss_equiv

def log_updated_parameters(model, optimizer, log_file_path):
    """
    记录每次更新的参数及其变化值。

    Args:
        model (nn.Module): 当前模型。
        optimizer (Optimizer): 优化器。
        log_file_path (str): 日志文件路径。
    """
    # 存储优化前的参数值
    old_params = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    # 执行优化
    optimizer.step()

    # 打开文件记录更新
    with open(log_file_path, "a") as log_file:
        log_file.write("\n--- 更新参数 ---\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_value = param.data
                old_value = old_params[name]
                diff = new_value - old_value
                update_magnitude = diff.abs().sum().item()
                if update_magnitude > 1e-6:  # 可设置阈值，忽略过小的变化
                    log_file.write(f"参数名: {name}, 更新值差异: {update_magnitude}\n")

def train(net: torch.nn.Module, optimizer: torch.optim, train_loader: torch.utils.data.DataLoader,
          device: torch.device, epoch: int, epoch_leftoff: int, loss_fn: torch.nn.Module, loss_hyperparams: dict,
          writer: torch.utils.tensorboard.SummaryWriter,model_name: str, all_losses: [float] = None) -> (torch.nn.Module, [float]):
    """
    Model trainer, saves losses to file
    Parameters
    ----------
    net: torch.nn.Module
        The model to train
    optimizer: torch.optim
        Optimizer used for training
    train_loader: torch.utils.data.DataLoader
        Data loader for the training set
    device: torch.device
        The device on which the network is trained
    epoch: int
        Current epoch, used for the running loss
    epoch_leftoff: int
        Starting epoch of the training function, used if a training run was
        stopped at e.g. epoch 10 and then later continued from there
    loss_fn: torch.nn.Module
        Loss function
    loss_hyperparams: dict
        Indicates, per loss, its hyperparameter
    writer: torch.utils.tensorboard.SummaryWriter
        The object to write performance metrics to
    all_losses: [float]
        The list of all running losses, used to display (not backprop)
    Returns
    ----------
    net: torch.nn.Module
        The model with updated weights
    all_losses: [float]
        The list of all running losses, used to display (not backprop)
    """
    # Training
    if all_losses:
        running_loss_conc, running_loss_pres, running_loss_class, running_loss_equiv, running_loss_orth,running_loss_com = all_losses
    elif not all_losses and epoch != 0:
        print(
            'Please pass the losses of the previous epoch to the training function')
    net.train()
    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    top_class1 = []
    top_class2 = []
    top_class3 = []
    top_class_concat = []
    top_class_all = []
    l_class = loss_hyperparams['l_class']
    l_pres = loss_hyperparams['l_pres']
    l_conc = loss_hyperparams['l_conc']
    l_orth = loss_hyperparams['l_orth']
    l_equiv = loss_hyperparams['l_equiv']
    num_correct = [0] * 4
    num_correctc = [0] * 4
    # 定义日志文件路径
    log_files = {
        "stage_1": "results_" + model_name + "/stage_1_updates.log",
        "stage_2": "results_" + model_name + "/stage_2_updates.log",
        "stage_3": "results_" + model_name + "/stage_3_updates.log",
        "stage_4": "results_" + model_name + "/stage_4_updates.log",
        "stage": "results_" + model_name + "/stage_updates.log",
    }
    for i, (X, lab) in enumerate(train_loader):
        lab = lab.to(device)

        optimizer.zero_grad()
        landmark_features1, maps1, scores1, landmark_features2, maps2, scores2,  landmark_features3, maps3, scores3,all_features_concat,scores_concat, \
        f1_m, f2_m,f3_m, f1,f2, f3, y1, y2,y3,y4, yc1, yc2,yc3,yc4 = net(X.to(device))
         # landmark_features1, maps1, scores1,_,_,_,_, _, _,_,_,\
            # f1_m,_,_,f1,_,_,y1,_,_,_= net(X.to(device))
        # Equivariance loss: calculate rotated landmarks distance
        loss_equiv1 =torch.tensor(0) # equiv_loss1(X, maps1, net, device, net.num_landmarks) * l_equiv
        # Classification accuracy
        yp1=scores1[:, :, 0:-1].mean(-1)
        # loss_class1=smooth_CE(yp1, lab, 0.7) * 1
        loss_class1 = loss_fn(scores1[:, :, 0:-1].mean(-1), lab).mean()#
        loss_class1 = loss_class1 * l_class
        loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps1, device)
        loss_conc1 = conc_loss(loc_x, loc_y, grid_x, grid_y, maps1) * l_conc
        loss_orth1 = orth_loss(net.num_landmarks, landmark_features1, device) * l_orth
        loss_pres1 = torch.nn.functional.avg_pool2d(maps1[:, :, 2:-2, 2:-2], 3, stride=1).max(-1)[0].max(-1)[0].max(0)[0].mean()
        loss_pres1 = (1 - loss_pres1) * l_pres
        part_loss1 = loss_conc1 + loss_pres1 + loss_orth1 + loss_equiv1 + loss_class1
        # 对比损失
        p, q = F.log_softmax(f1_m, dim=-1), F.softmax(f1, dim=-1)
        loss_reg1 = torch.mean(-torch.sum(p * q, dim=-1))   # 0.1
        #y
        loss_y1 = smooth_CE(y1, lab, 0.7) * 1
        loss_yc1 = smooth_CE(yc1, lab, 0.7) * 1
        total_loss1=part_loss1+loss_y1+loss_yc1#+loss_reg1
        # total_loss1.backward()
        # optimizer.step()
        #
        # optimizer.zero_grad()
        # _, _, _ , landmark_features2, maps2, scores2,_, _, _,_,_,\
        #     _,f2_m,_,_,f2,_,_,y2,_,_= net(X.to(device))
        loss_equiv2 = torch.tensor(0) #equiv_loss2(X,maps2, net, device, net.num_landmarks) * l_equiv
        yp2 = scores2[:, :, 0:-1].mean(-1)
        # loss_class2=smooth_CE(yp2, lab, 0.8) * 1
        loss_class2 = loss_fn(scores2[:, :, 0:-1].mean(-1), lab).mean()
        loss_class2 = loss_class2 * l_class
        loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps2, device)
        loss_conc2 = conc_loss(loc_x, loc_y, grid_x, grid_y, maps2) * l_conc
        loss_pres2 = torch.nn.functional.avg_pool2d(maps2[:, :, 2:-2, 2:-2], 3, stride=1).max(-1)[0].max(-1)[0].max(0)[0].mean()
        loss_pres2 = (1 - loss_pres2) * l_pres
        loss_orth2 = orth_loss(net.num_landmarks, landmark_features2, device) * l_orth
        part_loss2 = loss_conc2 + loss_pres2 + loss_orth2 + loss_equiv2 + loss_class2
        # 对比损失
        p, q = F.log_softmax(f2_m, dim=-1), F.softmax(f2, dim=-1)
        loss_reg2 = torch.mean(-torch.sum(p * q, dim=-1))
        # y
        loss_y2 = smooth_CE(y2, lab, 0.8) * 1
        loss_yc2 = smooth_CE(yc2, lab, 0.8) * 1
        total_loss2 = part_loss2 +loss_y2+ loss_yc2#+ loss_reg2
        # total_loss2.backward()
        # optimizer.step()
        #
        # optimizer.zero_grad()
        # _, _, _,_, _, _, landmark_features3, maps3, scores3,_,_,\
        #     _,_,f3_m,_,_,f3,_,_,y3,_ = net(X.to(device))
        loss_equiv3= torch.tensor(0) #equiv_loss3(X, maps3, net, device, net.num_landmarks) * l_equiv
        yp3 = scores3[:, :, 0:-1].mean(-1)
        # loss_class3 = smooth_CE(yp3, lab, 0.9) * 1
        loss_class3 = loss_fn(scores3[:, :, 0:-1].mean(-1), lab).mean()
        loss_class3 = loss_class3 * l_class
        loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps3, device)
        loss_conc3 = conc_loss(loc_x, loc_y, grid_x, grid_y, maps3) * l_conc
        loss_pres3= torch.nn.functional.avg_pool2d(maps3[:, :, 2:-2, 2:-2], 3, stride=1).max(-1)[0].max(-1)[0].max(0)[0].mean()
        loss_pres3 = (1 - loss_pres3) * l_pres
        loss_orth3 = orth_loss(net.num_landmarks, landmark_features3, device) * l_orth
        part_loss3 = loss_conc3 + loss_pres3 + loss_orth3 + loss_equiv3 + loss_class3
        #对比损失
        p, q = F.log_softmax(f3_m, dim=-1), F.softmax(f3, dim=-1)
        loss_reg3 = torch.mean(-torch.sum(p * q, dim=-1))  # 0.1
        # y
        loss_y3 = smooth_CE(y3, lab, 0.9) * 1
        loss_yc3 = smooth_CE(yc3, lab, 0.9) * 1
        total_loss3 = part_loss3+loss_y3+ loss_yc3#+ loss_reg3
        # total_loss3.backward()
        # optimizer.step()
        #
        # optimizer.zero_grad()
        # landmark_features1, maps1, scores1,  landmark_features2, maps2, scores2 ,landmark_features3, maps3, scores3,all_features_concat,scores_concat,\
        #     f1_m,f2_m,f3_m,f1,f2,f3,y1,y2,y3,y4 = net(X.to(device))
        preds_concat = scores_concat[:, :, :-1].mean(-1).argmax(dim=1)
        top_class_concat.append((preds_concat == lab).float().mean().cpu())
        yp4 = scores_concat[:, :, 0:-1].mean(-1)
        # loss_class_concat = smooth_CE(yp4, lab, 1) * 1
        loss_class_concat = loss_fn(scores_concat[:, :, 0:-1].mean(-1), lab).mean()
        loss_class_concat = loss_class_concat * l_class*2
        part_loss_concat = loss_class_concat  # +loss_conc3 + loss_pres3 + loss_orth3 + loss_equiv3
        # y
        loss_y4 = smooth_CE(y4, lab, 1) * 1
        loss_yc4 = smooth_CE(yc4, lab, 1) * 1
        total_loss_concat=(part_loss_concat+loss_y4+loss_yc4)*2#
        # total_loss_concat.backward()
        # optimizer.step()

        total_loss=total_loss1+total_loss2+total_loss3+total_loss_concat
        total_loss.backward()
        optimizer.step()

        if i == len(train_loader) - 1:
            log_file_path = log_files['stage']
            log_updated_parameters(net, optimizer, log_file_path)


        # # # KL_loss
        # # optimizer.zero_grad()
        # landmark_features1, maps1, scores1,  landmark_features2, maps2, scores2 ,landmark_features3, maps3, scores3,_,_,\
        #     _,_,_,_,_,_,_,_,_,_= net(X.to(device))
        # batch, sd1, sd2 = scores1.shape[0], scores1.shape[1], scores1.shape[2]
        # # scores1_soft = scores1 - scores1.max(dim=-1, keepdim=True)[0]
        # # scores2_soft = scores2 - scores2.max(dim=-1, keepdim=True)[0]
        # # loss_scores_KL = F.kl_div(F.log_softmax(scores1_soft.reshape(batch, sd1 * sd2), dim=1),
        # #                           F.softmax(scores2_soft.reshape(batch, sd1 * sd2), dim=1),
        # #                           reduction='batchmean')
        # # loss_features_mse = F.mse_loss(landmark_features1, landmark_features2)
        # # loss_features_KL=F.kl_div(F.log_softmax(f1, dim=2), F.softmax(f2, dim=2), reduction='batchmean')
        # # loss_maps_mse = F.mse_loss(maps1, maps2)
        # # 计算每对之间的 MSE
        # loss_score1_2 = F.mse_loss(scores1, scores2)
        # loss_score1_3 = F.mse_loss(scores1, scores3)
        # loss_score2_3 = F.mse_loss(scores2, scores3)
        # # 可以将它们加起来或取平均
        # loss_scores_mse = loss_score1_2 + loss_score1_3 + loss_score2_3  # 或者可以取平均
        loss_com =torch.tensor(0)#loss_scores_mse#*1000#loss_features_mse*1000  # (loss_scores_KL)
        # loss_com.backward()
        # optimizer.step()

        preds1 = scores1[:, :, :-1].mean(-1).argmax(dim=1)
        top_class1.append((preds1 == lab).float().mean().cpu())
        preds2 = scores2[:, :, :-1].mean(-1).argmax(dim=1)
        top_class2.append((preds2 == lab).float().mean().cpu())
        preds3 = scores3[:, :, :-1].mean(-1).argmax(dim=1)
        top_class3.append((preds3 == lab).float().mean().cpu())


        scores_all = (scores1 + scores2+scores3) / 2
        preds_all = scores_all[:, :, :-1].mean(-1).argmax(dim=1)
        top_class_all.append((preds_all == lab).float().mean().cpu())  # 不计算loss
        loss_conc = loss_conc1 + loss_conc2+loss_conc3
        loss_pres = loss_pres1 + loss_pres2+loss_pres3
        loss_orth = loss_orth1 + loss_orth2+loss_orth3
        loss_equiv = loss_equiv1 + loss_equiv2+loss_equiv3
        loss_class = loss_class1 + loss_class2+loss_class3+loss_class_concat
        loss_fuse=loss_yc1+loss_yc2+loss_yc3+loss_yc4

        _, p1 = torch.max(y1.data, 1)
        _, p2 = torch.max(y2.data, 1)
        _, p3 = torch.max(y3.data, 1)
        _, p4 = torch.max(y4.data, 1)
        num_correct[0] += p1.eq(lab.data).cpu().sum()
        num_correct[1] += p2.eq(lab.data).cpu().sum()
        num_correct[2] += p3.eq(lab.data).cpu().sum()
        num_correct[3] += p4.eq(lab.data).cpu().sum()

        _, pc1 = torch.max(yc1.data, 1)
        _, pc2 = torch.max(yc2.data, 1)
        _, pc3 = torch.max(yc3.data, 1)
        _, pc4 = torch.max(yc4.data, 1)
        num_correctc[0] += pc1.eq(lab.data).cpu().sum()
        num_correctc[1] += pc2.eq(lab.data).cpu().sum()
        num_correctc[2] += pc3.eq(lab.data).cpu().sum()
        num_correctc[3] += pc4.eq(lab.data).cpu().sum()


        torch.cuda.empty_cache()
        if epoch == epoch_leftoff and i == 0:
            running_loss_conc = loss_conc.item()
            running_loss_pres = loss_pres.item()
            running_loss_class = loss_class.item()
            running_loss_equiv = loss_equiv.item()
            running_loss_orth = loss_orth.item()
            running_loss_com=loss_com.item()
        else:
            running_loss_conc = 0.99 * running_loss_conc + 0.01 * loss_conc.item()
            running_loss_pres = 0.99 * running_loss_pres + 0.01 * loss_pres.item()
            running_loss_class = 0.99 * running_loss_class + 0.01 * loss_class.item()
            running_loss_equiv = 0.99 * running_loss_equiv + 0.01 * loss_equiv.item()
            running_loss_orth = 0.99 * running_loss_orth + 0.01 * loss_orth.item()
            running_loss_com = 0.99 * running_loss_com + 0.01 * loss_com.item()
        pbar.update()

    top1acc1 = np.mean(np.array(top_class1))
    top1acc2 = np.mean(np.array(top_class2))
    top1acc_concat = np.mean(np.array(top_class_concat))
    top1acc3 = np.mean(np.array(top_class3))
    top1acc_all = np.mean(np.array(top_class_all))
    total = len(train_loader.dataset)  # 1002
    # acc_train =float(num_correct) / total
    acc1 = 100. * float(num_correct[0]) / total
    acc2 = 100. * float(num_correct[1]) / total
    acc3 = 100. * float(num_correct[2]) / total
    acc4 = 100. * float(num_correct[3]) / total
    accc1 = 100. * float(num_correctc[0]) / total
    accc2 = 100. * float(num_correctc[1]) / total
    accc3 = 100. * float(num_correctc[2]) / total
    accc4 = 100. * float(num_correctc[3]) / total
    # writer.add_scalar('Concentration loss1', loss_conc1, epoch)
    # writer.add_scalar('Presence loss1', loss_pres1, epoch)
    # writer.add_scalar('Classification loss1', loss_class1, epoch)
    # writer.add_scalar('Equivariance loss1', loss_equiv1, epoch)
    # writer.add_scalar('Orthogonality loss1', loss_orth1, epoch)
    writer.add_scalar('Training Accuracy1', top1acc1, epoch)
    #
    # writer.add_scalar('Concentration loss2', loss_conc2, epoch)
    # writer.add_scalar('Presence loss2', loss_pres2, epoch)
    # writer.add_scalar('Classification loss2', loss_class2, epoch)
    # writer.add_scalar('Equivariance loss2', loss_equiv2, epoch)
    # writer.add_scalar('Orthogonality loss2', loss_orth2, epoch)
    writer.add_scalar('Training Accuracy2', top1acc2, epoch)
    writer.add_scalar('Training Accuracy3', top1acc3, epoch)
    writer.add_scalar('Training Accuracy', top1acc_all, epoch)
    writer.add_scalar('Training Accuracy_concat', top1acc_concat, epoch)
    writer.add_scalar('Training Accuracy_y4', acc4, epoch)

    writer.add_scalar('Training Accuracyc1', accc1, epoch)
    writer.add_scalar('Training Accuracyc2', accc2, epoch)
    writer.add_scalar('Training Accuracyc3', accc3, epoch)
    writer.add_scalar('Training Accuracyc_concat', accc4, epoch)

    writer.add_scalar('Concentration loss', running_loss_conc, epoch)
    writer.add_scalar('Presence loss', running_loss_pres, epoch)
    writer.add_scalar('Classification loss', running_loss_class/l_class, epoch)
    writer.add_scalar('Equivariance loss', running_loss_equiv, epoch)
    writer.add_scalar('Orthogonality loss', running_loss_orth, epoch)
    writer.add_scalar('Training loss_com', running_loss_com, epoch)
    writer.add_scalar('Training loss_fuse', loss_fuse, epoch)

    result_str = 'Iteration %d | acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f | top1acc_all = %.5f | top1acc_cnocat = %.5f\n' % (
        epoch, acc1, acc2, acc3, acc4, top1acc_all, top1acc_concat)
    print(result_str)
    with open(f'./results_{model_name}/' + '/results_train.txt', 'a') as file:
        file.write(result_str)
    loss_str = 'Iteration %d (train) | total_loss1 = %.5f | total_loss2 = %.5f | total_loss3 = %.5f | total_loss_concat = %.5f\n' % (
        epoch, total_loss1, total_loss2, total_loss3, total_loss_concat)
    print(loss_str)
    with open(f'./results_{model_name}/' + '/results_trainLoss.txt', 'a') as file:
        file.write(loss_str)
    map_str = 'Iteration %d (train) | loss_conc = %.5f | loss_pres = %.5f | loss_equiv = %.5f | loss_orth = %.5f\n' % (
        epoch, loss_conc, loss_pres, loss_equiv, loss_orth)
    print(map_str)
    with open(f'./results_{model_name}/' + '/results_trainLossMap.txt', 'a') as file:
        file.write(map_str)

    pbar.close()
    all_losses = running_loss_conc, running_loss_pres, running_loss_class, running_loss_equiv, running_loss_orth,running_loss_com
    writer.flush()
    return net, all_losses


def validation(device, net, val_loader, epoch, model_name, save_figures, writer,loss_fn,fixed_indices):
    """
    Calculates validation accuracy for trained model, writes it to Tensorboard Summarywriter.
    Also saves figures with attention maps if save_figures is set to True.
    Parameters
    ----------
    device: torch.device
        The device on which the network is loaded
    net: torch.nn.Module
        The model to evaluate
    val_loader: torch.utils.data.DataLoader
        Data loader for the validation set
    epoch: int
        Current epoch, used to save results
    model_name: str
        Name of the model, used to save results
    save_figures: bool
        Whether to save the attention maps
    writer: torch.utils.tensorboard.SummaryWriter
        The object to write metrics to
    """
    net.eval()
    net.to(device)
    pbar = tqdm(val_loader, position=0, leave=True)

    top_class1 = []
    all_scores1 = []
    top_class2 = []
    all_scores2 = []
    top_class3 = []
    all_scores3 = []

    top_class_concat = []
    all_scores_concat = []
    top_class_all = []
    all_labels = []

    all_maxes1 = torch.Tensor().to(device)
    all_maxes2 = torch.Tensor().to(device)
    all_maxes3 = torch.Tensor().to(device)
    all_maxes_concat = torch.Tensor().to(device)
    all_maxes_all = torch.Tensor().to(device)

    total_loss1 = 0.0  # To accumulate the total loss
    total_loss2 = 0.0
    total_loss3 = 0.0
    total_loss_concat = 0.0
    total_loss_all = 0.0

    num_correct = [0] * 5
    num_correctc = [0] * 5
    num_correct_sum=0

    total_batches = len(val_loader)  # Number of batches for averaging the loss
    loss_com=0
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(val_loader)):
            all_features1, maps1, scores1,all_features2, maps2, scores2,all_features3, maps3, scores3,all_features_concat,scores_concat, \
            _, _, _, _, _, _, y1, y2, y3, y4,yc1,yc2,yc3,yc4= net(X.to(device))
            scores1 = scores1.detach().cpu()
            all_scores1.append(scores1)
            scores2 = scores2.detach().cpu()
            all_scores2.append(scores2)
            scores3 = scores3.detach().cpu()
            all_scores3.append(scores3)
            scores_concat = scores_concat.detach().cpu()
            all_scores_concat.append(scores_concat)

            y1 = y1.detach().cpu()
            y2 = y2.detach().cpu()
            y3 = y3.detach().cpu()
            y4 = y4.detach().cpu()
            yc1 = yc1.detach().cpu()
            yc2 = yc2.detach().cpu()
            yc3 = yc3.detach().cpu()
            yc4 = yc4.detach().cpu()

            lab = y
            all_labels.append(lab)

            for j in range(scores1.shape[0]):
                probs1 = scores1[j, :, :-1].mean(-1).softmax(dim=0).cpu()
                preds1 = torch.argmax(probs1, dim=-1).cpu()
                top_class1.append(1 if preds1 == lab[j].cpu() else 0)
                probs2 = scores2[j, :, :-1].mean(-1).softmax(dim=0).cpu()
                preds2 = torch.argmax(probs2, dim=-1).cpu()
                top_class2.append(1 if preds2 == lab[j].cpu() else 0)
                probs3 = scores3[j, :, :-1].mean(-1).softmax(dim=0).cpu()
                preds3 = torch.argmax(probs3, dim=-1).cpu()
                top_class3.append(1 if preds3 == lab[j].cpu() else 0)
                probs_concat = scores_concat[j, :, :-1].mean(-1).softmax(dim=0).cpu()
                preds_concat = torch.argmax(probs_concat, dim=-1).cpu()
                top_class_concat.append(1 if preds_concat == lab[j].cpu() else 0)

                scores_all = (scores1 + scores2+scores3)/ 2
                probs_all= scores_all[j, :, :-1].mean(-1).softmax(dim=0).cpu()
                preds_all = torch.argmax(probs_all, dim=-1).cpu()
                top_class_all.append(1 if preds_all == lab[j].cpu() else 0)

                global_index = i * val_loader.batch_size + j  # 计算全局索引
                if global_index in fixed_indices:  # 检查是否是固定图片
                    # 保存注意力图
                    save_mapsone(X[j:j + 1], maps1[j:j + 1], epoch, model_name + '/map1', device, global_index)
                    save_mapsone(X[j:j + 1], maps2[j:j + 1], epoch, model_name + '/map2', device, global_index)
                    save_mapsone(X[j:j + 1], maps3[j:j + 1], epoch, model_name + '/map3', device, global_index)

            map_max1 = maps1.max(-1)[0].max(-1)[0][:, :-1].detach()
            all_maxes1 = torch.cat((all_maxes1, map_max1), 0)
            map_max2 = maps2.max(-1)[0].max(-1)[0][:, :-1].detach()
            all_maxes2 = torch.cat((all_maxes2, map_max2), 0)
            map_max3 = maps3.max(-1)[0].max(-1)[0][:, :-1].detach()
            all_maxes3 = torch.cat((all_maxes3, map_max3), 0)

            # _, p = torch.max(y3.data, 1)
            _, p1 = torch.max(y1.data, 1)
            _, p2 = torch.max(y2.data, 1)
            _, p3 = torch.max(y3.data, 1)
            _, p4 = torch.max(y4.data, 1)
            _, p5 = torch.max((y1 + y2 + y3 + y4).data, 1)
            # num_correct += p.eq(lab.data).cpu().sum()
            num_correct[0] += p1.eq(lab.data).cpu().sum()
            num_correct[1] += p2.eq(lab.data).cpu().sum()
            num_correct[2] += p3.eq(lab.data).cpu().sum()
            num_correct[3] += p4.eq(lab.data).cpu().sum()
            num_correct[4] += p5.eq(lab.data).cpu().sum()

            _, pc1 = torch.max(yc1.data, 1)
            _, pc2 = torch.max(yc2.data, 1)
            _, pc3 = torch.max(yc3.data, 1)
            _, pc4 = torch.max(yc4.data, 1)
            _, pc5 = torch.max((yc1 + yc2 + yc3 + yc4).data, 1)
            # num_correct += p.eq(lab.data).cpu().sum()
            num_correctc[0] += pc1.eq(lab.data).cpu().sum()
            num_correctc[1] += pc2.eq(lab.data).cpu().sum()
            num_correctc[2] += pc3.eq(lab.data).cpu().sum()
            num_correctc[3] += pc4.eq(lab.data).cpu().sum()
            num_correctc[4] += pc5.eq(lab.data).cpu().sum()

            _, p_sum = torch.max((y1 + y2 + y3 + y4+yc1 + yc2 + yc3 + yc4).data, 1)
            num_correct_sum += p_sum.eq(lab.data).cpu().sum()

            # # Step 1: 计算 maps1 和 maps2 的和，并除以 2
            # maps_all = (maps1 + maps2+maps3) /3
            # map_max_all = maps_all.max(-1)[0].max(-1)[0][:, :-1].detach()
            # all_maxes_all = torch.cat((all_maxes_all, map_max_all), 0)

            # KL_loss
            batch, sd1, sd2 = scores1.shape[0], scores1.shape[1], scores1.shape[2]
            # loss_scores_KL = F.kl_div(F.log_softmax(scores1.reshape(batch, sd1 * sd2), dim=1),
            #                           F.softmax(scores2.reshape(batch, sd1 * sd2), dim=1),
            #                           reduction='batchmean')
            # loss_features_mse = F.mse_loss(all_features1, all_features2)
            # loss_features_KL = F.kl_div(F.log_softmax(f1, dim=2), F.softmax(f2, dim=2), reduction='batchmean')
            # loss_maps_mse = F.mse_loss(maps1, maps2)
            # 计算每对之间的 MSE
            loss_score1_2 = F.mse_loss(scores1, scores2)
            loss_score1_3 = F.mse_loss(scores1, scores3)
            loss_score2_3 = F.mse_loss(scores2, scores3)
            # 可以将它们加起来或取平均
            loss_scores_mse = loss_score1_2 + loss_score1_3 + loss_score2_3  # 或者可以取平均

            loss_com +=loss_scores_mse.item()# loss_features_mse.item()#loss_scores_KL.item()
            # Calculate loss
            loss1 = loss_fn(scores1[:, :, 0:-1].mean(-1), lab).mean()
            total_loss1 += loss1.item()
            loss2 = loss_fn(scores2[:, :, 0:-1].mean(-1), lab).mean()
            total_loss2 += loss2.item()
            loss3 = loss_fn(scores3[:, :, 0:-1].mean(-1), lab).mean()
            total_loss3 += loss3.item()
            loss_concat = loss_fn(scores_concat[:, :, 0:-1].mean(-1), lab).mean()
            total_loss_concat += loss_concat.item()

            loss_all = loss_fn(scores_all[:, :, 0:-1].mean(-1), lab).mean()
            total_loss_all += loss_all.item()

            # Saving the attention maps
            # if save_figures and i % 100 == 0:  # 每100个batch，一个batch=8
            #     save_maps(X, maps1, epoch, model_name, device)
            #     save_maps(X, maps2, epoch, model_name, device)
            #     save_maps(X, maps3, epoch, model_name, device)

    total = len(val_loader.dataset)  # 840
    # acc_test = float(num_correct) / total  # 476/840
    acc1 = float(num_correct[0]) / total
    acc2 = float(num_correct[1]) / total
    acc3 = float(num_correct[2]) / total
    acc4 = float(num_correct[3]) / total
    acc_test = float(num_correct[4]) / total

    accc1 = float(num_correctc[0]) / total
    accc2 = float(num_correctc[1]) / total
    accc3 = float(num_correctc[2]) / total
    accc4 = float(num_correctc[3]) / total
    accc_test = float(num_correctc[4]) / total

    test_acc_ACsum=float(num_correct_sum) / total

    top1acc1 = np.mean(np.array(top_class1))
    writer.add_scalar('Validation Accuracy1', top1acc1, epoch)
    top1acc2 = np.mean(np.array(top_class2))
    writer.add_scalar('Validation Accuracy2', top1acc2, epoch)
    top1acc3 = np.mean(np.array(top_class3))
    writer.add_scalar('Validation Accuracy3', top1acc3, epoch)
    top1acc_concat = np.mean(np.array(top_class_concat))
    writer.add_scalar('Validation Accuracy concat', top1acc_concat, epoch)
    top1acc_all = np.mean(np.array(top_class_all))
    writer.add_scalar('Validation Accuracy', top1acc_all, epoch)
    writer.add_scalar('all Accuracy', acc_test, epoch)

    writer.add_scalar('c1 Accuracy', accc1, epoch)
    writer.add_scalar('c2 Accuracy', accc2, epoch)
    writer.add_scalar('c3 Accuracy', accc3, epoch)
    writer.add_scalar('c4 Accuracy', accc4, epoch)
    writer.add_scalar('c Accuracy', accc_test, epoch)
    writer.add_scalar('Validation ACsum', test_acc_ACsum, epoch)


    avg_loss_com=loss_com/ total_batches
    writer.add_scalar('Validation loss_com', avg_loss_com, epoch)
    avg_loss1 = total_loss1 / total_batches
    writer.add_scalar('Validation loss1', avg_loss1, epoch)
    avg_loss2 = total_loss2 / total_batches
    writer.add_scalar('Validation loss2', avg_loss2, epoch)
    avg_loss3 = total_loss3 / total_batches
    writer.add_scalar('Validation loss3', avg_loss3, epoch)

    avg_loss_concat = total_loss_concat / total_batches
    writer.add_scalar('Validation loss_concat', avg_loss_concat, epoch)
    avg_loss_all = total_loss_all / total_batches
    writer.add_scalar('Validation loss_all', avg_loss_all, epoch)


    result_str = 'Iteration %d | acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f | acc_test = %.5f | top1acc_all = %.5f | top1acc_cnocat = %.5f\n' % (
        epoch, acc1, acc2, acc3, acc4, acc_test, top1acc_all, top1acc_concat)
    print(result_str)
    with open(f'./results_{model_name}/' + '/results_test.txt', 'a') as file:
        file.write(result_str)
    resultc_str = 'Iteration %d | accc1 = %.5f | accc2 = %.5f | accc3 = %.5f | accc4 = %.5f | accc_test = %.5f | top1acc_all = %.5f | top1acc_cnocat = %.5f\n' % (
        epoch, accc1, accc2, accc3, accc4, accc_test, top1acc_all, top1acc_concat)
    print(resultc_str)
    with open(f'./results_{model_name}/' + '/resultsc_test.txt', 'a') as file:
        file.write(resultc_str)

    pbar.close()
    writer.flush()
    return acc_test


if __name__ == "__main__":
    pass