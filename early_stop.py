import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    # def __call__(self, val_loss, model):
    #     if self.best_score is None:
    #         self.best_score = val_loss
    #         self.best_model_state = model.state_dict()
    #     elif val_loss > self.best_score + self.min_delta:
    #         self.counter += 1
    #         if self.counter >= self.patience:
    #             self.early_stop = True
    #     else:
    #         self.best_score = val_loss
    #         self.best_model_state = model.state_dict()
    #         self.counter = 0
    def __call__(self, val_accuracy, model):
        if self.best_score is None:
            self.best_score = val_accuracy
            self.best_model_state = model.state_dict()
        elif val_accuracy < self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.best_model_state = model.state_dict()
            self.counter = 0
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)

        target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)

        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)


class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Multi-class Focal Loss for classification tasks.
        Args:
            alpha: Scaling factor for balancing the class contribution (float or list of floats).
            gamma: Focusing parameter to reduce the impact of easy samples.
            reduction: Specifies the reduction to apply ('none', 'mean', 'sum').
        """
        super(FocalLossMultiClass, self).__init__()
        if isinstance(alpha, (float, int)):  # Single alpha for all classes
            self.alpha = torch.tensor([alpha])
        elif isinstance(alpha, list):  # Per-class alpha
            self.alpha = torch.tensor(alpha)
        else:
            raise TypeError("Alpha must be a float, int, or list of floats.")

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute the multi-class Focal Loss.
        Args:
            logits: Predicted unnormalized scores (logits) of shape [batch_size, num_classes].
            targets: Ground truth labels (integer indices) of shape [batch_size].
        Returns:
            Focal loss value.
        """
        device = logits.device
        batch_size, num_classes = logits.size()

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]

        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).to(dtype=torch.float32,
                                                                         device=device)  # [batch_size, num_classes]

        # Extract probabilities of the target classes
        pt = (probs * targets_one_hot).sum(dim=-1)  # [batch_size]

        # Compute the focal weight
        focal_weight = (1 - pt) ** self.gamma  # [batch_size]

        # Compute log probabilities
        log_probs = torch.log(pt + 1e-12)  # Add epsilon for numerical stability

        # Apply alpha weighting (optional per-class alpha)
        if len(self.alpha) == 1:
            alpha_t = self.alpha[0]  # Single alpha for all classes
        else:
            alpha_t = self.alpha.to(device)[targets]  # Per-class alpha
        loss = -alpha_t * focal_weight * log_probs  # [batch_size]

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# 定义函数从 train_loader 计算类别样本数量和类别数
def calculate_class_weights(train_loader):
    """
    动态统计类别样本数量并计算权重和类别总数
    :param train_loader: DataLoader, 训练数据加载器
    :return: Tuple[Tensor, int], (权重向量, 类别总数)
    """
    class_counts = Counter()
    for _, labels in train_loader:
        class_counts.update(labels.tolist())  # 统计所有出现的类别

    num_classes = len(class_counts)  # 类别总数
    total_count = sum(class_counts.values())  # 总样本数
    weights = [total_count / (class_counts[i] * num_classes) for i in range(num_classes)]
    # return torch.tensor(weights, dtype=torch.float32)
    return weights