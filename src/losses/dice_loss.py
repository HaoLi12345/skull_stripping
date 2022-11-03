import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_3d(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 5D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"
    target = target.view(target.size(0), target.size(1), target.size(2), -1)
    input = input.view(input.size(0), input.size(1), input.size(2), -1)
    probs = F.softmax(input, dim=1)
    # a = F.softmax(input, dim=1)
    # print((probs.cpu().detach().numpy() - a.cpu().detach().numpy()).sum())

    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)  #
    num = torch.sum(num, dim=0)  # b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c,1,1
    den1 = torch.sum(den1, dim=0)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c,1,1
    den2 = torch.sum(den2, dim=0)

    dice = 2 * ((num + 0.0000001) / (den1 + den2 + 0.0000001))

    # # dice = dice[1:]  # we ignore bg dice val, and take the fg
    # dice_total = torch.sum(1 - dice) / 1  # divide by batch_sz
    # dice_total = dice_total
    # dice_eso = dice[1:]  # we ignore bg dice val, and take the fg
    # dice_total = -1 * torch.sum(dice_eso) / 1  # divide by batch_sz
    # dice_total = dice_total

    dice_eso = dice  # we ignore bg dice val, and take the fg
    dice_total = 1 - torch.mean(dice_eso) / 1  # divide by batch_sz
    dice_total = dice_total

    ### For compute the each organ's training dice coefficients
    dice_right_thalamus = dice[1]
    dice_right_thalamus = torch.sum(dice_right_thalamus)  # divide by channels

    dice_right_caudate = dice[2]
    dice_right_caudate = torch.sum(dice_right_caudate)

    dice_right_pallidum = dice[3]
    dice_right_pallidum = torch.sum(dice_right_pallidum)

    dice_right_putamen = dice[4]
    dice_right_putamen = torch.sum(dice_right_putamen)

    dice_left_thalamus = dice[5]
    dice_left_thalamus = torch.sum(dice_left_thalamus)  # divide by channels

    dice_left_caudate = dice[6]
    dice_left_caudate = torch.sum(dice_left_caudate)

    dice_left_pallidum = dice[7]
    dice_left_pallidum = torch.sum(dice_left_pallidum)

    dice_left_putamen = dice[8]
    dice_left_putamen = torch.sum(dice_left_putamen)
    return dice_total
    # return dice_total, dice_right_thalamus, dice_right_caudate, dice_right_pallidum, dice_right_putamen, \
    #        dice_left_thalamus, dice_left_caudate, dice_left_pallidum, dice_left_putamen

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)


    return 2 * (intersect / denominator.clamp(min=epsilon))

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, sigmoid_normalization=False):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)
        # return torch.sum(1. - per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, weight=None, sigmoid_normalization=False):
        super().__init__(weight, sigmoid_normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, sigmoid_normalization=False, epsilon=1e-6):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())




def test_dice():

    # model1d = torch.nn.Linear(nodes, num_class).cuda()
    model3d = torch.nn.Conv3d(16, 9, 3, padding=1).cuda()

    for i in range(1):

        input = torch.rand(3, 16, 32, 32, 32).cuda()
        target = torch.rand(3, 9, 32, 32, 32).random_(2).cuda()
        target = target.long().cuda()
        output = model3d(input)
        # output = F.softmax(output, dim=1)
        loss = dice_loss_3d(output, target)
        m = DiceLoss()
        loss1 = m(output, target)
        n = GeneralizedDiceLoss()
        loss2 = n(output, target)

        print(loss.item())
        print(loss1.item())
        print(loss2.item())

if __name__ == '__main__':
    test_dice()