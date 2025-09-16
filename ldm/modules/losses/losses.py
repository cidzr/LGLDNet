import torch
import torch.nn.functional as F

def loss_masks(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        rescale: bool = True,
        activated: bool = True,
) -> torch.tensor:
    if rescale:
        outputs = (outputs + 1.0) / 2.0
    targets = (targets + 1.0) / 2.0  # targets from dataset scale from -1 to 1
    loss_bce = bce_loss(outputs, targets, activated=activated)
    loss_dice = dice_loss(outputs, targets, activated=activated)
    return loss_bce, loss_dice

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    activated: bool = False,
):
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    if not activated:
        inputs = F.sigmoid(inputs)
    intersection = (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (2 * intersection + eps) / (denominator + eps)
    return loss.mean()

def iou_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    activated: bool = False,
):
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    if not activated:
        inputs = F.sigmoid(inputs)
    intersection = (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1) - intersection
    loss = 1 - (intersection + eps) / (denominator + eps)
    return loss.mean()

def bce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    activated: bool = False,
):
    if not activated:
        inputs = F.sigmoid(inputs)
    loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    return loss.mean()

def loss_diff_smooth(z, delta=1.):
    """
        Compute 8-neighborhood Huber Total Variation (TV) regularization
        via convolution, preserving (H, W) shape for all directions.

        Args:
            z: Tensor of shape (B, C, H, W)
            delta: Huber threshold

        Returns:
            Scalar TV loss
        """
    B, C, H, W = z.shape
    diffs = []

    # Down: diff between (i+1,j) and (i,j)
    d_down = z[:, :, 1:, :] - z[:, :, :-1, :]  # (B, C, H-1, W)
    d_down = F.pad(d_down, (0, 0, 0, 1))  # pad bottom -> (B, C, H, W)
    diffs.append(d_down)

    # Up
    d_up = z[:, :, :-1, :] - z[:, :, 1:, :]  # (B, C, H-1, W)
    d_up = F.pad(d_up, (0, 0, 1, 0))  # pad top
    diffs.append(d_up)

    # Right
    d_right = z[:, :, :, 1:] - z[:, :, :, :-1]  # (B, C, H, W-1)
    d_right = F.pad(d_right, (0, 1, 0, 0))  # pad right
    diffs.append(d_right)

    # Left
    d_left = z[:, :, :, :-1] - z[:, :, :, 1:]  # (B, C, H, W-1)
    d_left = F.pad(d_left, (1, 0, 0, 0))  # pad left
    diffs.append(d_left)

    # Down-Right
    d_dr = z[:, :, 1:, 1:] - z[:, :, :-1, :-1]  # (B, C, H-1, W-1)
    d_dr = F.pad(d_dr, (0, 1, 0, 1))  # pad right & bottom
    diffs.append(d_dr)

    # Up-Left
    d_ul = z[:, :, :-1, :-1] - z[:, :, 1:, 1:]  # (B, C, H-1, W-1)
    d_ul = F.pad(d_ul, (1, 0, 1, 0))  # pad left & top
    diffs.append(d_ul)

    # Down-Left
    d_dl = z[:, :, 1:, :-1] - z[:, :, :-1, 1:]  # (B, C, H-1, W-1)
    d_dl = F.pad(d_dl, (1, 0, 0, 1))  # pad left & bottom
    diffs.append(d_dl)

    # Up-Right
    d_ur = z[:, :, :-1, 1:] - z[:, :, 1:, :-1]  # (B, C, H-1, W-1)
    d_ur = F.pad(d_ur, (0, 1, 1, 0))  # pad right & top
    diffs.append(d_ur)

    # Huber function
    def huber(x):
        abs_x = x.abs()
        mask = abs_x <= delta
        loss_quad = 0.5 * (x ** 2) / delta
        loss_lin = abs_x - 0.5 * delta
        return torch.where(mask, loss_quad, loss_lin)

    # Compute average Huber loss over 8 directions
    huber_diff = torch.stack([huber(d) for d in diffs], dim=0)
    sloss = huber_diff.sum() / (huber_diff > 0).sum()
    return sloss
