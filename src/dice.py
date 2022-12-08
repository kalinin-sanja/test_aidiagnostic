import torch


def calc_dice_coef_batch(mask1, mask2):
    intersect = torch.sum(mask1*mask2, dim=(2,3)).squeeze()
    fsum = torch.sum(mask1, dim=(2,3)).squeeze()
    ssum = torch.sum(mask2, dim=(2,3)).squeeze()
    denom = fsum + ssum

    # If both masks are empty then dice=1
    intersect[denom == 0] = 1
    denom[denom == 0] = 2

    dice = (2 * intersect ) / denom
    return dice
