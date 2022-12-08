import numpy as np
import torch
from torch import nn
import segmentation_models_pytorch as smp

from dice import calc_dice_coef_batch


class LungModel(nn.Module):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image = batch[0]

        # [batch_size, num_channels, height, width]
        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, f'{h}, {w}'

        mask = batch[1]

        # [batch_size, num_classes, height, width]
        assert mask.ndim == 4

        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        dice_coefs = calc_dice_coef_batch(pred_mask, mask)

        stats = dict()
        if stage == 'train':
            stats['loss'] = loss
        stats['dice'] = dice_coefs

        return stats

    def shared_epoch_end(self, outputs, stage):

        metrics = dict()

        # aggregate step metics
        def __clean_tensor(x):
            return x if len(x.shape) > 0 else x[None]

        if stage == 'train':
            metrics['loss'] = np.mean([x['loss'] for x in outputs])

        dice_coef = torch.cat([__clean_tensor(x["dice"]) for x in outputs])

        metrics['dice'] = torch.mean(dice_coef).data.cpu().numpy()

        return metrics

    def training_step(self, batch, optimizer):
        self.model.train()
        optimizer.zero_grad()

        stats = self.shared_step(batch, "train")

        loss = stats['loss']
        loss.backward()
        optimizer.step()

        stats['loss'] = stats['loss'].data.cpu().numpy()
        return stats

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")
