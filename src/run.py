import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader

from data import get_datasets, prepare_batch
from model import LungModel


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing pleural effusion segmentation neural network',
        usage='run.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', type=int, default=-1, help='Avaiable GPU ID')

    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--mask_path', type=str, default=None)

    parser.add_argument('--train_batch_size', default=3, type=int)
    parser.add_argument('--val_batch_size', default=6, type=int)
    parser.add_argument('--val_ratio', default=0.3, type=float)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)

    parser.add_argument('-save', '--save_path', default='./output', type=str)
    parser.add_argument('--n_epoch', default=200, type=int)

    return parser.parse_args(args)


def save_model(model, optimizer, epoch, args):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )


def set_logger(args):
    log_file = os.path.join(args.save_path, 'train.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def save_progress_chart(train_dice_list, val_dice_list, args):
    plt.plot(list(range(args.n_epoch)), train_dice_list, label='train')
    plt.plot(list(range(args.n_epoch)), val_dice_list, label='valid')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Dice coef.')
    plt.savefig(os.path.join(args.save_path, 'training.png'))


def main(args):
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args)

    train_dataset, val_dataset = get_datasets(args)

    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.cpu_num,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.cpu_num,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = LungModel("Unet", "resnet34", in_channels=1, out_classes=1).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.learning_rate)
    ])

    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda != -1:
        device = torch.device(f"cuda:{args.cuda}")
        cuda.set_device(device)
        model = model.cuda(device)

    logging.info('Start Training...')

    best_dice = 0
    train_dice_list = []
    val_dice_list = []

    for epoch in np.arange(0, args.n_epoch):

        all_batch_stats = []

        for batch in training_loader:
            images, masks = prepare_batch(batch, device)

            batch_stats = model.training_step((images, masks), optimizer)
            all_batch_stats.append(batch_stats)

        metrics = model.training_epoch_end(all_batch_stats)
        log_metrics('Train', epoch, metrics)
        train_dice_list.append(metrics['dice'])

        val_batch_stats = []
        for batch in val_loader:
            images, masks = prepare_batch(batch, device)
            stats = model.validation_step((images, masks))
            val_batch_stats.append(stats)

        metrics = model.validation_epoch_end(val_batch_stats)
        log_metrics('Valid', epoch, metrics)
        dice_coef = metrics['dice']
        val_dice_list.append(dice_coef)

        if best_dice < dice_coef:
            best_dice = dice_coef
            save_model(model, optimizer, epoch, args)

    save_progress_chart(train_dice_list, val_dice_list, args)


if __name__ == '__main__':
    main(parse_args())
