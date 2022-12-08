import os
from pathlib import Path

import SimpleITK as sitk
import torch
import torchio as tio

from transform import train_transform, val_transform


def load_dicom(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    return image_itk


def get_img_dirs(current_dir):
    all_files = os.listdir(current_dir)
    img_dirs = [os.path.join(current_dir, name[:-5]) for name in all_files if name.endswith('.json')]

    if len(img_dirs) > 0:
        return img_dirs

    for name in all_files:
        path = os.path.join(current_dir, name)
        if os.path.isdir(path):
            img_dirs += get_img_dirs(path)

    return img_dirs


def load_data(args):
    img_dirs = get_img_dirs(args.image_path)

    subjects = []
    val_subjects = []

    num_subjects = len(img_dirs)
    num_val_subjects = int(args.val_ratio * num_subjects)
    num_train_subjects = num_subjects - num_val_subjects

    for i, img_dir in enumerate(img_dirs):
        img_itk = load_dicom(img_dir)

        template = os.path.join(args.image_path, 'LUNG1-001/')
        picture_dir = Path(img_dir[:len(template)]).parts[-1]
        mask_dir = os.path.join(args.mask_path, picture_dir)
        mask_file = os.listdir(mask_dir)[0]
        mask_path = os.path.join(mask_dir, mask_file)

        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(img_itk),
            mask=tio.LabelMap(mask_path)
        )

        if i < num_train_subjects:
            subjects.append(subject)
        else:
            val_subjects.append(subject)

    return subjects, val_subjects


def prepare_batch(batch, device):
    inputs = batch['image'][tio.DATA].to(device)
    targets = batch['mask'][tio.DATA].to(device)

    t = targets[:, 1, :, :, :]
    sums = t.sum(axis=(1, 2))
    z_max = torch.max(sums, dim=1).indices

    img_list = []
    mask_list = []

    for i, z in enumerate(z_max):
        img_list.append(inputs[i, :, :, :, z])
        mask_list.append(targets[i, 1, :, :, z])

    images = torch.concat(img_list)
    masks = torch.stack(mask_list)

    return images.view(-1, 1, images.shape[1], images.shape[2]),\
        masks.view(-1, 1, images.shape[1], images.shape[2])


def get_datasets(args):
    train_subjects, val_subjects = load_data(args)
    train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
    val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)

    print('Train dataset size:', len(train_dataset), 'subjects')
    print('Val dataset size:', len(val_dataset), 'subjects')

    return train_dataset, val_dataset
