import torchio as tio

RESAMPLE_COEF = 4
CROP_SHAPE = (96, 96, 80)

train_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(RESAMPLE_COEF),
    tio.CropOrPad(CROP_SHAPE),
    tio.RandomMotion(p=0.2),
    tio.RandomBiasField(p=0.3),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.RandomNoise(p=0.5),
    tio.RandomFlip(),
    tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    }),
    tio.OneHot()
])

val_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(RESAMPLE_COEF),
    tio.CropOrPad(CROP_SHAPE),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.OneHot()
])
