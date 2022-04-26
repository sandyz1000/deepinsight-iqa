# %%
%load_ext autoreload
%autoreload 2
%reload_ext autoreload
# %%
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
# sys.path.append(os.path.realpath(os.path.pardir))
from deepinsight_iqa.diqa.data import get_iqa_datagen
from deepinsight_iqa.diqa.utils.tf_imgutils import image_normalization, image_preprocess
from deepinsight_iqa.diqa.trainer import Trainer
from deepinsight_iqa.cli import parse_config
from deepinsight_iqa.common.utility import set_gpu_limit
set_gpu_limit(10)

job_dir = os.path.realpath(os.path.curdir)

# ## Set image directory and path
# %%
image_dir = "/Volumes/SDM/Dataset/image_quality_data/data"
csv_path = "combine.csv"
# cfg_path = "configs/diqa/default.json"
cfg_path = "configs/diqa/inceptionv3.json"
# cfg_path = "configs/diqa/mobilenet.json"
# cfg_path = "configs/diqa/resnetv2.json"
# resolve_config_path = (lambda cfg_path: Path(os.path.dirname(__file__)) / cfg_path)
# cfg = parse_config(resolve_config_path(cfg_path))
cfg = parse_config(cfg_path)
# %%
from functools import partial, reduce

img_preprocess_fn = partial(
    reduce,
    lambda x, y: y(x),
    [image_preprocess, partial(image_normalization, new_min=0, new_max=1)]
)

train, valid = get_iqa_datagen(
    image_dir,
    os.path.join(image_dir, csv_path),
    do_augment=cfg['use_augmentation'],
    image_preprocess=image_preprocess,
    image_normalization=partial(image_normalization, new_min=0, new_max=1),
    input_size=cfg['input_size'],
    batch_size=cfg['batch_size'],
    channel_dim=cfg['channel_dim'],
    do_train=True
)
# %%
it = iter(train)
(X_dist, X_ref, e_gt, r), Y = next(it)
# %%
plt.imshow(X_ref[0], cmap='gray')
# %%
network = 'objective'
if 'network' in cfg:
    cfg.pop('network')
model_dir = cfg.pop('model_dir', 'weights/diqa')

# %%
# cfg['epochs'] = 3
# cfg["train_bottleneck"] = True
trainer = Trainer(
    train, valid,
    network=network,
    model_dir=model_dir,
    **cfg
)
# %%
diqa = trainer.compile(network='objective')  # Model
diqa.build()
# %%
# trainer.load_weights(diqa, model_path=Path("weights/inception/objective/"))
diqa.load_weights("weights/inception/objective/")
# %%
# trainer.slow_trainer(diqa)
trainer.train(diqa, checkpoint_dir='chk_tmp/customv2/objective')
# %%
diqa.save_weights("weights/inception/objective/", save_format='tf')
# trainer.save_weights(diqa, model_path=Path("weights/diqa/custom/"))

# %%
trainer.bottleneck.trainable = False
subj = trainer.compile(network='subjective')
subj.build()
# %%
trainer.epochs = 20
trainer.train(subj, checkpoint_dir='chk_tmp/inception/subjective/')
# trainer.slow_trainer(subj)
# %%
subj.save_weights("weights/inception/subjective/", save_format='tf')

# %%
subj.load_weights("weights/customv2/subjective/")
# %%
