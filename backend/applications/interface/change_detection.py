import os.path as osp
import numpy as np
import paddlers as pdrs

from paddlers.transforms import decode_image
from skimage.io import imsave


def Transform_Color(out_dir, names):
    for name in names:
        img = decode_image(osp.join(out_dir, name))
        save_img = np.where(img == 0, img, 255)
        imsave(osp.join(out_dir, name), save_img)


def execute(model_path, data_path, out_dir, names, WINDOW_SIZE, STRIDE):
    predictor = pdrs.deploy.Predictor(model_path, use_gpu=True)
    predictor.slider_predict(
        data_path,
        save_dir=out_dir,
        transforms=None,
        block_size=WINDOW_SIZE,
        overlap=STRIDE)
    Transform_Color(out_dir, names)
