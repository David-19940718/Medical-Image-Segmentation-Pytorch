import os
import cv2
import torch
from tqdm import tqdm

from opts import Opts
from utils.utils import test_time_aug, make_dir, img_predict


def predict():
    with tqdm(total=args.n_test, desc=f'Predict', unit='img') as p_bar:
        for index, i in enumerate(os.listdir(args.dir_test)):
            save_path = os.path.join(args.dir_result, i)
            image = cv2.imread(os.path.join(args.dir_test, i))
            img_predict(args, image, save_path=save_path)
            p_bar.update(1)


if __name__ == '__main__':
    args = Opts().init()
    args.dir_test = os.path.join(args.dir_data, 'test')
    args.n_test = len(os.listdir(args.dir_test))
    args.net.load_state_dict(
        torch.load(
            os.path.join(args.dir_log, f'{args.dataset}_{args.arch}_{args.exp_id}.pth'), map_location=args.device
        )
    )
    if args.tta:
        args.net = test_time_aug(args.net, merge_mode='mean')
    make_dir(dir_path=args.dir_result)
    predict()
