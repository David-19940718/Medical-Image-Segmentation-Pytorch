import os
import cv2
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from runx.logx import logx
from datetime import datetime
from prettytable import PrettyTable
from ptflops import get_model_complexity_info

import torch

from opts import Opts
from metrics import *
from dataloader import get_dataset
from utils.utils import test_time_aug, make_dir, img_predict


def get_id():
    return json.load(open(os.path.join('data', args.dataset, 'dataset.json'), 'r'))['test']


def eval_all_dataset():
    args.net.eval()
    header = ["file_name", "iou", "dc", "pr", "acc", "sp", "se", "auc"]
    total_metrics, gt_list, pd_list, time_list, total_list = {}, [], [], [], []
    for h in header[1:]:
        total_metrics[h] = []
    file_name = get_id()

    with torch.no_grad():
        with tqdm(total=n_test, desc='Test', unit='img', leave=False) as p_bar:
            for index, batch in enumerate(test_loader):
                # load the picture
                image, label = batch['image'], batch['label']
                image = image.to(device=args.device, dtype=torch.float32)
                label = label.to(device=args.device, dtype=torch.float32)

                # statistics inference time
                torch.cuda.synchronize(args.device)
                start = time.time()
                output = args.net(image)
                output = torch.sigmoid(output)
                torch.cuda.synchronize(args.device)
                time_list.append(time.time() - start)

                # save as the numpy array for plot the auc roc curve
                if args.roc:
                    np_output = output.cpu().detach().numpy()[0, 0, :, :]
                    np_label = label.cpu().detach().numpy()[0, 0, :, :]
                    np_output = np.resize(np_output, np_label.shape)
                    gt_list += list(np_label.flatten())
                    pd_list += list(np_output.flatten())  # value between 0. and 1.

                # calculate the metrics
                rows = [file_name[index]]
                for h in header[1:]:
                    score = get_score(output, label, mode=h)
                    total_metrics[h] += [score]
                    rows.append(score)
                total_list.append(rows)
                p_bar.update(image.shape[0])

                # predict and save the result
                image = cv2.imread(os.path.join(args.dir_img, file_name[index]))
                img_predict(args, image, save_path=os.path.join(args.dir_result, file_name[index]))

    # return the results
    if args.roc:
        np.save(os.path.join(args.dir_log, "gt.npy"), gt_list)
        np.save(os.path.join(args.dir_log, "pd.npy"), pd_list)
    for h in header[1:]:
        total_metrics[h] = np.round(np.mean(total_metrics[h]), 4)
    data = pd.DataFrame(total_list)
    data.to_csv(
        os.path.join(os.path.join(args.dir_log, 'scores.csv')),
        header=header,
        index=True,
        mode='w',
        encoding='utf-8'
    )
    fps = np.mean(time_list)
    try:
        flops, params = get_model_complexity_info(
            args.net,
            (args.n_channels, args.height, args.width),
            print_per_layer_stat=False
        )
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            flops, params = 0., 0.
        else:
            raise exception
    results = total_metrics
    results['fps'] = round(1.0 / fps, 0)
    results['flops'] = flops
    results['params'] = params

    return results


if __name__ == '__main__':
    args = Opts().init()
    args.net.load_state_dict(
        torch.load(
            os.path.join(args.dir_log, f'{args.dataset}_{args.arch}_{args.exp_id}.pth'), map_location=args.device
        )
    )
    if args.tta:
        args.net = test_time_aug(args.net, merge_mode='mean')
    make_dir(dir_path=args.dir_result)

    test_loader, n_test, properties = get_dataset(args=args, flag='test')
    args.mean, args.std = properties[0], properties[1]

    logx.initialize(logdir=args.dir_log, coolname=True, tensorboard=True)
    logx.msg('Start testing...\n')
    table = PrettyTable(["key", "value"])
    table.align = 'l'
    infos = {
        'test size': n_test,
        'dataset': args.dataset,
        'experiment id': args.exp_id,
        'checkpoint dir': args.dir_log,
        'test time augmentation': args.tta,
        'device id': args.device.type + ':/' + args.gpus,
        'datetime': datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    }
    for key, value in infos.items():
        table.add_row([key, value])
    logx.msg(str(table) + '\n')

    scores = eval_all_dataset()
    res = PrettyTable(scores.keys())
    res.align = 'l'
    res.add_row(scores.values())
    logx.msg(str(res) + '\n')
