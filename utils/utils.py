import os
import cv2
import torch
import random
import warnings
import numpy as np
import pandas as pd
import ttach as tta
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from albumentations import Resize

from metrics import *
from utils.preprocess import get_properties


def setup_seed(seed):
    """
    support to reproduce an experiment.
    :param seed: random seed
    :return: None
    """
    if seed != 0:
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        os.environ['PYTHONHASHSEED'] = str(seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        torch.backends.cudnn.benchmark = True


def test_time_aug(net, merge_mode='mean'):
    """
    More operations please assess to this url: https://github.com/qubvel/ttach
    """
    print("Using the test time augmentation! [Default: HorizontalFlip]")
    trans = tta.Compose(
        [
            tta.HorizontalFlip(),
            # tta.Rotate90(angles=[0, 180]),
            # tta.Scale(scales=[1, 2]),
            # tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )
    net = tta.SegmentationTTAWrapper(net, trans, merge_mode=merge_mode)
    return net


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_mean_std(args):
    results = get_properties(args=args)
    mean, std = results['mean'], results['sd']
    return mean, std


def resume(args):
    checkpoint = torch.load(os.path.join(args.dir_ckpt, 'INTERRUPT.pth'))
    args.net.load_state_dict(checkpoint["net"])
    try:
        args.scheduler.load_state_dict(checkpoint["scheduler"])
    except AttributeError:
        pass
    args.optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint['epoch']
    return start_epoch


def get_lr(args, epoch):
    def poly_lr(e, max_epochs, initial_lr, exponent=0.9):
        return initial_lr * (1 - e / max_epochs) ** exponent

    if args.sche == "Poly":
        new_lr = poly_lr(epoch, args.epochs + 1, args.lr, 0.9)
        args.optimizer.param_groups[0]['lr'] = new_lr
    else:
        new_lr = args.optimizer.state_dict()['param_groups'][0]['lr']

    return new_lr


def img_process(args, image):
    mean, std = get_mean_std(args)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    aug = Resize(args.height, args.width)
    augmented = aug(image=image)
    image = augmented['image']
    if image.max() > 1:
        image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)
    return image


def img_predict(args, image, save_path, threshold=0.5):
    h, w, _ = image.shape
    args.net.eval()
    img = torch.from_numpy(img_process(args=args, image=image))
    img = img.unsqueeze(0)
    img = img.to(device=args.device, dtype=torch.float32)

    with torch.no_grad():
        # predict the image
        output = args.net(img)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()
        # save the binary image
        output = cv2.resize(output, (w, h))
        output = output > threshold
        output = output.astype(np.int) * 255
        cv2.imwrite(save_path, output)


def save_model(args, epoch, lr, interval=10):
    if epoch % interval == 0:
        checkpoint = {
            "net": args.net.state_dict(),
            "optimizer": args.optimizer.state_dict(),
            "scheduler": args.scheduler.state_dict() if args.sche != "Poly" else lr,
            "epoch": epoch + 1,
        }
        torch.save(checkpoint, os.path.join(args.dir_log, 'INTERRUPT.pth'))


def update_score(args, best_score, val_score, logx, epoch, epoch_start_time):
    t = time() - epoch_start_time
    if best_score < val_score['dc']:
        torch.save(args.net.state_dict(), f'{args.dir_log}/{args.dataset}_{args.arch}_{args.exp_id}.pth')
        logx.msg("val_score improved from {:.4f} to {:.4f} "
                 "and the epoch {} took {:.2f}s.\n".
                 format(best_score, val_score['dc'], epoch + 1, t))
        best_score = val_score['dc']
    else:
        logx.msg("val_score did not improved from {:.4f} "
                 "and the epoch {} took {:.2f}s.\n".
                 format(best_score, epoch + 1, t))
    return best_score, t


def eval_net(criterion, logx, epoch, val_loader, n_val, args):
    dice, sensitivity, specificity, iou, accuracy = 0., 0., 0., 0., 0.
    step, epoch_loss = 0, 0.

    with torch.no_grad():
        with tqdm(total=n_val, desc='Validation', unit='img', leave=False) as p_bar:
            args.net.eval()

            for batch in val_loader:
                image = batch['image']
                label = batch['label']

                image = image.to(device=args.device, dtype=torch.float32)
                data_type = torch.float32 if args.n_classes == 1 else torch.long
                label = label.to(device=args.device, dtype=data_type)

                output = args.net(image)
                loss = criterion(output, label)

                step += 1
                logx.add_scalar('Loss/val', loss.item(), epoch * n_val / image.shape[0] + step)
                epoch_loss += loss.item()

                iou += get_iou_score(output, label)
                dice += get_f1_score(output, label)
                accuracy += get_accuracy(output, label)
                sensitivity += get_sensitivity(output, label)
                specificity += get_specificity(output, label)

                p_bar.update(image.shape[0])

    num = n_val // image.shape[0]
    val_score = {
        'iou': iou / num,
        'dc': dice / num,
        'acc': accuracy / num,
        'sp': specificity / num,
        'se': sensitivity / num,
    }
    mean_epoch_loss = epoch_loss / num

    return val_score, mean_epoch_loss


def plot_curve(file_path, save_path, show=False):
    data = pd.read_csv(file_path)
    epoch = list(data['epoch'])
    train_loss = list(data['train_loss'])
    val_loss = list(data['val_loss'])

    plt.title('epoch vs. loss')
    plt.plot(epoch, train_loss, label='train_loss')
    plt.plot(epoch, val_loss, label='val_loss')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    if show:
        plt.show()
