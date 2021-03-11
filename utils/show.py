import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from opts import Opts
from utils.utils import make_dir, img_process


args = Opts().init()


def out():
    args.net.eval()
    img = torch.from_numpy(img_process(args=args, image=image))
    img = img.unsqueeze(0)
    img = img.to(device=args.device, dtype=torch.float32)

    with torch.no_grad():
        output = args.net(img)
        output = torch.sigmoid(output)
        output = output.squeeze(0)
        output = output.cpu().detach().numpy()
        output = np.mean(output, axis=0)
        output = cv2.resize(output, (w, h))

        save_file = os.path.join(save_path, save_name)
        plt.axis('off')
        # plt.colorbar()  # 显示颜色条
        plt.imsave(fname=save_file, arr=output, cmap='jet')  # plt.cm.hot / jet / ...


if __name__ == '__main__':
    img_name = "00077.png"
    img_path = os.path.join(args.dir_img, img_name)
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    save_path = os.path.join(os.getcwd(), 'visualization')
    save_name = f'{args.arch}_{args.exp_id}_layer3_after_ccm.png'
    make_dir(save_path)

    args.net.load_state_dict(
        torch.load(
            os.path.join(args.dir_log, f'{args.dataset}_{args.arch}_{args.exp_id}.pth'), map_location=args.device
        )
    )

    out()
