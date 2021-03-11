from runx.logx import logx
from datetime import datetime
from prettytable import PrettyTable

import torch.nn as nn

from opts import Opts
from dataloader import *
from utils.utils import *
from utils.visualizer import Visualizer


def train_net():
    header = ['epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou', 'lr', 'time(s)']
    start_epoch, global_step, best_score, total_list = -1, 1, 0.0, []
    if args.vis:
        viz = Visualizer(port=args.port, env=f"EXP_{args.exp_id}_NET_{args.arch}")

    # Resume the training process
    if args.resume:
        start_epoch = resume(args=args)

    # automatic mixed-precision training
    if args.amp_available:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch + 1, args.epochs):
        args.net.train()

        epoch_loss, epoch_start_time, rows = 0., time(), [epoch + 1]

        # get the current learning rate
        new_lr = get_lr(args=args, epoch=epoch)

        # Training process
        with tqdm(total=n_train, desc=f'Epoch-{epoch + 1}/{args.epochs}', unit='img') as p_bar:
            for batch in train_loader:
                # args.optimizer.zero_grad()
                image, label = batch['image'], batch['label']
                assert image.shape[1] == args.n_channels

                # Prepare the image and the corresponding label.
                image = image.to(device=args.device, dtype=torch.float32)
                mask_type = torch.float32 if args.n_classes == 1 else torch.long
                label = label.to(device=args.device, dtype=mask_type)

                # Forward propagation.
                if args.amp_available:
                    with torch.cuda.amp.autocast():
                        try:
                            output = args.net(image)
                        except RuntimeError as exception:
                            if "out of memory" in str(exception):
                                print("WARNING: out of memory")
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                                exit(0)
                            else:
                                raise exception
                        loss = criterion(output, label)
                else:
                    output = args.net(image)
                    loss = criterion(output, label)

                # visualize the image.
                if args.vis:
                    try:
                        viz.img(name='ground_truth', img_=label[0])
                        tmp = output[0]
                        tmp[tmp > 0.5] = 1.0
                        tmp[tmp < 0.5] = 0.0
                        viz.img(name='prediction', img_=tmp)
                    except ConnectionError:
                        pass

                args.optimizer.zero_grad()
                # Back propagation.
                if args.amp_available:
                    scaler.scale(loss).backward()
                    scaler.step(args.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    args.optimizer.step()

                global_step += 1
                epoch_loss += loss.item()
                logx.add_scalar('Loss/train', loss.item(), global_step)
                p_bar.set_postfix(**{'loss (batch)': loss.item()})
                p_bar.update(image.shape[0])

        # Calculate  the train loss
        train_loss = epoch_loss / (n_train // args.batch_size)
        metrics = {'train_loss': train_loss}
        logx.metric(phase='train', metrics=metrics, epoch=epoch)

        # Validate process
        val_score, val_loss = eval_net(criterion, logx, epoch, val_loader, n_val, args)

        # Update the current learning rate and
        # you should write the monitor metrics in step() if you use the ReduceLROnPlateau scheduler.
        if args.sche != "Poly":
            args.scheduler.step()

        # Calculating and logging the metrics
        metrics = {
            'val_loss': val_loss,
            'iou': val_score['iou'],
            'dc': val_score['dc'],
            'sp': val_score['sp'],
            'se': val_score['se'],
            'acc': val_score['acc'],
        }
        logx.metric(phase='val', metrics=metrics, epoch=epoch)

        # Print the metrics
        print("\033[1;33;44m=============================Evaluation result=============================\033[0m")
        logx.msg("[Train] Loss: %.4f | LR: %.6f" % (train_loss, new_lr))
        logx.msg("[Valid] Loss: %.4f | ACC: %.4f | IoU: %.4f | DC: %.4f" % (
            val_loss, metrics['acc'], metrics['iou'], metrics['dc'],))
        rows += [train_loss, val_loss, metrics['dc'], metrics['iou'], new_lr]

        # Logging the image to tensorboard
        logx.add_image('image', torch.cat([i for i in image], 2), epoch)
        logx.add_image('label/gt', torch.cat([j for j in label], 2), epoch)
        logx.add_image('label/pd', torch.cat([k > 0.5 for k in output], 2), epoch)

        # Update the best score
        best_score, tm = update_score(args, best_score, val_score, logx, epoch, epoch_start_time)
        rows.append(tm)
        total_list.append(rows)

        # Saving the model with relevant parameters
        save_model(args, epoch, new_lr, interval=10)

    data = pd.DataFrame(total_list)
    file_path = os.path.join(os.path.join(args.dir_log, 'metrics.csv'))
    data.to_csv(
        file_path,
        header=header,
        index=False,
        mode='w',
        encoding='utf-8'
    )
    plot_curve(file_path, args.dir_log, show=True)


if __name__ == '__main__':
    args = Opts().init()

    # load the dataset
    train_loader, n_train, properties = get_dataset(args=args, flag='train')
    val_loader, n_val, _ = get_dataset(args=args, flag='val')
    mean, std = properties[0], properties[1]

    # criterion
    criterion = nn.CrossEntropyLoss() if args.n_classes > 1 else args.loss_function

    # initialize the information
    logx.initialize(logdir=args.dir_log, coolname=True, tensorboard=True)
    logx.msg('Start training...\n')

    table = PrettyTable(["key", "value"])
    table.align = 'l'
    infos = {
        'vis': args.vis,
        'seed': args.seed,
        'epoch': args.epochs,
        'data aug': args.aug,
        'resume': args.resume,
        'optimizer': args.optim,
        'dataset': args.dataset,
        'training size': n_train,
        'validation size': n_val,
        'learning rate': args.lr,
        'parameters': args.param,
        'architecture': args.arch,
        'loss function': args.loss,
        'experiment id': args.exp_id,
        'mean std': f'{(mean, std)}',
        'batch size': args.batch_size,
        'output class': args.n_classes,
        'input channel': args.n_channels,
        'number worker': args.num_workers,
        'amp_available': args.amp_available,
        'image size': f'{(args.height, args.width)}',
        'device id': args.device.type + ':/' + args.gpus,
        'date time': datetime.now().strftime('%y-%m-%d-%H-%M-%S'),
    }
    for key, value in infos.items():
        table.add_row([key, value])

    logx.msg(str(table) + '\n')

    train_net()
