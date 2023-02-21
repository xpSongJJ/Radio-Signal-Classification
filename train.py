import json
import os
import time
import argparse
import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from utils import split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
from importlib import import_module
from config import Config


def main(args):
    config = Config()
    print(f"using {config.device} device...")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    tb_writer = SummaryWriter(log_dir='./runs/' + args.model + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    # 获得数据所在的文件路径
    data_dir = os.path.dirname(os.path.abspath(args.data_path))
    # train_indexes, train_labels, val_indexes, val_labels都是list，存储的是索引值
    reload_data = True  # 设置是否重新加载数据集
    if os.path.exists(data_dir + '/train_indexes.json') and not reload_data:
        with open(data_dir + '/train_indexes.json') as f:
            train_indexes = json.load(f)
        with open(data_dir + '/train_label.json') as f:
            train_labels = json.load(f)
        with open(data_dir + '/val_indexes.json') as f:
            val_indexes = json.load(f)
        with open(data_dir + '/val_label.json') as f:
            val_labels = json.load(f)
    else:
        # 不分配测试集, 如果修改了ratio参数，需要修改reload_data为True重新加载一次数据集
        train_indexes, train_labels, val_indexes, val_labels = split_data(args.data_path, args.snr,
                                                                          ratio=[0.875, 0.125, 0.],
                                                                          test=False, one_hot=False)
    print("using train data size: {}".format(len(train_labels)))
    print("using valid data size: {}".format(len(val_labels)))
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(hdf5_path=args.data_path,
                              mod_class=train_labels,
                              indexes=train_indexes,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(hdf5_path=args.data_path,
                            mod_class=val_labels,
                            indexes=val_indexes,
                            transform=data_transform["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=config.num_works)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=config.num_works)
    model = import_module('VisionTransformer.' + args.model)
    net = model.net(num_classes=args.num_classes).to(config.device)
    pg = get_params_groups(net, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=net,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=config.device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=net,
                                     data_loader=val_loader,
                                     device=config.device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(net.state_dict(), "./weights/" + args.model + ".pth")
            best_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='convnext', help='convnext or vision_transformer')  # 默认使用convnext网络
    parser.add_argument('--snr', type=int, default=30)
    parser.add_argument('--num-classes', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    # 数据集目录全称
    parser.add_argument('--data-path', type=str,
                        default="../data/2018/GOLD_XYZ_OSC.0001_1024.hdf5")  # hdf5.File()读取文件不支持相对路径

    opt = parser.parse_args()

    main(opt)
