import os
import sys
import pickle
import random
import math
import h5py
import torch
from tqdm import tqdm


def split_data(root: str, snr: int, ratio: list, test: bool = False):
    # 合法检测 ################
    if len(ratio) != 3:
        ratio = [0.7, 0.1, 0.2]
    list_sum = 0
    for i in ratio:
        list_sum += i
    assert list_sum == 1, "ratio 列表和必须为1."
    ##########################

    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    f = h5py.File(root, 'r')
    y, z = f['Y'], f['Z']
    indexes = [index for index, value in enumerate(z) if value == snr]  # 某个信噪比数据的全部索引
    mod_class = []  # 每一个索引对应的调制类别
    mod_num = 0  # 每种调制样本个数
    for index in indexes:
        if y[index, 0] == 1:
            mod_num += 1
        for i in range(24):
            if y[index, i] == 1:
                mod_class.append(i)

    train_indexes = []  # 训练集索引
    train_label = []  # 存储训练集图片对应索引信息
    val_indexes = []  # 验证集索引
    val_label = []  # 验证集对应的标签索引
    test_indexes = []  # 存储验证集的所有图片路径
    test_label = []  # 存储验证集图片对应索引信息
    if test:
        for i in range(24):
            start_index = int(mod_num*(1-ratio[2])) + i*mod_num
            end_index = (i + 1)*mod_num
            for j in range(start_index, end_index):
                test_indexes.append(indexes[j])
                test_label.append(mod_class[j])
        return test_indexes, test_label
    else:
        for i in range(24):
            start_index = i*mod_num  # 训练集（包括验证集）起始索引
            end_index = int(mod_num*(1-ratio[2])) + i*mod_num  # 训练集（包括验证集）结束索引

            train_index = random.sample(range(start_index, end_index), int(mod_num*ratio[0]))
            for j in range(start_index, end_index):
                if j in train_index:
                    train_indexes.append(indexes[j])
                    train_label.append(mod_class[j])
                else:
                    val_indexes.append(indexes[j])
                    val_label.append(mod_class[j])
        return train_indexes, train_label, val_indexes, val_label


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


