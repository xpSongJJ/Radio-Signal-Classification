from torch.utils.data import Dataset
import h5py


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, hdf5_path: str, mod_class: list, indexes: list, transform=None):
        self.hdf5_path = hdf5_path
        self.indexes = indexes
        self.mod_class = mod_class
        self.transform = transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        f = h5py.File(self.hdf5_path, 'r')
        x = f['X']
        index = self.indexes[item]
        signal = x[index]
        '''
        # 此操作将数据转成(32, 32, 2)，通道1是I信号，通道2是Q信号。执行此操作需要调整model中的卷积核及其它参数。
        signal1 = np.reshape(signal[:, 0], [32, 32])
        signal1 = np.expand_dims(signal, axis=2)
        signal2 = np.reshape(signal[:, 1], [32, 32])
        signal2 = np.expand_dims(signal, axis=2)
        signal = np.concatenate((signal1, signal2), axis=2)
        '''
        label = self.mod_class[item]
        if self.transform is not None:
            signal = self.transform(signal)

        return signal, label
