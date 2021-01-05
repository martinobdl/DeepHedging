from flags import FLAGS
from path_generator import BS_Generator, DscGenerator
import torch
from torch.utils.data import Dataset
import numpy as np


class AssetsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, generatorStock, generatorDsc, contract):
        """__init__.

        :param generatorStock:
        :param generatorDsc:
        :param contract: function from [n_assets, n_steps] -> float
        """
        self.generatorDsc = generatorDsc
        self.generatorStock = generatorStock
        self.contract = contract
        self.dates = generatorStock.dates

    def __len__(self):
        """__len__."""
        return self.generatorStock.n_paths

    def __getitem__(self, idx):
        """__getitem__.

        :param idx:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        dsc, _ = self.generatorDsc.generate(idx)
        stk, _ = self.generatorStock.generate(idx)
        x = np.vstack((dsc, stk)).astype(np.float32)
        y = self.contract(x)

        return torch.tensor(x), torch.tensor(y)

    def take(self, n):
        X = []
        # Y1 = []
        Y2 = []
        for i in range(n):
            x, y = self.__getitem__(i)
            X.append(x.unsqueeze(0))
            # Y1.append(y[0].unsqueeze(0))
            Y2.append(y.unsqueeze(0))
        return torch.cat(X, dim=0), torch.cat(Y2, dim=0)


if __name__ == '__main__':
    generatorStock = BS_Generator(n_paths=FLAGS.SMALL_SAMPLE)
    generatorDsc = DscGenerator(n_paths=FLAGS.SMALL_SAMPLE)

    def contract(x):
        return np.maximum(x[1, -1] - FLAGS.SPOT, 0).astype(np.float32)

    df = AssetsDataset(generatorStock, generatorDsc, contract)
    x, (_, y) = df[np.random.randint(len(df))]
