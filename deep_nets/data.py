import os
import torch
import torch.utils.data as td
import numpy as np
from torchvision import datasets, transforms
from robustbench.data import load_cifar10c


class DatasetWithLabelNoise(torch.utils.data.Dataset):
    def __init__(self, data, split, transform):
        self.data = data
        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        x = self.data.data[index]
        x1 = self.transform(x) if self.transform is not None else x
        if self.split == 'train':
            x2 = self.transform(x) if self.transform is not None else x
        else:  # to save a bit of computations
            x2 = x1
        y = self.data.targets[index]
        y_correct = self.data.targets_correct[index]
        label_noise = self.data.label_noise[index]
        return x1, x2, y, y_correct, label_noise

    def __len__(self):
        return len(self.data.targets)


def uniform_noise(*args, **kwargs):
    shape = [1000, 1, 28, 28]
    x = torch.from_numpy(np.random.rand(*shape)).float()
    # y_train = np.random.randint(0, 10, size=shape_train[0])
    y = np.floor(10 * x[:, 0, 0, 0].numpy())  # take the first feature
    y = torch.from_numpy(y).long()
    data = td.TensorDataset(x, y)
    return data


def dataset_gaussians_binary(*args, **kwargs):
    shape = shapes_dict['gaussians_binary']
    n, d = shape[0], shape[3]
    std = 0.1

    v = v_global.copy()
    v /= (v**2).sum()**0.5  # make it unit norm
    mu_zero, mu_one = v, -v

    x = np.concatenate([mu_zero + std*np.random.randn(n // 2, d), mu_one + std*np.random.randn(n // 2, d)])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    indices = np.random.permutation(np.arange(n))
    x, y = x[indices], y[indices]
    x = x[:, None, None, :]  # make it image-like

    data = td.TensorDataset()
    data.data, data.targets = torch.from_numpy(x).float(), torch.from_numpy(y).long()
    return data


def dataset_cifar10c(*args, **kwargs):
    dir_ = args[0]
    x, y = load_cifar10c(data_dir=dir_, n_examples=150000, severity=5, shuffle=True)

    data = td.TensorDataset()
    data.data, data.targets = x, y.long()
    return data


def asym_label_noise(dataset, label):
    if dataset == 'cifar10':
        if label == 9:
            return 1
        # bird -> airplane
        elif label == 2:
            return 0
        # cat -> dog
        elif label == 3:
            return 5
        # dog -> cat
        elif label == 5:
            return 3
        # deer -> horse
        elif label == 4:
            return 7
        else:
            return label
    elif dataset == 'cifar100':
        return (label + 1) % 100
    elif dataset == 'svhn':
        return (label + 1) % 10
    else:
        raise ValueError('This dataset does not yet support asymmetric label noise.')


def get_loaders(dataset, n_ex, batch_size, split, shuffle, data_augm, val_indices=None, p_label_noise=0.0,
                noise_type='sym', drop_last=False):
    dir_ = '/tmlscratch/andriush/data'  
    # dir_ = '/tmldata1/andriush/data'
    dataset_f = datasets_dict[dataset]
    batch_size = n_ex if n_ex < batch_size and n_ex != -1 else batch_size
    num_workers_train, num_workers_val, num_workers_test = 4, 4, 4

    data_augm_transforms = [transforms.RandomCrop(32, padding=4)]
    if dataset not in ['mnist', 'svhn']:
        data_augm_transforms.append(transforms.RandomHorizontalFlip())
    base_transforms = [transforms.ToPILImage()] if dataset != 'gaussians_binary' else []
    transform_list = base_transforms + data_augm_transforms if data_augm else base_transforms
    transform = transforms.Compose(transform_list + [transforms.ToTensor()])

    if dataset == 'cifar10_horse_car':
        cl1, cl2 = 7, 1  # 7=horse, 1=car
    elif dataset == 'cifar10_dog_cat':
        cl1, cl2 = 5, 3  # 5=dog, 3=cat
    if split in ['train', 'val']:
        if dataset != 'svhn':
            data = dataset_f(dir_, train=True, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='train', transform=transform, download=True)
            data.data = data.data.transpose([0, 2, 3, 1])
            data.targets = data.labels
        data.targets = np.array(data.targets)
        n_cls = max(data.targets) + 1

        if dataset in ['cifar10_horse_car', 'cifar10_dog_cat']:
            data.targets = np.array(data.targets)
            idx = (data.targets == cl1) + (data.targets == cl2)
            data.data, data.targets = data.data[idx], data.targets[idx]
            data.targets[data.targets == cl1], data.targets[data.targets == cl2] = 0, 1
            n_cls = 2
        n_ex = len(data.targets) if n_ex == -1 else n_ex
        if '_gs' in dataset:
            data.data = data.data.mean(3).astype(np.uint8)

        if val_indices is not None:
            assert len(val_indices) < len(data.targets), '#val has to be < total #train pts'
            val_indices_mask = np.zeros(len(data.targets), dtype=bool)
            val_indices_mask[val_indices] = True
            if split == 'train':
                data.data, data.targets = data.data[~val_indices_mask], data.targets[~val_indices_mask]
            else:
                data.data, data.targets = data.data[val_indices_mask], data.targets[val_indices_mask]
        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]  # so the #pts can be in [n_ex-n_eval, n_ex]
        # e.g., when frac_train=1.0, for training set, n_ex=50k while data.data.shape[0]=45k bc of val set
        if n_ex > data.data.shape[0]:
            n_ex = data.data.shape[0]

        data.label_noise = np.zeros(n_ex, dtype=bool)
        data.targets_correct = data.targets.copy()
        if p_label_noise > 0.0:
            print('Split: {}, number of examples: {}, noisy examples: {}'.format(split, n_ex, int(n_ex*p_label_noise)))
            print('Dataset shape: x is {}, y is {}'.format(data.data.shape, data.targets.shape))
            assert n_ex == data.data.shape[0]  # there was a mistake previously here leading to a larger noise level

            # gen random indices
            indices = np.random.permutation(np.arange(len(data.targets)))[:int(n_ex*p_label_noise)]
            for index in indices:
                if noise_type == 'sym':
                    lst_classes = list(range(n_cls))
                    cls_int = data.targets[index] if type(data.targets[index]) is int else data.targets[index].item()
                    lst_classes.remove(cls_int)
                    data.targets[index] = np.random.choice(lst_classes)
                else:
                    data.targets[index] = asym_label_noise(dataset, data.targets[index])
            data.label_noise[indices] = True
        print(data.data.shape)
        data = DatasetWithLabelNoise(data, split, transform if dataset != 'gaussians_binary' else None)
        loader = torch.utils.data.DataLoader(
            dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
            num_workers=num_workers_train if split == 'train' else num_workers_val, drop_last=drop_last)

    elif split == 'test':
        if dataset != 'svhn':
            data = dataset_f(dir_, train=False, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='test', transform=transform, download=True)
            data.data = data.data.transpose([0, 2, 3, 1])
            data.targets = data.labels
        n_ex = len(data) if n_ex == -1 else n_ex

        if dataset in ['cifar10_horse_car', 'cifar10_dog_cat']:
            data.targets = np.array(data.targets)
            idx = (data.targets == cl1) + (data.targets == cl2)
            data.data, data.targets = data.data[idx], data.targets[idx]
            data.targets[data.targets == cl1], data.targets[data.targets == cl2] = 0, 1
            data.targets = list(data.targets)  # to reduce memory consumption
        if '_gs' in dataset:
            data.data = data.data.mean(3).astype(np.uint8)
        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]
        data.targets_correct = data.targets.copy()

        data.label_noise = np.zeros(n_ex)
        data = DatasetWithLabelNoise(data, split, transform if dataset != 'gaussians_binary' else None)
        loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                             num_workers=num_workers_test, drop_last=drop_last)

    else:
        raise ValueError('wrong split')

    return loader


def create_loader(x, y, ln, n_ex, batch_size, shuffle, drop_last):
    if n_ex > 0:
        x, y, ln = x[:n_ex], y[:n_ex], ln[:n_ex]
    data = td.TensorDataset(x, y, ln)
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=False,
                                         num_workers=2, drop_last=drop_last)
    return loader


def get_xy_from_loader(loader, cuda=True, n_batches=-1):
    tuples = [(x, y, y_correct, ln) for i, (x, x_augm2, y, y_correct, ln) in enumerate(loader) if n_batches == -1 or i < n_batches]
    x_vals = torch.cat([x for (x, y, y_correct, ln) in tuples])
    y_vals = torch.cat([y for (x, y, y_correct, ln) in tuples])
    y_correct_vals = torch.cat([y_correct for (x, y, y_correct, ln) in tuples])
    ln_vals = torch.cat([ln for (x, y, y_correct, ln) in tuples])
    if cuda:
        x_vals, y_vals, y_correct_vals, ln_vals = x_vals.cuda(), y_vals.cuda(), y_correct_vals.cuda(), ln_vals.cuda()
    return x_vals, y_vals, y_correct_vals, ln_vals


shapes_dict = {'mnist': (60000, 1, 28, 28),
               'mnist_binary': (13007, 1, 28, 28),
               'svhn': (73257, 3, 32, 32),
               'cifar10': (50000, 3, 32, 32),
               'cifar10_horse_car': (10000, 3, 32, 32),
               'cifar10_dog_cat': (10000, 3, 32, 32),
               'cifar100': (50000, 3, 32, 32),
               'uniform_noise': (1000, 1, 28, 28),
               'gaussians_binary': (1000, 1, 1, 100),
               }
np.random.seed(0)
v_global = np.random.randn(shapes_dict['gaussians_binary'][3])  # needed for consistency between train and test
datasets_dict = {'mnist': datasets.MNIST,
                 'mnist_binary': datasets.MNIST,
                 'svhn': datasets.SVHN,
                 'cifar10': datasets.CIFAR10,
                 'cifar10_horse_car': datasets.CIFAR10,
                 'cifar10_dog_cat': datasets.CIFAR10,
                 'cifar10c': dataset_cifar10c,
                 'cifar10c_binary': dataset_cifar10c,
                 'cifar100': datasets.CIFAR100,
                 'uniform_noise': uniform_noise,
                 'gaussians_binary': dataset_gaussians_binary,
                 }
classes_dict = {'cifar10': {0: 'airplane',
                            1: 'automobile',
                            2: 'bird',
                            3: 'cat',
                            4: 'deer',
                            5: 'dog',
                            6: 'frog',
                            7: 'horse',
                            8: 'ship',
                            9: 'truck',
                            }
                }

