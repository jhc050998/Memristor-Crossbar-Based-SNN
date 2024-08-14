import numpy
import pandas

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


# WDBC(Wisconsin Diagnostic Breast Cancer) dataset
def WDBC_loader(isSave, isLoad):
    sn, fn = 569, 30  # sample num, feature num
    row_data = numpy.mat(pandas.read_csv("D:/Dataset/UCI/wdBC/wdBC.data", header=None, sep=','))  # read file for data
    row_inputs, row_tars = row_data[:, 2:32], row_data[:, 1]

    tensor_inputs, tensor_tars = torch.from_numpy(row_inputs.astype('float32')), torch.ones((sn, 2))*0.01
    # tensor_inputs (569, 30)   tensor_tars (569, 2)
    for i in range(sn):  # label one-hot coding
        if row_tars[i] == 'M':
            tensor_tars[i, 1] = 0.99
        elif row_tars[i] == 'B':
            tensor_tars[i, 0] = 0.99

    data_max, _ = torch.max(tensor_inputs, dim=0)  # (30) 每条数据里最大的
    data_min, _ = torch.min(tensor_inputs, dim=0)
    tensor_inputs = (tensor_inputs - data_min) / (data_max - data_min) + 0.01
    tot_max, tot_min = torch.max(tensor_inputs), torch.min(tensor_inputs)
    tensor_inputs = (tensor_inputs - tot_min) / (tot_max - tot_min) + 0.01

    if isLoad:
        rand_index = torch.load("./parameters_record/WDBC/WDBC_rand_index")
    else:
        rand_index = torch.randperm(sn)  # disrupt the order
    if isSave:
        torch.save(rand_index, "./parameters_record/WDBC/WDBC_rand_index")
    tensor_tars = torch.gather(tensor_tars, dim=0, index=torch.tile(torch.reshape(rand_index, [sn, 1]), [1, 2]))
    tensor_inputs = torch.gather(tensor_inputs, dim=0, index=torch.tile(torch.reshape(rand_index, [sn, 1]), [1, fn]))

    return tensor_inputs, tensor_tars


# JHUI(Johns Hopkins University Ionosphere) dataset
def JHUI_loader(isSave, isLoad):
    sn, fn = 351, 34  # sample num, feature num
    row_data = numpy.mat(pandas.read_csv(
        "D:/Dataset/UCI/ionosphere/ionosphere.data", header=None, sep=','))  # (351, 35)
    row_inputs, row_tars = row_data[:, 0:34], row_data[:, 34]
    # print(row_data)

    tensor_inputs, tensor_tars = torch.from_numpy(row_inputs.astype('float32')), torch.ones((sn, 2)) * 0.01
    # tensor_inputs (351, 34)   tensor_tars (351, 2)
    for i in range(sn):  # label one-hot coding
        if row_tars[i] == 'g':
            tensor_tars[i, 1] = 0.99
        elif row_tars[i] == 'b':
            tensor_tars[i, 0] = 0.99

    data_max, _ = torch.max(tensor_inputs, dim=0)  # return value and index, take value
    data_min, _ = torch.min(tensor_inputs, dim=0)
    tensor_inputs = (tensor_inputs - data_min) / (data_max - data_min + 1e-5) + 0.01
    tot_max, tot_min = torch.max(tensor_inputs), torch.min(tensor_inputs)
    tensor_inputs = (tensor_inputs - tot_min) / (tot_max - tot_min + 1e-5) + 0.01

    if isLoad:
        rand_index = torch.load("./parameters_record/JHUI/JHUI_rand_index")
    else:
        rand_index = torch.randperm(sn)  # disrupt the order
    if isSave:
        torch.save(rand_index, "parameters_record/JHUI/JHUI_rand_index")
    tensor_tars = torch.gather(tensor_tars, dim=0, index=torch.tile(torch.reshape(rand_index, [sn, 1]), [1, 2]))
    tensor_inputs = torch.gather(tensor_inputs, dim=0, index=torch.tile(torch.reshape(rand_index, [sn, 1]), [1, fn]))

    return tensor_inputs, tensor_tars


# BUPA liver disorders dataset
def BUPA_loader(isSave, isLoad):
    sn, fn = 345, 5  # sample num
    row_data = numpy.mat(pandas.read_csv("D:/Dataset/UCI/LD/bupa.data", header=None, sep=','))  # (345, 7)
    row_inputs, row_tars = row_data[:, 0:5], row_data[:, 5]  # 第6个数据是标签，分数值>5或<=5，第7个数据是分划数据集用的，依据不明
    # 原始划分中有1类数据145个，2类数据200个（备用信息）

    tensor_inputs, tensor_tars = torch.from_numpy(row_inputs.astype('float32')), torch.ones((sn, 2)) * 0.01
    # tensor_inputs (345, 5)   tensor_tars (345, 2)
    for i in range(sn):  # label one-hot coding
        if row_tars[i] > 5.0:
            tensor_tars[i, 1] = 0.99
        elif row_tars[i] <= 5.0:
            tensor_tars[i, 0] = 0.99

    data_max, _ = torch.max(tensor_inputs, dim=0)  # return value and index, take value
    data_min, _ = torch.min(tensor_inputs, dim=0)
    tensor_inputs = (tensor_inputs - data_min) / (data_max - data_min) + 0.01
    tot_max, tot_min = torch.max(tensor_inputs), torch.min(tensor_inputs)
    tensor_inputs = (tensor_inputs - tot_min) / (tot_max - tot_min) + 0.01

    if isLoad:
        rand_index = torch.load("./parameters_record/BUPA/BUPA_rand_index")
    else:
        rand_index = torch.randperm(sn)  # disrupt the order
    if isSave:
        torch.save(rand_index, "./parameters_record/BUPA/BUPA_rand_index")
    tensor_tars = torch.gather(tensor_tars, dim=0, index=torch.tile(torch.reshape(rand_index, [sn, 1]), [1, 2]))
    tensor_inputs = torch.gather(tensor_inputs, dim=0, index=torch.tile(torch.reshape(rand_index, [sn, 1]), [1, fn]))

    return tensor_inputs, tensor_tars


# MNIST
mnist_train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("D:/Dataset/mnist", train=True, download=True,  # in MNIST/raw
                   transform=transforms.Compose([transforms.ToTensor()])), batch_size=60000, shuffle=True)
mnist_test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("D:/Dataset/mnist", train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])), batch_size=10000, shuffle=False)

# FashionMNIST
fashion_mnist_train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST("D:/Dataset/mnist", train=True, download=True,  # in FashionMNIST/raw
                          transform=transforms.Compose([transforms.ToTensor()])), batch_size=60000, shuffle=True)
fashion_mnist_test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST("D:/Dataset/mnist", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()])), batch_size=10000, shuffle=False)


# Show images in MNIST & FashionMNIST dataset
def MNIST_show(data_set="mnist", show_part="train"):
    train_data_loader = mnist_train_loader
    if data_set == "fashion_mnist":
        train_data_loader = fashion_mnist_train_loader
    test_data_loader = mnist_test_loader
    if data_set == "fashion_mnist":
        test_data_loader = fashion_mnist_test_loader

    X_train, y_train = [], []
    for idx, (data, target) in enumerate(train_data_loader):  # read out all data in a time
        X_train, y_train = data, target
    print("Form of train set data: " + str(X_train.shape))  # (60000,1,28,28)
    print("Form of train set label: " + str(y_train.shape))  # (60000)
    print("")

    X_test, y_test = [], []
    for idx, (data, target) in enumerate(test_data_loader):
        X_test, y_test = data, target
    print("Form of test set data: " + str(X_test.shape))  # (10000,1,28,28)
    print("Form of test set label: " + str(y_test.shape))  # (10000)
    print("")

    if show_part == "train":
        for i in range(X_train.size()[0]):   # show train set
            data, target = X_train[i], y_train[i]
            print("Image" + str(i+1) + "label: " + str(target))
            img = data.permute(1, 2, 0)
            x_img = data[0]
            x_img = torch.where(x_img > 0.5, 0.01, 1.79)
            # x_img = numpy.array(data[0])
            # x_img = feature.canny(x_img, sigma=0.5)  # border detection
            print(x_img)
            plt.imshow(x_img)
            plt.show()
    else:
        for i in range(X_test.size()[0]):  # show test set
            data, target = X_test[i], y_test[i]
            print("Image" + str(i+1) + "label: " + str(target))
            img = data.permute(1, 2, 0)
            plt.imshow(img)
            plt.show()


def main():
    MNIST_show(data_set="fashion_mnist", show_part="train")


if __name__ == "__main__":
    main()
