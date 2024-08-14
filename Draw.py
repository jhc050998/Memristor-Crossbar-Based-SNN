import itertools

import math
import numpy
import torch

import matplotlib.pyplot as plt


def MNIST_conMx():
    matrix = torch.load("./parameters_record/MNIST/MNIST_M_conMx")
    classes = numpy.array(torch.arange(0, 10))
    title = 'Confusion matrix'
    c_map = plt.cm.Blues  # 绘制的颜色

    matrix_d = torch.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if matrix[i, j] > 200:
                matrix_d[i, j] = 20#matrix[i, j]/5
            else:
                matrix_d[i, j] = matrix[i, j]

    plt.imshow(matrix_d, interpolation='nearest', cmap=c_map)  # , cmap=c_map
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.ylim(len(classes) - 0.5, -0.5)

    fmt = '.0f'
    thresh = torch.max(matrix) / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):  # 写矩阵中数字
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center", color="white" if matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def WDBC_loss():
    loss = torch.tensor(torch.load("./parameters_record/WDBC/WDBC_loss"))
    loss_Q = torch.tensor(torch.load("./parameters_record/WDBC/WDBC_Q_loss"))
    loss_IF = torch.tensor(torch.load("./parameters_record/WDBC/WDBC_IF_loss"))
    # print(loss.size())
    # print(loss)

    plt.figure(figsize=(20, 10), dpi=100)
    x = numpy.arange(loss.size()[0])
    plt.plot(x, loss, linewidth=1, label="HDSNN")
    plt.plot(x, loss_Q, linewidth=1, label="HDSNNq")
    plt.plot(x, loss_IF, c="grey", linewidth=1, label="SpikeProp")
    plt.legend()
    plt.show()


def WDBC_acc():
    acc = torch.tensor(torch.load("./parameters_record/WDBC/WDBC_acc"))
    acc_Q = torch.tensor(torch.load("./parameters_record/WDBC/WDBC_Q_acc"))
    acc_IF = torch.tensor(torch.load("./parameters_record/WDBC/WDBC_IF_acc"))

    plt.figure(figsize=(20, 10), dpi=100)
    x = numpy.arange(acc.size()[0])
    plt.plot(x, acc, linewidth=1, label="HDSNN")
    plt.plot(x, acc_Q, linewidth=1, label="HDSNNq")
    plt.plot(x, acc_IF, c="grey", linewidth=1, label="SpikeProp")
    plt.legend()
    plt.show()


def WDBC_roc():
    TPR_y = torch.tensor(torch.load("./parameters_record/WDBC/WDBC_TPR"))
    FPR_x = torch.tensor(torch.load("./parameters_record/WDBC/WDBC_FPR"))

    plt.figure(figsize=(10, 10), dpi=100)
    x = FPR_x
    plt.plot(x, TPR_y)
    plt.show()


def JHUI_loss():
    loss = torch.tensor(torch.load("./parameters_record/JHUI/JHUI_loss"))
    loss_Q = torch.tensor(torch.load("./parameters_record/JHUI/JHUI_Q_loss"))
    loss_IF = torch.tensor(torch.load("./parameters_record/JHUI/JHUI_IF_loss"))

    plt.figure(figsize=(20, 10), dpi=100)
    x = numpy.arange(loss.size()[0])
    plt.plot(x, loss, linewidth=1, label="HDSNN")
    plt.plot(x, loss_Q, linewidth=1, label="HDSNNq")
    plt.plot(x, loss_IF, c="grey", linewidth=1, label="SpikeProp")
    plt.legend()
    plt.show()


def JHUI_acc():
    acc = torch.tensor(torch.load("./parameters_record/JHUI/JHUI_acc"))
    acc_Q = torch.tensor(torch.load("./parameters_record/JHUI/JHUI_Q_acc"))
    acc_IF = torch.tensor(torch.load("./parameters_record/JHUI/JHUI_IF_acc"))

    plt.figure(figsize=(20, 10), dpi=100)
    x = numpy.arange(acc.size()[0])
    plt.plot(x, acc, linewidth=1, label="HDSNN")
    plt.plot(x, acc_Q, linewidth=1, label="HDSNNq")
    plt.plot(x, acc_IF, c="grey", linewidth=1, label="SpikeProp")
    plt.legend()
    plt.show()


def BUPA_loss():
    loss = torch.tensor(torch.load("./parameters_record/BUPA/BUPA_loss"))
    loss_Q = torch.tensor(torch.load("./parameters_record/BUPA/BUPA_Q_loss"))
    loss_IF = torch.tensor(torch.load("./parameters_record/BUPA/BUPA_IF_loss"))

    plt.figure(figsize=(20, 10), dpi=100)
    x = numpy.arange(loss.size()[0])
    plt.plot(x, loss, linewidth=1, label="HDSNN")
    plt.plot(x, loss_Q, linewidth=1, label="HDSNNq")
    plt.plot(x, loss_IF, c="grey", linewidth=1, label="SpikeProp")
    plt.legend()
    plt.show()


def BUPA_acc():
    acc = torch.tensor(torch.load("./parameters_record/BUPA/BUPA_acc"))
    acc_Q = torch.tensor(torch.load("./parameters_record/BUPA/BUPA_Q_acc"))
    acc_IF = torch.tensor(torch.load("./parameters_record/BUPA/BUPA_IF_acc"))

    plt.figure(figsize=(20, 10), dpi=100)
    x = numpy.arange(acc.size()[0])
    plt.plot(x, acc, linewidth=1, label="HDSNN")
    plt.plot(x, acc_Q, linewidth=1, label="HDSNNq")
    plt.plot(x, acc_IF, c="grey", linewidth=1, label="SpikeProp")
    plt.legend()
    plt.show()


def MNIST_loss():
    loss = torch.tensor(torch.load("./parameters_record/MNIST/MNIST_loss", map_location=torch.device('cpu')))
    loss_Q = torch.tensor(torch.load("./parameters_record/MNIST/MNIST_Q_loss", map_location=torch.device('cpu')))

    loss_10, loss_Q_10 = [], []
    for i in range(loss.size()[0]):
        if i % 10 == 0:
            loss_10.append(loss[i])
            loss_Q_10.append(loss_Q[i])
    loss_10 = torch.tensor(loss_10)
    loss_Q_10 = torch.tensor(loss_Q_10)

    plt.figure(figsize=(20, 10), dpi=100)
    x = numpy.arange(loss.size()[0]/10)
    plt.plot(x, loss_10, c="blue", linewidth=1.5, label="HDSNN")
    plt.plot(x, loss_Q_10, c="red", linewidth=1.5, label="HDSNNq")
    plt.legend()
    plt.show()


def FMNIST_loss():
    loss = torch.tensor(torch.load("./parameters_record/FMNIST/FMNIST_loss", map_location=torch.device('cpu')))
    loss_Q = torch.tensor(torch.load("./parameters_record/FMNIST/FMNIST_Q_loss", map_location=torch.device('cpu')))

    loss_10, loss_Q_10 = [], []
    for i in range(loss.size()[0]):
        if i % 10 == 0:
            loss_10.append(loss[i])
            loss_Q_10.append(loss_Q[i])
    loss_10 = torch.tensor(loss_10)
    loss_Q_10 = torch.tensor(loss_Q_10)

    plt.figure(figsize=(20, 10), dpi=100)
    x = numpy.arange(loss.size()[0]/10)
    plt.plot(x, loss_10, c="blue", linewidth=1.5, label="HDSNN")
    plt.plot(x, loss_Q_10, c="red", linewidth=1.5, label="HDSNNq")
    plt.legend()
    plt.show()


def main():
    # WDBC_loss()
    # WDBC_acc()
    # WDBC_roc()

    # JHUI_loss()
    # JHUI_acc()
    # BUPA_loss()
    # BUPA_acc()

    # MNIST_loss()
    # FMNIST_loss()

    # MNIST_conMx()

    '''# 准备数据
    categories = ['BinaryConnect', 'BinaryNet', 'Gated-XNOR', 'HDSNN']
    values = [17.32, 13.12, 11.89, 8.19]
    # colors = ['red', 'blue', 'green', 'yellow']
    # colors = [(0.2, 0.4, 0.6), 'orange', (0.1, 0.6, 0.3), 'grey']
    colors = [(0.2, 0.4, 0.6), (0.2, 0.4, 0.6), (0.2, 0.4, 0.6), 'orange']

    # 创建柱状图
    plt.bar(categories, values, width=0.45, color=colors)  # , color=colors
    plt.show()'''

    n1 = math.log(5.571e-3, 10)
    n2 = math.log(1.327e-4, 10)
    n3 = math.log(5.0e-7, 10)

    n1_d, n2_d, n3_d = n1 + 8, n2 + 8, n3 + 8
    print(n1)
    print(n2)
    print(n3)

    categories = ['CPU', 'GPU', 'Memristor Crossbar']
    values = [n1_d, n2_d, n3_d]
    colors = [(0.2, 0.4, 0.6), (0.2, 0.4, 0.6), 'orange']

    plt.bar(categories, values, width=0.45, color=colors)
    plt.show()


if __name__ == "__main__":
    main()
