import math
import time
import torch

import Dataset as Ds
import SNN


def MNIST_LIF_train(X_train, y_train, X_test, y_test):
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        print("Training on cpu.")
        device = torch.device("cpu")
    else:
        print("Training on gpu.")
        device = torch.device("cuda:7")

    # Network layout
    ly1 = SNN.LIFLayer(inF=784, outF=800, T=4.0, dt=0.01)
    ly2 = SNN.LIFLayer(inF=800, outF=10, T=4.0, dt=0.01)

    # Send the data to the device
    ly1.t = ly1.t.to(device)
    ly2.t = ly2.t.to(device)
    ly1.wt = ly1.wt.to(device)
    ly2.wt = ly2.wt.to(device)
    ly1.cause_mask = ly1.cause_mask.to(device)
    ly2.cause_mask = ly2.cause_mask.to(device)
    ly1.adam_m_wt = ly1.adam_m_wt.to(device)
    ly2.adam_m_wt = ly2.adam_m_wt.to(device)
    ly1.adam_v_wt = ly1.adam_v_wt.to(device)
    ly2.adam_v_wt = ly2.adam_v_wt.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)

    # Training process
    epoch_num = 40
    interval = 50
    lr_start, lr_end = 1e-3, 1e-6  # decaying learning rate
    lr_decay = (lr_end / lr_start) ** (1.0 / epoch_num)

    sn, bs = X_train.size()[0], 128  # number of data samples, batch size
    bs_const = bs
    bn = int(math.ceil(sn / bs))  # number of batches
    loss, total_loss = 0, []
    time_start = time.time()  # time when training process start
    for epoch in range(epoch_num):
        bs = bs_const
        lr = lr_start * lr_decay ** epoch
        for bi in range(bn):
            # input data
            if (bi + 1) * bs > sn:  # for the last batch with unusual size
                data, tar = X_train[bi * bs:sn], y_train[bi * bs:sn]
            else:  # for other batches
                data, tar = X_train[bi * bs:(bi + 1) * bs], y_train[bi * bs:(bi + 1) * bs]
            z0 = torch.exp(1.0 - data.view(-1, 28 * 28))  # processing data (bs,1,28,28) --> (bs,784)
            tar_10 = (torch.ones(tar.size()[0], 10) * 0.99).to(device)  # the prepared label
            for i in range(data.size()[0]):
                tar_10[i, tar[i]] = 0.01

            bs = z0.size()[0]

            # Forward propagation
            t0 = 1.0 - data.view(-1, 28 * 28)
            t1 = ly1.forward(bs, t0)
            t2 = ly2.forward(bs, t1)
            z1, z2 = torch.exp(t1), torch.exp(t2)

            # Backward propagation (Gradient for Weight)
            z2_lo, z_tar = torch.softmax(z2, dim=1), torch.softmax(torch.exp(tar_10), dim=1)

            wt2Ex = torch.tile(torch.reshape(ly2.wt, [1, ly2.inF, ly2.outF]), [bs, 1, 1])
            t2Ex_2 = torch.tile(torch.reshape(t2, [bs, 1, ly2.outF]), [1, ly2.inF, 1])
            t1Ex_2 = torch.tile(torch.reshape(t1, [bs, ly2.inF, 1]), [1, 1, ly2.outF])
            delta2 = (z2_lo - z_tar) * z2 / torch.sum(ly2.g(wt2Ex, t1Ex_2, t2Ex_2) * ly2.cause_mask, dim=1)

            wt1Ex = torch.tile(torch.reshape(ly1.wt, [1, ly1.inF, ly1.outF]), [bs, 1, 1])
            t1Ex_1 = torch.tile(torch.reshape(t1, [bs, 1, ly1.outF]), [1, ly1.inF, 1])
            t0Ex_1 = torch.tile(torch.reshape(t0, [bs, ly1.inF, 1]), [1, 1, ly1.outF])
            delta1 = ly2.pass_delta(bs, ly2.g(
                wt2Ex, t1Ex_2, t2Ex_2), delta2) / torch.sum(ly1.g(wt1Ex, t0Ex_1, t1Ex_1) * ly1.cause_mask, dim=1)

            ly2.backward(bs, delta2, t1, t2, lr)
            ly1.backward(bs, delta1, t0, t1, lr)

            CE = -1.0 * torch.sum(torch.log(torch.clamp(z2_lo, 1e-5, 1.0)) * z_tar) / data.size()[0]
            CE_min = -1.0 * torch.sum(torch.log(torch.clamp(z_tar, 1e-5, 1.0)) * z_tar) / data.size()[0]
            loss = abs(CE - CE_min)
            if bi % interval == 0:
                print("Current Training epoch: " + str(epoch + 1), end="\t")
                print("Progress: [" + str(bi * bs) + "/" + str(sn), end="")
                print("(%.0f %%)]" % (100.0 * bi * bs / sn), end="\t")
                print("Loss: " + str(loss))
                total_loss.append(loss)
        pass
        time_epoch_end = time.time()
        print("Time consuming: %.3f s" % (time_epoch_end - time_start))
        torch.save(ly1.wt, "./parameters_record/MNIST/MNIST_LIF_wt1")
        torch.save(ly2.wt, "./parameters_record/MNIST/MNIST_LIF_wt2")
        print("Accuracy on train data: ")
        MNIST_LIF_test(X_train, y_train)
        print("Accuracy on test data: ")
        MNIST_LIF_test(X_test, y_test)
    pass


def MNIST_LIF_test(X, y):
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        print("Testing on cpu.")
        device = torch.device("cpu")
    else:
        print("Testing on gpu.")
        device = torch.device("cuda:7")

    # Network layout
    wt1 = torch.load("./parameters_record/MNIST/MNIST_LIF_wt1")
    wt2 = torch.load("./parameters_record/MNIST/MNIST_LIF_wt2")

    ly1 = SNN.LIFLayer(inF=784, outF=800, T=4.0, dt=0.01, wt=wt1)
    ly2 = SNN.LIFLayer(inF=800, outF=10, T=4.0, dt=0.01, wt=wt2)

    ly1.t = ly1.t.to(device)
    ly2.t = ly2.t.to(device)
    X, y = X.to(device), y.to(device)

    # Testing Process
    correct = 0
    sn, bs = X.size()[0], 100  # number of data samples, batch size
    bn = int(math.ceil(sn / bs))  # number of batches
    for bi in range(bn):
        if (bi + 1) * bs > sn:
            data, tar = X[bi * bs:sn], y[bi * bs:sn]
        else:
            data, tar = X[bi * bs:(bi + 1) * bs], y[bi * bs:(bi + 1) * bs]

        # Forward propagation
        t0 = 1.0 - data.view(-1, 28 * 28)
        t1 = ly1.forward(bs, t0)
        t2 = ly2.forward(bs, t1)

        prediction = torch.argmin(t2, dim=1)
        correct += prediction.eq(tar.data).sum()

        print(bi)
    pass
    print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
    print("(%.3f %%)" % (100. * correct / sn))
    print(" ")
    return correct


def MNIST_LIF():  #
    # Data prepare
    X_train, y_train = [], []  # (60000,1,28,28)
    for idx, (data, target) in enumerate(Ds.mnist_train_loader):  # read out all data in one time
        X_train, y_train = data, target
    X_test, y_test = [], []
    for idx, (data, target) in enumerate(Ds.mnist_test_loader):
        X_test, y_test = data, target

    '''# Train
    time_start = time.time()
    MNIST_LIF_train(X_train, y_train, X_test, y_test)
    time_end = time.time()
    print("Time consuming: %.3f s" % (time_end - time_start))

    # Test
    print("Train set: ")
    MNIST_LIF_test(X_train, y_train)'''
    print("Test set: ")
    MNIST_LIF_test(X_test, y_test)


def main():
    MNIST_LIF()


if __name__ == "__main__":
    main()
