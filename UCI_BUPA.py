import math
import time
import numpy
import torch

import Dataset as Ds
import SNN


def BUPA_LIF_train(X_train, y_train, X_test, y_test):
    # Network layout
    ly1 = SNN.LIFLayer(inF=5, outF=10, T=4.0, dt=0.01)
    ly2 = SNN.LIFLayer(inF=10, outF=2, T=4.0, dt=0.01)

    # Training process
    epoch_num = 600
    interval = 50
    lr_start, lr_end = 1e-3, 1e-4  # decaying learning rate
    lr_decay = (lr_end / lr_start) ** (1.0 / epoch_num)

    sn, bs = X_train.size()[0], X_train.size()[0]  # number of data samples, batch size
    bn = int(math.ceil(sn / bs))  # number of batches
    loss, total_loss = 0, []
    # acc, total_acc = 0, []
    for epoch in range(epoch_num):
        lr = lr_start * lr_decay ** epoch
        for bi in range(bn):
            # input data
            if (bi + 1) * bs > sn:  # for the last batch with unusual size
                data, tar_2 = X_train[bi * bs:sn], y_train[bi * bs:sn]
            else:  # for other batches
                data, tar_2 = X_train[bi * bs:(bi + 1) * bs], y_train[bi * bs:(bi + 1) * bs]

            # Forward propagation
            t0 = 1.0 - data
            t1 = ly1.forward(bs, t0)
            t2 = ly2.forward(bs, t1)
            z1, z2 = torch.exp(t1), torch.exp(t2)

            # Backward propagation (Gradient for Weight)
            z2_lo, z_tar = torch.softmax(z2, dim=1), torch.softmax(torch.exp(1.0 - tar_2), dim=1)

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
            total_loss.append(loss)
        if epoch % interval == 0:
            print("Current Training epoch: " + str(epoch + 1), end="\t")
            print("Progress: [" + str(epoch) + "/" + str(epoch_num), end="")
            print("(%.0f %%)]" % (100.0 * epoch / epoch_num), end="\t")
            print(" ")
            print("Loss: " + str(loss))
        pass
        '''# 每轮测试一次
        torch.save(ly1.wt, "./parameters_record/BUPA/BUPA_LIF_wt1")
        torch.save(ly2.wt, "./parameters_record/BUPA/BUPA_LIF_wt2")
        acc = BUPA_LIF_test(X_test, y_test)
        total_acc.append(acc)'''
    pass
    torch.save(ly1.wt, "./parameters_record/BUPA/BUPA_LIF_wt1")
    torch.save(ly2.wt, "./parameters_record/BUPA/BUPA_LIF_wt2")
    torch.save(total_loss, "./parameters_record/BUPA/BUPA_loss")
    # torch.save(total_acc, "./parameters_record/BUPA/BUPA_acc")


def BUPA_LIF_test(X, y):
    # Network layout
    wt1 = torch.load("./parameters_record/BUPA/BUPA_LIF_wt1")
    wt2 = torch.load("./parameters_record/BUPA/BUPA_LIF_wt2")

    ly1 = SNN.LIFLayer(inF=5, outF=10, T=4.0, dt=0.01, wt=wt1)
    ly2 = SNN.LIFLayer(inF=10, outF=2, T=4.0, dt=0.01, wt=wt2)

    # Testing Process
    correct = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    sn, bs = X.size()[0], X.size()[0]  # number of data samples, batch size
    bn = int(math.ceil(sn / bs))  # number of batches
    for bi in range(bn):
        if (bi + 1) * bs > sn:
            data, tar_2 = X[bi * bs:sn], y[bi * bs:sn]
        else:
            data, tar_2 = X[bi * bs:(bi + 1) * bs], y[bi * bs:(bi + 1) * bs]
        tar = torch.argmax(tar_2, dim=1)

        # Forward propagation
        t0 = 1.0 - data
        t1 = ly1.forward(bs, t0)
        t2 = ly2.forward(bs, t1)

        prediction = torch.argmin(t2, dim=1)
        correct_temp = prediction.eq(tar.data).sum()
        correct += correct_temp

        # 算4项
        TP_temp = (tar.data * prediction.eq(tar.data)).sum()
        FP_temp = (tar.data * prediction.eq(1.0 - tar.data)).sum()

        TP += TP_temp  # 正例，且分类正确
        TN += correct_temp - TP_temp  # 反例，且分类正确
        FP += FP_temp  # 正例，且分类错误
        FN += (bs - correct_temp) - FP_temp
    pass
    '''print("TP: " + str(TP))
    print("TN: " + str(TN))
    print("FP: " + str(FP))
    print("FN: " + str(FN))
    print(" ")

    Pre = TP / (TP + FP)
    print("Pre: " + str(int(TP)) + "/" + str(int(TP + FP)), end="")
    print("(%.3f %%)" % (100. * Pre))
    FPR = FP / (TN + FP)
    print("FPR: " + str(int(FP)) + "/" + str(int(TN + FP)), end="")
    print("(%.3f %%)" % (100. * FPR))
    FNR = FN / (TP + FN)
    print("FNR: " + str(int(FN)) + "/" + str(int(TP + FN)), end="")
    print("(%.3f %%)" % (100. * FNR))
    F1 = 2 * TP / (2 * TP + FP + FN)
    print("F1: " + str(int(2 * TP)) + "/" + str(int(2 * TP + FP + FN)), end="")
    print("(%.3f %%)" % (100. * F1))
    print(" ")

    print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
    print("(%.3f %%)" % (100. * correct / sn))
    print(" ")'''
    return TP, TN, FP, FN  # correct/sn


def BUPA_LIF():  # 78.70/75.00
    # Data prepare
    X, y = Ds.BUPA_loader(isSave=True, isLoad=False)

    # Read out data
    f = 0
    siz = 173
    X_train, y_train = torch.cat(
        (X[0:f * siz], X[f * siz + siz:345]), dim=0), torch.cat((y[0:f * siz], y[f * siz + siz:345]), dim=0)
    X_test, y_test = X[f * siz:f * siz + siz], y[f * siz:f * siz + siz]

    '''# Train
    time_start = time.time()
    BUPA_LIF_train(X_train, y_train, X_test, y_test)
    time_end = time.time()
    print("Time consuming: %.3f s" % (time_end - time_start))

    # Test
    print("Train set: ")
    BUPA_LIF_test(X_train, y_train)
    print("Test set: ")
    BUPA_LIF_test(X_test, y_test)'''

    ep = 24
    time_start = time.time()
    T4 = torch.zeros(4, ep)  # TP, TN, FP, FN
    for i in range(ep):
        BUPA_LIF_train(X_train, y_train, X_test, y_test)
        T4[0, i], T4[1, i], T4[2, i], T4[3, i] = BUPA_LIF_test(X_test, y_test)
        print("第" + str(i + 1) + "轮结束。")
    pass
    time_end = time.time()
    print("Time consuming: %.3f s" % (time_end - time_start))
    print(T4)
    torch.save(T4, "./parameters_record/BUPA/BUPA_T4")


def main():
    # BUPA_LIF()

    T4 = torch.load("./parameters_record/BUPA/BUPA_T4")
    print(T4.size())
    print(T4)

    T4_g = []
    for i in range(24):
        if T4[0, i] > 50:
            T4_g.append(numpy.array(T4[:, i]))
    #T4_g = torch.tensor(T4_g).transpose(0, 1)
    #print(T4_g.size())

    TP, TN, FP, FN = T4[0], T4[1], T4[2], T4[3]

    Acc = (TP + TN) / (TP + TN + FP + FN)
    Pre = TP / (TP + FP)
    FPR = FP / (TN + FP)
    FNR = FN / (TP + FN)
    F1 = 2 * TP / (2 * TP + FP + FN)

    print("Acc: " + str(torch.mean(Acc)) + ", " + str(torch.std(Acc)))
    print("Pre: " + str(torch.mean(Pre)) + ", " + str(torch.std(Pre)))
    print("FPR: " + str(torch.mean(FPR)) + ", " + str(torch.std(FPR)))
    print("FNR: " + str(torch.mean(FNR)) + ", " + str(torch.std(FNR)))
    print("F1: " + str(torch.mean(F1)) + ", " + str(torch.std(F1)))


if __name__ == "__main__":
    main()
