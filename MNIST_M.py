import math
import numpy
import random
import time
import torch

import Dataset as Ds
import SNN

from memristor import memristor_vi_crossbar as mc


def MNIST_LIF_Q_train(X_train, y_train, X_test, y_test):
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        print("Training on cpu.")
        device = torch.device("cpu")
    else:
        print("Training on gpu.")
        device = torch.device("cuda:7")

    # Network layout
    ly1 = SNN.LIF_QLayer(inF=784, outF=800, T=4.0, dt=0.01, Lb=-0.05, Hb=0.30, QN=17)
    ly2 = SNN.LIF_QLayer(inF=800, outF=10, T=4.0, dt=0.01, Lb=-0.05, Hb=0.30, QN=17)

    # Send the data to the device
    ly1.t = ly1.t.to(device)
    ly2.t = ly2.t.to(device)
    ly1.wt, ly1.wt_q = ly1.wt.to(device), ly1.wt_q.to(device)
    ly2.wt, ly2.wt_q = ly2.wt.to(device), ly2.wt_q.to(device)
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
    wt_max, wt_min = 0.0, 0.0
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
            t0 = 1.0 - data
            t1 = ly1.forward(bs, t0)
            t2 = ly2.forward(bs, t1)
            z1, z2 = torch.exp(t1), torch.exp(t2)

            # Backward propagation (Gradient)
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

            if torch.max(ly1.wt) > wt_max:
                wt_max = torch.max(ly1.wt)
            if torch.min(ly1.wt) < wt_min:
                wt_min = torch.min(ly1.wt)

            if torch.max(ly2.wt) > wt_max:
                wt_max = torch.max(ly2.wt)
            if torch.min(ly2.wt) < wt_min:
                wt_min = torch.min(ly2.wt)

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
        torch.save(ly1.wt_q, "./parameters_record/MNIST/MNIST_LIF_Q_wt1")
        torch.save(ly2.wt_q, "./parameters_record/MNIST/MNIST_LIF_Q_wt2")
        print("Accuracy on train data: ")
        MNIST_LIF_Q_test(X_train, y_train, ly1.wt_q, ly2.wt_q, True)
        print("Accuracy on test data: ")
        MNIST_LIF_Q_test(X_test, y_test, ly1.wt_q, ly2.wt_q, True)
    pass

    # 对量化训练后所得权值的评估
    print("wt_min: " + str(wt_min) + ", wt_max: " + str(wt_max))
    wt_q1_min, wt_q1_max = torch.min(ly1.wt_q), torch.max(ly1.wt_q)
    print("wt_q1_min: " + str(wt_q1_min) + ", wt_q1_max: " + str(wt_q1_max))
    wt_q2_min, wt_q2_max = torch.min(ly2.wt_q), torch.max(ly2.wt_q)
    print("wt_q2_min: " + str(wt_q2_min) + ", wt_q2_max: " + str(wt_q2_max))

    inv = (ly1.Hb - ly1.Lb) / (ly1.QN - 1)
    print("inv: " + str(inv))
    stepN1 = (wt_q1_max - wt_q1_min) / inv + 1
    stepN2 = (wt_q2_max - wt_q2_min) / inv + 1
    print("stepN1: " + str(stepN1))
    print("stepN2: " + str(stepN2))


def MNIST_LIF_Q_test(X, y, wt1, wt2, not_SA):
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        print("Testing on cpu.")
        device = torch.device("cpu")
    else:
        print("Testing on gpu.")
        device = torch.device("cuda:7")

    # Network layout
    ly1 = SNN.LIF_QLayer(inF=784, outF=800, T=4.0, dt=0.01, Lb=-0.05, Hb=0.30, QN=17, wt=wt1)
    ly2 = SNN.LIF_QLayer(inF=800, outF=10, T=4.0, dt=0.01, Lb=-0.05, Hb=0.30, QN=17, wt=wt2)

    ly1.t = ly1.t.to(device)
    ly2.t = ly2.t.to(device)
    X, y = X.to(device), y.to(device)

    if not_SA:
        print("wt_1: ")
        print(ly1.wt_q.size())
        # print(ly1.wt_q[177])
        wt_q1_min, wt_q1_max = torch.min(ly1.wt_q), torch.max(ly1.wt_q)
        print("wt_q1_min: " + str(wt_q1_min) + ", wt_q1_max: " + str(wt_q1_max))

        print("wt_2: ")
        print(ly2.wt_q.size())
        # print(ly2.wt_q)
        wt_q2_min, wt_q2_max = torch.min(ly2.wt_q), torch.max(ly2.wt_q)
        print("wt_q2_min: " + str(wt_q2_min) + ", wt_q2_max: " + str(wt_q2_max))

        inv = (ly1.Hb - ly1.Lb) / (ly1.QN - 1)
        print("inv: " + str(inv))
        stepN1 = (wt_q1_max - wt_q1_min) / inv + 1
        stepN2 = (wt_q2_max - wt_q2_min) / inv + 1
        print("stepN1: " + str(stepN1))
        print("stepN2: " + str(stepN2))

    # Testing Process
    correct = 0
    sn, bs = X.size()[0], 100  # number of data samples, batch size
    bn = int(math.ceil(sn / bs))  # number of batches

    conMx = torch.zeros((10, 10))
    print("conMx: " + str(conMx))
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
        print("pred: " + str(prediction))
        print("tar: " + str(tar.data))
        for i in range(bs):
            conMx[tar.data[i], prediction[i]] += 1

        correct += prediction.eq(tar.data).sum()
        print("bi: " + str(bi))
    pass
    print("conMx: " + str(conMx))
    torch.save(conMx, "./parameters_record/MNIST/MNIST_M_conMx")
    if not_SA:
        print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
        print("(%.3f %%)" % (100. * correct / sn))
        print(" ")
    return correct / sn


def Weight_cut(inv, u_bound, l_bound):
    wt1 = torch.load("./parameters_record/MNIST/MNIST_LIF_Q_wt1")
    wt2 = torch.load("./parameters_record/MNIST/MNIST_LIF_Q_wt2")

    wt1_cut = torch.where(wt1 >= u_bound - inv * 29.0, u_bound - inv * 29.0, wt1)
    wt1_cut = torch.where(wt1_cut <= l_bound + inv * 17.0, l_bound + inv * 17.0, wt1_cut)
    wt2_cut = torch.where(wt2 >= u_bound - inv * 29.0, u_bound - inv * 29.0, wt2)
    wt2_cut = torch.where(wt2_cut <= l_bound + inv * 17.0, l_bound + inv * 17.0, wt2_cut)

    torch.save(wt1_cut, "./parameters_record/MNIST/MNIST_LIF_Q_wt1_cut")
    torch.save(wt2_cut, "./parameters_record/MNIST/MNIST_LIF_Q_wt2_cut")


def MNIST_Simulated_Annealing_Optimization(
        wt1_raw, wt2_raw, X_train, y_train, wt_max, wt_min, step):  # 模拟退火优化链接权值
    alpha = 0.99  # 衰减率
    T0, Tf = 97.0, 3.0  # 起始与结束温度

    T = T0  # 当下温度
    cor_raw = MNIST_LIF_Q_test(X_train, y_train, wt1_raw, wt2_raw, not_SA=False)
    print("起始训练集正确率：" + str(cor_raw))
    E_cur = (1.0 - cor_raw) * 1000.0  # 初始内能
    wt1_cur, wt2_cur = wt1_raw, wt2_raw  # 初始化
    time_start = time.time()  # time when training process start
    while T > Tf:
        T = T * alpha  # 温度下降
        print("当下温度：" + str(T))

        # 对权值进行扰动，计算前后误差结果
        changeP1 = torch.rand_like(wt1_raw)
        changeP1 = torch.where(changeP1 > 1.0 - 1e-4, 1.0, 0.0)
        changeM1 = torch.rand_like(wt1_raw)
        changeM1 = torch.where(changeM1 < 1e-4, -1.0, 0.0)

        changeP2 = torch.rand_like(wt2_raw)
        changeP2 = torch.where(changeP2 > 1.0 - 1e-4, 1.0, 0.0)
        changeM2 = torch.rand_like(wt2_raw)
        changeM2 = torch.where(changeM2 < 1e-4, -1.0, 0.0)

        wt1_new = wt1_cur + changeP1 * step
        wt1_new = wt1_new + changeM1 * step
        wt1_new = torch.where(wt1_new - wt_max > 0.0, wt_max, wt1_new)
        wt1_new = torch.where(wt1_new - wt_min < 0.0, wt_min, wt1_new)

        wt2_new = wt2_cur + changeP2 * step
        wt2_new = wt2_new + changeM2 * step
        wt2_new = torch.where(wt2_new - wt_max > 0.0, wt_max, wt2_new)
        wt2_new = torch.where(wt2_new - wt_min < 0.0, wt_min, wt2_new)

        cor_new = MNIST_LIF_Q_test(X_train, y_train, wt1_new, wt2_new, not_SA=False)
        E_new = (1.0 - cor_new) * 1000.0  # 扰动后的内能
        if E_new < E_cur:  # 内能变小时直接接受扰动后结果
            wt1_cur, wt2_cur = wt1_new, wt2_new
            print("一般接受")
            E_cur = E_new
        else:  # 内能变大时依据麦彻玻利斯准则决定是否接受
            Random = random.random()
            if Random <= math.exp((E_cur - E_new) / T):
                print("特殊接受：R: " + str(Random) + ", diff_E:" + str(math.exp((E_cur - E_new) / T)))
                wt1_cur, wt2_cur = wt1_new, wt2_new
                E_cur = E_new
            pass
        pass
        print("当下内能：" + str(E_cur))
        print(" ")
    pass
    print("最终训练集正确率：" + str(MNIST_LIF_Q_test(X_train, y_train, wt1_cur, wt2_cur, not_SA=False)))
    torch.save(wt1_cur, "./parameters_record/MNIST/MNIST_LIF_Q_wt1_SA")
    torch.save(wt2_cur, "./parameters_record/MNIST/MNIST_LIF_Q_wt2_SA")

    time_end = time.time()
    print("Time consuming: %.3f s" % (time_end - time_start))


def MNIST_LIF_Q():  # 98.66/97.32 -->
    # Data prepare
    X_train, y_train = [], []  # (60000,1,28,28)
    for idx, (data, target) in enumerate(Ds.mnist_train_loader):  # read out all data in one time
        X_train, y_train = data, target
    X_test, y_test = [], []
    for idx, (data, target) in enumerate(Ds.mnist_test_loader):
        X_test, y_test = data, target

    '''# Train
    time_start = time.time()
    MNIST_LIF_Q_train(X_train, y_train, X_test, y_test)
    time_end = time.time()
    print("Time consuming: %.3f s" % (time_end - time_start))

    wt1 = torch.load("./parameters_record/MNIST/MNIST_LIF_Q_wt1_cut")
    wt2 = torch.load("./parameters_record/MNIST/MNIST_LIF_Q_wt2_cut")
    MNIST_Simulated_Annealing_Optimization(wt1, wt2, X_train, y_train, 0.2906, -0.2125, 0.021875)'''

    # Test
    wt1 = torch.load("./parameters_record/MNIST/MNIST_LIF_Q_wt1_SA", map_location=torch.device('cpu'))
    wt2 = torch.load("./parameters_record/MNIST/MNIST_LIF_Q_wt2_SA", map_location=torch.device('cpu'))
    # print("Train set: ")
    # MNIST_LIF_Q_test(X_train, y_train, wt1, wt2, not_SA=True)
    print("Test set: ")
    MNIST_LIF_Q_test(X_test, y_test, wt1, wt2, not_SA=False)


def MNIST_LIF_Q_memristor():  # 最终训练好的权值放到忆阻器硬件上后的结果
    # 将电阻分为24个档位
    tap_num = 24
    cond_24 = torch.ones(tap_num) / mc.R_off
    cond_inv = (1.0 / mc.R_on - 1.0 / mc.R_off) / (tap_num - 1.0)
    for i in range(tap_num):
        cond_24[i] += cond_inv * i
    print(cond_24)

    wt1 = torch.load("./parameters_record/MNIST/MNIST_LIF_Q_wt1_SA")
    wt2 = torch.load("./parameters_record/MNIST/MNIST_LIF_Q_wt2_SA")

    '''print("wt1: ")
    print(wt1.size())
    print(wt1)
    wt1_min, wt1_max = torch.min(wt1), torch.max(wt1)
    print("wt1_min: " + str(wt1_min) + ", wt1_max: " + str(wt1_max))

    print("wt2: ")
    print(wt2.size())
    print(wt2)
    wt2_min, wt2_max = torch.min(wt2), torch.max(wt2)
    print("wt2_min: " + str(wt2_min) + ", wt2_max: " + str(wt2_max))

    Hb, Lb = torch.max(wt1_max, wt2_max), torch.min(wt1_min, wt2_min)
    print("Hb: " + str(Hb) + ", Lb: " + str(Lb))
    inv = 0.1
    print("inv: " + str(inv))
    stepN1 = (wt1_max - wt1_min) / inv + 1
    stepN2 = (wt2_max - wt2_min) / inv + 1
    print("stepN1: " + str(stepN1))
    print("stepN2: " + str(stepN2))'''

    # 权值的24个档位
    weight_24 = numpy.arange(-5, 19, 1)
    print(weight_24)
    weight_24 = weight_24 / 10.0  # 权值应是的值
    print(weight_24)
    weight_24 = (weight_24 + 0.6) / 250.0  # 调整使权值落在忆阻器各态范围内
    print(weight_24)

    # 对忆阻器进行调制
    tap_num = 24  # 档位数量
    vol = 1.0  # 调制忆阻器所用电流
    cond_tar = numpy.array(weight_24.reshape((6, 4)))  # 调制的目标电导
    cond_res = mc.adjust_memristor_crossbar(cond_tar, vol)  # 对忆阻器阵列进行调制，得到调制好的电导矩阵
    # print("目标电导矩阵：")
    # print(cond_tar)
    # print("实际电导矩阵：")
    # print(cond_res)

    # 用忆阻器来存放权值
    cond_res = torch.tensor(cond_res).view(-1, tap_num)[0]  # 拉平成24格
    # print(cond_res)
    wt1_m = wt1 * 10.0 + 5.0
    wt2_m = wt2 * 10.0 + 5.0  # 对应0-23，共24个档位
    for i in range(tap_num):
        # print("档位" + str(i) + "记录完成。")
        wt1_m = torch.where(torch.abs(wt1_m - i) < 1e-3, cond_res[i], wt1_m)
        wt2_m = torch.where(torch.abs(wt2_m - i) < 1e-3, cond_res[i], wt2_m)
    print(torch.max(wt1_m))
    print(torch.min(wt1_m))
    print(torch.max(wt2_m))
    print(torch.min(wt2_m))

    # Data prepare
    X_train, y_train = [], []  # (60000,1,28,28)
    for idx, (data, target) in enumerate(Ds.mnist_train_loader):  # read out all data in one time
        X_train, y_train = data, target
    X_test, y_test = [], []
    for idx, (data, target) in enumerate(Ds.mnist_test_loader):
        X_test, y_test = data, target

    ly1 = SNN.LIF_QLayer(inF=784, outF=800, T=4.0, dt=0.01, Lb=-0.05, Hb=0.30, QN=16, wt=None, wt_q=wt1_m)
    ly2 = SNN.LIF_QLayer(inF=800, outF=10, T=4.0, dt=0.01, Lb=-0.05, Hb=0.30, QN=16, wt=None, wt_q=wt2_m)

    # Testing Process
    print("Train set: ")
    correct = 0
    sn, bs = X_train.size()[0], 1  # number of data samples, batch size
    bn = int(math.ceil(sn / bs))  # number of batches

    for bi in range(bn):
        if (bi + 1) * bs > sn:
            data, tar_2 = X_train[bi * bs:sn], y_train[bi * bs:sn]
        else:
            data, tar_2 = X_train[bi * bs:(bi + 1) * bs], y_train[bi * bs:(bi + 1) * bs]
        tar = torch.argmax(tar_2, dim=1)

        # Forward propagation
        t0 = 1.0 - data
        t1 = ly1.forward_Memristor(bs, t0, factor_V=-6.0, R0=2500.0)
        t2 = ly2.forward_Memristor(bs, t1, factor_V=-6.0, R0=2500.0)

        prediction = torch.argmin(t2, dim=1)
        correct += prediction.eq(tar.data).sum()
    pass
    print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
    print("(%.3f %%)" % (100. * correct / sn))
    print(" ")

    print("Test set: ")
    correct = 0
    sn, bs = X_test.size()[0], 1  # number of data samples, batch size
    bn = int(math.ceil(sn / bs))  # number of batches

    for bi in range(bn):
        if (bi + 1) * bs > sn:
            data, tar_2 = X_test[bi * bs:sn], y_test[bi * bs:sn]
        else:
            data, tar_2 = X_test[bi * bs:(bi + 1) * bs], y_test[bi * bs:(bi + 1) * bs]
        tar = torch.argmax(tar_2, dim=1)

        # Forward propagation
        t0 = 1.0 - data
        t1 = ly1.forward_Memristor(bs, t0, factor_V=-6.0, R0=2500.0)
        t2 = ly2.forward_Memristor(bs, t1, factor_V=-6.0, R0=2500.0)

        prediction = torch.argmin(t2, dim=1)
        correct += prediction.eq(tar.data).sum()
    pass
    print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
    print("(%.3f %%)" % (100. * correct / sn))
    print(" ")


def main():
    # Weight_cut(inv=0.021875, u_bound=0.9250, l_bound=-0.5844)
    MNIST_LIF_Q()
    # MNIST_LIF_Q_memristor()


if __name__ == "__main__":
    main()
