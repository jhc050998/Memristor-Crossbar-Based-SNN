import math
import numpy
import random
import time
import torch

import Dataset as Ds
import SNN

from memristor import memristor_vi_crossbar as mc


def WDBC_LIF_Q_train(X_train, y_train, X_test, y_test):
    # Network layout
    ly1 = SNN.LIF_QLayer(inF=30, outF=40, T=4.0, dt=0.01, Lb=0.0, Hb=0.40, QN=5)
    ly2 = SNN.LIF_QLayer(inF=40, outF=2, T=4.0, dt=0.01, Lb=0.0, Hb=0.40, QN=5)

    # Training process
    epoch_num = 600
    interval = 50
    lr_start, lr_end = 1e-2, 1e-3  # decaying learning rate
    lr_decay = (lr_end / lr_start) ** (1.0 / epoch_num)

    sn, bs = X_train.size()[0], X_train.size()[0]  # number of data samples, batch size
    bn = int(math.ceil(sn / bs))  # number of batches
    loss, total_loss = 0, []
    # acc, total_acc = 0, []
    wt_max, wt_min = 0.0, 0.0
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

            # Backward propagation (Gradient)
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
            total_loss.append(loss)
        if epoch % interval == 0:
            print("Current Training epoch: " + str(epoch + 1), end="\t")
            print("Progress: [" + str(epoch) + "/" + str(epoch_num), end="")
            print("(%.0f %%)]" % (100.0 * epoch / epoch_num), end="\t")
            print(" ")
            print("Loss: " + str(loss))
        pass
        '''#每轮测试一次
        acc = WDBC_LIF_Q_test(X_test, y_test, ly1.wt, ly2.wt, False)
        total_acc.append(acc)'''
    pass
    torch.save(ly1.wt_q, "./parameters_record/WDBC/WDBC_LIF_Q_wt1")
    torch.save(ly2.wt_q, "./parameters_record/WDBC/WDBC_LIF_Q_wt2")
    torch.save(total_loss, "./parameters_record/WDBC/WDBC_Q_loss")
    # torch.save(total_acc, "./parameters_record/WDBC/WDBC_Q_acc")

    # 对量化训练后所得权值的评估
    '''print("wt_min: " + str(wt_min) + ", wt_max: " + str(wt_max))
    wt_q1_min, wt_q1_max = torch.min(ly1.wt_q), torch.max(ly1.wt_q)
    print("wt_q1_min: " + str(wt_q1_min) + ", wt_q1_max: " + str(wt_q1_max))
    wt_q2_min, wt_q2_max = torch.min(ly2.wt_q), torch.max(ly2.wt_q)
    print("wt_q2_min: " + str(wt_q2_min) + ", wt_q2_max: " + str(wt_q2_max))

    inv = (ly1.Hb - ly1.Lb) / (ly1.QN - 1)
    print("inv: " + str(inv))
    stepN1 = (wt_q1_max - wt_q1_min) / inv + 1
    stepN2 = (wt_q2_max - wt_q2_min) / inv + 1
    print("stepN1: " + str(stepN1))
    print("stepN2: " + str(stepN2))'''


def WDBC_LIF_Q_test(X, y, wt1, wt2, not_SA):
    # Network layout
    ly1 = SNN.LIF_QLayer(inF=30, outF=40, T=4.0, dt=0.01, Lb=0.0, Hb=0.40, QN=5, wt=wt1)
    ly2 = SNN.LIF_QLayer(inF=40, outF=2, T=4.0, dt=0.01, Lb=0.0, Hb=0.40, QN=5, wt=wt2)

    if not_SA:
        print("wt_1: ")
        print(ly1.wt_q.size())
        # print(ly1.wt_q)
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
    if not_SA:
        print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
        print("(%.3f %%)" % (100. * correct/sn))
        print(" ")
    return TP, TN, FP, FN  # correct/sn


def Weight_cut(inv, u_bound, l_bound):
    wt1 = torch.load("./parameters_record/WDBC/WDBC_LIF_Q_wt1")
    wt2 = torch.load("./parameters_record/WDBC/WDBC_LIF_Q_wt2")

    wt1_cut = torch.where(wt1 >= u_bound - inv * 0.0, u_bound - inv * 0.0, wt1)
    wt1_cut = torch.where(wt1_cut <= l_bound + inv * 0.0, l_bound + inv * 0.0, wt1_cut)
    wt2_cut = torch.where(wt2 >= u_bound - inv * 0.0, u_bound - inv * 0.0, wt2)
    wt2_cut = torch.where(wt2_cut <= l_bound + inv * 0.0, l_bound + inv * 0.0, wt2_cut)

    torch.save(wt1_cut, "./parameters_record/WDBC/WDBC_LIF_Q_wt1_cut")
    torch.save(wt2_cut, "./parameters_record/WDBC/WDBC_LIF_Q_wt2_cut")


def WDBC_Simulated_Annealing_Optimization(wt1_raw, wt2_raw, X_train, y_train, wt_max, wt_min, step):  # 模拟退火优化链接权值
    alpha = 0.99  # 衰减率
    T0, Tf = 97.0, 3.0  # 起始与结束温度

    T = T0  # 当下温度
    cor_raw = WDBC_LIF_Q_test(X_train, y_train, wt1_raw, wt2_raw, not_SA=False)
    print("起始训练集正确率：" + str(cor_raw))
    E_cur = (1.0 - cor_raw) * 1000.0  # 初始内能
    wt1_cur, wt2_cur = wt1_raw, wt2_raw  # 初始化
    while T > Tf:
        T = T*alpha  # 温度下降
        print("当下温度：" + str(T))

        # 对权值进行扰动，计算前后误差结果
        changeP1 = torch.rand_like(wt1_raw)
        changeP1 = torch.where(changeP1 > 0.999, 1.0, 0.0)
        changeM1 = torch.rand_like(wt1_raw)
        changeM1 = torch.where(changeM1 < 0.001, -1.0, 0.0)

        changeP2 = torch.rand_like(wt2_raw)
        changeP2 = torch.where(changeP2 > 0.999, 1.0, 0.0)
        changeM2 = torch.rand_like(wt2_raw)
        changeM2 = torch.where(changeM2 < 0.001, -1.0, 0.0)

        wt1_new = wt1_cur + changeP1 * step
        wt1_new = wt1_new + changeM1 * step
        wt1_new = torch.where(wt1_new - wt_max > 0.0, wt_max, wt1_new)
        wt1_new = torch.where(wt1_new - wt_min < 0.0, wt_min, wt1_new)

        wt2_new = wt2_cur + changeP2 * step
        wt2_new = wt2_new + changeM2 * step
        wt2_new = torch.where(wt2_new - wt_max > 0.0, wt_max, wt2_new)
        wt2_new = torch.where(wt2_new - wt_min < 0.0, wt_min, wt2_new)

        cor_new = WDBC_LIF_Q_test(X_train, y_train, wt1_new, wt2_new, not_SA=False)
        E_new = (1.0 - cor_new) * 1000.0  # 扰动后的内能
        if E_new < E_cur:  # 内能变小时直接接受扰动后结果
            wt1_cur, wt2_cur = wt1_new, wt2_new
            E_cur = E_new
        else:  # 内能变大时依据麦彻玻利斯准则决定是否接受
            Random = random.random()
            if Random <= math.exp((E_cur - E_new)/T):
                print("特殊接受：R: " + str(Random) + ", diff_E:" + str(math.exp((E_cur - E_new)/T)))
                wt1_cur, wt2_cur = wt1_new, wt2_new
                E_cur = E_new
            pass
        pass
        print("当下内能：" + str(E_cur))
        print(" ")
    pass
    print("最终训练集正确率：" + str(WDBC_LIF_Q_test(X_train, y_train, wt1_cur, wt2_cur, not_SA=False)))
    torch.save(wt1_cur, "./parameters_record/WDBC/WDBC_LIF_Q_wt1_SA")
    torch.save(wt2_cur, "./parameters_record/WDBC/WDBC_LIF_Q_wt2_SA")


def WDBC_LIF_Q():  # 96.48/95.09  -->  97.54/96.49
    # Data prepare
    X, y = Ds.WDBC_loader(isSave=False, isLoad=True)

    # Read out data
    f = 0
    siz = 285
    X_train, y_train = torch.cat(
        (X[0:f * siz], X[f * siz + siz:569]), dim=0), torch.cat((y[0:f * siz], y[f * siz + siz:569]), dim=0)
    X_test, y_test = X[f * siz:f * siz + siz], y[f * siz:f * siz + siz]

    '''# Train
    time_start = time.time()
    WDBC_LIF_Q_train(X_train, y_train, X_test, y_test)
    time_end = time.time()
    print("Time consuming: %.3f s" % (time_end - time_start))

    wt1 = torch.load("./parameters_record/WDBC/WDBC_LIF_Q_wt1_cut")
    wt2 = torch.load("./parameters_record/WDBC/WDBC_LIF_Q_wt2_cut")
    WDBC_Simulated_Annealing_Optimization(wt1, wt2, X_train, y_train, 1.8, -0.5, 0.1)

    # Test
    wt1 = torch.load("./parameters_record/WDBC/WDBC_LIF_Q_wt1")
    wt2 = torch.load("./parameters_record/WDBC/WDBC_LIF_Q_wt2")
    print("Train set: ")
    WDBC_LIF_Q_test(X_train, y_train, wt1, wt2, not_SA=True)
    print("Test set: ")
    WDBC_LIF_Q_test(X_test, y_test, wt1, wt2, not_SA=True)'''

    ep = 24
    time_start = time.time()
    T4 = torch.zeros(4, ep)  # TP, TN, FP, FN
    for i in range(ep):
        WDBC_LIF_Q_train(X_train, y_train, X_test, y_test)
        wt1 = torch.load("./parameters_record/WDBC/WDBC_LIF_Q_wt1")
        wt2 = torch.load("./parameters_record/WDBC/WDBC_LIF_Q_wt2")

        T4[0, i], T4[1, i], T4[2, i], T4[3, i] = WDBC_LIF_Q_test(X_test, y_test, wt1, wt2, False)
        print("第" + str(i + 1) + "轮结束。")
    pass
    time_end = time.time()
    print("Time consuming: %.3f s" % (time_end - time_start))
    # print(T4)
    torch.save(T4, "./parameters_record/WDBC/WDBC_Q_T4")


def WDBC_LIF_Q_memristor():  # 最终训练好的权值放到忆阻器硬件上后的结果
    # 将电阻分为24个档位
    tap_num = 24
    cond_24 = torch.ones(tap_num)/mc.R_off
    cond_inv = (1.0/mc.R_on - 1.0/mc.R_off) / (tap_num - 1.0)
    for i in range(tap_num):
        cond_24[i] += cond_inv * i
    print(cond_24)

    wt1 = torch.load("./parameters_record/WDBC/WDBC_LIF_Q_wt1_SA")
    wt2 = torch.load("./parameters_record/WDBC/WDBC_LIF_Q_wt2_SA")

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
    X, y = Ds.WDBC_loader(isSave=False, isLoad=True)

    # Read out data
    f = 0
    siz = 285
    X_train, y_train = torch.cat(
        (X[0:f * siz], X[f * siz + siz:569]), dim=0), torch.cat((y[0:f * siz], y[f * siz + siz:569]), dim=0)
    X_test, y_test = X[f * siz:f * siz + siz], y[f * siz:f * siz + siz]

    ly1 = SNN.LIF_QLayer(inF=30, outF=40, T=4.0, dt=0.01, Lb=0.0, Hb=0.40, QN=5, wt=None, wt_q=wt1_m)
    ly2 = SNN.LIF_QLayer(inF=40, outF=2, T=4.0, dt=0.01, Lb=0.0, Hb=0.40, QN=5, wt=None, wt_q=wt2_m)

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
    # Weight_cut(inv=0.1, u_bound=1.5, l_bound=-0.8)
    WDBC_LIF_Q()
    # WDBC_LIF_Q_memristor()

    T4 = torch.load("./parameters_record/WDBC/WDBC_Q_T4")
    # print(T4.size())
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
