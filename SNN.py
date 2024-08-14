import numpy
import torch
import matplotlib.pyplot as plt


class LIFLayer:  # 有泄露累积触发模型
    def __init__(self, inF, outF, T, dt, wt=None):
        self.inF, self.outF = inF, outF  # 输入、输出通道数

        self.tau_m, self.tau_s = 4.0, 1.0  # 膜与突触脉冲常数 tau_m = 4.0, tau_s = 1.0
        self.th = 1.0  # 脉冲神经元发射阈值
        self.T = T  # 模拟时长
        self.dt = dt  # 时间片大小
        self.tN = int(T / dt) + 1  # 时间片的数量
        self.tL = (self.tN - 1) / self.T  # 时间粒度，单位时间被分成的份数
        self.t = torch.linspace(0.0, T, self.tN)  # 时间轴

        if wt is None:  # 权值需随机初始化的情况
            self.wt = torch.rand(self.inF, self.outF) * (8.0 / self.inF)  # 均匀分布初始化
        else:
            self.wt = wt  # 权值使用传入参数的情况
        self.cause_mask = torch.tensor(0)  # 引发集记录
        # 学习率自适应（Adam）相关参数
        self.b1, self.b2, self.ep = 0.9, 0.9, 1e-8
        self.step, self.adam_m_wt, self.adam_v_wt = 0, torch.zeros_like(self.wt), torch.zeros_like(self.wt)

    def forward(self, bs, t_in):  # 前向传播过程（由输入得输出），基于模拟膜电位
        bs, inF, outF = bs, self.inF, self.outF  # 批大小，输入、输出通道数
        tau_0 = self.tau_m * self.tau_s / (self.tau_m - self.tau_s)
        T, tN, tL = self.T, self.tN, self.tL

        tEx = torch.tile(torch.reshape(self.t, [1, 1, tN]), [bs, inF, 1])  # 时间轴的扩展 (bs, inF, tN)
        t_inTEx = torch.tile(torch.reshape(t_in, [bs, inF, 1]), [1, 1, tN])  # 输入对时间轴的扩展 (bs, inF, tN)
        t_in_range = torch.where(t_inTEx > tEx, 0, 1)  # (bs, inF, tN) 截断的实现

        t_in_spike = torch.sum(t_in_range * torch.exp((t_inTEx-tEx)/self.tau_s), dim=1)  # (bs, inF, tN)*
        t_in_effect = tau_0 * t_in_range * self.f(t_inTEx, tEx)  # (bs, inF, tN)
        u = torch.bmm(torch.tile(  # (bs, outF, tN) 模拟的膜电位
            torch.reshape(self.wt.T, [1, outF, inF]), [bs, 1, 1]), t_in_effect)
        tEx2 = torch.tile(torch.reshape(self.t, [1, 1, tN]), [bs, outF, 1])
        t_out = torch.argmax(torch.where(u > self.th, self.T-tEx2, -1e1), dim=2) / tL  # (bs, outF)
        t_out = torch.where(t_out <= 0.0, 12.0, t_out)  # 不发射认为是晚发射

        # 完成引发集记录
        t_inEx = torch.tile(torch.reshape(t_in, [bs, inF, 1]), [1, 1, outF])
        t_outEx = torch.tile(torch.reshape(t_out, [bs, 1, outF]), [1, inF, 1])
        self.cause_mask = torch.where(t_inEx <= t_outEx, 1, 0)  # 引发集记录

        '''print("t_in: " + str(t_in[0]))  # 0均是指取第1个批
        print("wt: " + str(self.wt))

        # print("u: " + str(u[0]))
        # print(torch.where(u > self.th, self.T - tEx2, -1e5)[0, catch])

        # print("输出: " + str(t_out[0]))
        # print("引发集: " + str(self.cause_mask[0]))

        print("t_out: " + str(t_out))
        # print(self.cause_mask)
        plt.figure(figsize=(20, 10), dpi=100)
        x = numpy.arange(tN) * T / (tN - 1)
        plt.plot(x, torch.zeros(tN) + 0.21, c="black", linestyle="--")  # 画阈值线
        plt.plot(x, u[0, 0], c="blue")  # 画的是第1批的第1个神经元的膜电位图形
        # plt.plot(x, t_in_spike[0], c="red")
        plt.show()'''

        return t_out

    def pass_delta(self, bs, FB, delta):
        bs, inF, outF = bs, self.inF, self.outF  # 批大小，输入、输出通道数
        deltaEx = torch.tile(torch.reshape(delta, [bs, 1, outF]), [1, inF, 1])
        delta_out = torch.sum(deltaEx * FB * self.cause_mask, dim=2)
        return delta_out

    def backward(self, bs, delta, t_in, t_out, lr):
        bs, inF, outF = bs, self.inF, self.outF  # 批大小，输入、输出通道数
        deltaEx = torch.tile(torch.reshape(delta, [bs, 1, outF]), [1, inF, 1])
        t_inEx = torch.tile(torch.reshape(t_in, [bs, inF, 1]), [1, 1, outF])
        t_outEx = torch.tile(torch.reshape(t_out, [bs, 1, outF]), [1, inF, 1])
        adj_wt = torch.sum(deltaEx * self.f(t_inEx, t_outEx) * self.cause_mask, dim=0)

        # 不用自适应
        # self.wt -= lr * adj_wt

        # Adam
        self.step += 1
        self.adam_m_wt = self.b1 * self.adam_m_wt + (1.0 - self.b1) * adj_wt
        self.adam_v_wt = self.b2 * self.adam_v_wt + (1.0 - self.b2) * adj_wt * adj_wt
        M_wt = self.adam_m_wt / (1.0 - self.b1 ** self.step)
        V_wt = self.adam_v_wt / (1.0 - self.b2 ** self.step)
        self.wt -= lr * (M_wt / (torch.sqrt(V_wt) + self.ep))

    def f(self, t1, t2):  # 见笔记
        Tm, Ts = self.tau_m, self.tau_s
        return torch.exp((t1-t2)/Tm) - torch.exp((t1-t2)/Ts)  # 返回结果与t1、t2皆同型

    def g(self, w, t1, t2):  # 见笔记
        Tm, Ts = self.tau_m, self.tau_s
        return w * (torch.exp((t1-t2)/Tm)/Tm - torch.exp((t1-t2)/Ts)/Ts)  # 返回结果与w、t1、t2皆同型


class LIF_QLayer:  # 量化突触权值模型
    def __init__(self, inF, outF, T, dt, Lb, Hb, QN, wt=None, wt_q=None):
        self.QN = QN
        self.inF, self.outF = inF, outF  # 输入、输出通道数

        self.tau_m, self.tau_s = 4.0, 1.0  # 膜与突触脉冲常数
        self.th = 1.0  # 脉冲神经元发射阈值
        self.T = T  # 模拟时长
        self.dt = dt  # 时间片大小
        self.tN = int(T / dt) + 1  # 时间片的数量
        self.tL = (self.tN - 1) / self.T  # 时间粒度，单位时间被分成的份数
        self.t = torch.linspace(0.0, T, self.tN)  # 时间轴

        if wt_q is None:
            if wt is None:
                self.wt = torch.rand(self.inF, self.outF) * (8.0 / self.inF)  # 均匀分布初始化
            else:
                self.wt = wt  # 权值使用传入参数的情况
            self.wt_q = torch.zeros_like(self.wt)  # 量化后的权值
            self.wt_inv = (Hb - Lb) / (QN - 1)
            self.Lb, self.Hb = Lb, Hb
            self.weight_quantize()
        else:
            if wt is None:
                self.wt = wt_q
            else:
                self.wt = wt
            self.wt_q = wt_q

        self.cause_mask = torch.tensor(0)  # 引发集记录
        # 学习率自适应（Adam）相关参数
        self.b1, self.b2, self.ep = 0.9, 0.9, 1e-8
        self.step, self.adam_m_wt, self.adam_v_wt = 0, torch.zeros_like(self.wt), torch.zeros_like(self.wt)

    def weight_quantize(self):
        # print("量化操作1次")
        # 限制范围
        # self.wt_q = torch.where(self.wt - self.Hb > 0.0, self.Hb, self.wt)
        # self.wt_q = torch.where(self.wt_q - self.Lb < 0.0, self.Lb, self.wt_q)

        # 量化到各档位上
        self.wt_q = torch.round((self.wt + self.Lb) / self.wt_inv)
        self.wt_q = self.wt_q * self.wt_inv - self.Lb

    def forward(self, bs, t_in):  # 前向传播过程（由输入得输出），基于模拟膜电位
        bs, inF, outF = bs, self.inF, self.outF  # 批大小，输入、输出通道数
        tau_0 = self.tau_m * self.tau_s / (self.tau_m - self.tau_s)
        T, tN, tL = self.T, self.tN, self.tL

        tEx = torch.tile(torch.reshape(self.t, [1, 1, tN]), [bs, inF, 1])  # 时间轴的扩展 (bs, inF, tN)
        t_inTEx = torch.tile(torch.reshape(t_in, [bs, inF, 1]), [1, 1, tN])  # 输入对时间轴的扩展 (bs, inF, tN)
        t_in_range = torch.where(t_inTEx > tEx, 0, 1)  # (bs, inF, tN) 截断的实现
        t_in_effect = tau_0 * t_in_range * self.f(t_inTEx, tEx)  # (bs, inF, tN)
        u = torch.bmm(torch.tile(  # (bs, outF, tN) 模拟的膜电位
            torch.reshape(self.wt_q.T, [1, outF, inF]), [bs, 1, 1]), t_in_effect)
        tEx2 = torch.tile(torch.reshape(self.t, [1, 1, tN]), [bs, outF, 1])
        t_out = torch.argmax(torch.where(u > self.th, self.T - tEx2, -1e1), dim=2) / tL  # (bs, outF)
        t_out = torch.where(t_out <= 0.0, 12.0, t_out)  # 不发射认为是晚发射

        # 完成引发集记录
        t_inEx = torch.tile(torch.reshape(t_in, [bs, inF, 1]), [1, 1, outF])
        t_outEx = torch.tile(torch.reshape(t_out, [bs, 1, outF]), [1, inF, 1])
        self.cause_mask = torch.where(t_inEx <= t_outEx, 1, 0)  # 引发集记录

        return t_out

    def forward_Memristor(self, bs, t_in, factor_V, R0):  # 前向过程中不出现wt，仅用wt_q（量化后的值）
        bs, inF, outF = bs, self.inF, self.outF  # 批大小，输入、输出通道数
        tau_0 = self.tau_m * self.tau_s / (self.tau_m - self.tau_s)
        T, tN, tL = self.T, self.tN, self.tL

        tEx = torch.tile(torch.reshape(self.t, [1, 1, tN]), [bs, inF, 1])  # 时间轴的扩展 (bs, inF, tN)
        t_inTEx = torch.tile(torch.reshape(t_in, [bs, inF, 1]), [1, 1, tN])  # 输入对时间轴的扩展 (bs, inF, tN)
        t_in_range = torch.where(t_inTEx > tEx, 0, 1)  # (bs, inF, tN) 截断的实现
        t_in_effect = tau_0 * t_in_range * self.f(t_inTEx, tEx)  # (bs, inF, tN)

        # 需将t_in_effect编成电压信号，权值已是忆阻器阻值
        V_Memristor_in = t_in_effect / 10.0  # 计算时电压需小于阈值电压
        G_Memristor = torch.reshape(self.wt_q.T, [1, outF, inF])

        I_out = torch.bmm(G_Memristor, V_Memristor_in)  # 忆阻器阵列的输出电流
        V_adjust = factor_V * torch.bmm(torch.ones_like(G_Memristor), V_Memristor_in)  # 对其进行调整的电压
        V_Memristor_out = R0 * I_out + V_adjust  # 硬件的最后输出

        # 回到软件处理
        u = V_Memristor_out
        tEx2 = torch.tile(torch.reshape(self.t, [1, 1, tN]), [bs, outF, 1])
        t_out = torch.argmax(torch.where(u > self.th, self.T - tEx2, -1e1), dim=2) / tL  # (bs, outF)
        t_out = torch.where(t_out <= 0.0, 12.0, t_out)  # 不发射认为是晚发射

        return t_out

    def pass_delta(self, bs, FB, delta):
        bs, inF, outF = bs, self.inF, self.outF  # 批大小，输入、输出通道数
        deltaEx = torch.tile(torch.reshape(delta, [bs, 1, outF]), [1, inF, 1])
        delta_out = torch.sum(deltaEx * FB * self.cause_mask, dim=2)
        return delta_out

    def backward(self, bs, delta, t_in, t_out, lr):
        bs, inF, outF = bs, self.inF, self.outF  # 批大小，输入、输出通道数
        deltaEx = torch.tile(torch.reshape(delta, [bs, 1, outF]), [1, inF, 1])
        t_inEx = torch.tile(torch.reshape(t_in, [bs, inF, 1]), [1, 1, outF])
        t_outEx = torch.tile(torch.reshape(t_out, [bs, 1, outF]), [1, inF, 1])
        adj_wt = torch.sum(deltaEx * self.f(t_inEx, t_outEx) * self.cause_mask, dim=0)

        # 不用自适应
        # self.wt -= lr * adj_wt

        # Adam
        self.step += 1
        self.adam_m_wt = self.b1 * self.adam_m_wt + (1.0 - self.b1) * adj_wt
        self.adam_v_wt = self.b2 * self.adam_v_wt + (1.0 - self.b2) * adj_wt * adj_wt
        M_wt = self.adam_m_wt / (1.0 - self.b1 ** self.step)
        V_wt = self.adam_v_wt / (1.0 - self.b2 ** self.step)
        self.wt -= lr * (M_wt / (torch.sqrt(V_wt) + self.ep))

        self.weight_quantize()

    def f(self, t1, t2):  # 见笔记
        Tm, Ts = self.tau_m, self.tau_s
        return torch.exp((t1-t2)/Tm) - torch.exp((t1-t2)/Ts)  # 返回结果与t1、t2皆同型

    def g(self, w, t1, t2):  # 见笔记
        Tm, Ts = self.tau_m, self.tau_s
        return w * (torch.exp((t1-t2)/Tm)/Tm - torch.exp((t1-t2)/Ts)/Ts)  # 返回结果与w、t1、t2皆同型


def main():
    wt = torch.tensor([[0.6469], [0.7015], [0.6190], [0.7578], [0.3983],
                       [0.4506], [0.0016], [0.2745], [0.7095], [0.5000]])
    neuron = LIFLayer(inF=10, outF=1, T=4.0, dt=0.01)

    # sp_in_rand = torch.reshape(torch.rand(10) * 4.0, (1, 10))
    # sp_out = neuron.forward(bs=1, t_in=sp_in_rand)

    sp_in = torch.reshape(torch.tensor(
        [0.7195, 0.9184, 1.8563, 3.7564, 3.4255, 1.9255, 2.4868, 3.2514, 2.1617, 1.6805]), (1, 10))
    sp_out = neuron.forward(bs=1, t_in=sp_in)


if __name__ == "__main__":
    main()
