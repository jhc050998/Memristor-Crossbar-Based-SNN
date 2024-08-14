import numpy as np

# 惠普忆阻器模型的参数
R_on = 100  # 最小电阻值
R_off = 16000  # 最大电阻值
a = 0.0021  # 斜率因子
V_th = 0.2  # 阈值电压


def adjust_conductance(current_conductance, target_conductance, input_voltage):  # 对单个器件的单次调整
    """
    根据惠普忆阻器模型调整电导

    参数:
    current_conductance (float): 当前电导
    target_conductance (float): 目标电导
    input_voltage (float): 输入电压

    返回:
    调整后的电导
    """
    if input_voltage < V_th:  # 小于阈值的电压不能改变忆阻器电导值
        return current_conductance

    if current_conductance < target_conductance:
        new_conductance = current_conductance + a * (target_conductance - current_conductance) * input_voltage
        return max(new_conductance, 1 / R_off)
    else:
        new_conductance = current_conductance - a * (current_conductance - target_conductance) * input_voltage
        return min(new_conductance, 1 / R_on)


def calculate_current(input_voltage, conductances):  # 用忆阻器阵列进行计算
    """
    计算每行的电流值

    参数:
    input_voltage (float): 输入电压
    conductances (list): 该行的电导值

    返回:
    该行的电流值
    """
    return [input_voltage * g for g in conductances]


def create_and_adjust_array(input_voltage, target_conductances, max_iterations=10000000, tolerance=1e-9,
                            error_tolerance=0.00001):
    """
    创建和调制忆阻器交叉阵列

    参数:
    input_voltage (float): 输入电压
    target_conductances (numpy.ndarray): 目标电导矩阵
    max_iterations (int, optional): 最大迭代次数
    tolerance (float, optional): 收敛容差
    error_tolerance (float, optional): 错误容差

    返回:
    column_currents (numpy.ndarray): 每列的总电流值
    conductances (numpy.ndarray): 调整后的电导矩阵
    """
    num_rows, num_cols = target_conductances.shape

    # 创建忆阻器阵列
    conductances = np.ones((num_rows, num_cols)) * 1 / R_off

    # 执行多次调整
    for iteration in range(max_iterations):
        prev_conductances = conductances.copy()

        # 调整电导值
        for row in range(num_rows):
            for col in range(num_cols):
                target_conductance = target_conductances[row, col]
                conductances[row, col] = adjust_conductance(conductances[row, col], target_conductance, input_voltage)

        # 计算电导矩阵与上一次的差异
        diff = np.abs(conductances - prev_conductances)
        if np.max(diff) < tolerance:
            break

        # 更新输入电压
        if np.abs(conductances - target_conductances).max() > error_tolerance:
            input_voltage = min(input_voltage + 0.1, 5)
        else:
            input_voltage = max(input_voltage - 0.1, -5)

    # 计算每列的总电流值
    column_currents = np.zeros(num_cols)
    for row in range(num_rows):
        row_currents = calculate_current(input_voltage, conductances[row])
        column_currents += row_currents

    return column_currents, conductances


def adjust_conductance_error(target_matrix, actual_matrix):  # 评估调制操作的误差
    """
    Calculates the error matrix and the relative error matrix between the target and actual conductance matrices.

    Args:
        target_matrix (numpy.ndarray): The target conductance matrix.
        actual_matrix (numpy.ndarray): The actual conductance matrix.

    Returns:
        error_matrix (list): The error matrix between the target and actual matrices.
        relative_error_matrix (list): The relative error matrix between the target and actual matrices.
    """
    # Calculate the error matrix
    error_matrix = [[target_matrix[i][j] - actual_matrix[i][j] for j in range(actual_matrix.shape[1])] for i in
                    range(actual_matrix.shape[0])]

    # Calculate the relative error matrix
    relative_error_matrix = [
        [abs((target_matrix[i][j] - actual_matrix[i][j]) / target_matrix[i][j]) for j in range(actual_matrix.shape[1])]
        for i in range(actual_matrix.shape[0])]

    return error_matrix, relative_error_matrix

# -----------------------------------------------------------------------------------------------------------


def adjust_single_memristor():
    # 单器件调制
    cond = 1.0 / R_off
    cond_tar = 0.00417032
    vol = 1.0

    cond_new = cond
    print(cond_new)
    print("")

    epoch = 2000
    for i in range(epoch):
        cond_new = adjust_conductance(cond_new, cond_tar, vol)
        print(cond_new)


def adjust_memristor_crossbar(target_conductances, input_voltage):
    # 输入电压和目标电导矩阵
    column_currents, conductances = create_and_adjust_array(input_voltage, target_conductances)  # 进行调制
    error_conduct, error_matrix = adjust_conductance_error(target_conductances, conductances)  # 评估调制误差

    with open("output.txt", "w") as f:  # 调制好的结果写入文件中保存
        for item in conductances:
            f.write(str(item) + "\n")

    '''print("每列的总电流值:")
    print(column_currents)
    print("电导矩阵:")
    print(conductances)
    print("误差电导矩阵:")
    print(error_conduct)
    print("误差电导矩阵百分比:")
    print(error_matrix)'''

    return conductances


def use_memristor_crossbar():
    # 读取 "output.txt" 文件
    with open('memristor/output.txt', 'r') as file:
        lines = file.readlines()

    # 创建一个空的 NumPy 数组
    data = []

    # 遍历每一行
    for line in lines:
        # 去掉行末的换行符
        line = line.strip()
        # 将行拆分为元素列表
        elements = line.strip('[]').split()
        # 将每个元素转换为浮点数
        row = [float(x) for x in elements]
        data.append(row)

    # 将数据转换为 NumPy 数组
    real_conductances = np.array(data)

    print(real_conductances)

    input_voltages = [1, 2, 3, 4, 5, 6, 7, 8]
    print(input_voltages)

    num_cols = len(real_conductances[0])
    num_rows = len(input_voltages)

    column_currents = np.zeros(num_cols)
    for row in range(num_rows):
        row_currents = calculate_current(input_voltages[row], real_conductances[row])
        column_currents += row_currents

    print(column_currents)


def main():
    vol = 1.0
    cond_tar = np.array(
        [
            [1.0000e-04, 5.2424e-04, 9.4848e-04, 1.3727e-03],
            [1.7970e-03, 2.2212e-03, 2.6455e-03, 3.0697e-03],
            [3.4939e-03, 3.9182e-03, 4.3424e-03, 4.7667e-03],
            [5.1909e-03, 5.6152e-03, 6.0394e-03, 6.4636e-03],
            [6.8879e-03, 7.3121e-03, 7.7364e-03, 8.1606e-03],
            [8.5848e-03, 9.0091e-03, 9.4333e-03, 9.8576e-03]
        ])

    cond_res = adjust_memristor_crossbar(cond_tar, vol)
    print("电导矩阵:")
    print(cond_res)

    # use_memristor_crossbar()

    '''# 将电阻分为24个档位
    tap_num = 24
    cond_24 = torch.ones(tap_num) / R_off
    cond_inv = (1.0/R_on - 1.0/R_off) / (tap_num - 1.0)
    for i in range(tap_num):
        cond_24[i] += cond_inv * i
    # print(cond_24)

    # 权值的24个档位
    weight_24 = torch.arange(-9, 15, 1)
    weight_24 = (weight_24 * 7.0 - 6.0) / 300.0
    weight_24 = (weight_24 + 0.2355) / 55.0
    print(weight_24)'''


if __name__ == "__main__":
    main()
