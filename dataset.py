import os
from torch.utils.data.dataset import Dataset
import torch
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler

# 数据预处理
scaler = StandardScaler()


# 填平补齐
def pad_and_trun(input):
    output = []
    for i in input:
        if i.shape[0] > 256:
            i = i[:256][:]
            output.append(i)
        elif i.shape[0] < 256:
            i = np.concatenate((i, np.zeros((256 - i.shape[0], 3))))
            output.append(i)
    return np.array(output)


# 取数据
def get_sensor_data(file_path, frequency=8):
    acc_file = open(file_path, 'r', encoding='utf-8')
    # gyro_file = open(file_path.replace('acc','gyro'), 'r', encoding='utf-8')
    # ori = open(file_path.replace('acc','ori'), 'r', encoding='utf-8')
    data = []
    for idx, i in enumerate(acc_file.readlines()[16:]):
        if idx % frequency == 0:
            acc_data = i.strip().split(', ')[1:]
            acc_data = [np.float(i) for i in acc_data]
            # print(acc_data)
            data.append(acc_data)

    return scaler.fit_transform(data)


# 遍历数据
class state_dataset(Dataset):
    def __init__(self, data_path, frequency=8):
        self.user_path = os.listdir(data_path)
        self.adl_path = []
        self.fall_path = []
        self.adl_x_list = []
        self.fall_x_list = []
        for self.secend_path in tqdm(self.user_path):
            self.third_path = os.listdir(data_path + self.secend_path)
            for path in self.third_path:
                if path == 'ADL':
                    self.fourth_path = os.listdir(data_path + self.secend_path + '/' + path)
                    for i in self.fourth_path:
                        self.final_path = os.listdir(data_path + self.secend_path + '/' + path + '/' + i)
                        self.adl_path.extend(
                            [data_path + self.secend_path + '/' + path + '/' + i + '/' + j for j in self.final_path])
                        for item in self.adl_path:
                            if 'gyro' in item:
                                assert get_sensor_data(item, frequency).shape[1] == 3
                                self.adl_x_list.append(get_sensor_data(item, frequency))

                elif path == 'FALLS':
                    self.fourth_path = os.listdir(data_path + self.secend_path + '/' + path)
                    for i in self.fourth_path:
                        self.final_path = os.listdir(data_path + self.secend_path + '/' + path + '/' + i)
                        self.fall_path.extend(
                            [data_path + self.secend_path + '/' + path + '/' + i + '/' + j for j in self.final_path])
                        # print(self.fall_path)
                        for item in self.fall_path:
                            # print(item)
                            if 'acc' in item:
                                assert get_sensor_data(item, frequency).shape[1] == 3
                                self.fall_x_list.append(get_sensor_data(item, frequency))

        self.adl_x_list = pad_and_trun(self.adl_x_list)
        self.fall_x_list = pad_and_trun(self.fall_x_list)
        self.adl_y = np.zeros((self.adl_x_list.shape[0]))
        self.fall_y = np.ones((self.fall_x_list.shape[0]))
        self.x_data = torch.tensor(np.concatenate((self.adl_x_list, self.fall_x_list), axis=0), dtype=torch.float)
        self.y_data = torch.tensor(np.concatenate((self.adl_y, self.fall_y), axis=0), dtype=torch.int64)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)
