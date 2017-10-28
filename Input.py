import Data_preprocessing as dp
import Data_reader as reader
import numpy as np


class Input():

    def __init__(self, probability, file_name='./resource/data.csv', feature_name=None):
        if probability <= 0.05:
            self.scale = 50
        elif probability > 0.05 and probability <= 0.5:
            self.scale = 35
        else:
            self.scale = 10
        self.flag = []
        if feature_name is None:
            feature_name = ['POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT', 'RR_002', 'A_002']
        data = reader.data_formatting(
            reader.data_extracter(feature_num=91, file_name=file_name, data_size=614))
        data = dp.shuffle(data)
        self.feature, self.label = reader.feature_extracter(data, feature_name, has_name=False)

        self.feature = reader.data_formatting(self.feature)
        self.label = reader.data_formatting(self.label)
        self.feature, self.label = dp.kill_missing(self.feature, self.label)
        self.feature = (dp.fill_missing(self.feature, method='mean'))
        for i in range(4):
            self.flag.append(self.feature.pop(-2))
        #print(self.flag)
        self.vaild_list = []



    def get_feature(self):
        return self.feature

    def normalize(self):
        self.feature = dp.normalize(self.feature, type='normal')
        self.label = dp.normalize(self.label, type='constant', scale=self.scale)

    def get_duration(self):
        return self.feature[2]

    def delete_duration(self):
        self.feature.pop(2)

    def get_vaild_flag(self, probability):
        for i in range(len(self.flag[3])):
            if self.flag[3][i] <= probability and self.flag[2][i] >= probability and self.flag[3][i] != 0:
                self.vaild_list.append(i)



    def data_generator(self, is_split=True, probability=0.01):
        feature = np.transpose(np.array(self.feature))
        label = np.transpose(np.array(self.label))
        print('total numbers of data is', len(feature))
        total_list = []
        for t in range(len(feature)):
            total_list.append(t)
        test_num = int(len(feature) * 0.2)
        vail_num = int(len(feature) * 0.1)
        self.get_vaild_flag(probability)
        for v in self.vaild_list:
            if v in total_list:
                total_list.remove(v)
            if len(total_list) <= len(feature) * 0.8:
                break
        for iv in self.vaild_list:
            if iv in total_list:
                self.vaild_list.remove(iv)



        test_feature = []
        test_label = []
        train_feature = []
        train_label = []
        vaildation_feature = []
        vaildation_label = []
        for vaild_num in self.vaild_list:
            test_feature.append(feature[vaild_num])
            test_label.append(label[vaild_num])
        for tra in total_list[:int(0.875 * len(total_list))]:
            train_feature.append(feature[tra])
            train_label.append(label[tra])
        for va in total_list[int(0.875 * len(total_list)):]:
            vaildation_feature.append(feature[va])
            vaildation_label.append(label[va])
        if is_split:
            return np.array(train_feature), np.array(train_label), np.array(vaildation_feature), np.array(vaildation_label), np.array(test_feature), np.array(test_label)
        return feature, label


