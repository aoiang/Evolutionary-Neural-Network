import Data_preprocessing as dp
import Data_reader as reader
import numpy as np
import random

class Input():

    def __init__(self, file_name='./resource/C2_1_v9_20171025.csv', feature_name=None):
        if feature_name is None:
            feature_name = ['POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT', 'RR_002', 'A_002']
        data = reader.data_formatting(
            reader.data_extracter(feature_num=90, file_name=file_name, data_size=614))
        attribute = [data[0]]
        data.pop(0)
        random.shuffle(data)
        attribute.extend(data)
        data = attribute
        self.feature, self.label = reader.feature_extracter(data, feature_name, has_name=False)
        self.feature = reader.data_formatting(self.feature)
        self.label = reader.data_formatting(self.label)

        self.feature, self.label = dp.kill_missing(self.feature, self.label)
        self.feature = (dp.fill_missing(self.feature, method='mean'))

    def get_feature(self):
        return self.feature

    def normalize(self):
        self.feature = dp.normalize(self.feature, type='normal')
        self.label = dp.normalize(self.label, type='constant')

    def get_duration(self):
        return self.feature[2]

    def delete_duration(self):
        self.feature.pop(2)

    def data_generator(self, is_split=True, split_num=30):
        feature = np.transpose(np.array(self.feature))
        label = np.transpose(np.array(self.label))
        print(len(feature))
        test_num = int(len(feature) * 0.2)
        vail_num = int(len(feature) * 0.1)
        #random.shuffle(fe)
        if is_split:
            return feature[test_num:-(vail_num)], label[test_num:-(vail_num)], feature[-(vail_num):], label[-(vail_num):],  feature[:test_num], label[:test_num]
        return feature, label


