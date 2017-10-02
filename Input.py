import Data_preprocessing as dp
import Data_reader as reader
import numpy as np

class Input():

    def __init__(self, file_name='./resource/data.csv'):
        data = reader.data_formatting(
            reader.data_extracter(feature_num=71, file_name=file_name, data_size=91))
        self.feature, self.label = reader.feature_extracter(data, ['PATH_LTH', 'FREQUENCY', 'RR_01', 'A_01'], has_name=False)
        self.feature = reader.data_formatting(self.feature)
        self.label = reader.data_formatting(self.label)
        self.feature, self.label = dp.kill_missing(self.feature, self.label)
        # print(len(feature[0]), len(label[0]))
        self.feature = (dp.fill_missing(self.feature, method='mean'))
        self.label = (dp.fill_missing(self.label, method='mean'))
        self.feature = dp.normalize(self.feature, is_max=True)
        self.label = dp.normalize(self.label, is_max=True)

    def data_generator(self, is_split=True):
        feature = np.transpose(np.array(self.feature))
        label = np.transpose(np.array(self.label))
        if is_split:
            return feature[10:82], label[10:82], feature[:10], label[:10]
        return feature, label


