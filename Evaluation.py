import Evolution as ep
import Input as inputs
import math



def get_output(probability):
    feature_name = probability_selector(probability)
    input_init = inputs.Input(probability, feature_name=feature_name)
    duration = input_init.get_duration()
    input_init.delete_duration()
    input_init.normalize()
    tra_feature, tra_label, vaildation_feature, vaildation_label, test_feature, test_label = input_init.data_generator()
    vaild_list = input_init.vaild_list
    vaild_dur = []
    for vaild_num in vaild_list:
        vaild_dur.append(duration[vaild_num])

    optimal_net = ep.evolve(30, 1, tra_feature, tra_label, vaildation_feature, vaildation_label, test_feature, test_label, type='random')
    w, b, e, r, predict, test = ep.Evolution().network_optimizer(tra_feature, tra_label, test_feature, test_label, optimal_net, 18600, final=True)
    print(vaild_dur)
    return e, predict, test, vaild_dur

def format_output(predict_old, test_old):
    predict = []
    test = []
    for i in range(len(predict_old)):
        predict.append(predict_old[i][0])
        test.append(test_old[i][0])
    return predict, test

def evaluation(predict, test, duration):
    v = []
    s = []
    square = []
    error = []


    for i in range(len(predict)):
        if predict[i] < 0:
            predict[i] = 0.1
        s.append(predict[i] / test[i])
        if test[i] <= 10:
            v.append(100 * (test[i] / 10)**0.2 * math.log(s[i], math.e))
        else:
            v.append(100 * math.log(s[i], math.e))
    for i in range(len(predict)):
        square.append(duration[i]*(v[i]**2))
        error.append(v[i]*duration[i])
    sum_VS = sum(square)
    sum_VR = sum(error)
    sum_WY = sum(duration)
    mean_error = sum_VR / sum_WY
    rms = (sum_VS / sum_WY) ** 0.5
    std = (sum_VS / sum_WY - mean_error ** 2) ** 0.5
    print('rms is', rms, file=f)
    print('mean_error is', mean_error, file=f)
    print('std is', std, file=f)

def probability_selector(probability):
    feature_name = None
    if probability == 0.01:
        feature_name = ['DURATION', 'ATT001', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2',  'RR_001', 'A_001', 'FLAG1', 'FLAG2', 'FLAG3', 'FLAG4']
    if probability == 0.02:
        feature_name = ['DURATION', 'ATT002', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_002', 'A_002', 'FLAG1', 'FLAG2',
         'FLAG3', 'FLAG4']
    if probability == 0.03:
        feature_name = ['DURATION', 'ATT003', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_003', 'A_003',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 0.05:
        feature_name = ['DURATION', 'ATT005', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_005', 'A_005',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 0.1:
        feature_name = ['DURATION', 'ATT01', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_01', 'A_01',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 0.2:
        feature_name = ['DURATION', 'ATT02', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_02', 'A_02',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 0.3:
        feature_name = ['DURATION', 'ATT03', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_03', 'A_03',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 0.5:
        feature_name = ['DURATION', 'ATT05', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_05', 'A_05',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == '_1':
        feature_name = ['DURATION', 'ATT_1', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_1', 'A_1',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == '_2':
        feature_name = ['DURATION', 'ATT_2', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_2', 'A_2',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == '_3':
        feature_name = ['DURATION', 'ATT_3', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_3', 'A_3',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == '_5':
        feature_name = ['DURATION', 'ATT_5', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR_5', 'A_5',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 1:
        feature_name = ['DURATION', 'ATT1', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR1', 'A1',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 2:
        feature_name = ['DURATION', 'ATT2', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR2', 'A2',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 3:
        feature_name = ['DURATION', 'ATT3', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR3', 'A3',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 5:
        feature_name = ['DURATION', 'ATT5', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR5', 'A5',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 10:
        feature_name = ['DURATION', 'ATT10', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR10', 'A10',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 20:
        feature_name = ['DURATION', 'ATT20', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR20', 'A20',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 30:
        feature_name = ['DURATION', 'ATT30', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR30', 'A30',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    if probability == 50:
        feature_name = ['DURATION', 'ATT50', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT2', 'RR50', 'A50',
                        'FLAG1', 'FLAG2',
                        'FLAG3', 'FLAG4']
    return feature_name

if __name__ == "__main__":
    f = open('./result.txt', 'a')
    testlist = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50]
    # for test_p in testlist:
    #     if test_p <= 0.05:
    #         scale = 50
    #     elif test_p > 0.05 and test_p <= 0.5:
    #         scale = 35
    #     else:
    #         scale = 10
    #     for i in range(5):
    #         error, predict, test, duration = get_output(test_p)
    #         predict *= scale
    #         test *= scale
    #         predict, test = format_output(predict, test)
    #         for i in range(len(predict)):
    #             print(predict[i], test[i])
    #         print(error)
    #         evaluation(predict, test, duration)
    test_p = 50
    print('A', test_p, file=f)
    if test_p <= 0.05:
        scale = 50
    elif test_p > 0.05 and test_p <= 0.5:
        scale = 35
    else:
        scale = 15

    error, predict, test, duration = get_output(test_p)
    predict *= scale
    test *= scale
    predict, test = format_output(predict, test)
    print('predict   actual', file=f)
    for i in range(len(predict)):
        print(predict[i],'   ', test[i], file=f)
    evaluation(predict, test, duration)
    f.close()