import Evolution as ep
import Input as inputs
import math



def get_output():
    input_init = inputs.Input(feature_name=['DURATION', 'ATT_001', 'POLAR', 'ELEVATION', 'FREQUENCY', 'R_RAINHEIGHT', 'RR_001', 'A_001'])
    duration = input_init.get_duration()
    duration = duration[:int(len(duration)*0.2)]
    input_init.delete_duration()
    input_init.normalize()
    tra_feature, tra_label, vaildation_feature, vaildation_label, test_feature, test_label = input_init.data_generator()
    optimal_net = ep.evolve(30, 3, tra_feature, tra_label, vaildation_feature, vaildation_label, test_feature, test_label, type='optimal')
    w, b, e, r, predict, test = ep.Evolution().network_optimizer(tra_feature, tra_label, test_feature, test_label, optimal_net, 18600, final=True)
    return e, predict, test, duration

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
    print(rms)



if __name__ == "__main__":
    error, predict, test, duration = get_output()
    predict, test = format_output(predict, test)
    for i in range(len(predict)):
        print(predict[i], test[i])
    print(error)
    evaluation(predict, test, duration)