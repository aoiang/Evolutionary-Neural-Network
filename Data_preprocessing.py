def cal_mean(data):
    mean_num = []
    for i in range(len(data)):
        sum = 0.0
        for j in range(len(data[i])):
            try:
                sum += data[i][j]
            except:
                pass
        mean_num.append(sum / len(data[i]))
    return mean_num


def fill_missing(data, method='mean'):
    mean_num = cal_mean(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if type(data[i][j]) is not float:
                if method == '0':
                    data[i][j] = 0
                if method == 'mean':
                    data[i][j] = mean_num[i]
    return data


def normalize(data, is_max=False):
    maxnum = []
    for i in range((len(data))):
        maxnum.append(max(data[i]))

    for i in range(len(data)):
        for j in range(len(data[i])):
            if is_max:
                data[i][j] /= maxnum[i]
            else:
                data[i][j] /= 50.0
    return data

def kill_missing(feature, label):
    i = 0
    j = 0
    while 1:
        if type(label[i][j]) is not float or type(feature[-1][j]) is not float:
            for r in range(len(feature)):
                del feature[r][j]
            del label[i][j]
            j -= 1
        if j + 1 == len(label[i]):
            break
        j += 1
    return feature, label


