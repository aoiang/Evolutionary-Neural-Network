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
    for i in range(len(data)):
        for j in range(len(data[i])):
            if is_max:
                data[i][j] /= max(data[i])
            else:
                data[i][j] /= 100.0
    return data
