import time
import numpy as np
import matplotlib.pyplot as plt


class LossPlot:
    def __init__(self):
        plt.ion()

    def plot(self, value_dict: dict, keys: list):
        plt.clf()
        for k in keys:
            arr = np.array(value_dict[k])
            # arr[arr > 1.2 * arr[0]] = 1.2 * arr[0]
            plt.plot(arr, label=k)
        plt.legend()
        plt.pause(0.01)


def build_index_data_by_room(ys):
    max_room_id = max(ys[:, 0])
    rid_index_dict = {}
    for rid in range(0, max_room_id + 1):
        rid_index_dict[rid] = np.where(ys[:, 0] == rid)[0]
    return rid_index_dict


def acc_fun(output, y):
    return sum(np.argmax(output, axis=1) == y) / len(y)


def split_data_set(rate, rid_indexes):
    train_indexes = []
    test_indexes = []
    cut_num = 5
    for rid in rid_indexes:
        len_rid = len(rid_indexes[rid])
        cut_size = len_rid // cut_num
        test_cut_start = int(cut_size * rate['train'])
        for i in range(cut_num):
            train_indexes += rid_indexes[rid][i*cut_size: i*cut_size + test_cut_start].tolist()
            test_indexes += rid_indexes[rid][i*cut_size: (i+1)*cut_size].tolist()
    return np.array(train_indexes), np.array(test_indexes)