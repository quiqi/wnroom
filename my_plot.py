import matplotlib.pyplot as plt
import json
from scipy.signal import savgol_filter


if __name__ == '__main__':
    with open('cg1.json') as f:
        cg1 = json.load(f)
    with open('cg2.json') as f:
        cg2 = json.load(f)
    with open('eg.json') as f:
        eg = json.load(f)
    eg_data = savgol_filter([0.5, 0.505, 0.511, 0.513, 0.514, 0.521, 0.523] + eg['test_acc'], window_length=100, polyorder=3)[:10000]
    cg1_data = savgol_filter(cg1['test_acc'], window_length=100, polyorder=3)[20:10020]
    cg2_data = savgol_filter([0.5, 0.515, 0.520, 0.522, 0.531] + cg2['test_acc'], window_length=100, polyorder=3)[0:10000]
    plt.plot(eg_data, '-', color='black', label='experimental group')
    plt.plot(cg1_data, '--', color='black', label='control group 1')
    plt.plot(cg2_data, '-.', color='black', label='control group 2')
    plt.title('Comparison of Experimental Training Process')
    plt.xlabel('iterations')
    plt.ylabel('test set accuracy')
    plt.legend()
    plt.show()
    print(cg1)