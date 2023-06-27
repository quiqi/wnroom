import os
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt


def mp3_to_wav(mp3_path, output_dir):
    song = AudioSegment.from_mp3(mp3_path)
    wav_path = output_dir + mp3_path.split('/')[-1].split('.')[0] + '.wav'
    song.export(wav_path, format="wav")


def split_wav(wav_path: str, output_dir: str):
    audio = AudioSegment.from_wav(wav_path)
    segment_length = 1 * 1000  # 1 seconds in milliseconds
    start = 0
    end = segment_length
    counter = 1
    wav_name = wav_path.split('/')[-1].split('.')[0]
    while start < len(audio):
        segment = audio[start:end]
        segment.export(f"{output_dir}/{wav_name}_{counter}.wav", format="wav")
        print('\t' + f"{output_dir}/{wav_name}_{counter}.wav")
        start += segment_length
        end += segment_length
        counter += 1


def wav_fft(wav_path, output_dir):
    sound = AudioSegment.from_wav(wav_path)
    samples = np.array(sound.get_array_of_samples())
    fft_result = np.fft.fft(samples,  n=5000)[2500::]
    fft_result = abs(fft_result)
    # plt.plot(fft_result)
    # plt.show()
    wav_path = output_dir + wav_path.split('/')[-1].split('.')[0]
    np.save(wav_path, fft_result)


def batching(input_root, output_path, method):
    data_list = os.listdir(input_root)
    for data_name in data_list:
        print('{}_{}'.format(method, data_name))
        method(input_root+data_name, output_path)


if __name__ == '__main__':
    ##############################
    #   1. 将 mp3 转为 wav 的脚本
    ##############################
    # mp3_path_root = './data/data/DSRDataset/MP3/'
    # wav_path_root = './data/data/DSRDataset/WAV/'
    # batching(mp3_path_root, wav_path_root, mp3_to_wav)


    ############################
    #   2. wav 切片
    ############################
    # wav_path_root = './data/data/DSRDataset/WAV/'
    # wav_split_root = './data/data/DSRDataset/WAVSplit'
    # batching(wav_path_root, wav_split_root, split_wav)

    ############################
    #   3. 傅里叶变换
    ############################
    # wav_split_path = './data/data/DSRDataset/WAVSplit/'
    # fft_path = './data/data/DSRDataset/FFT/'
    # batching(wav_split_path, fft_path, wav_fft)

    ############################
    #   4. 数据整合
    ############################
    fft_path = './data/data/DSRDataset/FFT/'
    npy_list = os.listdir(fft_path)
    all_x = []
    all_y = []
    for npy_name in npy_list:
        data = np.load(fft_path + npy_name)
        all_x.append(data)
        all_y.append([int(i) for i in npy_name.split('_')[:-1]])

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    np.save('data/data/DSRDataset/X.npy', all_x)
    np.save('data/data/DSRDataset/Y.npy', all_y)









