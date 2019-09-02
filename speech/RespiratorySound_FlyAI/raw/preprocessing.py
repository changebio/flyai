#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/5/14 14:17
#@Author: yangjian
#@File  : preprocessing.py

import os
import re
import shutil
import numpy as np
import soundfile as sf
import pandas as pd
import tensorflow as tf

from pydub import AudioSegment
#import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# print(os.listdir('data'))
# print(os.listdir('data/respiratory_sound_database/Respiratory_Sound_Database'))

# Play an audio file
audio_file = '213_1p5_Pr_mc_AKGC417L.wav'
path = 'data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + audio_file

# Read the 'demographic_info.txt' file
# demographic_info.txt 患者信息表
col_names = ['patient_id', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height']
# df_demo = pd.read_csv('data/demographic_info.txt', sep=" ", header=None, names=col_names)
# print(df_demo.head(10))

# Read the 'patient_diagnosis.csv' file
# patient_diagnosis.csv 患者-疾病表
# df_diag = pd.read_csv('data/respiratory_sound_database/Respiratory_Sound_Database/patient_diagnosis.csv',
#                       header=None, names=['patient_id', 'diagnosis'])
# print(df_diag.head(10))

# Read the 'filename_differences.txt' file
# 对应音频文件名表
# df_diff = pd.read_csv('data/respiratory_sound_database/Respiratory_Sound_Database/filename_differences.txt',
#                       sep=" ", header=None, names=['file_names'])
# print(df_diff.head(10))

# Read the 'filename_format.txt' file
# 文件名信息
# -Recording index患者编号
# -Chest location胸部部位（Trachea (Tc)气管, {Anterior (A)前, Posterior (P)后, Lateral (L)测}{left (l)左, right (r)右}）
# -Acquisition mode采集模式 (sequential/single channel (sc)顺序/单通道, simultaneous/multichannel (mc)同步/多通道)
# -Recording equipment 录音设备(AKG C417L MicrophoneAKG C417L(AKGC417L)麦克风,
#                               3M Littmann Classic II SE Stethoscope3M Littmann Classic II SE(LittC2SE)听诊器,
#                               3M Litmmann 3200 Electronic Stethoscope3M Litmmann 3200(Litt3200)电子听诊器,
#                               WelchAllyn Meditron Master Elite Electronic StethoscopeWelchAllyn Meditron Master Elite(Meditron)电子听诊器)
# data = open('data/respiratory_sound_database/Respiratory_Sound_Database/filename_format.txt', 'r').read()
# print(data)

# List the files in the 'audio_and_txt_files' folder
# print(os.listdir('data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files'))

# Display the contents of one annotation .txt file
#音频标注表
# -Beginning of respiratory cycle呼吸循环开始
# -End of respiratory cycle呼吸循环结束
# -Presence/absence_of_crackles是否有裂纹
# -Presence/absence_of_wheezes是否有喘息
# col_names = ['Beginning_of_respiratory_cycle', 'End_of_respiratory_cycle',
#              'Presence/absence_of_crackles', 'Presence/absence_of_wheezes']
# df_annot = pd.read_csv(
#     'data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/154_2b4_Al_mc_AKGC417L.txt',
#     sep="\t", header=None, names=col_names)
# print(df_annot.head(10))

# Create an Audio Spectrogram
# 创建音频频谱图
# col_names = ['Beginning_of_respiratory_cycle', 'End_of_respiratory_cycle',
#              'Presence/absence_of_crackles', 'Presence/absence_of_wheezes']
# df_annot = pd.read_csv(
#     'data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/154_2b4_Al_mc_AKGC417L.txt',
#     sep="\t", header=None, names=col_names)
# print(df_annot.head(20))


# Define helper functions
# Load a .wav file.
# These are 24 bit files. The PySoundFile library is able to read 24 bit files.
# https://pysoundfile.readthedocs.io/en/0.9.0/
def get_wav_info(wav_file):
    data, rate = sf.read(wav_file)
    return data, rate

# source: Andrew Ng Deep Learning Specialization, Course 5
#def graph_spectrogram(wav_file):
#    data, rate = get_wav_info(wav_file)
#    nfft = 200 # Length of each window segment
#    fs = 8000 # Sampling frequencies
#    noverlap = 120 # Overlap between windows
#    nchannels = data.ndim
#    if nchannels == 1:
#        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
#    # elif nchannels == 2:
#    else:
#        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
#    return pxx

# plot the spectrogram
# Time is on the x axis and Frequencies are on the y axis.
# The intensity of the different colours shows the amount of energy i.e. how loud the sound is,
# at different frequencies, at different times.
# x = graph_spectrogram(
#     'data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/154_2b4_Al_mc_AKGC417L.wav')

# Read an audio file as a numpy array
# # choose an audio file
# audio_file = '154_2b4_Al_mc_AKGC417L.wav'
# # read the file
# data, rate = sf.read('data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + audio_file)
# # display the numpy array
# print(data)


# How to slice a section from an audio file
# https://stackoverflow.com/questions/37999150/
# python-how-to-split-a-wav-file-into-multiple-wav-files
# note: Time is given in seconds. Will be converted to milliseconds later.
# start_time = 0
# end_time = 7
#
# t1 = start_time * 1000 # pydub works in milliseconds
# t2 = end_time * 1000
# newAudio = AudioSegment.from_wav(
#     'data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + audio_file) # path is defined above
# newAudio = newAudio[t1:t2]
# newAudio.export('new_slice.wav', format="wav")



print("-------------------------------------------------------------------")

# df_demo 患者信息表
# df_diag 患者疾病表
# df_annot 单个患者音频表
# print(df_demo)
# print(df_diag)
# print(df_annot)

# df = df_demo.merge(df_diag, how='left')
# print(df)
# df.to_csv('data.csv', index=False)


# print("-------------------------------------------------------------------")
# dirs = os.listdir('data/audio_and_txt_files')
# for d in dirs:
#     ls = os.listdir('data/audio_and_txt_files/' + d)
#     print(ls)
#     for l in ls:
#         if re.match('.*.zip', l) != None:
#             print('begin copy.')
#             shutil.copyfile('data/audio_and_txt_files/' + d + '/' + l,
#                             'data/audio_and_txt_files/' + l)

# print(dirs)
# txt_dirs = []
# audio_dirs = []

# file_path = ['audio_and_txt_files/' + str(x) for x in range(101, 227)]
# df_files_path = pd.DataFrame(file_path, columns=['audio_and_txt_files_path'])
# print(df_files_path)
# df = pd.concat([df_files_path, df], axis=1, sort=False)
# print(df)
# df.to_csv('dataset.csv', index=False)
# for d in dirs:
#     if re.match('.*.txt', d) != None:
#         txt_dirs.append('Annot/'+ d)
#     if re.match('.*.wav', d) != None:
#         audio_dirs.append('Audio/' + d)
#     ls = d.split('_')
#     print(ls[0])
#     this_path = str(ls[0]) + '/'
#     if not os.path.exists('data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + this_path):
#         os.makedirs('data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + this_path)
#     shutil.copyfile('data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + d,
#                     'data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + this_path + d)


# print(txt_dirs)
# print(audio_dirs)
# txt = pd.DataFrame(txt_dirs, columns=['txt_path'])
# print(txt)
# audio = pd.DataFrame(audio_dirs, columns=['audio_path'])
# print(audio)


# for dir in txt_dirs:
#     df_annot = pd.read_csv(
#         'data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + dir,
#         sep="\t", header=None, names=col_names)
#
#     print(df_annot)

    # print('***********************************************************************************')
    #
    # count = 0
    # for index, row in df_annot.iterrows():
    #     # print(row)
    #     start_time = row[0]
    #     end_time = row[1]
    #     # print(start_time)
    #     # print(end_time)
    #     count += 1
    #     t1 = start_time * 1000  # pydub works in milliseconds
    #     t2 = end_time * 1000
    #     newAudio = AudioSegment.from_wav(
    #         'data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + dir.split('.')[0] + '.wav' )  # path is defined above
    #     newAudio = newAudio[int(t1):int(t2)]
    #     newAudio.export('new_slice_' + dir.split('.')[0] + '_' + str(count) + '.wav', format="wav")
    # count = 0
    #
    # for an in df_annot[2:]:
    #     print(an)
        # start_time = an[0]
        # end_time = an[1]
        # count += 1
        # t1 = start_time * 1000  # pydub works in milliseconds
        # t2 = end_time * 1000
        # newAudio = AudioSegment.from_wav(
        #     'data/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files/' + dir.split('.')[0] + '.wav' )  # path is defined above
        # newAudio = newAudio[int(t1):int(t2)]
        # newAudio.export('new_slice_' + dir.split('.')[0] + '_' + str(count) + '.wav', format="wav")
