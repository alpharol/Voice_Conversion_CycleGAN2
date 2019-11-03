import librosa
import numpy as np
import os
import pyworld
from tqdm import trange
from utils import *
import argparse
import time
import pickle

def preprocess(train_A_dir, train_B_dir, training_data_dir, random_seed):

    np.random.seed(random_seed)
    sampling_rate = 16000
    num_mcep = 24
    frame_period = 5.0
    n_frames = 128

    print('----------------------Data Preprocessing------------------------')
    start_time = time.time()

    print('Loading Wavs...')
    wavs_A = load_wavs(wav_dir = train_A_dir, sr = sampling_rate)
    wavs_B = load_wavs(wav_dir = train_B_dir, sr = sampling_rate)

    print('Encoding Wavs...')
    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs = wavs_A, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs = wavs_B, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)

    print('Calculating Statistics of F0...')
    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    print('Log Pitch A: Mean %f, Std %f' %(log_f0s_mean_A, log_f0s_std_A))
    print('Log Pitch B: Mean %f, Std %f' %(log_f0s_mean_B, log_f0s_std_B))

    print('Data Transpose...')
    coded_sps_A_transposed = transpose_in_list(lst = coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(lst = coded_sps_B)

    print('Data Normalizing...')
    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_A_transposed)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_B_transposed)


    if not os.path.exists(training_data_dir):
        os.makedirs(training_data_dir)
    np.savez(os.path.join(training_data_dir, 'logf0s_normalization.npz'), mean_A = log_f0s_mean_A, std_A = log_f0s_std_A, mean_B = log_f0s_mean_B, std_B = log_f0s_std_B)
    np.savez(os.path.join(training_data_dir, 'mcep_normalization.npz'), mean_A = coded_sps_A_mean, std_A = coded_sps_A_std, mean_B = coded_sps_B_mean, std_B = coded_sps_B_std)

    with open(os.path.join(training_data_dir, 'A_coded_norm.pk'),"wb") as fa:
        pickle.dump(coded_sps_A_norm,fa)

    with open(os.path.join(training_data_dir, 'B_coded_norm.pk'),"wb") as fb:
        pickle.dump(coded_sps_B_norm,fb)

    
    end_time = time.time()
    time_elapsed = end_time - start_time


    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')

    train_A_dir_default = './data/vcc2016_training/SF1'
    train_B_dir_default = './data/vcc2016_training/TM1'
    training_data_dir_default = './training_data'

    parser.add_argument('--train_A_dir', type = str, help = 'Directory for A.', default = train_A_dir_default)
    parser.add_argument('--train_B_dir', type = str, help = 'Directory for B.', default = train_B_dir_default)
    parser.add_argument('--training_data_dir', type = str, help = 'Directory for saving data.', default = training_data_dir_default)
 
    argv = parser.parse_args()

    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    training_data_dir = argv.training_data_dir
   
    preprocess(train_A_dir = train_A_dir, train_B_dir = train_B_dir, training_data_dir = training_data_dir, random_seed = 0)
