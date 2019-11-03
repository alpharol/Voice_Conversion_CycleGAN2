import argparse
import os
import numpy as np

from model import CycleGAN2
from utils import *

def conversion(training_data_dir, model_dir, model_name, data_dir, conversion_direction, output_dir, pc):

    num_features = 24
    sampling_rate = 16000
    frame_period = 5.0

    model = CycleGAN2(num_features = num_features, mode = 'test')

    if os.path.exists(os.path.join(model_dir, "checkpoint")) == True:
        f = open(os.path.join(model_dir, "checkpoint"),"r")
        all_ckpt = f.readlines()
        f.close()
        pretrain_ckpt = all_ckpt[-1].split("\n")[0].split("\"")[1]
        assert os.path.exists(os.path.join(model_dir, (pretrain_ckpt+".index"))) == True, "The checkpoint is not exist."
        model.load(filepath=os.path.join(model_dir, pretrain_ckpt))
        print("Loading pretrained model {}".format(pretrain_ckpt))


    mcep_normalization_params = np.load(os.path.join(training_data_dir, 'mcep_normalization.npz'))
    mcep_mean_A = mcep_normalization_params['mean_A']
    mcep_std_A = mcep_normalization_params['std_A']
    mcep_mean_B = mcep_normalization_params['mean_B']
    mcep_std_B = mcep_normalization_params['std_B']

    logf0s_normalization_params = np.load(os.path.join(training_data_dir, 'logf0s_normalization.npz'))
    logf0s_mean_A = logf0s_normalization_params['mean_A']
    logf0s_std_A = logf0s_normalization_params['std_A']
    logf0s_mean_B = logf0s_normalization_params['mean_B']
    logf0s_std_B = logf0s_normalization_params['std_B']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in trange(len(os.listdir(data_dir))):
        file = os.listdir(data_dir)[i]
        filepath = os.path.join(data_dir, file)
        wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_features)
        coded_sp_transposed = coded_sp.T

        frame_size = 128
        if conversion_direction == 'A2B':
            # pitch
            print("AtoB")
            if pc == True:
                print("pitch convert")
                f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_A, std_log_src = logf0s_std_A,
                 mean_log_target = logf0s_mean_B, std_log_target = logf0s_std_B)
            else:
                print("pitch same")
                f0_converted = f0

            # normalization A Domain
            coded_sp_norm = (coded_sp_transposed - mcep_mean_A) / mcep_std_A

            # padding
            remain, padd = frame_size - coded_sp_norm.shape[1] % frame_size, False
            if coded_sp_norm.shape[1] % frame_size != 0:
                coded_sp_norm = np.concatenate((coded_sp_norm, np.zeros((24, remain))), axis=1)
                padd = True

            # inference for segmentation
            coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm[:, 0:frame_size]]), direction=conversion_direction)[0]
            for i in range(1, coded_sp_norm.shape[1] // frame_size):
                ccat = model.test(inputs=np.array([coded_sp_norm[:, i * frame_size:(i + 1) * frame_size]]),
                                  direction=conversion_direction)[0]
                coded_sp_converted_norm = np.concatenate((coded_sp_converted_norm, ccat), axis=1)

            if padd == True:
                coded_sp_converted_norm = coded_sp_converted_norm[:,:-remain]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_B + mcep_mean_B
        else:
            print("BtoA")
            if pc == True:
                print("pitch convert")
                f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_A, std_log_src = logf0s_std_A,
                mean_log_target = logf0s_mean_B, std_log_target = logf0s_std_B)
            else:
                f0_converted = f0

            # normalization B Domain
            coded_sp_norm = (coded_sp_transposed - mcep_mean_B) / mcep_std_B

            # padding
            remain, padd = frame_size - coded_sp_norm.shape[1] % frame_size, False
            if coded_sp_norm.shape[1] % frame_size != 0:
                coded_sp_norm = np.concatenate((coded_sp_norm, np.zeros((24, remain))), axis=1)
                padd = True

            # inference for segmentation
            coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm[:, 0:frame_size]]), direction=conversion_direction)[0]
            for i in range(1, coded_sp_norm.shape[1] // frame_size):
                ccat = model.test(inputs=np.array([coded_sp_norm[:, i * frame_size:(i + 1) * frame_size]]),
                                  direction=conversion_direction)[0]
                coded_sp_converted_norm = np.concatenate((coded_sp_converted_norm, ccat), axis=1)

            if padd == True:
                coded_sp_converted_norm = coded_sp_converted_norm[:,:-remain]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_A + mcep_mean_A

        # output translation value processing
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)

        # World vocoder synthesis
        wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
        librosa.output.write_wav(os.path.join(output_dir, os.path.basename(file)), wav_transformed, sampling_rate)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Convert voices using pre-trained CycleGAN model.')

    training_data_dir_default = "./training_data" 
    model_dir_default = './model'
    model_name_default = 'sf1_tm1'
    data_dir_default = './data/evaluation_all/SF1'
    conversion_direction_default = 'A2B'
    output_dir_default = './converted_voices'
    pc_default = True

    parser.add_argument('--training_data_dir', type = str, help = 'Directory for the data.', default = training_data_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for the pre-trained model.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'Filename for the pre-trained model.', default = model_name_default)
    parser.add_argument('--data_dir', type = str, help = 'Directory for the voices for conversion.', default = data_dir_default)
    parser.add_argument('--conversion_direction', type=str,
                        help='Conversion direction for CycleGAN. A2B or B2A. The first object in the model file name is A, and the second object in the model file name is B.',
                        default=conversion_direction_default)
    parser.add_argument('--output_dir', type = str, help = 'Directory for the converted voices.', default = output_dir_default)
    parser.add_argument('--pc', type=bool, help='True: using pitch conversion in DomainB',
                        default=pc_default)

    argv = parser.parse_args()

    training_data_dir = argv.training_data_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    data_dir = argv.data_dir
    conversion_direction = argv.conversion_direction
    output_dir = argv.output_dir
    pc = argv.pc

    # Conversion coder
    conversion(training_data_dir = training_data_dir, model_dir = model_dir, model_name = model_name, data_dir = data_dir, conversion_direction = conversion_direction, output_dir = output_dir, pc=pc)
