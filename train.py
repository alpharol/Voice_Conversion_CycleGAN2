import os
import numpy as np
import argparse
import time
import librosa
import pickle
from utils import *
from model import CycleGAN2
from tqdm import trange


def train(train_A_dir, train_B_dir, training_data_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir,
          tensorboard_log_dir, MCEPs_dim, lambda_list):

    gen_loss_thres = 100.0
    np.random.seed(random_seed)
    num_epochs = 2000
    mini_batch_size = 1
    generator_learning_rate = 0.0002
    generator_learning_rate_decay = generator_learning_rate / 200000
    discriminator_learning_rate = 0.0001
    discriminator_learning_rate_decay = discriminator_learning_rate / 200000
    sampling_rate = 16000
    num_mcep = MCEPs_dim
    frame_period = 5.0
    n_frames = 128
    lambda_cycle = lambda_list[0]
    lambda_identity = lambda_list[1]


    # ****************************************************************
    # *************************Loading DATA***************************
    # ****************************************************************
    with open(os.path.join(training_data_dir, 'A_coded_norm.pk'),"rb") as fa:
        coded_sps_A_norm = pickle.load(fa)

    with open(os.path.join(training_data_dir, 'B_coded_norm.pk'),"rb") as fb:
        coded_sps_B_norm = pickle.load(fb)

    mcep_normalization_params = np.load(os.path.join(training_data_dir, 'mcep_normalization.npz'))
    coded_sps_A_mean = mcep_normalization_params['mean_A']
    coded_sps_A_std = mcep_normalization_params['std_A']
    coded_sps_B_mean = mcep_normalization_params['mean_B']
    coded_sps_B_std = mcep_normalization_params['std_B']

    logf0s_normalization_params = np.load(os.path.join(training_data_dir, 'logf0s_normalization.npz'))
    log_f0s_mean_A = logf0s_normalization_params['mean_A']
    log_f0s_std_A = logf0s_normalization_params['std_A']
    log_f0s_mean_B = logf0s_normalization_params['mean_B']
    log_f0s_std_B = logf0s_normalization_params['std_B']


    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)

    if validation_B_dir is not None:
        validation_B_output_dir = os.path.join(output_dir, 'converted_B')
        if not os.path.exists(validation_B_output_dir):
            os.makedirs(validation_B_output_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("****************************************************************")
    print("*************************Start Training*************************")
    print("****************************************************************")

    # Model define
    model = CycleGAN2(num_features = num_mcep, log_dir=tensorboard_log_dir, model_name=model_name)

    epoch = 0
    # load model
    if os.path.exists(os.path.join(model_dir, "checkpoint")) == True:
        f = open(os.path.join(model_dir, "checkpoint"),"r")
        all_ckpt = f.readlines()
        f.close()
        pretrain_ckpt = all_ckpt[-1].split("\n")[0].split("\"")[1]
        epoch = int(pretrain_ckpt.split("-")[1].split(".")[0])
        if os.path.exists(os.path.join(model_dir, (pretrain_ckpt+".index"))) == True:
            model.load(filepath=os.path.join(model_dir, pretrain_ckpt))
            print("Loading pretrained model {}".format(pretrain_ckpt))
    else:
        print("Training model from 1 epoch")
    
    for k in range(epoch+1, num_epochs):
        print('Epoch: %d' % k)

        start_time_epoch = time.time()

        dataset_A, dataset_B = sample_train_data(dataset_A = coded_sps_A_norm, dataset_B = coded_sps_B_norm, n_frames = n_frames)

        n_samples = dataset_A.shape[0]
        # -------------------------------------------- one epoch learning -------------------------------------------- #
        for i in trange(n_samples // mini_batch_size):

            num_iterations = n_samples // mini_batch_size * epoch + i

            if num_iterations > 10000:
                lambda_identity = 0
            if num_iterations > 200000:
                generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss, generator_loss_A2B = model.train\
                (input_A = dataset_A[start:end], input_B = dataset_B[start:end],
                 lambda_cycle = lambda_cycle, lambda_identity = lambda_identity,
                 generator_learning_rate = generator_learning_rate, discriminator_learning_rate = discriminator_learning_rate)
            #model.summary()

            # # Minimum AtoB loss model save
            # if gen_loss_thres > generator_loss_A2B:
            #     gen_loss_thres = generator_loss_A2B
            #     best_model_name = 'Bestmodel' + model_name
            #     model.save(directory=model_dir, filename=best_model_name)
            #     print("generator loss / generator A2B loss ", generator_loss, generator_loss_A2B)

            if i % (n_samples // 2) == 0:
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(num_iterations, generator_learning_rate, discriminator_learning_rate, generator_loss, discriminator_loss))

        # Last model save
        if k == 1 or k % 100 == 0:
            print("Saving Epoch {}".format(k))
            ckpt_name = model_name + "-" + str(k) + ".ckpt"
            model.save(directory = model_dir, filename = ckpt_name)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))
        # -------------------------------------------- one epoch learning -------------------------------------------- #
        # ------------------------------------------- validation inference ------------------------------------------- #
        if validation_A_dir is not None:
            if k % 500 == 0:
                print('Generating Validation Data B from A...')
                for i in trange(len(os.listdir(validation_A_dir))):
                    file = os.listdir(validation_A_dir)[i]
                    filepath = os.path.join(validation_A_dir, file)
                    wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                    wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                    f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                    f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_A, std_log_src = log_f0s_std_A, mean_log_target = log_f0s_mean_B, std_log_target = log_f0s_std_B)
                    coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                    coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = 'A2B')[0]
                    coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                    wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                    librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)
                    # break

        if validation_B_dir is not None:
            if k % 500 == 0:
                print('Generating Validation Data A from B...')
                for i in trange(len(os.listdir(validation_B_dir))):
                    file = os.listdir(validation_B_dir)[i]
                    filepath = os.path.join(validation_B_dir, file)
                    wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                    wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                    f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                    f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_B, std_log_src = log_f0s_std_B, mean_log_target = log_f0s_mean_A, std_log_target = log_f0s_std_A)
                    coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_B_mean) / coded_sps_B_std
                    coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = 'B2A')[0]
                    coded_sp_converted = coded_sp_converted_norm * coded_sps_A_std + coded_sps_A_mean
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                    wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                    librosa.output.write_wav(os.path.join(validation_B_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train CycleGAN-VC2 model')

    train_A_dir_default = './data/vcc2016_training/SF1'
    train_B_dir_default = './data/vcc2016_training/TM1'
    training_data_dir_default = "./training_data"
    model_dir_default = './model'
    model_name_default = 'sf1_tm1'
    random_seed_default = 0
    validation_A_dir_default = './data/evaluation_all/SF1'
    validation_B_dir_default = './data/evaluation_all/TM1'
    output_dir_default = './validation_output'
    tensorboard_log_dir_default = './log'
    MCEPs_dim_default = 24
    lambda_cycle_defalut = 10.0
    lambda_identity_defalut = 5.0


    parser.add_argument('--train_A_dir', type = str, help = 'Directory for A.', default = train_A_dir_default)
    parser.add_argument('--train_B_dir', type = str, help = 'Directory for B.', default = train_B_dir_default)
    parser.add_argument('--training_data_dir', type = str, help = 'Directory for saving data.', default = training_data_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = model_name_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)
    parser.add_argument('--validation_A_dir', type=str,
                        help='Convert validation A after each training epoch. If set none, no conversion would be done during the training.',
                        default=validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type=str,
                        help='Convert validation B after each training epoch. If set none, no conversion would be done during the training.',
                        default=validation_B_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation voices.', default = output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)
    parser.add_argument('--MCEPs_dim', type=int, help='input dimension', default=MCEPs_dim_default)
    parser.add_argument('--lambda_cycle', type=float, help='lambda cycle', default=lambda_cycle_defalut)
    parser.add_argument('--lambda_identity', type=float, help='lambda identity', default=lambda_identity_defalut)

    argv = parser.parse_args()

    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    training_data_dir = argv.training_data_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_A_dir = None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
    validation_B_dir = None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir
    MCEPs_dim = argv.MCEPs_dim
    lambda_cycle = argv.lambda_cycle
    lambda_identity = argv.lambda_identity

    train(train_A_dir=train_A_dir, train_B_dir=train_B_dir, training_data_dir = training_data_dir_default, model_dir=model_dir, model_name=model_name,
          random_seed=random_seed, validation_A_dir=validation_A_dir, validation_B_dir=validation_B_dir,
          output_dir=output_dir, tensorboard_log_dir = tensorboard_log_dir, MCEPs_dim=MCEPs_dim, lambda_list=[lambda_cycle, lambda_identity])