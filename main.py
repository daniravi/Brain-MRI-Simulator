import tensorflow as tf
from DaniNet import DaniNet
from os import environ
import os
import argparse
from progression_net import progression_net
import pickle
import pprint
import numpy as np
from GUI import GUI
from datetime import datetime
import MRI_assembler as MRI_assembler
import re

environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['FSLOUTPUTTYPE'] = 'NIFTI'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='DaniNet')
parser.add_argument('--conf', type=int, default=1)
parser.add_argument('--phase', type=int, default=3)
parser.add_argument('--epoch', type=int, default=300, help='number of epochs')
parser.add_argument('--dataset', type=str, default='TrainingSetMRI', help='training dataset name that stored in ./data')
parser.add_argument('--datasetTL', type=str, default='TransferLr', help='transfer learning dataset name that stored in ./data')
parser.add_argument('--datasetGT', type=str, default='PredictionGT', help='testing dataset name that stored in ./data')
parser.add_argument('--savedir', type=str, default='save', help='dir of saving checkpoints and intermediate training results')
parser.add_argument('--use_trained_model', type=str2bool, default=True, help='whether train from an existing model or from scratch')
parser.add_argument('--use_init_model', type=str2bool, default=True, help='whether train from the init model if cannot find an existing model')
parser.add_argument('--slice', type=int, default=100, help='slice')

FLAGS = parser.parse_args()


def train_regressors(max_regional_expansion, map_disease, regressor_type):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    maxNumberOfRegion = 1000
    if regressor_type == 0:
        if not os.path.isdir('Regressor_0'):
            os.mkdir('Regressor_0')
    else:
        if not os.path.isdir('Regressor_1'):
            os.mkdir('Regressor_1')
    for j in range(40, 143):
        if regressor_type == 0:
            filehandler = open('Regressor_0/' + str(j), 'wb')
            pickle.dump([[], []], filehandler)
            filehandler.close()
        else:
            filehandler = open('Regressor_1/' + str(j), 'wb')
            pickle.dump([[], []], filehandler)
            filehandler.close()
        print(j)
        for i in range(0, maxNumberOfRegion):
            tf.reset_default_graph()
            with tf.Session(config=config):
                model = progression_net(current_slice=j, max_regional_expansion=max_regional_expansion, map_disease=map_disease, regressor_type=regressor_type)
                if model.train_and_save(current_region=i):
                    print(i)
                    model.test(current_region=i)
                else:
                    break


def main(_):
    conditioned_enabled = False
    progression_enabled = False
    attention_loss_function = False
    V2_enabled = False  # fuzzy + logistic regressor
    test_label = ''
    num_epochs_transfer_learning = 1500
    if FLAGS.conf == 0:
        conditioned_enabled = False
        progression_enabled = False
        attention_loss_function = False
        V2_enabled = False
        test_label = '_baseline'
        print('Select baseline configuration')
    elif FLAGS.conf == 1:
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = False
        V2_enabled = False
        test_label = '_DaniNet-V1'
        print('Select DaniNet-V1 configuration')
    elif FLAGS.conf == 2:
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = False
        V2_enabled = True
        test_label = '_DaniNet-V2'
        print('Select DaniNet-V2 configuration')
    elif FLAGS.conf == 3:
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = True
        V2_enabled = True  # fuzzy + logistic regressor
        test_label = '_DaniNet-V2_AL'
        print('Select DaniNet-V2_AL configuration')
    else:
        print('Please select one of the available modality.')
        exit()

    if V2_enabled:
        regressor_type = 1  # 1 logistic regressor
    else:
        regressor_type = 0  # 0 support vector regressor

    # ResearchGroup = {'Cognitive normal', 'Subjective memory concern', 'Early mild cognitive Impairment', 'Mild cognitive impairment',
    #                 'Late mild cognitive impairment', 'Alzheimer''s disease'};
    FLAGS.savedir = FLAGS.savedir + test_label
    max_regional_expansion = 10
    map_disease = (0, 1, 2, 2, 2, 3)
    age_intervals = (63, 66.4, 69.1, 71, 72.4, 74, 75.6, 77.4, 79.4, 81.7, 87)

    if FLAGS.phase == 0:
        train_regressors(max_regional_expansion=max_regional_expansion, map_disease=map_disease, regressor_type=regressor_type)
    if FLAGS.phase == 1:
        training(curr_slice=FLAGS.slice, conditioned_enabled=conditioned_enabled, progression_enabled=progression_enabled,
                 attention_loss_function=attention_loss_function, max_regional_expansion=max_regional_expansion, map_disease=map_disease,
                 age_intervals=age_intervals, V2_enabled=V2_enabled, output_dir='sample_Train', regressor_type=regressor_type)
        # testing(curr_slice=FLAGS.slice, conditioned_enabled=conditioned_enabled, output_dir='sample_Test', max_regional_expansion=max_regional_expansion,
        #        map_disease=map_disease, age_intervals=age_intervals, V2_enabled=V2_enabled, regressor_type=regressor_type)
    if FLAGS.phase == 2:
        transfer_learning(curr_slice=FLAGS.slice, conditioned_enabled=conditioned_enabled, progression_enabled=progression_enabled,
                          attention_loss_function=attention_loss_function, output_dir='sample_TrLearn', num_epochs=num_epochs_transfer_learning,
                          max_regional_expansion=max_regional_expansion, map_disease=map_disease, age_intervals=age_intervals, V2_enabled=V2_enabled,
                          regressor_type=regressor_type)
        testing(curr_slice=FLAGS.slice, conditioned_enabled=conditioned_enabled, output_dir='sample_Test_after_TL',
                max_regional_expansion=max_regional_expansion,
                map_disease=map_disease, age_intervals=age_intervals, V2_enabled=V2_enabled, regressor_type=regressor_type)
    if FLAGS.phase == 3:
        test_label = 'sample_Test_after_TL'
        outputFolder = 'SyntheticMRI_V1'
        type_of_assembly = 0  # 0: input image; 1: 3d spatial-consistency
        if type_of_assembly == 0:
            outputFolder = 'InputMRI'
        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
        MRI_assembler.assemblyAll(test_label, age_intervals, outputFolder, type_of_assembly, FLAGS)

    if FLAGS.phase == 4:
        GUI()
    if FLAGS.phase == 5:
        # fsl_bin_dir = '/share/apps/fsl-6.0.1' #server
        fsl_bin_dir = '/usr/local/fsl'
        extract_volumes('ADNI_100_S_0015_MR_MPR-R____N3_Br_20061213151820274_S13884_I33040.nii', fsl_bin_dir)


def testing(curr_slice, conditioned_enabled, output_dir, max_regional_expansion, map_disease, age_intervals, V2_enabled, regressor_type):
    pprint.pprint(FLAGS)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        model = DaniNet(
            session,  # TensorFlow session
            is_training=False,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to load checkpoints, samples, and summary
            dataset_name=FLAGS.datasetTL,  # name of the dataset in the folder ./data
            slice_number=curr_slice,
            output_dir=output_dir,
            max_regional_expansion=max_regional_expansion,
            map_disease=map_disease,
            age_intervals=age_intervals,
            V2_enabled=V2_enabled,
            regressor_type=regressor_type
        )
        print('\n\tTesting Mode')
        model.testing(
            testing_samples_dir='./data/' + FLAGS.datasetTL + '/',
            current_slice=curr_slice,
            conditioned_enabled=conditioned_enabled
        )
    tf.reset_default_graph()


def transfer_learning(curr_slice, conditioned_enabled, progression_enabled, attention_loss_function, output_dir, num_epochs, max_regional_expansion,
                      map_disease, age_intervals, V2_enabled, regressor_type):
    pprint.pprint(FLAGS)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if attention_loss_function == 1:
        attention_loss_function = 2
    with tf.Session(config=config) as session:
        model = DaniNet(
            session,  # TensorFlow session
            is_training=True,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to load and save checkpoints, samples, and summary
            dataset_name=FLAGS.datasetTL,  # name of the dataset in the folder ./data
            slice_number=curr_slice,
            output_dir=output_dir,
            max_regional_expansion=max_regional_expansion,
            map_disease=map_disease,
            age_intervals=age_intervals,
            V2_enabled=V2_enabled,
            regressor_type=regressor_type,
            attention_loss_function=attention_loss_function
        )
        print('\n\tTransfer learning mode')
        model.train(
            num_epochs=num_epochs,
            use_trained_model=FLAGS.use_trained_model,
            use_init_model=FLAGS.use_init_model,
            n_epoch_to_save=num_epochs / 50,
            conditioned_enabled=conditioned_enabled,
            progression_enabled=progression_enabled
        )
    tf.reset_default_graph()


def extract_volumes(input_file, fsl_bin_dir):
    environ['PATH'] = environ['PATH'] + ':' + fsl_bin_dir + '/bin'
    environ['FSLDIR'] = fsl_bin_dir

    start_time = datetime.now()
    os.system('mkdir FSL_file')
    os.system('cp ' + input_file + ' ./FSL_file/input.nii')
    # Vol_name=['L_hippocampus','R_hippocampus','Peripheral_grey','Ventricular_csf','Grey','White','Brain']
    os.system('run_first_all -i ./FSL_file/input.nii -b -s L_Hipp,R_Hipp -o ./FSL_file/regions')
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    start_time = datetime.now()
    os.system('fslstats ./FSL_file/regions_all_fast_firstseg.nii -h 64 > ./FSL_file/region_volume.txt')
    text_file = open("./FSL_file/region_volume.txt", "r")
    lines = text_file.readlines()
    text_file.close()
    Vol = []
    for i in [20, 63]:
        Vol = np.append(Vol, np.float32(lines[i]))
    os.system('sienax ./FSL_file/input.nii -o ./FSL_file/tissue -r')
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    text_file = open("./FSL_file/tissue/report.sienax", "r")
    lines = text_file.readlines()
    text_file.close()
    for i in range(np.size(lines) - 5, np.size(lines)):
        Vol = np.append(Vol, re.findall('\d+\.\d+', lines[i])[0])
    print(Vol)


def training(curr_slice, conditioned_enabled, progression_enabled, attention_loss_function, max_regional_expansion, map_disease, age_intervals, V2_enabled,
             output_dir, regressor_type):
    pprint.pprint(FLAGS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        model = DaniNet(
            session,  # TensorFlow session
            is_training=True,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
            dataset_name=FLAGS.dataset,  # name of the dataset in the folder ./data
            slice_number=curr_slice,
            output_dir=output_dir,
            max_regional_expansion=max_regional_expansion,
            map_disease=map_disease,
            age_intervals=age_intervals,
            V2_enabled=V2_enabled,
            regressor_type=regressor_type,
            attention_loss_function=attention_loss_function
        )

        print('\n\tTraining Mode')
        if not FLAGS.use_trained_model:
            print('\n\tPre-train the network')
            model.train(
                num_epochs=5,  # number of epochs
                use_trained_model=FLAGS.use_trained_model,
                use_init_model=False,
                conditioned_enabled=conditioned_enabled,
                progression_enabled=progression_enabled
            )
            print('\n\tPre-train is done! The training will start.')
            FLAGS.use_trained_model = True
        model.train(
            num_epochs=FLAGS.epoch,  # number of epochs
            use_trained_model=FLAGS.use_trained_model,
            use_init_model=FLAGS.use_init_model,
            conditioned_enabled=conditioned_enabled,
            progression_enabled=progression_enabled
        )
    tf.reset_default_graph()


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    tf.app.run()
