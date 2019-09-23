import tensorflow as tf
from DaniNet import DaniNet
from os import environ
import argparse
from progression_net import progression_net
import pickle
import pprint
import numpy as np
from matplotlib.pyplot import imread
import nibabel as nib
from scipy.misc import imread
import os
from GUI import GUI

environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='DaniNet')
parser.add_argument('--phase', type=int, default=1)
parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('--dataset', type=str, default='TrainingSetMRI', help='training dataset name that stored in ./data')
parser.add_argument('--datasetTL', type=str, default='TransferLr', help='training dataset name that stored in ./data')
parser.add_argument('--savedir', type=str, default='save', help='dir of saving checkpoints and intermediate training results')
parser.add_argument('--use_trained_model', type=str2bool, default=True, help='whether train from an existing model or from scratch')
parser.add_argument('--use_init_model', type=str2bool, default=True, help='whether train from the init model if cannot find an existing model')
parser.add_argument('--slice', type=int, default=100, help='slice')

FLAGS = parser.parse_args()


def train_regressors():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    maxNumberOfRegion = 1000
    if not os.path.isdir('Regressor'):
        os.mkdir('Regressor')
    for j in range(40, 143):
        filehandler = open('Regressor/' + str(j), 'wb')
        pickle.dump([[], []], filehandler)
        filehandler.close()
        print(j)
        for i in range(0, maxNumberOfRegion):
            tf.reset_default_graph()
            with tf.Session(config=config):
                model = progression_net(j)
                if model.train_and_save(i):
                    print(i)
                    # model.test(i)
                else:
                    break


def main(_):

    conditioned_enabled = True
    progression_enabled = True
    outputFolder = 'test_PR_CD_TF'
    max_regional_expansion = 10
    # ResearchGroup = {'Cognitive normal', 'Subjective memory concern', 'Early mild cognitive Impairment', 'Mild cognitive impairment',
    #                 'Late mild cognitive impairment', 'Alzheimer''s disease'};
    map_disease = (0, 1, 2, 2, 2, 3)
    if FLAGS.phase == 0:
        exit()
        # train_regressors(max_regional_expansion,map_disease)
    if FLAGS.phase == 1:
        training(FLAGS.slice, conditioned_enabled, progression_enabled, max_regional_expansion, map_disease)
    if FLAGS.phase == 2:
        transfer_learning(FLAGS.slice, conditioned_enabled, progression_enabled, 'sample_TF', 1000, max_regional_expansion, map_disease)
    if FLAGS.phase == 3:
        testing(FLAGS.slice, conditioned_enabled, outputFolder, max_regional_expansion, map_disease)
    if FLAGS.phase == 4:
        assemblyMri('71.0712_1_2_1_ADNI_126_S_5243_MR_MT1__N3m_Br_20130724140336799_S195168_I382272.nii.png', outputFolder)
    if FLAGS.phase == 5:
        g = GUI()


def testing(curr_slice, conditioned_enabled, output_dir, max_regional_expansion, map_disease):
    pprint.pprint(FLAGS)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        model = DaniNet(
            session,  # TensorFlow session
            is_training=False,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
            dataset_name=FLAGS.datasetTL,  # name of the dataset in the folder ./data
            slice_number=curr_slice,
            output_dir=output_dir,
            max_regional_expansion=max_regional_expansion,
            map_disease=map_disease

        )
        print('\n\tTesting Mode')
        model.testing(
            testing_samples_dir='./data/' + FLAGS.datasetTL + '/',
            current_slice=curr_slice,
            conditioned_enabled=conditioned_enabled
        )
    tf.reset_default_graph()


def transfer_learning(curr_slice, conditioned_enabled, progression_enabled, output_dir, num_epochs, max_regional_expansion, map_disease):
    pprint.pprint(FLAGS)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        model = DaniNet(
            session,  # TensorFlow session
            is_training=True,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
            dataset_name=FLAGS.datasetTL,  # name of the dataset in the folder ./data
            slice_number=curr_slice,
            output_dir=output_dir,
            max_regional_expansion=max_regional_expansion,
            map_disease=map_disease
        )
        print('\n\tTransfer learning mode')
        model.train(
            num_epochs=num_epochs,
            use_trained_model=FLAGS.use_trained_model,
            use_init_model=FLAGS.use_init_model,
            n_epoch_to_save=num_epochs,
            conditioned_enabled=conditioned_enabled,
            progression_enabled=progression_enabled
        )
    tf.reset_default_graph()


def assemblyMri(fileName, folder):
    currSliceSlice = 103
    numb_Slice = 30
    a = np.ones((5, 1280, 1280), dtype=np.float)
    finalMri = np.ones((numb_Slice, 128, 128), dtype=np.int16)
    for i in range(0, numb_Slice):
        a[0, :, :] = imread('./save/' + str(currSliceSlice - 2) + '/' + folder + '/test_2_' + fileName)
        a[1, :, :] = imread('./save/' + str(currSliceSlice - 1) + '/' + folder + '/test_1_' + fileName)
        a[2, :, :] = imread('./save/' + str(currSliceSlice) + '/' + folder + '/test_0_' + fileName)
        a[3, :, :] = imread('./save/' + str(currSliceSlice + 1) + '/' + folder + '/test_-1_' + fileName)
        a[4, :, :] = imread('./save/' + str(currSliceSlice + 2) + '/' + folder + '/test_-2_' + fileName)
        MRIAll = a[0, :, :] * 0.05 + a[1, :, :] * 0.15 + a[2, :, :] * 0.6 + a[3, :, :] * 0.15 + a[4, :, :] * 0.05

        index_progression = 9
        finalMri[i, :, :] = np.int16(MRIAll[:128, 128 * index_progression:128 * (index_progression + 1)])

        currSliceSlice = currSliceSlice + 1
        img = nib.Nifti1Image(finalMri, np.eye(4))
        nib.save(img, 'out.nii.gz')


def training(curr_slice, conditioned_enabled, progression_enabled, max_regional_expansion, map_disease):
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
            max_regional_expansion=max_regional_expansion,
            map_disease=map_disease
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
