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
import skfuzzy
import fnmatch
from skimage import measure
from matplotlib import pyplot

environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='DaniNet')
parser.add_argument('--phase', type=int, default=6)
parser.add_argument('--epoch', type=int, default=300, help='number of epochs')
parser.add_argument('--dataset', type=str, default='TrainingSetMRI', help='training dataset name that stored in ./data')
parser.add_argument('--datasetTL', type=str, default='TransferLr', help='transfer learning dataset name that stored in ./data')
parser.add_argument('--datasetGT', type=str, default='PredictionGT', help='testing dataset name that stored in ./data')
parser.add_argument('--savedir', type=str, default='save', help='dir of saving checkpoints and intermediate training results')
parser.add_argument('--use_trained_model', type=str2bool, default=True, help='whether train from an existing model or from scratch')
parser.add_argument('--use_init_model', type=str2bool, default=True, help='whether train from the init model if cannot find an existing model')
parser.add_argument('--slice', type=int, default=100, help='slice')

FLAGS = parser.parse_args()


def train_regressors(max_regional_expansion, map_disease, classifier):
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
                model = progression_net(current_slice=j, max_regional_expansion=max_regional_expansion, map_disease=map_disease, classifier=classifier)
                if model.train_and_save(current_region=i):
                    print(i)
                    model.test(current_region=i)
                else:
                    break


def main(_):
    conditioned_enabled = True
    progression_enabled = True
    test_label = 'test'
    max_regional_expansion = 10
    classifier = 1  # classifier 0 svr, 1 logistic regressor
    # ResearchGroup = {'Cognitive normal', 'Subjective memory concern', 'Early mild cognitive Impairment', 'Mild cognitive impairment',
    #                 'Late mild cognitive impairment', 'Alzheimer''s disease'};
    map_disease = (0, 1, 2, 2, 2, 3)
    age_intervals = (63, 66, 68, 70, 72, 74, 76, 78, 80, 83, 87)

    if FLAGS.phase == 0:
        train_regressors(max_regional_expansion=max_regional_expansion, map_disease=map_disease, classifier=classifier)
    if FLAGS.phase == 1:
        training(curr_slice=FLAGS.slice, conditioned_enabled=conditioned_enabled, progression_enabled=progression_enabled,
                 max_regional_expansion=max_regional_expansion, map_disease=map_disease, age_intervals=age_intervals)
    if FLAGS.phase == 2:
        transfer_learning(curr_slice=FLAGS.slice, conditioned_enabled=conditioned_enabled, progression_enabled=progression_enabled, output_dir='sample_TF',
                          num_epochs=1000, max_regional_expansion=max_regional_expansion, map_disease=map_disease, age_intervals=age_intervals)
    if FLAGS.phase == 3:
        testing(curr_slice=FLAGS.slice, conditioned_enabled=conditioned_enabled, output_dir=test_label, max_regional_expansion=max_regional_expansion,
                map_disease=map_disease, age_intervals=age_intervals)
    if FLAGS.phase == 4:
        assembly_MRI('71.0712_1_2_1_ADNI_126_S_5243_MR_MT1__N3m_Br_20130724140336799_S195168_I382272.nii.png', test_label, 65, age_intervals)
    if FLAGS.phase == 5:
        GUI()
    if FLAGS.phase == 6:
        validation_folder = os.path.join('./data', FLAGS.datasetGT)
        TL_folder = os.path.join('./data', FLAGS.datasetTL)
        validation(validation_folder, TL_folder, FLAGS.slice, test_label, age_intervals)


def testing(curr_slice, conditioned_enabled, output_dir, max_regional_expansion, map_disease, age_intervals):
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
            map_disease=map_disease,
            age_intervals=age_intervals

        )
        print('\n\tTesting Mode')
        model.testing(
            testing_samples_dir='./data/' + FLAGS.datasetTL + '/',
            current_slice=curr_slice,
            conditioned_enabled=conditioned_enabled
        )
    tf.reset_default_graph()


def transfer_learning(curr_slice, conditioned_enabled, progression_enabled, output_dir, num_epochs, max_regional_expansion, map_disease, age_intervals):
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
            map_disease=map_disease,
            age_intervals=age_intervals
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


def generate_MRI_slice(generated_images, age_to_generate, age_intervals):
    bin_centers = np.convolve(age_intervals, [0.5, 0.5], 'valid')
    generated_image = np.zeros([128, 128])
    batch_fuzzy_membership = np.zeros(10)
    for t in range(10):
        batch_fuzzy_membership[t] = skfuzzy.membership.gaussmf(age_to_generate, bin_centers[t], 1.5)
        generated_image = generated_image + generated_images[:128, (t * 128):((t + 1) * 128)] * batch_fuzzy_membership[t]

    generated_image = (generated_image - np.min(generated_image)) / (np.max(generated_image) - np.min(generated_image))
    return generated_image


def generate_MRI_slice_average_5(curr_slice, folder, fileName):
    a = np.ones((5, 128 * 10, 128 * 10), dtype=np.float)
    a[0, :, :] = imread('./' + FLAGS.savedir + '/' + str(curr_slice - 2) + '/' + folder + '/test_2_' + fileName)
    a[1, :, :] = imread('./' + FLAGS.savedir + '/' + str(curr_slice - 1) + '/' + folder + '/test_1_' + fileName)
    a[2, :, :] = imread('./' + FLAGS.savedir + '/' + str(curr_slice) + '/' + folder + '/test_0_' + fileName)
    a[3, :, :] = imread('./' + FLAGS.savedir + '/' + str(curr_slice + 1) + '/' + folder + '/test_-1_' + fileName)
    a[4, :, :] = imread('./' + FLAGS.savedir + '/' + str(curr_slice + 2) + '/' + folder + '/test_-2_' + fileName)
    return a[0, :, :] * 0.05 + a[1, :, :] * 0.15 + a[2, :, :] * 0.6 + a[3, :, :] * 0.15 + a[4, :, :] * 0.05


def hist_norm(source, template):
    old_type = source.dtype
    old_shape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    interp_t_values = interp_t_values.astype(old_type)

    return interp_t_values[bin_idx].reshape(old_shape)


def validation(validation_folder, TL_folder, curr_slice, output_dir, age_intervals):
    all_GT = os.listdir(validation_folder + '/' + str(curr_slice) + '/')
    final_similarity = 0
    for currentGT in all_GT:

        current_patient_id = currentGT.split('ADNI_')[1][:10]
        currentGT_image = imread(validation_folder + '/' + str(curr_slice) + '/' + currentGT)
        first_scan_name = fnmatch.filter(os.listdir(TL_folder + '/' + str(curr_slice) + '/'), '*' + current_patient_id + '*')
        first_scan_name = first_scan_name[0]
        input_image = imread(TL_folder + '/' + str(curr_slice) + '/' + first_scan_name)
        generate_pred_scan = generate_MRI_slice_average_5(curr_slice, output_dir, first_scan_name)
        generate_pred_scan = generate_MRI_slice(generate_pred_scan, np.float32(currentGT.split('_')[0]), age_intervals)
        generate_pred_scan = hist_norm(generate_pred_scan, input_image)
        #pyplot.imshow(generate_pred_scan)
        #pyplot.show()
        #pyplot.imshow(currentGT_image)
        #pyplot.show()
        #pyplot.imshow(input_image)
        #pyplot.show()
        final_similarity = [final_similarity, measure.compare_ssim(generate_pred_scan, currentGT_image)]
    return final_similarity


def assembly_MRI(fileName, folder, age_to_generate, age_intervals):
    curr_slice = 103
    numb_Slice = 30

    final_MRI = np.ones((numb_Slice, 128, 128), dtype=np.int16)
    for i in range(0, numb_Slice):
        progression_MRI = generate_MRI_slice_average_5(curr_slice, folder, fileName)
        final_MRI[i, :, :] = np.int16(generate_MRI_slice(progression_MRI, age_to_generate, age_intervals) * 32767 * 2 - 32767)

        curr_slice = curr_slice + 1
        img = nib.Nifti1Image(final_MRI, np.eye(4))
        nib.save(img, str(age_to_generate) + 'out.nii.gz')


def training(curr_slice, conditioned_enabled, progression_enabled, max_regional_expansion, map_disease, age_intervals):
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
            map_disease=map_disease,
            age_intervals=age_intervals
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
