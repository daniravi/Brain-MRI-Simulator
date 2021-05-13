import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath('./DCSRN/'))
from DaniNet import DaniNet
from os import environ
import argparse
from progression_net import progression_net
import pprint
import numpy as np
import MRI_assembler as MRI_assembler
import re
import glob as glob
from joblib import Parallel, delayed
import pickle
import statsmodels.formula.api as smf
from sklearn import linear_model
from sklearn.svm import SVR
from scipy.stats import iqr


import statsmodels.api as sm

environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
environ['FSLOUTPUTTYPE'] = 'NIFTI'


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    tuples1 = []
    tuples2 = []
    length = len(ages)
    for i in range(length):
        tuples1.append(net_worths[i])
        tuples2.append(ages[i])

    differences_tuples = []
    for i in range(length):
        differences_tuples.append((abs(net_worths[i] - predictions[i]), i))
    differences_sorted = sorted(differences_tuples)

    # Return the indices of the datapoints to be removed

    indices_to_remove = []
    for i in range(int(length / 20)):
        indices_to_remove.append(differences_sorted[length - 1 - i][1])
    indices_to_remove = sorted(indices_to_remove, reverse=True)

    # Remove the relevant tuples
    for i in indices_to_remove:
        del tuples1[i]
        del tuples2[i]
    return [tuples2, tuples1]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# model initialization: initialization_step=0, use_init_model=false,phase 1, config=2 ,epoch =100
# model initialization: initialization_step=1, use_init_model=True,phase 1, config=2 ,epoch=300
# model initialization: initialization_step=2, use_init_model=True,phase 1, config=2,epoch =300
# phase2 conf 7
# phase3 conf -1, #create synthetic images
# phase3 conf -3, #create synthetic images
# phase4 conf 7 regression -1  #run_FSL= true
# phase4 conf 7 regression  any

parser = argparse.ArgumentParser(description='DaniNet')
parser.add_argument('--conf', type=int, default=7)
parser.add_argument('--phase', type=int, default=4)
parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('--dataset', type=str, default='TrainingSetMRI', help='training dataset name that stored in ./data')
parser.add_argument('--datasetTL', type=str, default='TransferLr', help='transfer learning dataset name that stored in ./data')
parser.add_argument('--datasetGT', type=str, default='PredictionGT', help='testing dataset name that stored in ./data')
parser.add_argument('--savedir', type=str, default='save', help='dir of saving checkpoints and intermediate training results')
parser.add_argument('--use_trained_model', type=str2bool, default=True, help='whether train from an existing model or from scratch')
parser.add_argument('--use_init_model', type=str2bool, default=True, help='whether train from the init model if cannot find an existing model')
parser.add_argument('--slice', type=int, default=100, help='slice')
parser.add_argument('--super_resolution_3D', type=str2bool, default=True, help='activate 3D super-resolution')
parser.add_argument('--run_FSL', type=str2bool, default=True, help='run_FSL')
parser.add_argument('--initialization_step', type=int, default=-1, help='run_FSL')

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
    outputFolder = ''
    num_epochs_transfer_learning = 1500
    type_of_assembly = 0  # 0: input image; 1: no consistecny 2: 3d spatial-consistency
    correct_model_bias = False
    remove_outlier = True
    type_of_regressor = -1  # linear_regressor = 0,mixed_effect_model = 1,svm = 2
    if FLAGS.super_resolution_3D:
        super_resolution_label = '3D'
    else:
        super_resolution_label = ''
    if FLAGS.conf == 0:
        conditioned_enabled = False
        progression_enabled = False
        outputFolder = 'SyntheticMRI_V0' + super_resolution_label
        attention_loss_function = False
        V2_enabled = False
        test_label = '_baseline'
        print('Selected Face-Ageing configuration')
    elif FLAGS.conf == 1:
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = False
        V2_enabled = False
        test_label = '_DaniNet-V1'
        outputFolder = 'SyntheticMRI_V1' + super_resolution_label
        type_of_assembly = 1
        print('Selected DaniNet-V1 configuration')
    elif FLAGS.conf == 2:
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = False
        V2_enabled = True
        test_label = '_DaniNet-V2'
        outputFolder = 'SyntheticMRI_V2' + super_resolution_label
        type_of_assembly = 1
        print('Selected DaniNet-V2 configuration')
    elif FLAGS.conf == 3:
        num_epochs_transfer_learning = 1500
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = True
        V2_enabled = True  # fuzzy + logistic regressor
        test_label = '_DaniNet-V2_AL'
        outputFolder = 'SyntheticMRI_V2_AL' + super_resolution_label
        type_of_assembly = 2
        print('Select DaniNet-V2_AL configuration')
    elif FLAGS.conf == 4:
        num_epochs_transfer_learning = 1500
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = False
        V2_enabled = True  # fuzzy + logistic regressor
        outputFolder = 'SyntheticMRI_V1_NO_PWF' + super_resolution_label
        type_of_assembly = 2
        test_label = '_DaniNet-V1'
        print('Select DaniNet-V1_AL_no_PWF configuration')
    elif FLAGS.conf == 5:
        num_epochs_transfer_learning = 600
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = False
        V2_enabled = True  # fuzzy + logistic regressor
        test_label = '_DaniNet-V2'
        outputFolder = 'SyntheticMRI_V2_NO_PWF' + super_resolution_label
        type_of_assembly = 2
        print('Select DaniNet-V2_AL_no_PWF configuration')
    elif FLAGS.conf == 6:
        # num_epochs_transfer_learning = 600
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = True
        V2_enabled = True  # fuzzy + logistic regressor
        test_label = '_DaniNet-V2_AL'
        outputFolder = 'SyntheticMRI_V2_AL_noSpatial' + super_resolution_label
        type_of_assembly = 1
        print('Select DaniNet-V2_AL_AL_noSpatial configuration')
    elif FLAGS.conf == 7:
        num_epochs_transfer_learning = 1500
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = True
        V2_enabled = True  # fuzzy + logistic regressor
        test_label = '_DaniNet-V2_AL'
        outputFolder = 'SyntheticMRI_V2_AL_x2TF' + super_resolution_label
        type_of_assembly = 2
        print('Select DaniNet-V2_ALLA_x2TF configuration')
    elif FLAGS.conf == 8:
        num_epochs_transfer_learning = 1500
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = True
        V2_enabled = True  # fuzzy + logistic regressor
        test_label = '_DaniNet-V2_AL_bin'
        outputFolder = 'SyntheticMRI_V2_AL_bin' + super_resolution_label
        type_of_assembly = 2
        print('Select DaniNet-V2_AL_bin configuration')
    elif FLAGS.conf == 9:
        num_epochs_transfer_learning = 1500
        conditioned_enabled = True
        progression_enabled = True
        attention_loss_function = True
        V2_enabled = True  # fuzzy + logistic regressor
        test_label = '_DaniNet-V2_AL_x2TF3D_no_Loss4'
        outputFolder = 'SyntheticMRI_V2_AL_x2TF3D_no_Loss4' + super_resolution_label
        type_of_assembly = 2
        print('Select DaniNet-V2_AL_x2TF3D_no_Loss4 configuration')
    elif FLAGS.conf == -1:
        outputFolder = 'SyntheticRealFollowUp'
        type_of_assembly = 0
        print('Selected assembly input modality')
        FLAGS.super_resolution_3D = False
    elif FLAGS.conf == -2:
        outputFolder = 'SyntheticProgressionMRI' + super_resolution_label
        test_label = '_DaniNet-V2_AL'
        print('Selected assembly progression modality')
    elif FLAGS.conf == -3:
        outputFolder = 'SyntheticRealTest'
        type_of_assembly = 3
        print('Selected assembly linear regressor')
        FLAGS.super_resolution_3D = False
    else:
        print('Please select one of the available modality.')
        exit()

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    if V2_enabled:
        regressor_type = 1  # 1 logistic regressor
    else:
        regressor_type = 0  # 0 support vector regressor

    # ResearchGroup = {'Cognitive normal', 'Subjective memory concern', 'Early mild cognitive Impairment', 'Mild cognitive impairment',
    #                 'Late mild cognitive impairment', 'Alzheimer''s disease'};
    FLAGS.savedir = FLAGS.savedir + test_label
    max_regional_expansion = 10
    map_disease = (0, 1, 2, 2, 2, 3)
    age_intervals = (63, 65.5, 68, 71.5, 73, 75.5, 78, 80.5, 83, 85.5, 88)

    if FLAGS.phase == 0:
        train_regressors(max_regional_expansion=max_regional_expansion, map_disease=map_disease, regressor_type=regressor_type)
    if FLAGS.phase == 1:
        training(curr_slice=FLAGS.slice, conditioned_enabled=conditioned_enabled, progression_enabled=progression_enabled,
                 attention_loss_function=attention_loss_function, max_regional_expansion=max_regional_expansion, map_disease=map_disease,
                 age_intervals=age_intervals, V2_enabled=V2_enabled, output_dir='sample_Train', regressor_type=regressor_type,initialization_step=FLAGS.initialization_step)
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
        if FLAGS.conf == -2:
            MRI_assembler.assemblyAll_progression(test_label, age_intervals, outputFolder, FLAGS)
        elif FLAGS.conf == -3:
            MRI_assembler.assemblyTraining(test_label, age_intervals, outputFolder, type_of_assembly, FLAGS)
        else:
            MRI_assembler.assemblyAll(test_label, age_intervals, outputFolder, type_of_assembly, FLAGS)
    if FLAGS.phase == 4:
        evaluation(outputFolder, FLAGS.run_FSL, type_of_regressor, correct_model_bias, remove_outlier)


def get_follow_up_age(fileName, quaries_for_progression):
    for i in range(np.size(quaries_for_progression)):
        if os.path.basename(fileName)[:-7] == quaries_for_progression[i][0]:
            return i
    else:
        return -1


def extract_data_frame(file_list):
    ID = ["" for _ in range(np.size(file_list))]
    ages = np.zeros([np.size(file_list)])
    gender = np.zeros([np.size(file_list)])
    diagnos = np.zeros([np.size(file_list)])
    for i, currFileName in enumerate(file_list):
        ages[i] = os.path.basename(currFileName).split('_')[0]
        ID[i] = currFileName.split('ADNI')[1][1:11]
        if os.path.basename(currFileName).split('_')[1].isdigit():
            gender[i] = os.path.basename(currFileName).split('_')[1]
        else:
            gender[i] = -1
        diagnos[i] = os.path.basename(currFileName).split('_')[2]
    ID = [ID.index(l) for l in ID]
    return ID, ages, gender, diagnos


def evaluation(outputFolder, run_FSL, type_of_regressor, correct_model_bias, remove_outliers):
    # fsl_bin_dir = '/share/apps/fsl-6.0.1' #server
    number_of_considered_brain_regions = 7
    number_of_parallel_jobs = 12
    fsl_bin_dir = '/usr/local/fsl'
    environ['PATH'] = environ['PATH'] + ':' + fsl_bin_dir + '/bin'
    environ['FSLDIR'] = fsl_bin_dir
    saved_folder = '/media/dravi/data/CVPR/FLS_RealFollowUp_MRI/'

    input_files = glob.glob('./SyntheticRealFollowUp/' + "*.nii*")
    if not os.path.exists('./FLS_RealFollowUp_MRI/'):
        os.system('mkdir ./FLS_RealFollowUp_MRI/')
    if run_FSL:
        Parallel(n_jobs=number_of_parallel_jobs)(delayed(extract_volumes)('./FLS_RealFollowUp_MRI/', i) for i in input_files)
    result = np.zeros((np.size(input_files), number_of_considered_brain_regions))
    for i, file in enumerate(input_files):
        result[i, :] = (extract_volumes('./FLS_RealFollowUp_MRI/', file))

    if type_of_regressor >= 0:
        synthetic_input_files = glob.glob('./SyntheticRealTest/' + "*.nii*")
        if run_FSL:
            Parallel(n_jobs=number_of_parallel_jobs)(delayed(extract_volumes)(saved_folder, i) for i in synthetic_input_files)
        synthetic_result = np.zeros((np.size(synthetic_input_files), number_of_considered_brain_regions))

        for i, file in enumerate(synthetic_input_files):
          synthetic_result[i, :] = (extract_volumes(saved_folder, file))
        with open('objs.pkl', 'wb') as f:
           pickle.dump(synthetic_result, f)
        f = open('objs.pkl', 'rb')
        synthetic_result = pickle.load(f)
        f.close()
        [ID, ages, gender, diagnos] = extract_data_frame(synthetic_input_files)

        for i in range(number_of_considered_brain_regions):
            synthetic_result[synthetic_result[:, i] < 0] = 0
        model = np.array([object() for _ in range(number_of_considered_brain_regions)])
        data = {'ages': ages, 'ID': ID, 'gender': gender, 'diagnos': diagnos, 'vol0': synthetic_result[:, 0], 'vol1': synthetic_result[:, 1],
                'vol2': synthetic_result[:, 2],
                'vol3': synthetic_result[:, 3], 'vol4': synthetic_result[:, 4], 'vol5': synthetic_result[:, 5], 'vol6': synthetic_result[:, 5]}
        tupla = np.concatenate([ages.reshape(1,-1), gender.reshape(1,-1), diagnos.reshape(1,-1)]).T

        for i in range(number_of_considered_brain_regions):
            if type_of_regressor == 0:
                model[i] = linear_model.LinearRegression()
                model[i].fit(tupla, synthetic_result[:, i])  # linear regressor population based
                if remove_outliers:
                    clean_tupla = outlierCleaner(model[i].predict(tupla), tupla, synthetic_result[:, i])
                    model[i].fit(np.asarray(clean_tupla[0]), np.asarray(clean_tupla[1]))
            if type_of_regressor == 1:
                # vc = {'classroom': '0 + C(classroom)', 'pretest': '0 + pretest', 'ID': '0 + ID'}
                # >> > MixedLM.from_formula('test_score ~ age + gender + ID', vc_formula=vc,
                model[i] = smf.mixedlm("vol" + str(i) + "~ages+diagnos+gender+ID", data, groups=(data["diagnos"]))
                model[i]=model[i].fit()
                if remove_outliers:
                    tupla = np.concatenate([ages.reshape(1,-1),np.asarray(ID).reshape(1,-1), gender.reshape(1,-1), diagnos.reshape(1,-1)]).T
                    data_out = {'ages': ages, 'ID': ID, 'gender': gender, 'diagnos': diagnos}
                    clean_tupla =outlierCleaner(model[i].predict(data_out),tupla,synthetic_result[:, i])
                    data_out = {'ages': np.asarray(clean_tupla[0])[:,0], 'ID': np.asarray(clean_tupla[0])[:,1], 'gender': np.asarray(clean_tupla[0])[:,2], 'diagnos': np.asarray(clean_tupla[0])[:,3], 'vol0':clean_tupla[1]}
                    model[i] = smf.mixedlm("vol0"+ "~ages+diagnos+gender+ID", data_out, groups=(data_out["diagnos"]))
                    model[i]=model[i].fit()
            if type_of_regressor == 2:
                model[i] = SVR(C=10, coef0=0.0, degree=1, epsilon=0.05, gamma='auto',
                               kernel='rbf', max_iter=-1, shrinking=True, tol=0.0001, verbose=True)
                model[i].fit(tupla, synthetic_result[:, i])
                if remove_outliers:
                    clean_tupla = outlierCleaner(model[i].predict(tupla), tupla, synthetic_result[:, i])
                    model[i].fit(np.asarray(clean_tupla[0]), np.asarray(clean_tupla[1]))

        [ID, ages, gender, diagnos] = extract_data_frame(input_files)
        result2 = np.zeros((np.size(input_files), number_of_considered_brain_regions))
        for i in range(number_of_considered_brain_regions):
            if type_of_regressor == 0:
                result2[:, i] = model[i].predict(np.concatenate([ages.reshape(1, -1), gender.reshape(1, -1), diagnos.reshape(1, -1)]).T)
            if type_of_regressor == 1:
                data = {'ages': ages, 'ID': ID, 'gender': gender, 'diagnos': diagnos}
                result2[:, i] = model[i].predict(data)
            if type_of_regressor == 2:
                result2[:, i] = model[i].predict(np.concatenate([ages.reshape(1, -1), gender.reshape(1, -1), diagnos.reshape(1, -1)]).T)
    else:
        input_files = glob.glob('./' + outputFolder + '/' + "*.nii*")
        if not os.path.exists('./FLS_' + outputFolder + '/'):
            os.system('mkdir ./FLS_' + outputFolder + '/')
        if run_FSL:
            Parallel(n_jobs=number_of_parallel_jobs)(delayed(extract_volumes)('./FLS_' + outputFolder + '/', i) for i in input_files)
        result2 = np.zeros((np.size(input_files), number_of_considered_brain_regions))
        for i, file in enumerate(input_files):
            result2[i, :] = (extract_volumes('./FLS_' + outputFolder + '/', file))

    totErr = np.zeros((np.size(input_files), number_of_considered_brain_regions))
    totErrAbs = np.zeros((np.size(input_files), number_of_considered_brain_regions))
    index = [False for _ in range(np.size(input_files))]
    for j in range(number_of_considered_brain_regions):
        for i in range(np.size(input_files)):
            totErrAbs[i, j] = abs(result[i, j] - result2[i, j])
            totErr[i, j] = (result[i, j] - result2[i, j])
        if correct_model_bias:
            result[:, j] = result[:, j] - np.mean(totErr[:, j])
            result[:, j] = [0 if a_ < 0 else a_ for a_ in result[:, j]]
        #index = index + (totErr[:, j] > np.sort(totErr[:, j])[-3])  # remove sample where the estimation of the volumes fails
        index = index + (totErrAbs[:, j] > (np.median(totErrAbs[:, j]) + 3 * iqr(totErrAbs[:, j])))
    if np.shape(index)[0]>1:
        result[index, :] = 0

    totErr = np.zeros((np.size(input_files), number_of_considered_brain_regions))
    for j in range(number_of_considered_brain_regions):
        curr_index = 0
        for i in range(np.size(input_files)):
            if not (result[i, j] <= 0) and not (result2[i, j] <= 0):
                # positionInlist = get_follow_up_age(input_files[i], MRI_assembler.quaries_for_progression)
                # normalization_by_age = float(MRI_assembler.quaries_for_progression[positionInlist][1]) - float(
                #    MRI_assembler.quaries_for_progression[positionInlist][0].split('_')[0])
                totErr[curr_index, j] = abs(result[i, j] - result2[i, j])  # * normalization_by_age
                curr_index = curr_index + 1
        print("{0:.3f}".format(np.mean(totErr[:curr_index - 1, j])) + ' $\pm$ ' + "{0:.3f}".format(np.std(totErr[:curr_index - 1, j])) + '&')


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


def extract_volumes(FSL_output, input_file):
    # Vol_name=['L_hippocampus','R_hippocampus','Peripheral_grey','Ventricular_csf','Grey','White','Brain']
    number_of_considered_brain_regions = 7
    input_fileNew = os.path.basename(input_file)[:-15]
    print(input_fileNew)
    if not os.path.exists(FSL_output + input_fileNew):
        os.system('mkdir ' + FSL_output + input_fileNew)
    os.system('cp ' + input_file + ' ' + FSL_output + input_fileNew + '/input.nii.gz')

    input_file = input_fileNew
    Vol = np.zeros(number_of_considered_brain_regions)
    if not os.path.exists('' + FSL_output + input_file + '/tissue'):
        os.system('run_first_all -i ' + FSL_output + input_file + '/input.nii.gz -b -s L_Hipp -o ' + FSL_output + input_file + '/regions')
    if os.path.exists('' + FSL_output + input_file + '/regions-L_Hipp_first.nii'):
        os.system(
            'fslstats ' + FSL_output + input_file + '/regions-L_Hipp_first.nii -h 64 > ' + FSL_output + input_file + '/region_volume1.txt')
        text_file = open('' + FSL_output + input_file + '/region_volume1.txt', "r")
        lines = text_file.readlines()
        text_file.close()
        Vol[0] = np.float32(lines[63])
    if not os.path.exists('' + FSL_output + input_file + '/tissue'):
        os.system('run_first_all -i ' + FSL_output + input_file + '/input.nii.gz -b -s R_Hipp -o ' + FSL_output + input_file + '/regions')
    if os.path.exists('' + FSL_output + input_file + '/regions-R_Hipp_first.nii'):
        os.system(
            'fslstats ' + FSL_output + input_file + '/regions-R_Hipp_first.nii -h 64 > ' + FSL_output + input_file + '/region_volume2.txt')
        text_file = open('' + FSL_output + input_file + '/region_volume2.txt', "r")
        lines = text_file.readlines()
        text_file.close()
        Vol[1] = np.float32(lines[63])
    if not os.path.exists('' + FSL_output + input_file + '/tissue'):
        os.system('./sienax_mod ' + FSL_output + input_file + '/input.nii.gz -o ' + FSL_output + input_file + '/tissue -r')
    text_file = open('' + FSL_output + input_file + '/tissue/report.sienax', "r")
    lines = text_file.readlines()
    text_file.close()
    t = 2
    for i in range(np.size(lines) - 5, np.size(lines)):
        if np.size(re.findall('\d+\.\d+', lines[i])) > 0:
            Vol[t] = float(re.findall('\d+\.\d+', lines[i])[0])
        elif np.size(re.findall('\d+', lines[i])) > 0:
            Vol[t] = float(re.findall('\d+', lines[i])[0])
        else:
            Vol[t] = -1
        t = t + 1
    return Vol / Vol[6] * 100


def training(curr_slice, conditioned_enabled, progression_enabled, attention_loss_function, max_regional_expansion, map_disease, age_intervals, V2_enabled,
             output_dir, regressor_type,initialization_step):
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
            attention_loss_function=attention_loss_function,
            initialization_step=initialization_step
        )

        print('\n\tTraining Mode')
        if not FLAGS.use_trained_model:
            print('\n\tPre-train the network')
            model.train(
                num_epochs=5,  # number of epochs
                use_trained_model=FLAGS.use_trained_model,
                use_init_model=False,
                conditioned_enabled=conditioned_enabled,
                progression_enabled=progression_enabled,
                initialization_step=initialization_step
            )
            print('\n\tPre-train is done! The training will start.')
            FLAGS.use_trained_model = True
        model.train(
            num_epochs=FLAGS.epoch,  # number of epochs
            use_trained_model=FLAGS.use_trained_model,
            use_init_model=FLAGS.use_init_model,
            conditioned_enabled=conditioned_enabled,
            progression_enabled=progression_enabled,
            initialization_step=initialization_step
        )
    tf.reset_default_graph()


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    tf.app.run()
