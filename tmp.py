from matplotlib.pyplot import imread
from scipy.misc import imread
import fnmatch
import MRI_assembler
import numpy as np

import os
l1=os.listdir('./input/40/')
l2=os.listdir('./output/40/')
for i in range(0,179):
	matching = [s for s in l2 if l1[i].split('ADNI')[1][1:11] in s]
	print("assembly_MRI('"+l1[i]+"', test_label,"+matching[0].split('_')[0]+", age_intervals)")


# validation_folder = os.path.join('./data', FLAGS.datasetGT)
# TL_folder = os.path.join('./data', FLAGS.datasetTL)
# validation(validation_folder, TL_folder, FLAGS.slice, test_label, age_intervals)

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
    from skimage import measure
    from matplotlib import pyplot

    all_GT = os.listdir(validation_folder + '/' + str(curr_slice) + '/')
    final_similarity = 0
    for currentGT in all_GT:
        current_patient_id = currentGT.split('ADNI_')[1][:10]
        currentGT_image = imread(validation_folder + '/' + str(curr_slice) + '/' + currentGT)
        first_scan_name = fnmatch.filter(os.listdir(TL_folder + '/' + str(curr_slice) + '/'), '*' + str(current_patient_id) + '*')
        first_scan_name = first_scan_name[0]
        input_image = imread(TL_folder + '/' + str(curr_slice) + '/' + first_scan_name)
        generate_pred_scan = MRI_assembler.generate_MRI_slice_average_5(curr_slice, output_dir, first_scan_name, FLAGS)
        generate_pred_scan = MRI_assembler.generate_MRI_slice(generate_pred_scan, np.float32(currentGT.split('_')[0]), age_intervals)
        generate_pred_scan = hist_norm(generate_pred_scan, input_image)
        show_results = 0
        if show_results:
            pyplot.imshow(generate_pred_scan)
            pyplot.show()
            pyplot.imshow(currentGT_image)
            pyplot.show()
            pyplot.imshow(input_image)
            pyplot.show()
        final_similarity = [final_similarity, measure.compare_ssim(generate_pred_scan, currentGT_image)]
    return final_similarity