from __future__ import print_function, division, absolute_import, unicode_literals
import nibabel as nib
import numpy as np
from matplotlib.pyplot import imread
from scipy.misc import imread
import skfuzzy
import glob as glob
import os
from skimage.exposure import match_histograms
from DCSRN.tf_dcsrn import dcsrn, image_util

output_path = "./DCSRN/snapshots/"
dataset_HCP = "./DCSRN/HCP_NPY_Augment/"
net = dcsrn.DCSRN(channels=1)
path = os.path.join(output_path, "model.cpkt")

# todo the name of the folder for the synthetic data are equal to the input name rather than to the follow-up
transformationMatrix = [[0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]]

sigma_membership_function = 2.5
sigma_for_weigh_in_5_slice_average = 1.5

quaries_for_progression2 = [['71.6521_0_6_1_ADNI_005_S_1341_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070717180727152_S27674_I60421.nii.png', '73.7288'],
                            ['75.3808_0_6_1_ADNI_024_S_1307_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070731173528231_S27062_I63419.nii.png', '77.3836'],
                            ['70.7534_0_6_1_ADNI_002_S_1018_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070217031115360_S23127_I40821.nii.png', '72.7918'],
                            ['75.4493_0_6_1_ADNI_023_S_0084_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20061201101851665_S10764_I31204.nii.png', '77.4986'],
                            ['80.526_0_4_1_ADNI_032_S_0718_MR_MPR-R____N3_Br_20070118003551300_S16860_I36476.nii.png', '82.726'],
                            ['77.8877_0_2_1_ADNI_127_S_5228_MR_MT1__N3m_Br_20140218085638626_S193159_I414573.nii.png', '79.9315'],
                            ['81.8493_0_6_1_ADNI_036_S_0759_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070120004547956_S18095_I36973.nii.png', '83.9041'],
                            ['68.0959_1_3_1_ADNI_067_S_2195_MR_MT1__GradWarp__N3m_Br_20110308161908105_S95056_I223044.nii.png', '71.1425']]

quaries_for_progression = [['79.063_1_2_1_ADNI_006_S_5153_MR_MT1__N3m_Br_20130429150511942_S187543_I369218.nii.png', '81.3726'],
                           ['71.1178_1_1_1_ADNI_128_S_0522_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070821191050747_S33977_I69665.nii.png', '76.8849'],
                           ['66.0795_1_3_1_ADNI_031_S_2022_MR_MT1__N3m_Br_20110308135351043_S89098_I222954.nii.png', '68.0959'],
                           ['73.1096_1_3_1_ADNI_029_S_2395_MR_MT1__GradWarp__N3m_Br_20120110142103537_S131129_I277026.nii.png', '77.5041'],
                           ['67.6164_0_2_1_ADNI_037_S_5222_MR_MT1__N3m_Br_20140217130229125_S194549_I414492.nii.png', '69.7425'],
                           ['78.4767_1_1_1_ADNI_137_S_0972_MR_MPR-R__GradWarp__N3_Br_20070808222725581_S33060_I66265.nii.png', '80.9699'],
                           ['80.8247_1_1_1_ADNI_041_S_4014_MR_MT1__GradWarp__N3m_Br_20110427155103716_S104714_I229335.nii.png', '82.9589'],
                           ['76.2767_0_1_1_ADNI_029_S_0824_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20080308130404745_S18138_I96288.nii.png', '79.4'],
                           ['77.1397_0_1_1_ADNI_099_S_0352_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20061228144308428_S12993_I34540.nii.png', '80.1973'],
                           ['68.1507_1_2_1_ADNI_041_S_5253_MR_MT1__GradWarp__N3m_Br_20130909123604868_S195544_I388460.nii.png', '70.1616'],
                           ['69.4658_0_6_1_ADNI_036_S_0760_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070209144922688_S18263_I38655.nii.png', '71.5863'],
                           ['78.3452_0_1_1_ADNI_023_S_0031_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20071113171832266_S13260_I81860.nii.png', '80.9753'],
                           ['72.0247_0_3_1_ADNI_022_S_4805_MR_MT1__GradWarp__N3m_Br_20120803100315441_S157184_I321453.nii.png', '74.0466'],
                           ['75.6301_0_5_1_ADNI_016_S_4902_MR_MT1__GradWarp__N3m_Br_20121206135322230_S173860_I349864.nii.png', '78.7425'],
                           ['71.6521_0_6_1_ADNI_005_S_1341_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070717180727152_S27674_I60421.nii.png', '73.7288'],
                           ['65.2384_0_2_1_ADNI_012_S_5195_MR_MT1__N3m_Br_20130624165310232_S192516_I377881.nii.png', '67.3068'],
                           ['78.0658_0_1_1_ADNI_136_S_0196_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070215191203517_S13254_I40264.nii.png', '81.0959'],
                           ['70.7534_0_6_1_ADNI_002_S_1018_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070217031115360_S23127_I40821.nii.png', '72.7918'],
                           ['73.1068_0_3_1_ADNI_116_S_4635_MR_MT1__GradWarp__N3m_Br_20121025150702194_S159504_I342605.nii.png', '76.9041'],
                           ['80.0274_1_1_1_ADNI_014_S_0558_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070213011223680_S15790_I39687.nii.png', '83.074'],
                           ['70.526_0_2_1_ADNI_002_S_5256_MR_MT1__N3m_Br_20130801190358040_S195295_I384094.nii.png', '72.6822'],
                           ['75.3808_0_6_1_ADNI_024_S_1307_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070731173528231_S27062_I63419.nii.png', '77.3836'],
                           ['71.0712_1_2_1_ADNI_126_S_5243_MR_MT1__N3m_Br_20130724140336799_S195168_I382272.nii.png', '73.1671'],
                           ['80.526_0_4_1_ADNI_032_S_0718_MR_MPR-R____N3_Br_20070118003551300_S16860_I36476.nii.png', '82.726'],
                           ['68.1342_1_2_1_ADNI_100_S_5091_MR_MT1__N3m_Br_20130313092014289_S183968_I362900.nii.png', '70.3452'],
                           ['70.6466_0_3_1_ADNI_129_S_2332_MR_MT1__GradWarp__N3m_Br_20110330135716681_S101174_I225431.nii.png', '73.6192'],
                           ['77.8877_0_2_1_ADNI_127_S_5228_MR_MT1__N3m_Br_20140218085638626_S193159_I414573.nii.png', '79.9315'],
                           ['69.4055_1_4_1_ADNI_130_S_0289_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20080220155538554_S41275_I91163.nii.png', '71.92'],
                           ['72.4_0_1_1_ADNI_130_S_0886_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20071106112206789_S41102_I80948.nii.png', '76.5123'],
                           ['74.1616_1_1_1_ADNI_014_S_0519_MR_MPR__GradWarp__B1_Correction__N3_Br_20070213000353268_S14488_I39646.nii.png', '78.4658'],
                           ['81.8493_0_6_1_ADNI_036_S_0759_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070120004547956_S18095_I36973.nii.png', '83.9041'],
                           ['72.0849_1_4_1_ADNI_094_S_0921_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070802211921906_S32300_I64369.nii.png', '74.4822'],
                           ['73.5397_0_2_1_ADNI_002_S_5230_MR_MT1__N3m_Br_20130711141124934_S193616_I379928.nii.png', '75.9699'],
                           ['65.6192_1_2_1_ADNI_012_S_5157_MR_MT1__N3m_Br_20130530081001410_S189829_I374561.nii.png', '67.726'],
                           ['72.7699_1_4_1_ADNI_012_S_1175_MR_MPR____N3_Br_20070711171420642_S24756_I59231.nii.png', '74.8'],
                           ['68.0493_1_6_1_ADNI_018_S_0682_MR_MPR-R____N3_Br_20070101224407196_S16292_I35046.nii.png', '70.0959'],
                           ['59.8247_0_6_1_ADNI_032_S_0147_MR_MPR-R____N3_Br_20070123193235582_S11187_I37212.nii.png', '61.989'],
                           ['72.9425_0_2_1_ADNI_013_S_5171_MR_MT1__N3m_Br_20130618165504044_S191233_I377079.nii.png', '75.126'],
                           ['76.474_1_3_1_ADNI_068_S_4217_MR_MT1__GradWarp__N3m_Br_20110910142439801_S121337_I255437.nii.png', '78.7562'],
                           ['69.2849_0_2_1_ADNI_033_S_5198_MR_MT1__GradWarp__N3m_Br_20130618170006470_S191196_I377083.nii.png', '71.3205'],
                           ['69.9534_0_2_1_ADNI_127_S_5185_MR_MT1__N3m_Br_20140218085551069_S190934_I414572.nii.png', '71.9644'],
                           ['75.4493_0_6_1_ADNI_023_S_0084_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20061201101851665_S10764_I31204.nii.png', '77.4986'],
                           ['70.1562_1_1_1_ADNI_037_S_0327_MR_MPR__GradWarp__N3_Br_20071030155120722_S13473_I79731.nii.png', '72.2521'],
                           ['78.9233_1_4_1_ADNI_037_S_0566_MR_MPR-R__GradWarp__N3_Br_20071030170112484_S15456_I79806.nii.png', '81.0685'],
                           ['74.8795_1_4_1_ADNI_067_S_0176_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20061229173912445_S12798_I34802.nii.png', '77.9671'],
                           ['80.526_1_2_1_ADNI_100_S_5096_MR_MT1__N3m_Br_20130510150416760_S185434_I371986.nii.png', '82.7425'],
                           ['76.5781_1_1_1_ADNI_123_S_0298_MR_MPR-R____N3_Br_20070802222152867_S13153_I64426.nii.png', '79.5973'],
                           ['66.2411_0_1_1_ADNI_068_S_4424_MR_MT1__GradWarp__N3m_Br_20111228151925376_S134919_I274671.nii.png', '69.6849'],
                           ['80.3288_1_1_1_ADNI_005_S_0610_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070923122713025_S37098_I74586.nii.png', '83.8'],
                           ['90.1616_1_2_1_ADNI_135_S_5273_MR_MT1__GradWarp__N3m_Br_20130909172409309_S197355_I388923.nii.png', '92.2164'],
                           ['70.4411_0_2_1_ADNI_020_S_5140_MR_MT1__GradWarp__N3m_Br_20130430111940654_S187865_I369385.nii.png', '72.5233'],
                           ['77.5452_1_6_1_ADNI_099_S_1144_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20080410141128267_S24218_I102039.nii.png', '79.6521'],
                           ['71.6822_1_6_1_ADNI_031_S_0554_MR_MPR-R__GradWarp__N3_Br_20070804144329166_S14904_I64710.nii.png', '73.7781'],
                           ['71.9479_0_2_1_ADNI_027_S_5277_MR_MT1__N3m_Br_20130829111716132_S198494_I388029.nii.png', '73.9836'],
                           ['64.9699_1_2_1_ADNI_135_S_5269_MR_MT1__GradWarp__N3m_Br_20130808143026990_S196646_I384808.nii.png', '67.074'],
                           ['84.9918_0_6_1_ADNI_109_S_1157_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070808201939048_S24712_I66162.nii.png', '87.0247'],
                           ['78.6466_1_1_1_ADNI_130_S_0232_MR_MPR__GradWarp__B1_Correction__N3_Br_20070210212735744_S19172_I39158.nii.png', '82.16'],
                           ['76.8712_1_2_1_ADNI_012_S_5121_MR_MT1__N3m_Br_20130327161327024_S184614_I364667.nii.png', '78.9973'],
                           ['80.2849_0_6_1_ADNI_022_S_0129_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070712152831181_S11484_I59488.nii.png', '82.3205'],
                           ['76.4247_0_2_1_ADNI_130_S_5142_MR_MT1__N3m_Br_20140625140503335_S187633_I432883.nii.png', '78.4712'],
                           ['72.9342_1_1_1_ADNI_021_S_0647_MR_MPR-R__GradWarp__N3_Br_20061219093645477_S15958_I33651.nii.png', '75.0795'],
                           ['65.5479_0_2_1_ADNI_012_S_5213_MR_MT1__N3m_Br_20130801191233337_S195995_I384096.nii.png', '67.7178'],
                           ['73.0329_1_1_1_ADNI_137_S_0459_MR_MPR-R__GradWarp__N3_Br_20070323165833183_S13735_I46632.nii.png', '76.063'],
                           ['70.0493_0_2_1_ADNI_053_S_5272_MR_MT1__N3m_Br_20130829110738173_S197629_I388027.nii.png', '72.1425'],
                           ['79.6438_1_4_1_ADNI_130_S_0505_MR_MPR__GradWarp__B1_Correction__N3_Br_20070210210356220_S17292_I39143.nii.png', '82.9288'],
                           ['77.3808_1_1_1_ADNI_021_S_0984_MR_MPR-R__GradWarp__N3_Br_20071002181836795_S34235_I76618.nii.png', '79.8548'],
                           ['70.2055_0_6_1_ADNI_068_S_0109_MR_MPR-R____N3_Br_20070731181213576_S10771_I63456.nii.png', '72.4384'],
                           ['61.2904_1_1_1_ADNI_073_S_4795_MR_MT1__GradWarp__N3m_Br_20120626144852418_S153047_I312680.nii.png', '63.7836'],
                           ['69.3452_1_2_1_ADNI_053_S_5296_MR_MT1__N3m_Br_20140305152714432_S207828_I416077.nii.png', '71.3671'],
                           ['63.6822_1_3_1_ADNI_126_S_2360_MR_MT1__GradWarp__N3m_Br_20111216190951330_S130277_I272846.nii.png', '67.1781'],
                           ['71.526_1_1_1_ADNI_009_S_0751_MR_MPR-R__GradWarp__N3_Br_20070926180636377_S26994_I75397.nii.png', '77.0712'],
                           ['78.8767_1_4_1_ADNI_007_S_0344_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070118032630969_S12632_I36583.nii.png', '83.0932'],
                           ['81.589_0_1_1_ADNI_116_S_4092_MR_MT1__GradWarp__N3m_Br_20110701093218876_S112544_I242878.nii.png', '83.6685'],
                           ['80.3315_0_1_1_ADNI_033_S_0741_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070304112905373_S17007_I42454.nii.png', '83.4055'],
                           ['69.6795_0_4_1_ADNI_127_S_1427_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20080220144535548_S37934_I91130.nii.png', '72.6795'],
                           ['74.4466_0_2_1_ADNI_057_S_5292_MR_MT1__N3m_Br_20131114112019596_S204225_I398341.nii.png', '76.474'],
                           ['73.9726_0_6_1_ADNI_011_S_0010_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20061208115113738_S8801_I32278.nii.png', '75.9726'],
                           ['56.5479_1_5_1_ADNI_052_S_4945_MR_MT1__GradWarp__N3m_Br_20121206104425886_S172159_I349800.nii.png', '58.7534'],
                           ['63.8575_1_4_1_ADNI_137_S_0443_MR_MPR__GradWarp__N3_Br_20070808211136646_S27311_I66212.nii.png', '66.3096'],
                           ['69.6027_0_6_1_ADNI_094_S_1164_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070812144704499_S23872_I67227.nii.png', '71.7616'],
                           ['78.0137_0_2_1_ADNI_130_S_5258_MR_MT1__N3m_Br_20140217114647035_S195638_I414456.nii.png', '80.0959'],
                           ['65.5233_0_5_1_ADNI_094_S_4630_MR_MT1__GradWarp__N3m_Br_20170502150851665_S146114_I846435.nii.png', '67.6712'],
                           ['68.9534_1_4_1_ADNI_033_S_0513_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070302155253729_S14674_I42262.nii.png', '70.9836'],
                           ['71.8795_1_1_1_ADNI_011_S_0023_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20061208152249710_S8867_I32413.nii.png', '76.0658'],
                           ['75.3507_0_2_1_ADNI_032_S_5263_MR_MT1__GradWarp__N3m_Br_20130909115830760_S197801_I388432.nii.png', '77.7562'],
                           ['74.6219_1_1_1_ADNI_010_S_0067_MR_MPR-R____N3_Br_20080308121705559_S10346_I96213.nii.png', '76.7836'],
                           ['78.9425_0_2_1_ADNI_033_S_5259_MR_MT1__GradWarp__N3m_Br_20130909115928504_S196570_I388433.nii.png', '81.0658'],
                           ['65.7397_0_1_1_ADNI_136_S_4269_MR_MT1__N3m_Br_20111108150018959_S127664_I265203.nii.png', '67.8411'],
                           ['70.6959_0_6_1_ADNI_062_S_0730_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070424120901998_S17061_I50491.nii.png', '72.7425'],
                           ['68.0959_1_3_1_ADNI_067_S_2195_MR_MT1__GradWarp__N3m_Br_20110308161908105_S95056_I223044.nii.png', '71.1425'],
                           ['80.5068_1_2_1_ADNI_036_S_5271_MR_MT1__GradWarp__N3m_Br_20130909121615590_S197093_I388447.nii.png', '83.5151'],
                           ['76.0712_1_1_1_ADNI_128_S_4586_MR_MT1__GradWarp__N3m_Br_20120626144124947_S152929_I312674.nii.png', '78.2027'],
                           ['65.7479_0_2_1_ADNI_009_S_5176_MR_MT1__GradWarp__N3m_Br_20140625152001765_S189558_I432911.nii.png', '67.8795'],
                           ['77.3479_1_1_1_ADNI_062_S_0768_MR_MPR__GradWarp__B1_Correction__N3_Br_20070424122138787_S17527_I50505.nii.png', '79.3753'],
                           ['69.0932_1_3_1_ADNI_068_S_4332_MR_MT1__GradWarp__N3m_Br_20111121113457406_S129382_I267769.nii.png', '72.7589'],
                           ['77.1233_0_1_1_ADNI_062_S_0578_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070424114948584_S15036_I50463.nii.png', '79.1589'],
                           ['68.8822_0_2_1_ADNI_027_S_5110_MR_MT1__N3m_Br_20140218084909943_S184796_I414565.nii.png', '71.0329'],
                           ['71.8411_0_3_1_ADNI_130_S_2391_MR_MT1__N3m_Br_20120416132603467_S112324_I297599.nii.png', '73.989'],
                           ['69.0301_1_6_1_ADNI_062_S_0793_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070424124029550_S18188_I50528.nii.png', '71.3233'],
                           ['83.6658_0_6_1_ADNI_018_S_0335_MR_MPR-R____N3_Br_20070101214453591_S14587_I35011.nii.png', '85.7753'],
                           ['74.8521_1_5_1_ADNI_057_S_4888_MR_MT1__GradWarp__N3m_Br_20120808112702529_S159516_I322359.nii.png', '78.874'],
                           ['76.674_0_1_1_ADNI_114_S_0601_MR_MPR__GradWarp__B1_Correction__N3_Br_20070213200251690_S15201_I39849.nii.png', '78.7753'],
                           ['77.4493_1_2_1_ADNI_137_S_4862_MR_MT1__N3m_Br_20130909172814431_S195943_I388927.nii.png', '79.5507'],
                           ['74.2164_1_1_1_ADNI_131_S_0123_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20070403130049441_S21795_I47966.nii.png',
                            '77.4822'],
                           ['66.4877_0_2_1_ADNI_067_S_5159_MR_MT1__GradWarp__N3m_Br_20130507123742637_S188571_I371250.nii.png', '68.5616'],
                           ['73.2192_0_5_1_ADNI_136_S_4189_MR_MT1__N3m_Br_20110928092552578_S122679_I258683.nii.png', '75.3616'],
                           ['64.2822_1_3_1_ADNI_109_S_4455_MR_MT1__GradWarp__N3m_Br_20120221111634642_S140743_I285906.nii.png', '66.3123'],
                           ['85.011_1_4_1_ADNI_126_S_0709_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070922104214456_S37404_I74491.nii.png', '87.0192'],
                           ['69.7671_1_4_1_ADNI_098_S_0667_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070818140815913_S27462_I68572.nii.png', '72.2'],
                           ['69.6438_0_2_1_ADNI_126_S_5214_MR_MT1__N3m_Br_20130618084902877_S191969_I376906.nii.png', '71.7151'],
                           ['71.8795_1_6_1_ADNI_123_S_0162_MR_MPR-R____N3_Br_20070802221619449_S11551_I64421.nii.png', '73.9781'],
                           ['73.9014_1_1_1_ADNI_068_S_0127_MR_MPR____N3_Br_20090717151112105_S65209_I149595.nii.png', '76.8247'],
                           ['68.1644_1_3_1_ADNI_135_S_4356_MR_MT1__GradWarp__N3m_Br_20111121111949549_S128879_I267753.nii.png', '70.2466'],
                           ['68.7178_1_2_1_ADNI_002_S_5178_MR_MT1__N3m_Br_20130529154358875_S189796_I374467.nii.png', '70.8795'],
                           ['60.2137_0_4_1_ADNI_037_S_0552_MR_MPR-R__GradWarp__N3_Br_20071030165302495_S14889_I79799.nii.png', '62.2849'],
                           ['64.737_1_4_1_ADNI_037_S_0588_MR_MPR-R__GradWarp__N3_Br_20071030172305051_S23601_I79827.nii.png', '67.0082'],
                           ['80.1836_1_6_1_ADNI_011_S_0053_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070108231238897_S10064_I35485.nii.png', '82.2247'],
                           ['55.0767_0_5_1_ADNI_037_S_4381_MR_MT1__GradWarp__N3m_Br_20120125131543498_S137404_I280574.nii.png', '57.2493'],
                           ['80.9397_1_6_1_ADNI_007_S_0316_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070118030733120_S12583_I36572.nii.png', '83.0849'],
                           ['72.0055_0_4_1_ADNI_022_S_1351_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070712162435303_S28483_I59619.nii.png', '75.0301'],
                           ['86.4986_0_1_1_ADNI_013_S_0575_MR_MPR__GradWarp__B1_Correction__N3_Br_20070426091206571_S17858_I51155.nii.png', '88.589'],
                           ['70.811_0_6_1_ADNI_123_S_0094_MR_MPR-R____N3_Br_20070802220658596_S10713_I64412.nii.png', '72.8521'],
                           ['67.2658_1_2_1_ADNI_019_S_5242_MR_MT1__N3m_Br_20130709102339134_S193992_I379548.nii.png', '69.3781'],
                           ['68.5068_0_4_1_ADNI_128_S_0608_MR_MPR-R__GradWarp__N3_Br_20070819173459985_S15212_I69018.nii.png', '70.6384'],
                           ['67.4712_0_1_1_ADNI_073_S_4559_MR_MT1__GradWarp__N3m_Br_20120308101114630_S141768_I288879.nii.png', '69.5397'],
                           ['71.3342_1_1_1_ADNI_029_S_0843_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20071125125954274_S32382_I83048.nii.png', '73.8438'],
                           ['66.0082_1_2_1_ADNI_127_S_5266_MR_MT1__N3m_Br_20140218085828754_S196947_I414575.nii.png', '68.1041'],
                           ['79.5096_1_5_1_ADNI_023_S_4243_MR_MT1__GradWarp__N3m_Br_20111121111732075_S128504_I267751.nii.png', '81.811'],
                           ['65.9562_1_6_1_ADNI_018_S_0286_MR_MPR-R____N3_Br_20070119234045119_S12434_I36921.nii.png', '68.1041'],
                           ['66.5562_0_2_1_ADNI_100_S_5102_MR_MT1__N3m_Br_20130510151204686_S185774_I371988.nii.png', '68.7233'],
                           ['85.411_1_2_1_ADNI_021_S_5236_MR_MT1__GradWarp__N3m_Br_20130711142001516_S193604_I379936.nii.png', '87.4849'],
                           ['71.3452_1_2_1_ADNI_073_S_5227_MR_MT1__GradWarp__N3m_Br_20130910111939930_S192788_I389155.nii.png', '73.3808'],
                           ['80.1315_0_1_1_ADNI_029_S_0866_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070807155433987_S18915_I65615.nii.png', '82.2192'],
                           ['62.9068_0_5_1_ADNI_009_S_4324_MR_MT1__GradWarp__N3m_Br_20111216165843571_S129810_I272728.nii.png', '65.0356'],
                           ['73.263_1_6_1_ADNI_023_S_0083_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20061130194925284_S10569_I31147.nii.png', '75.3945'],
                           ['74.3178_1_1_1_ADNI_011_S_0005_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20061206162739115_S12036_I31889.nii.png', '76.8274'],
                           ['83.8603_1_1_1_ADNI_033_S_0923_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070809195626699_S30209_I66596.nii.png', '89.3205'],
                           ['78.1397_1_3_1_ADNI_072_S_4007_MR_MT1__GradWarp__N3m_Br_20120425143407156_S101141_I300516.nii.png', '80.1973'],
                           ['67.5671_1_2_1_ADNI_100_S_5280_MR_MT1__N3m_Br_20130910134227138_S199027_I389161.nii.png', '69.6877'],
                           ['66.7644_0_1_1_ADNI_068_S_4340_MR_MT1__GradWarp__N3m_Br_20111206104203287_S131144_I270045.nii.png', '70.6575'],
                           ['59.811_0_2_1_ADNI_032_S_5289_MR_MT1__GradWarp__N3m_Br_20131114104009767_S204015_I398319.nii.png', '62.0137'],
                           ['79.5233_0_6_1_ADNI_007_S_1339_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070607135220070_S27415_I56323.nii.png', '81.7589'],
                           ['72.4575_0_6_1_ADNI_036_S_0577_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070119233627442_S14975_I36918.nii.png', '74.4685'],
                           ['72.5945_0_2_1_ADNI_021_S_5177_MR_MT1__GradWarp__N3m_Br_20130530130555892_S189690_I374610.nii.png', '74.663'],
                           ['80.6795_1_4_1_ADNI_027_S_0307_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20080423173155047_S48851_I103676.nii.png', '82.7452'],
                           ['77.2575_1_3_1_ADNI_012_S_4188_MR_MT1__N3m_Br_20110928090612523_S121622_I258662.nii.png', '81.3452'],
                           ['84.3507_1_6_1_ADNI_057_S_1371_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070730190819800_S28667_I62997.nii.png', '86.4219'],
                           ['81.0356_0_4_1_ADNI_068_S_0802_MR_MPR-R____N3_Br_20070120023628251_S18267_I37051.nii.png', '83.3534'],
                           ['61.189_0_3_1_ADNI_137_S_4623_MR_MT1__GradWarp__N3m_Br_20120406133713328_S146004_I296411.nii.png', '65.2548'],
                           ['72.8493_0_1_1_ADNI_003_S_4081_MR_MT1__GradWarp__N3m_Br_20110715125926714_S114213_I244937.nii.png', '74.9397'],
                           ['65.3096_1_3_1_ADNI_067_S_4212_MR_MT1__GradWarp__N3m_Br_20120327110013354_S144607_I293697.nii.png', '68.8466'],
                           ['63.6521_1_4_1_ADNI_137_S_0669_MR_MPR__GradWarp__N3_Br_20070306175246416_S17100_I43007.nii.png', '66.7288'],
                           ['61.0_0_4_1_ADNI_128_S_1406_MR_MT1__GradWarp__N3m_Br_20140814091326353_S225550_I439324.nii.png', '64.2274'],
                           ['80.326_1_2_1_ADNI_021_S_5237_MR_MT1__GradWarp__N3m_Br_20130711142223754_S193595_I379938.nii.png', '82.3753'],
                           ['72.4932_1_4_1_ADNI_031_S_1066_MR_MPR-R__GradWarp__N3_Br_20070813152924349_S22390_I67437.nii.png', '75.5699'],
                           ['72.9589_0_2_1_ADNI_036_S_5283_MR_MT1__GradWarp__N3m_Br_20130909121814067_S199119_I388449.nii.png', '75.9671'],
                           ['65.7425_0_2_1_ADNI_053_S_5202_MR_MT1__N3m_Br_20130617143900308_S191745_I376832.nii.png', '68.0329'],
                           ['71.3863_1_3_1_ADNI_068_S_2168_MR_MT1__GradWarp__N3m_Br_20110308170256074_S94292_I223063.nii.png', '74.5233'],
                           ['84.3699_1_1_1_ADNI_082_S_4339_MR_MT1__GradWarp__N3m_Br_20111121111528565_S128898_I267749.nii.png', '86.5342'],
                           ['79.7616_1_6_1_ADNI_062_S_0690_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070424115616838_S16923_I50472.nii.png', '81.7945'],
                           ['78.1699_1_3_1_ADNI_014_S_2308_MR_MT1__GradWarp__N3m_Br_20140620085302036_S218674_I431751.nii.png', '81.3887'],
                           ['70.6_1_2_1_ADNI_041_S_5131_MR_MT1__GradWarp__N3m_Br_20130507144711248_S186341_I371434.nii.png', '72.6192'],
                           ['76.4904_0_2_1_ADNI_127_S_5200_MR_MT1__N3m_Br_20140625152650849_S191890_I432918.nii.png', '79.9096'],
                           ['76.126_0_1_1_ADNI_020_S_0883_MR_MPR__GradWarp__B1_Correction__N3_Br_20070718141242422_S19459_I60672.nii.png', '78.4932'],
                           ['71.4466_0_4_1_ADNI_128_S_0258_MR_MPR-R__GradWarp__N3_Br_20070819153607648_S12927_I68884.nii.png', '73.6247'],
                           ['56.6575_0_6_1_ADNI_137_S_0366_MR_MPR-R__GradWarp__N3_Br_20070323161434454_S14200_I46610.nii.png', '58.7507'],
                           ['69.2438_0_4_1_ADNI_073_S_0909_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070927080829184_S19717_I75489.nii.png', '72.4274'],
                           ['78.474_0_1_1_ADNI_129_S_4396_MR_MT1__GradWarp__N3m_Br_20111206105556292_S131489_I270065.nii.png', '80.5014'],
                           ['77.0932_0_1_1_ADNI_006_S_0681_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20080224150536783_S18450_I92309.nii.png', '79.2247'],
                           ['65.0_0_6_1_ADNI_010_S_0829_MR_MPR____N3_Br_20070731162759291_S26116_I63339.nii.png', '67.011'],
                           ['77.5726_1_1_1_ADNI_036_S_0576_MR_MPR__GradWarp__B1_Correction__N3_Br_20070119231131615_S15156_I36899.nii.png', '79.6986'],
                           ['82.0055_0_2_1_ADNI_027_S_5288_MR_MT1__N3m_Br_20130926153329974_S201422_I392161.nii.png', '84.0521'],
                           ['73.5616_1_4_1_ADNI_137_S_0668_MR_MPR__GradWarp__N3_Br_20070306173913456_S17610_I42999.nii.png', '79.6082'],
                           ['89.0658_1_1_1_ADNI_016_S_4121_MR_MT1__GradWarp__N3m_Br_20110804072621384_S115573_I248639.nii.png', '91.1342'],
                           ['68.7123_1_5_1_ADNI_126_S_4896_MR_MT1__N3m_Br_20130610152303186_S181635_I375894.nii.png', '72.2767'],
                           ['66.663_1_2_1_ADNI_073_S_5167_MR_MT1__GradWarp__N3m_Br_20130619095422052_S192007_I377169.nii.png', '68.8192'],
                           ['67.0932_0_2_1_ADNI_024_S_5290_MR_MT1__GradWarp__N3m_Br_20130919101714986_S200100_I391093.nii.png', '69.1616'],
                           ['79.0356_0_2_1_ADNI_053_S_5287_MR_MT1__N3m_Br_20130919094227753_S200252_I391059.nii.png', '81.0904'],
                           ['78.7205_0_2_1_ADNI_114_S_5234_MR_MT1__GradWarp__N3m_Br_20130910134614720_S193630_I389165.nii.png', '80.7863']]


def generate_MRI(generated_images, age_to_generate, age_intervals):
    bin_centers = np.convolve(age_intervals, [0.5, 0.5], 'valid')
    generated_image = np.zeros([128, 128])
    batch_fuzzy_membership = np.zeros(10)
    for t in range(10):
        batch_fuzzy_membership[t] = skfuzzy.membership.gaussmf(age_to_generate, bin_centers[t], sigma_membership_function)
        generated_image = generated_image + generated_images[:128, (t * 128):((t + 1) * 128)] * batch_fuzzy_membership[t]

    generated_image = (generated_image - np.min(generated_image)) / (np.max(generated_image) - np.min(generated_image))
    return generated_image


def averaging_5_slice(curr_slice, folder, fileName, FLAGS):
    a = np.ones((5, 128 * 10, 128 * 10), dtype=np.float)
    a[0, :, :] = imread('./' + FLAGS.savedir + '/' + str(curr_slice - 2) + '/' + folder + '/test_2_' + fileName)
    a[1, :, :] = imread('./' + FLAGS.savedir + '/' + str(curr_slice - 1) + '/' + folder + '/test_1_' + fileName)
    a[2, :, :] = imread('./' + FLAGS.savedir + '/' + str(curr_slice) + '/' + folder + '/test_0_' + fileName)
    a[3, :, :] = imread('./' + FLAGS.savedir + '/' + str(curr_slice + 1) + '/' + folder + '/test_-1_' + fileName)
    a[4, :, :] = imread('./' + FLAGS.savedir + '/' + str(curr_slice + 2) + '/' + folder + '/test_-2_' + fileName)
    return a[0, :, :] * gaussian(2) + a[1, :, :] * gaussian(1) + a[2, :, :] * gaussian(0) + a[3, :, :] * gaussian(1) + a[4, :,
                                                                                                                       :] * gaussian(2)


def gaussian(x):
    return np.exp(-np.power(x, 2.) / (2 * np.power(sigma_for_weigh_in_5_slice_average, 2.)))


def assembly_Progression(fileName, folder, age_intervals, outputFolder, FLAGS):
    numb_Slice = 95
    print(fileName)
    final_MRI = np.ones((numb_Slice, 128, 128, 10), dtype=np.int16)
    for j in range(10):
        curr_slice = 44
        print(str(j + 1) + '/10')
        for i in range(0, numb_Slice):
            progression_MRI = averaging_5_slice(curr_slice, folder, fileName, FLAGS)
            final_MRI[i, :, :, j] = np.int16(generate_MRI(progression_MRI, age_intervals[j], age_intervals) * 32767 * 2 - 32767)
            curr_slice = curr_slice + 1

    img = nib.Nifti1Image(final_MRI, transformationMatrix)
    nib.save(img, outputFolder + '/' + fileName + '.nii.gz')


def assembly_3D(file, folder, age):
    if not os.path.exists(folder + '_3D'):
        os.system('mkdir ' + folder + '_3D')
    curr_age = file.split('_')[0]
    img_in = nib.load('./SyntheticInputMRI/' + file)
    img_pre = nib.load('./'+folder +'/'+ os.path.basename(file))
    data_img_in = np.array(img_in.dataobj)
    data_img_pre = np.array(img_pre.dataobj)
    result = data_img_pre.reshape([1, np.size(data_img_pre, 0), np.size(data_img_pre, 1), np.size(data_img_pre, 2), 1])
    result = net.predict(path, result)
    matched = match_histograms(result.reshape([np.size(data_img_pre, 0), np.size(data_img_pre, 1), np.size(data_img_pre, 2)]),
                               data_img_in.reshape([np.size(data_img_pre, 0), np.size(data_img_pre, 1), np.size(data_img_pre, 2)]), multichannel=True)
    if np.double(curr_age) < age:
        matched = np.minimum(matched, data_img_in.reshape([np.size(data_img_pre, 0), np.size(data_img_pre, 1), np.size(data_img_pre, 2)]))
    else:
        matched = np.maximum(matched, data_img_in.reshape([np.size(data_img_pre, 0), np.size(data_img_pre, 1), np.size(data_img_pre, 2)]))
    array_img = nib.Nifti1Image(matched.reshape([np.size(data_img_in, 0), np.size(data_img_in, 1), np.size(data_img_in, 2)]), transformationMatrix)
    nib.save(array_img, folder + '_3D/' + os.path.basename(file))


def assembly_MRI(fileName, folder, age_to_generate, age_intervals, outputFolder, type_of_assembly, FLAGS):
    curr_slice = 44
    numb_Slice = 95
    print(fileName)
    final_MRI = np.ones((numb_Slice, 128, 128), dtype=np.int16)
    if FLAGS.create_MRI:
        for i in range(0, numb_Slice):
            if type_of_assembly == 0:
                followUpFile = glob.glob('./data/' + FLAGS.datasetGT + '/' + str(curr_slice) + '/' + str(age_to_generate) + '*.png')
                final_MRI[i, :, :] = imread('./data/' + FLAGS.datasetGT + '/' + str(curr_slice) + '/' + os.path.basename(followUpFile[0]))
            elif type_of_assembly == 1:
                progression_MRI = imread('./' + FLAGS.savedir + '/' + str(curr_slice) + '/' + folder + '/test_0_' + fileName)
                final_MRI[i, :, :] = np.int16(generate_MRI(progression_MRI, age_to_generate, age_intervals) * 32767 * 2 - 32767)
            elif type_of_assembly == 2:
                progression_MRI = averaging_5_slice(curr_slice, folder, fileName, FLAGS)
                final_MRI[i, :, :] = np.int16(generate_MRI(progression_MRI, age_to_generate, age_intervals) * 32767 * 2 - 32767)
            elif type_of_assembly == 3:
                followUpFile = glob.glob('./data/' + FLAGS.dataset + '/' + str(curr_slice) + '/' + str(age_to_generate) + '*.png')
                final_MRI[i, :, :] = imread('./data/' + FLAGS.dataset + '/' + str(curr_slice) + '/' + os.path.basename(followUpFile[0]))
            curr_slice = curr_slice + 1

        img = nib.Nifti1Image(final_MRI, transformationMatrix)
        nib.save(img, outputFolder + '/' + fileName + '.nii.gz')
    if FLAGS.super_resolution_3D:
        assembly_3D(fileName+ '.nii.gz', outputFolder, age_to_generate)


def assemblyAll(test_label, age_intervals, outputFolder, type_of_assembly, FLAGS):
    for i, j in quaries_for_progression:
        assembly_MRI(i, test_label, float(j), age_intervals, outputFolder, type_of_assembly, FLAGS)


def assemblyTraining(test_label, age_intervals, outputFolder, type_of_assembly, FLAGS):
    allTrainingFile = glob.glob('./data/' + FLAGS.dataset + '/' + str(42) + '/*.png')
    for i in allTrainingFile:
        currFile = os.path.basename(i)
        assembly_MRI(currFile, test_label, currFile.split('_')[0], age_intervals, outputFolder, type_of_assembly, FLAGS)


def assemblyAll_progression(test_label, age_intervals, outputFolder, FLAGS):
    for i, j in quaries_for_progression:
        assembly_Progression(i, test_label, age_intervals, outputFolder, FLAGS)
