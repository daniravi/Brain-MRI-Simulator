from matplotlib.widgets import Slider, Button
import numpy as np
from scipy import ndimage as ndimage
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import skfuzzy

class GUI(object):
    def __init__(self):
        file_path=self.open_new_mri()
        self.error_map = np.zeros([128, 128, 3])
        self.generated_images = plt.imread(file_path)
        self.age_intervals = [63, 66, 68, 70, 72, 74, 76, 78, 80, 83, 87]
        self.bin_centers = np.convolve(self.age_intervals, [0.5, 0.5], 'valid')
        self.researchGroup = ['Cognitively normal', 'Subjective memory concern', 'Early cognitive impairment', 'Mild cognitive impairment',
                              'Late cognitive impairment',
                              'Alzheimer''s disease']

        self.current_age_to_show = self.age_intervals[0]

        # SHOW
        self.fig = plt.figure(figsize=(10, 4))
        ax1 = self.fig.add_subplot(131)
        ax2 = self.fig.add_subplot(132)
        ax3 = self.fig.add_subplot(133)
        borderSize = 0.1
        sizeSlider = 0.03
        self.fig.subplots_adjust(left=borderSize, bottom=borderSize)

        self.real_age = float(str(file_path).split('/')[-1].split('_')[1])
        self.curr_diagnosis = int(str(file_path).split('/')[-1].split('_')[3]) - 1
        self.bin_of_input = np.min([np.max([np.digitize(self.real_age, self.bin_centers) - 1, 0]), 9])

        imageIn = plt.imread(file_path)
        self.currentImageIn = imageIn[:128, (self.bin_of_input * 128):(self.bin_of_input + 1) * 128]
        self.currentImageIn = np.rot90(self.currentImageIn, 1)

        self.labelBaselineAge = plt.text(-235, -35, "Baseline Age: " + str(round(self.real_age, 2)))
        self.label_curr_diagnosis = plt.text(-235, -25, "Diagnosis: " + self.researchGroup[self.curr_diagnosis])

        plt.text(-250, -5, "x", fontsize=14)
        plt.text(-100, -5, "g(i)", fontsize=14)
        plt.text(50, -5, "g(i)-x", fontsize=14)

        ax_position = plt.axes(rect=[0.1, 0.9, 0.1, 0.05])
        self.button = Button(ax_position, 'Load', color='lightgoldenrodyellow', hovercolor='0.975')

        ax_position = plt.axes(rect=[0.45, 0.12, 0.1, 0.05])
        self.buttonAnimate = Button(ax_position, 'Simulation', color='lightgoldenrodyellow', hovercolor='0.975')
        self.ageIncrement = 0.5
        self.button.on_clicked(self.reset)
        self.buttonAnimate.on_clicked(self.animate)
        self.imGUI_Diff = ax3.imshow(self.error_map)
        self.imGUI_Out = ax2.imshow(self.generated_images, cmap='gray')
        self.imGUI_IN = ax1.imshow(self.currentImageIn, cmap='gray')

        self.imGUI_Out.axes.get_xaxis().set_visible(False)
        self.imGUI_Out.axes.get_yaxis().set_visible(False)
        self.imGUI_Diff.axes.get_xaxis().set_visible(False)
        self.imGUI_Diff.axes.get_yaxis().set_visible(False)
        self.imGUI_IN.axes.get_xaxis().set_visible(False)
        self.imGUI_IN.axes.get_yaxis().set_visible(False)

        axAge = self.fig.add_axes([0.2, 0.06, 0.67, sizeSlider], facecolor='lightgoldenrodyellow')
        self.label_current_age_to_show = Slider(axAge, 'Progressed Age:', self.age_intervals[0], self.bin_centers[-1], valinit=0)
        self.label_current_age_to_show.valtext.set_text(str(round(self.current_age_to_show, 2)))
        self.label_current_age_to_show.on_changed(self.updateAge)
        self.update()
        plt.show()



    def updateAge(self, val):
        self.current_age_to_show = val
        self.update()
        return val

    def reset(self,_):
        file_path=self.open_new_mri()
        self.real_age = float(str(file_path).split('/')[-1].split('_')[1])
        self.curr_diagnosis = int(str(file_path).split('/')[-1].split('_')[3]) - 1
        self.label_curr_diagnosis.set_text("Diagnosis:" + self.researchGroup[self.curr_diagnosis])
        self.labelBaselineAge.set_text("Baseline Age:" + np.str(self.real_age))

        self.generated_images = plt.imread(file_path)
        imageIn = plt.imread(file_path)
        self.currentImageIn = imageIn[:128, (self.bin_of_input * 128):(self.bin_of_input + 1) * 128]
        self.currentImageIn = np.rot90(self.currentImageIn, 1)
        self.update()

    @staticmethod
    def open_new_mri():
        root = tk.Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename()
        root.destroy()
        return file_path

    def update(self):
        batch_fuzzy_membership = np.zeros(10)
        generated_image = np.zeros([128, 128])
        self.label_current_age_to_show.valtext.set_text(str(round(self.current_age_to_show, 2)))
        for t in range(10):
            batch_fuzzy_membership[t] = skfuzzy.membership.gaussmf(self.current_age_to_show, self.bin_centers[t], 1.5)
            generated_image = generated_image + self.generated_images[:128, (t * 128):((t + 1) * 128)] * batch_fuzzy_membership[t]

        all_image_min = np.ones([128, 128])
        all_image_max = np.zeros([128, 128])
        currentBin = np.min([np.max([np.digitize(self.real_age, self.bin_centers) - 1, 0]), 9])

        for i in range(self.bin_of_input + 1, currentBin):
            all_image_min = np.minimum(all_image_min, self.generated_images[:128, (i * 128):((i + 1) * 128)])

        for i in range(currentBin, self.bin_of_input):
            all_image_max = np.maximum(all_image_max, self.generated_images[:128, (i * 128):((i + 1) * 128)])

        generated_image = (generated_image - np.min(generated_image)) / (np.max(generated_image) - np.min(generated_image))
        generated_image = np.minimum(generated_image, all_image_min)
        generated_image = np.maximum(generated_image, all_image_max)
        generated_image = np.rot90(generated_image, 1)

        self.imGUI_Out.set_data(generated_image)

        diff = abs(self.currentImageIn - generated_image)

        if self.current_age_to_show < self.real_age:
            self.error_map[:, :, 1] = np.minimum(ndimage.filters.gaussian_filter(diff * (diff > 0.05), 4) * 4, 1)
            self.error_map[:, :, 0] = np.zeros([128, 128])
        elif self.current_age_to_show > self.real_age:
            self.error_map[:, :, 1] = np.zeros([128, 128])
            self.error_map[:, :, 0] = np.minimum(ndimage.filters.gaussian_filter(diff * (diff > 0.05), 4) * 4, 1)
        else:
            self.error_map[:, :, 1] = np.zeros([128, 128])
            self.error_map[:, :, 0] = np.zeros([128, 128])

        self.imGUI_Diff.set_data(self.error_map)
        self.imGUI_IN.set_data(self.currentImageIn)
        self.fig.canvas.draw()



    def animate(self,_):
        while self.label_current_age_to_show.val < self.bin_centers[-1]:
            self.label_current_age_to_show.set_val(self.label_current_age_to_show.val + self.ageIncrement)
            self.update()
        for i in range(20):
            self.update()
        while self.label_current_age_to_show.val > self.age_intervals[0]:
            self.label_current_age_to_show.set_val(self.label_current_age_to_show.val - self.ageIncrement)
            self.update()
