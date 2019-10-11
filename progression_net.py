import h5py
import numpy as np
import pickle
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing as preprocessing
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression


class progression_net(object):
    def __init__(self, current_slice, max_regional_expansion, map_disease, regressor_type):
        self.num_output = 1
        self.num_inputs = 4
        self.current_slice = current_slice
        self.in_data = []
        self.out_data = []
        self.load_matlab_data(current_slice)
        self.max_regional_expansion = max_regional_expansion
        self.map_disease = map_disease
        self.regressor_type = regressor_type

    def load_regressor(self):
        if self.regressor_type==0:
            filehandler = open('Regressor_0/' + str(self.current_slice), 'rb')
        else:
            filehandler = open('Regressor_1/' + str(self.current_slice), 'rb')
        [all_regressor, all_scaler] = pickle.load(filehandler)
        filehandler.close()
        return all_regressor, all_scaler

    def load_region_data(self, current_region):
        indexForCurrentRegion = (self.in_data[:, 3] == current_region + 1)
        X = np.float64(self.in_data[indexForCurrentRegion, :3])
        for j in range(np.size(X, 0)):
            X[j, 2] = self.map_disease[np.int(X[j, 2]) - 1] + 1
        Y = self.out_data[indexForCurrentRegion]
        return X, Y

    def test(self, current_region):
        [all_regressor, all_scaler] = self.load_regressor()
        X, Y = self.load_region_data(current_region)
        result = all_regressor[current_region].predict_proba(X)
        prediction = np.reshape(result[:, 0], [-1, 1])
        print(np.concatenate((X[:50, :], Y[:50], all_scaler[current_region].inverse_transform(prediction[:50])), axis=1))
        return

    def train_and_save(self, current_region):
        [all_regressor, all_scaler] = self.load_regressor()
        X, y = self.load_region_data(current_region)
        correct_index = [y < self.max_regional_expansion]

        y = y[np.reshape(correct_index, (-1))]
        X = X[np.reshape(correct_index, (-1)), :]
        if self.regressor_type == 0:
            clf = SVR(C=100, coef0=0.0, degree=1, epsilon=0.005, gamma='auto',
                      kernel='linear', max_iter=-1, shrinking=True, tol=0.000001, verbose=True)
        else:
            clf = LogisticRegression(max_iter=100000000, tol=0.000001,
                                     C=1.0, class_weight='balanced', fit_intercept=True,
                                     intercept_scaling=1, multi_class='ovr',
                                     penalty='l2', random_state=None, solver='lbfgs',
                                     verbose=1, warm_start=False
                                     )

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        pipeline = make_pipeline(
            preprocessing.StandardScaler(),
            clf,
        )
        if np.size(y) == 0:
            return 0
        y_n = preprocessing.StandardScaler().fit(y)
        y_fina = y_n.transform(y) >= np.mean(y_n.transform(y))

        pipeline.fit(X, y_fina.ravel())

        all_regressor.append(pipeline)
        all_scaler.append(y_n)
        if self.regressor_type==0:
            filehandler = open('Regressor_0/' + str(self.current_slice), 'wb')
        else:
            filehandler = open('Regressor_1/' + str(self.current_slice), 'wb')
        pickle.dump([all_regressor, all_scaler], filehandler)
        filehandler.close()
        return 1

    def load_matlab_data(self, current_slice):
        mat = h5py.File('data_TF/' + str(current_slice) + '.mat', 'r')
        in_data = mat['in']  # array
        out_Data = mat['out']  # structure containing an array
        in_data = in_data[:self.num_inputs, :]
        out_Data = out_Data[:self.num_output, :]
        self.in_data = in_data.T
        self.out_data = out_Data.T
        return
