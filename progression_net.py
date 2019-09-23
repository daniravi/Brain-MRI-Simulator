import h5py
import numpy as np
import pickle
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing as preprocessing
#from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression


class progression_net(object):
    def __init__(self, current_slice, max_regional_expansion, map_disease):
        self.num_output = 1
        self.num_inputs = 4
        self.current_slice = current_slice
        self.in_data = []
        self.out_data = []
        self.load_matlab_data(current_slice)
        self.max_regional_expansion = max_regional_expansion
        self.map_disease = map_disease

    def test(self, i):
        filehandler = open('Regressor/' + str(self.current_slice), 'rb')
        [allRegressor, allScaler] = pickle.load(filehandler)
        filehandler.close()

        indexForCurrentRegion = (self.in_data[:, 3] == i + 1)
        X = np.float64(self.in_data[indexForCurrentRegion, :3])
        for j in range(np.size(X, 0)):
            X[j, 2] = self.map_disease[np.int(X[j, 2]) - 1] + 1
        Y = self.out_data[indexForCurrentRegion, :3]
        result = allRegressor[i].predict_proba(X)
        prediction = np.reshape(result[:, 0], [-1, 1])
        print(np.concatenate((X[:50, :], Y[:50], allScaler[i].inverse_transform(prediction[:50])), axis=1))
        return

    def train_and_save(self, i):
        filehandler = open('Regressor/' + str(self.current_slice), 'rb')
        [allRegressor, allScaler] = pickle.load(filehandler)
        filehandler.close()

        indexForCurrentRegion = (self.in_data[:, 3] == i + 1)
        X = np.float64(self.in_data[indexForCurrentRegion, :3])
        for j in range(np.size(X, 0)):
            X[j, 2] = self.map_disease[np.int(X[j, 2]) - 1] + 1

        y = self.out_data[indexForCurrentRegion]
        corectIndex = [y < self.max_regional_expansion]

        y = y[np.reshape(corectIndex, (-1))]
        X = X[np.reshape(corectIndex, (-1)), :]
        # clf = SVR(C=100, coef0=0.0, degree=2, epsilon=0.005, gamma='auto',
        #          kernel='linear', max_iter=-1, shrinking=True, tol=0.000001, verbose=True)

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

        allRegressor.append(pipeline)
        allScaler.append(y_n)
        filehandler = open('Regressor/' + str(self.current_slice), 'wb')
        pickle.dump([allRegressor, allScaler], filehandler)
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
