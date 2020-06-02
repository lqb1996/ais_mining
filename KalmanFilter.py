import numpy as np
from DataLoader import *
import configparser


class KalmanFilter(object):
    def __init__(self, data, Q=1e-5, R=0.001**2):
        self.n_iter = len(data)
        self.data = data
        self.Q = Q
        self.R = R
        sz = (self.n_iter,)  # size of array
        # x = -3*np.sin(np.arange(n_iter)) # truth value (typo in example at top of p. 13 calls this z)
        self.z = np.array(self.data)    # observations (normal about x, sigma=0.1)
        # Q = 1e-5 # process variance
        # allocate space for arrays
        self.xhat=np.zeros(sz)      # a posteri estimate of x
        P=np.zeros(sz)         # a posteri error estimate
        xhatminus=np.zeros(sz) # a priori estimate of x
        Pminus=np.zeros(sz)    # a priori error estimate
        K=np.zeros(sz)         # gain or blending factor

        # R = 0.1**2 # estimate of measurement variance, change to see effect

        # intial guesses
        self.xhat[0] = 0.0
        P[0] = 1.0

        '''输入为z,输出为xhat'''
        for k in range(1, self.n_iter):
            # time update
            xhatminus[k] = self.xhat[k-1]
            Pminus[k] = P[k-1]+Q
            # measurement update
            K[k] = Pminus[k]/(Pminus[k]+R)
            self.xhat[k] = xhatminus[k]+K[k]*(self.z[k]-xhatminus[k])
            P[k] = (1-K[k])*Pminus[k]

    def get_res(self):
        return self.xhat


if __name__ == '__main__':
    proDir = os.path.split(os.path.realpath(__file__))[0]
    configPath = os.path.join(proDir, "setting.cfg")
    cf = configparser.ConfigParser()
    cf.read(configPath)
    csv_path = os.path.join(proDir, cf.get("path", "csv_path"))
    csv_loader = CSVDataSet(csv_path)
    print(csv_loader.np_total_x[0][:, 1:2].reshape(1, -1)[0])
    print(KalmanFilter(csv_loader.np_total_x[0][:, 1:2].reshape(1, -1)[0]).get_res())
