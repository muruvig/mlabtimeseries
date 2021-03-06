import numpy as np
import matplotlib.pyplot as plt

class Generator:
    def __init__(self, **kwargs):
        if "type" in kwargs:
            self._type = kwargs["type"]
        else:
            self._type = "autoregressive"
        if "seed" in kwargs:
            self.seed = kwargs["seed"]
            
    def __repr__(self):
        return "%s TimeSeries Generator"%self._type.title()

    def type(self):
        return self._type

    #generate a timeseries of my type
    #argument in kwargs = start, end, period
    def generate(self, n, **kwargs):
        if "start" in kwargs:
            start = kwargs["start"]
        else:
            start = 1
        if "end" in kwargs:
            end = kwargs["end"]
        else:
            end = n
        time = np.linspace(start, end, n)
        if self.type() == "autoregressive":
            data = self.generate_armodel(n, kwargs)
        if self.type() == "movingaverage":
            data = self.generate_mamodel(n, kwargs)
        return TimeSeries(time, data)

    def generate_armodel(self, n, kwargs):
        if "p" in kwargs:
            p = kwargs["p"]
        else:
            p = 1
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
        else:
            alpha = [0.8]
        if "sigma" in kwargs:
            sigma = kwargs["sigma"]
        else:
            sigma = 1.3
        if "period" in kwargs:
            period = kwargs["period"]
        else:
            period = 10

        X = np.zeros((n, 1))
        interval = n/period
        rem = n%period
        for i in xrange(interval):
            rangedata = autoregressive(period, p, alpha, sigma)
            start, end = i*period, (i+1)*period
            X[start:end, :] = rangedata
        if rem > 0:
            rangedata = autoregressive(rem, p, alpha, sigma)
            start = interval*period
            X[start:,:] = rangedata 
        return X


    def generate_mamodel(self, n, kwargs):
        if "q" in kwargs:
            q = kwargs["q"]
        else:
            q = 1
        if "mu" in kwargs:
            mu = kwargs["mu"]
        else:
            mu = 0
        if "theta" in kwargs:
            theta = kwargs["theta"]
        else:
            theta = [0.8]
        if "period" in kwargs:
            period = kwargs["period"]
        else:
            period = 10

        X = np.zeros((n, 1))
        interval = n/period
        rem = n%period
        for i in xrange(interval):
            rangedata = moving_average(period, q, mu, theta)
            start, end = i*period, (i+1)*period
            X[start:end, :] = rangedata
        if rem > 0:
            rangedata = moving_average(rem, q, mu, theta)
            start = interval*period
            X[start:,:] = rangedata 
        return X

def autoregressive(period, p, alpha, sigma):
        X = np.zeros((period, 1))
        X[0] = np.random.randn()
        for k in xrange(1, period):
            X[k] =  np.random.randn()*sigma + sum([alpha[i-1]*X[k-i] for i in xrange(1, p+1) if k-i >= 0])
        return X

def moving_average(period, q, mu, theta):
    X = np.zeros((period, 1))
    X[0] = mu + np.random.randn()
    epsilon = []
    for k in xrange(1, period):
        epsilon.append(np.random.randn())
        d = len(epsilon) - 1 
        X[k] = mu + sum([theta[i]*epsilon[d-i] for i in xrange(0, q) if i < q and i < len(epsilon)])
        if (len(epsilon) > q):
            epsilon = epsilon[1:]
    return X


class TimeSeries:
    def __init__(self, time, data, **kwargs):
        self.time = time
        self.data = data

    def plot(self, **kwargs):
        if "start" in kwargs:
            start = kwargs["start"]
        else:
            start = 0
        if "end" in kwargs:
            end = kwargs["end"]
        else:
            end = len(self.data)
        plt.figure()
        plt.plot(self.time[start:end], self.data[start:end])

    def data(self):
        return self.data

    def time(self):
        return self.time

