import numpy as np

#平均误差
def ME(X,Y):
    s1 = np.array(X)
    s2 = np.array(Y)
    s = (s1 - s2).mean()
    return s

#平均绝对值误差
def MAE(X,Y):
    s1 = np.array(X)
    s2 = np.array(Y)
    s = np.fabs(s1-s2).mean()
    return s

#均方根误差
def RMSE(X,Y):
    s1 = np.array(X)
    s2 = np.array(Y)
    s = np.sqrt(np.square(s1-s2).mean())
    return s

#平均百分误差
def MPE(X,Y):
    s1 = np.array(X)
    s2 = np.array(Y)
    s = ((s1-s2)/s2).mean()
    return s

#平均绝对值百分误差
def MAPE(X,Y):
    s1 = np.array(X)
    s2 = np.array(Y)
    s = np.fabs((s1-s2)/s2).mean()
    return s

if __name__ == '__main__':
    X = [100,200,200]
    Y = [200,200,200]
    print(MPE(X,Y))