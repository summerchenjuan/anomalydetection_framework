import numpy as np

#空值数据（None）删除，不参与评估

class evaluation:
    def __init__(self,X,Y):
        self.X = np.array(X)
        self.Y = np.array(Y)

    def deleteNone(self):
        if(None in self.X):
                 index = np.where(self.X == None)
                 self.X = np.delete(self.X,index)
                 self.Y = np.delete(self.Y,index)
                 if(None in self.Y):
                     index = np.where(self.Y == None)
                     self.X = np.delete(self.X, index)
                     self.Y = np.delete(self.Y, index)
        elif (None in self.Y):
                index = np.where(self.Y == None)
                X = np.delete(self.X, index)
                Y = np.delete(self.Y, index)
        return self.X,self.Y

    #空值数据Nan删除，不参与评估
    def deleteNan(self):
        if (np.sum(self.X!=self.X) > 0 ):
                 index = np.argwhere(np.isnan(self.X))
                 self.X = np.delete(self.X,index)
                 self.Y = np.delete(self.Y,index)
                 if (np.sum(self.Y!=self.Y) > 0 ):
                     index = np.argwhere(np.isnan(self.Y))
                     self.X = np.delete(self.X, index)
                     self.Y = np.delete(self.Y, index)
        elif (np.sum(self.Y!=self.Y) > 0 ):
                index = np.argwhere(np.isnan(self.Y))
                self.X = np.delete(self.X, index)
                self.Y = np.delete(self.Y, index)
        return self.X, self.Y

    #平均误差
    def ME(self):
        s = (self.X - self.Y).mean()
        return s

    #平均绝对值误差
    def MAE(self):
        s = np.fabs(self.X-self.Y).mean()
        return s

    #均方根误差
    def RMSE(self):
        s = np.sqrt(np.square(self.X-self.Y).mean())
        return s

    #平均百分误差
    def MPE(self):
        s = ((self.X-self.Y)/self.Y).mean()
        return s

    #平均绝对值百分误差
    def MAPE(self):
        s = np.fabs((self.X-self.Y)/self.Y).mean()
        return s

if __name__ == '__main__':
    X = [np.nan,np.nan,200,100,11,1,1]
    Y = [200,200,200,1,11,1,np.nan]
    a = evaluation(X,Y)
    a.deleteNan()
    print(X,Y)
    print(a.MAPE())