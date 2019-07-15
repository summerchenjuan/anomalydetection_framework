import numpy as np
import math

class Qfunction:
    modelname = 'QfunctionModel'

    def __init__(self,scorethreshold,series,windows):
        self.scorethreshold = scorethreshold
        self.series = series
        self.windows = windows

    def getmodelname(self,modelname):
        return modelname

    def setdistribution(self,wseries):
        """
        假设时间窗口的数据满足正态分布，计算均值和标准差
        :param wseries:
        :return:
        """
        wseries = np.delete(wseries, np.where(np.isnan(wseries))[0])
        mean = np.mean(wseries)
        std = np.std(wseries)
        distributionParams = {}
        distributionParams['mean'] = mean
        distributionParams['std'] = std
        return distributionParams

    def tailProbability(self,x, distributionParams):
        """
        计算x的右尾函数，即计算P(X>x|x>=mean)的概率或P(X<x|x<mean)
        """
        if x < distributionParams["mean"]:
            # Gaussian is symmetrical around mean, so flip to get the tail probability
            xp = 2 * distributionParams["mean"] - x
            return self.tailProbability(xp,distributionParams)
        if x == np.nan:
            return 0.5
        z = (x - distributionParams["mean"]) / distributionParams["std"]
        return 0.5 * math.erfc(z / 1.4142)

    def anomalyscore(self,ptail):
         """
         计算异常分数值
         :param ptail: 右尾函数得到的概率
         :return: 异常分数值，越接近1，异常的可能性越大
         """
         return 1-ptail

    def anomalylocation(self):
         wseries = []
         anomalylist = []
         for i in range(len(self.series)):
             if(i < self.windows):
                wseries.append(self.series[i])
                #小于windows暂时不计算异常分数#
             elif(i >= self.windows):
                 distributionParams = self.setdistribution(wseries)
                 ptail = self.tailProbability(self.series[i],distributionParams)
                 anomalyscore = self.anomalyscore(ptail)
                 if(anomalyscore >= self.scorethreshold):
                     anomalylist.append(i)
                     print(i,anomalyscore,len(wseries))
                 wseries.pop(0)
                 wseries.append(self.series[i])
         return anomalylist








