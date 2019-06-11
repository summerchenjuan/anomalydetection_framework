class PredictModel:
    def __init__(self):
        self.type = 'Predict'
        self.name = None

    def getModelName(self):
        return self.name

    def getModelType(self):
        return self.type



if __name__ == '__main__':
    p = PredictModel()
    p.x = 2
    print(p.x)