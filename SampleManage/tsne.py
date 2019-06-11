import numpy as np
from sklearn.manifold import TSNE
from .modelsql import modelsql
from .models import PriSample


def mul_visulize():
    df = modelsql(PriSample).select_data_nodelists(['alimds.ihep.ac.cn','cemds.ihep.ac.cn'])
    metrics = ['bytes_in_value','bytes_out_value']
    ds = df[metrics]
    print(ds)

    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_embedded = TSNE(n_components=2,verbose=1).fit_transform(X)
    print(X_embedded)



