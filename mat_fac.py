"""
@Time ： 2020/10/23 8:02
@Auth ： Yifan Hu
@File ：mat_fac.py
@IDE ：PyCharm
"""
from surprise import SVDpp, SVD
from surprise.model_selection import train_test_split
from data_loader import SurpriseDataLoader

# matrix_factorization.SVD	The famous SVD algorithm, as popularized by Simon Funk during the Netflix Prize.
# matrix_factorization.SVDpp	The SVD++ algorithm, an extension of SVD taking into account implicit ratings.
Model = SVD

data = SurpriseDataLoader.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.3)

algo = Model()
algo.fit(trainset)

uid = None
iid = None
algo.predict(uid, iid)
