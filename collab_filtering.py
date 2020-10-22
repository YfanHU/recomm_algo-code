"""
@Time ： 2020/10/22 16:40
@Auth ： Yifan Hu
@File ：collab_filtering.py
@IDE ：PyCharm
"""

from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from data_loader import SurpriseDataLoader



# collaborative filtering based on users or items
USER_BASED = True

# available models
# KNNBasic	A basic collaborative filtering algorithm.
# KNNWithMeans	A basic collaborative filtering algorithm, taking into account the mean ratings of each user.
# KNNWithZScore	A basic collaborative filtering algorithm, taking into account the z-score normalization of each user.
# KNNBaseline	A basic collaborative filtering algorithm taking into account a baseline rating.
model = KNNBaseline

data = SurpriseDataLoader.load_builtin('ml-100k')
trainset = data.build_full_trainset()

sim_options = {'name': 'pearson_baseline', 'user_based': USER_BASED}
# sim_options = {'name': 'pearson_baseline', 'user_based': False}

algo = model(sim_options=sim_options)
algo.fit(trainset)

# predict
raw_uid = None
raw_iid = None
inner_uid = algo.trainset.to_inner_uid(raw_uid)
inner_iid = algo.trainset.to_inner_iid(raw_iid)

algo.predict(raw_uid, raw_iid).est  # --> float
if USER_BASED:
    algo.get_neighbors(inner_uid, k=10)  # --> list of raw iids
else:
    algo.get_neighbors(inner_iid, k=10)  # --> list of raw uids
