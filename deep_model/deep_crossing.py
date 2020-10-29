"""
@Time ： 2020/10/29 9:27
@Auth ： Yifan Hu
@File ：deep_crossing.py
@IDE ：PyCharm
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 评价指标rmse
def rmse(pred_rate, real_rate):
    loss_func = nn.MSELoss()
    mse_loss = loss_func(pred_rate, real_rate.float())
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss


# from data_loader import SklearnDataLoader
# from sklearn.preprocessing import MinMaxScaler
# df_x,df_y = SklearnDataLoader.load_builtin('ml-100k')
# # cols :
# # uid iid age movie_type age occupation gender
# df = df_x.copy()
# df['rating'] = df_y
# #%%
# movie_type_series = pd.Series(index = df.index,dtype='int')
# for i,movie_type in enumerate(['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
#        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
#        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
#        'Western']):
#     movie_type_series += df_x[movie_type] * i
# #%%
# movie_type_series
#
# #%%
# df = df_x[['uid','iid','age','gender','occupation']]
# df['movie_type'] = movie_type_series
# df['rating'] = df_y['rating']
# #%%
# from sklearn.preprocessing import LabelEncoder
# label_enc_dic = dict()
# df_new = df.copy()
# for cate_col in ['uid','iid','gender','occupation','movie_type']:
#     label_enc =  LabelEncoder()
#     label_enc.fit(df[cate_col])
#     label_enc_dic[cate_col] = label_enc
#     df_new[cate_col] = label_enc.transform(df[cate_col])
# age_scaler = MinMaxScaler()
# df_new['age'] = age_scaler.fit_transform(df_new[['age']])
# df_new.to_csv('./data/df_deep_crossing_100k.csv',index = False)

class ResUnit(torch.nn.Module):
    def __init__(self, n, m=None):
        super(ResUnit, self).__init__()
        if not m:
            m = n
        self.layer1 = nn.Linear(n, m, bias=True)
        self.layer2 = nn.Linear(m, n, bias=True)

    def forward(self, x):
        output = torch.relu(self.layer1(x))
        output = torch.relu(self.layer2(output) + x)
        return output


class DeepCrossing(torch.nn.Module):
    def __init__(self, cate_num_dic, continuous_list, k, n_res=2, m_res=None):
        super(DeepCrossing, self).__init__()
        self.embedd_dict = dict()
        self.continuous_list = continuous_list
        # self.n_continuous = len(continuous_list)
        self.k = k
        for cate_index in cate_num_dic:
            self.embedd_dict[cate_index] = nn.Embedding(cate_num_dic[cate_index], k)
        n = k * len(cate_num_dic) + len(continuous_list)
        self.res_units = nn.Sequential(*[ResUnit(n, m_res) for _ in range(n_res)])
        self.scoring = nn.Linear(n, 1, bias=True)

    def forward(self, x):
        input_embedd = torch.cat(tuple(self.embedd_dict[cate_index](x[:, cate_index].long())
                                       for cate_index in self.embedd_dict), 1)
        input = torch.cat((input_embedd,
                           x[:, self.continuous_list]), 1)
        input = input.float()
        output_res = self.res_units(input)
        output_score = self.scoring(output_res)
        # 5 for movielens-100k
        return 5 * torch.sigmoid(output_score)


if __name__ == '__main__':

    BATCH_SIZE = 500
    n_epochs = 10
    learing_rating = 0.001
    k = 10
    n_res = 2
    l2_penalty = 0.2
    mom = 0.8
    loss_func = rmse

    df = pd.read_csv('../data/df_deep_crossing_100k.csv')
    df_x, df_y = df.drop('rating', axis=1), df[['rating']]

    cate_list = [0, 1, 3, 4, 5]
    cate_num_dic = dict()
    continuous_list = [2]
    for cate_index in cate_list:
        cate_num_dic[cate_index] = df_x.iloc[:, cate_index].nunique()

    model = DeepCrossing(cate_num_dic, continuous_list, k=k, n_res=n_res, m_res=None)

    x_train, x_test, y_train, y_test = train_test_split(df_x.values, df_y.values)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    # 训练集分批处理
    loader = DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # 最新批数据
        shuffle=False  # 是否随机打乱数据
    )
    # 测试集分批处理
    loader_test = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # optimizer = torch.optim.SGD(model.parameters(), lr=learing_rating, momentum=mom)
    optimizer = torch.optim.Adam(model.parameters(), lr=learing_rating)
    loss__train_set = []
    loss_test_set = []
    for epoch in range(n_epochs):  # 对数据集进行训练
        # 训练集
        for step, (batch_x, batch_y) in enumerate(loader):  # 每个训练步骤
            output = model(batch_x)
            loss_result = loss_func(output, batch_y)
            l2_regularization = torch.tensor(0).float()
            # 加入l2正则
            for param in model.parameters():
                l2_regularization += torch.norm(param, 2)
            loss = loss_result + l2_penalty * l2_regularization
            # loss = loss_result

            optimizer.zero_grad()  # 如果不置零，Variable的梯度在每次backwrd的时候都会累加
            loss.backward()
            optimizer.step()  # 进行更新
            # 将每一次训练的数据进行存储，然后用于绘制曲线
            if step % 20 == 0:
                print('Epoch:{} STEP:{},train_LOSS:{:.4f}'.format(epoch, step, loss))
            # print('Epoch:{} STEP:{},train_LOSS:{:.4f}'.format(epoch, step, loss))
        loss__train_set.append(loss)

        # 测试集
        for step, (batch_x, batch_y) in enumerate(loader_test):  # 每个训练步骤
            output = model(batch_x)
            # 平方差
            loss_result = loss_func(output, batch_y)
            l2_regularization = torch.tensor(0).float()
            # 加入l2正则
            for param in model.parameters():
                l2_regularization += torch.norm(param, 2)
            loss = loss_result + l2_penalty * l2_regularization
            # loss = loss_result
            # loss.backward()
            # optimizer.step()  # 进行更新
        # 将每一次训练的数据进行存储，然后用于绘制曲线
        print('Epoch:{} ; test_LOSS:{:.4f}'.format(epoch, loss))
        loss_test_set.append(loss)

    plt.clf()
    plt.plot(range(epoch + 1), loss__train_set, label='Training data')
    plt.plot(range(epoch + 1), loss_test_set, label='Test data')
    plt.title('The MovieLens Dataset Learing Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.show()
    print("train_loss", loss)
