"""
@Time ： 2020/10/28 17:10
@Auth ： Yifan Hu
@File ：auto_rec.py
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


# # construct matrix users * items from movielens-100k
# from data_loader import SklearnDataLoader
# from collections import defaultdict
# df = SklearnDataLoader.load_builtin('ml-100k',rating_only=True)
# uid_iid_rating_dic = defaultdict(dict)
# for _,item in df.iterrows():
#     uid = item.uid
#     iid = item.iid
#     rating = item.rating
#     uid_iid_rating_dic[uid][iid] = rating
# df_rating_dict_list = [{'uid':item[0],**item[1]} for item in uid_iid_rating_dic.items()]
# df_rating = pd.concat([pd.DataFrame(item,index=[i]) for i,item in enumerate(df_rating_dict_list)])
# df_rating = df_rating.set_index('uid')
# df_rating.sort_index(inplace=True)
# df_rating = df_rating[sorted(df_rating.columns.values)]

class AutoRecModel(torch.nn.Module):
    def __init__(self, n, m):
        super(AutoRecModel, self).__init__()
        self.n = n
        self.m = m
        self.layer_enc = nn.Linear(n, m, bias=True)
        self.layer_dec = nn.Linear(m, n, bias=True)

    def forward(self, x):
        x = x.float()
        x = torch.tanh(self.layer_enc(x))
        return torch.sigmoid(self.layer_dec(x))


if __name__ == '__main__':
    USER_BASED = False

    BATCH_SIZE = 100
    n_epochs = 100
    learing_rating = 0.05
    l2_penalty = 0.1
    mom = 0.8
    loss_func = rmse

    df_rating = pd.read_csv('../data/df_rating_100k.csv', index_col=0)
    df_rating = df_rating.fillna(0)

    if not USER_BASED:
        df_rating = df_rating.T

    n = df_rating.shape[1]  # 特征数目
    m = n // 10
    model = AutoRecModel(n, m)

    x_train, x_test, y_train, y_test = train_test_split(df_rating.values, df_rating.values)

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

    optimizer = torch.optim.SGD(model.parameters(), lr=learing_rating, momentum=mom)  # 学习率为
    # loss_func = nn.MSELoss()
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
            if step % 5 == 0:
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
