"""
@Time ： 2020/10/26 17:51
@Auth ： Yifan Hu
@File ：ffm.py
@IDE ：PyCharm
"""
import torch
from torch import nn
from 数据处理.缺失值处理 import df_fillna
from 数据处理.类别变量处理 import df_get_dummies
from 数据处理.归一化处理 import df_get_and_apply_scaler
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_loader import SklearnDataLoader


def feature_engi(data):
    # do feature_engineering on data
    data = data.drop(['uid', 'iid', 'timestamp', 'zip code',
                      'movie title', 'release date', 'video release date', 'IMDb URL'], axis=1)
    dummy_cols = ['gender', 'occupation']
    data = df_fillna(data)
    df_dummy = df_get_dummies(data[dummy_cols])
    data = data.drop(dummy_cols, axis=1)
    data = pd.concat([data, df_dummy], axis=1)
    data[['age']], _ = df_get_and_apply_scaler(data[['age']])
    return data


target_scaler = MinMaxScaler()


def scale_and_transform(target, inverse=False):
    global target_scaler
    if not inverse:
        # scale target to [0,1]
        return target_scaler.fit_transform(target)
    else:
        # inverse target to original range
        return target_scaler.inverse_transform(target)


# 评价指标rmse
def rmse(pred_rate, real_rate):
    loss_func = nn.MSELoss()
    mse_loss = loss_func(pred_rate, real_rate.float())
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss


class FFM_model(nn.Module):
    '''
    realization of ffm model
    '''

    def __init__(self, n, k, f, feature2fielddict):
        '''

        :param n: number of features
        :param k: dim of latent space
        :param f: number of fields
        '''
        super(FFM_model, self).__init__()
        self.n = n  # len(items) + len(users)
        self.k = k
        self.f = f
        self.linear = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.n, self.f, self.k))
        self.feature2fielddict = feature2fielddict

    def ffm_layer(self, x):
        linear_part = self.linear(x)
        field_aware_interaction_part = torch.tensor(0).float()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                field_aware_interaction_part = field_aware_interaction_part + torch.sum(
                    torch.mul(self.v[i, self.feature2fielddict[j]],
                              self.v[j, self.feature2fielddict[i]])) * torch.unsqueeze(torch.mul(x[:, i], x[:, j]), 1)
        output = torch.add(linear_part, field_aware_interaction_part)
        return output  # out_size = (batch, 1)

    def forward(self, x):
        x = x.float()
        output = self.ffm_layer(x)
        return output

if __name__ == '__main__':
    # 0,1-19,20-21,22-42
    feature2fielddict = {}
    for field, index_range in enumerate([[0, 0], [1, 19], [20, 21], [22, 42]]):
        for i in range(index_range[0], index_range[1] + 1):
            feature2fielddict[i] = field

    BATCH_SIZE = 1000
    n_epochs = 35
    learing_rating = 0.05
    mom = 0.5

    n = 43  # 特征数目
    k = 5  # 因子的数目
    f = field + 1
    ffm = FFM_model(n, k, f, feature2fielddict=feature2fielddict)

    df_data, df_label = SklearnDataLoader.load_builtin('ml-100k')
    df_data = feature_engi(df_data)
    df_label = scale_and_transform(df_label)

    # x_train, x_test, y_train, y_test = train_test_split(df_data.values, np.squeeze(df_label.values))
    x_train, x_test, y_train, y_test = train_test_split(df_data.values, df_label)

    # x_train = None
    # y_train = None
    # x_test = None
    # y_test = None
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

    # 训练网络

    optimizer = torch.optim.SGD(ffm.parameters(), lr=learing_rating, momentum=mom)  # 学习率为
    # loss_func = nn.MSELoss()
    # loss__train_set = []
    # loss_test_set = []
    for epoch in range(n_epochs):  # 对数据集进行训练
        # 训练集
        for step, (batch_x, batch_y) in enumerate(loader):  # 每个训练步骤
            output = ffm(batch_x)
            rmse_loss = rmse(output, batch_y)
            # l2_regularization = torch.tensor(0).float()
            # # 加入l2正则
            # for param in fm.parameters():
            #     l2_regularization += torch.norm(param, 2)
            # loss = rmse_loss + l2_regularization
            loss = rmse_loss

            optimizer.zero_grad()  # 如果不置零，Variable的梯度在每次backwrd的时候都会累加
            loss.backward()
            optimizer.step()  # 进行更新
            # 将每一次训练的数据进行存储，然后用于绘制曲线
            if step % 50 == 0:
                print('Epoch:{} STEP:{},train_LOSS:{:.4f}'.format(epoch, step, loss))
            # print('Epoch:{} STEP:{},train_LOSS:{:.4f}'.format(epoch, step, loss))
        # loss__train_set.append(loss)

        # 测试集
        for step, (batch_x, batch_y) in enumerate(loader_test):  # 每个训练步骤
            # 此处省略一些训练步骤
            optimizer.zero_grad()  # 如果不置零，Variable的梯度在每次backwrd的时候都会累加
            output = ffm(batch_x)
            # output = output.transpose(1, 0)
            # 平方差
            rmse_loss = rmse(output, batch_y)
            # l2_regularization = torch.tensor(0).float()
            # # print("l2_regularization type",l2_regularization.dtype)
            # # 加入l2正则
            # for param in fm.parameters():
            #     # print("param type",pt.norm(param,2).dtype)
            #     l2_regularization += torch.norm(param, 2)
            # # loss = rmse_loss + l2_regularization
            loss = rmse_loss
            loss.backward()
            optimizer.step()  # 进行更新
        # 将每一次训练的数据进行存储，然后用于绘制曲线
        print('Epoch:{} ; test_LOSS:{:.4f}'.format(epoch, loss))
        # loss_test_set.append(loss)
