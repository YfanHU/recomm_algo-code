"""
@Time ： 2020/10/29 14:59
@Auth ： Yifan Hu
@File ：neuralcf.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_loader import SklearnDataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

os.chdir('../')


# 评价指标rmse
def rmse(pred_rate, real_rate):
    loss_func = nn.MSELoss()
    mse_loss = loss_func(pred_rate, real_rate.float())
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss


class NeuralCF(torch.nn.Module):
    def __init__(self, n_users, n_items, k=8, mlp_layer_sizes=[16, 8, 4]):
        super(NeuralCF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedd_users_mf = nn.Embedding(n_users, k)
        self.embedd_users_mlp = nn.Embedding(n_users, mlp_layer_sizes[0] // 2)
        self.embedd_items_mf = nn.Embedding(n_items, k)
        if mlp_layer_sizes[0] % 2 == 0:
            self.embeed_items_mlp = nn.Embedding(n_items, mlp_layer_sizes[0] // 2)
        else:
            self.embeed_items_mlp = nn.Embedding(n_items, mlp_layer_sizes[0] // 2 + 1)
        self.mlp_layers = [nn.Linear(mlp_layer_sizes[i], mlp_layer_sizes[i + 1], bias=True)
                           for i in range(len(mlp_layer_sizes) - 1)]
        self.scoring_layer = nn.Linear(k + mlp_layer_sizes[-1], 1, bias=True)

    def forward(self, x):
        mf_user = self.embedd_users_mf(x[:, 0])
        mlp_user = self.embedd_users_mlp(x[:, 0])
        mf_item = self.embedd_items_mf(x[:, 1])
        mlp_item = self.embeed_items_mlp(x[:, 1])
        mf_output = torch.mul(mf_user, mf_item)
        mlp_input = torch.cat((mlp_user, mlp_item), 1)
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
        ncf_input = torch.cat((mf_output, mlp_input), 1)
        score = self.scoring_layer(ncf_input)
        return score


if __name__ == '__main__':

    BATCH_SIZE = 500
    n_epochs = 30
    learing_rating = 0.001
    k = 10
    n_res = 2
    l2_penalty = 0.2
    mom = 0.8
    loss_func = rmse

    df_rating = SklearnDataLoader.load_builtin('ml-100k', rating_only=True)
    n_users = df_rating['uid'].nunique()
    n_items = df_rating['iid'].nunique()
    df_x = df_rating[['uid', 'iid']]
    label_enc_uid = LabelEncoder()
    label_enc_iid = LabelEncoder()
    df_y = df_rating[['rating']]
    df_x['uid'] = label_enc_uid.fit_transform(df_x['uid'].values)
    df_x['iid'] = label_enc_iid.fit_transform(df_x['iid'].values)

    model = NeuralCF(n_users, n_items)

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
