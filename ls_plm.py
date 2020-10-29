"""
@Time ： 2020/10/28 16:23
@Auth ： Yifan Hu
@File ：ls_plm.py
@IDE ：PyCharm
"""

import torch
from torch import nn
# import torch.nn as nn
from data_loader import SklearnDataLoader
from logis_reg import feature_engi
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 评价指标rmse
def rmse(pred_rate, real_rate):
    loss_func = nn.MSELoss()
    mse_loss = loss_func(pred_rate, real_rate.float())
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss

def ce_loss(pred_rate,real_rate):
    loss_func = nn.BCELoss()
    ce_loss = loss_func(pred_rate,real_rate.float())
    return ce_loss

class LS_PLM_MODEL(nn.Module):
    '''
    realization of ls_plm model
    '''

    def __init__(self, n, m=4):
        '''

        :param n: number of features
        :param m: number of slices
        '''
        super(LS_PLM_MODEL, self).__init__()
        self.n = n
        self.m = m
        self.linear_soft = nn.Linear(self.n, m, bias=True)
        self.linear_sig = nn.Linear(self.n, m, bias=True)

    def forward(self, x):
        x = x.float()
        slice_output = torch.softmax(self.linear_soft(x), 1)
        sigmoid_output = torch.sigmoid(self.linear_sig(x))
        output = torch.mul(slice_output, sigmoid_output)
        return torch.sum(output, 1)


if __name__ == '__main__':

    BATCH_SIZE = 512
    n_epochs = 30
    learing_rating = 0.05
    mom = 0.8
    loss_func = ce_loss


    n = 43  # 特征数目
    m = 12  # 分区的数目
    model = LS_PLM_MODEL(n, m)

    df_data, df_label = SklearnDataLoader.load_builtin('ml-100k', classification=True)
    df_data = feature_engi(df_data)
    y_true = df_label.values.ravel()

    # df_data, df_label = SklearnDataLoader.load_builtin('ml-100k')
    # df_data = feature_engi(df_data)
    # df_label = scale_and_transform(df_label)

    # x_train, x_test, y_train, y_test = train_test_split(df_data.values, np.squeeze(df_label.values))
    x_train, x_test, y_train, y_test = train_test_split(df_data.values, y_true)

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

    optimizer = torch.optim.SGD(model.parameters(), lr=learing_rating, momentum=mom)  # 学习率为
    # loss_func = nn.MSELoss()
    loss__train_set = []
    loss_test_set = []
    for epoch in range(n_epochs):  # 对数据集进行训练
        # 训练集
        for step, (batch_x, batch_y) in enumerate(loader):  # 每个训练步骤
            output = model(batch_x)
            loss_result = loss_func(output, batch_y)
            # l2_regularization = torch.tensor(0).float()
            # # 加入l2正则
            # for param in fm.parameters():
            #     l2_regularization += torch.norm(param, 2)
            # loss = rmse_loss + l2_regularization
            loss = loss_result

            optimizer.zero_grad()  # 如果不置零，Variable的梯度在每次backwrd的时候都会累加
            loss.backward()
            optimizer.step()  # 进行更新
            # 将每一次训练的数据进行存储，然后用于绘制曲线
            if step % 50 == 0:
                print('Epoch:{} STEP:{},train_LOSS:{:.4f}'.format(epoch, step, loss))
            # print('Epoch:{} STEP:{},train_LOSS:{:.4f}'.format(epoch, step, loss))
        loss__train_set.append(loss)

        # 测试集
        for step, (batch_x, batch_y) in enumerate(loader_test):  # 每个训练步骤
            # 此处省略一些训练步骤
            optimizer.zero_grad()  # 如果不置零，Variable的梯度在每次backwrd的时候都会累加
            output = model(batch_x)
            # 平方差
            loss_result = loss_func(output, batch_y)
            # l2_regularization = torch.tensor(0).float()
            # # print("l2_regularization type",l2_regularization.dtype)
            # # 加入l2正则
            # for param in fm.parameters():
            #     # print("param type",pt.norm(param,2).dtype)
            #     l2_regularization += torch.norm(param, 2)
            # # loss = rmse_loss + l2_regularization
            loss = loss_result
            loss.backward()
            optimizer.step()  # 进行更新
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

    # # 保存训练好的模型
    # torch.save(model.state_dict(), "data/fm_params.pt")
    # test_save_net = model(n, m)
    #
    # # 读取模型
    # test_save_net.load_state_dict(torch.load("data/fm_params.pt"))
    # # 测试网络
    # pred = test_save_net(torch.tensor(x_test))
    # pred = pred.transpose(1, 0)
    # loss_result = loss_func(pred, y_test)
    # print("test_loss", loss_result)
