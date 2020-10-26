"""
@Time ： 2020/10/23 16:36
@Auth ： Yifan Hu
@File ：fm.py
@IDE ：PyCharm
"""

# ref : https://blog.csdn.net/u012969412/article/details/88684723
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from data_loader import SklearnDataLoader
from logis_reg import feature_engi, scale_and_transform
import matplotlib.pyplot as plt


# 评价指标rmse
def rmse(pred_rate, real_rate):
    loss_func = nn.MSELoss()
    mse_loss = loss_func(pred_rate, real_rate.float())
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss


class FM_model(nn.Module):
    def __init__(self, n, k):
        super(FM_model, self).__init__()
        self.n = n  # len(items) + len(users)
        self.k = k
        self.linear = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.k, self.n))

    def fm_layer(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v.t())  # out_size = (batch, k)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t())  # out_size = (batch, k)
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        return output  # out_size = (batch, 1)

    def forward(self, x):
        x = x.float()
        output = self.fm_layer(x)
        return F.sigmoid(output)


BATCH_SIZE = 1000
n_epochs = 35
learing_rating = 0.05
mom = 0.5

n = 43  # 特征数目
k = 5  # 因子的数目
fm = FM_model(n, k)

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

# #
# #train
# from tqdm import tqdm
# batch_size=64
# model = FM_model(p,k).cuda()
# loss_fn =nn.MSELoss()
# optimer = torch.optim.SGD(model.parameters(),lr=0.0001,weight_decay=0.001)
# epochs = 100
# for epoch in range(epochs):
#     loss_epoch = 0.0
#     loss_all = 0.0
#     perm = np.random.permutation(x_train.shape[0])
#     model.train()
#     for x,y in tqdm(batcher(x_train[perm], y_train[perm], batch_size)):
#         model.zero_grad()
#         x = torch.as_tensor(np.array(x.tolist()), dtype=torch.float,device=torch.device('cuda'))
#         y = torch.as_tensor(np.array(y.tolist()), dtype=torch.float,device=torch.device('cuda'))
#         x = x.view(-1, p)
#         y = y.view(-1, 1)
#         preds = model(x)
#         loss = loss_fn(preds,y)
#         loss_all += loss.item()
#         loss.backward()
#         optimer.step()
#     loss_epoch = loss_all/len(x)
#     print(f"Epoch [{epoch}/{10}], "
#               f"Loss: {loss_epoch:.8f} ")

# 训练网络

optimizer = torch.optim.SGD(fm.parameters(), lr=learing_rating,momentum=mom)  # 学习率为
# loss_func = nn.MSELoss()
loss__train_set = []
loss_test_set = []
for epoch in range(n_epochs):  # 对数据集进行训练
    # 训练集
    for step, (batch_x, batch_y) in enumerate(loader):  # 每个训练步骤
        output = fm(batch_x)
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
    loss__train_set.append(loss)

    # 测试集
    for step, (batch_x, batch_y) in enumerate(loader_test):  # 每个训练步骤
        # 此处省略一些训练步骤
        optimizer.zero_grad()  # 如果不置零，Variable的梯度在每次backwrd的时候都会累加
        output = fm(batch_x)
        output = output.transpose(1, 0)
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
    loss_test_set.append(loss)

plt.clf()
plt.plot(range(epoch + 1), loss__train_set, label='Training data')
plt.plot(range(epoch + 1), loss_test_set, label='Test data')
plt.title('The MovieLens Dataset Learing Curve')
plt.xlabel('Number of Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()
print("train_loss", loss)

# 保存训练好的模型
torch.save(fm.state_dict(), "data/fm_params.pt")
test_save_net = FM_model(n, k)

# 读取模型
test_save_net.load_state_dict(torch.load("data/fm_params.pt"))
# 测试网络
pred = test_save_net(torch.tensor(x_test))
pred = pred.transpose(1, 0)
rmse_loss = rmse(pred, y_test)
print("test_loss", rmse_loss)
