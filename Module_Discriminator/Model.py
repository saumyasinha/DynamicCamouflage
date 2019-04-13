import torch
from torch import nn

nn_initalization = nn.init.xavier_uniform


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn_initalization(m.weight.data)
    elif isinstance(m, nn.Linear):
        nn_initalization(m.weight.data)


class discrimator_net(nn.Module):
    def __init__(self):
        super(discrimator_net, self).__init__()
        self.bk_conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bk_conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.bk_conv3 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.bk_conv4 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.bk_conv5 = nn.Conv2d(4, 4, kernel_size=3, padding=1)

        self.conv_per_1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv_per_2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv_per_3 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(20, 4, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.lrelu = nn.LeakyReLU()
        self.do = nn.Dropout()
        self.linear = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()


    def forward(self, x, bk_1, bk_2, bk_3, bk_4):

        bk_1 = self.bk_conv1(bk_1)
        bk_1 = self.maxpool(bk_1)
        bk_1 = self.bk_conv2(bk_1)
        bk_1 = self.maxpool(bk_1)
        bk_1 = self.bk_conv3(bk_1)
        bk_1 = self.maxpool(bk_1)
        bk_1 = self.bk_conv4(bk_1)
        bk_1 = self.maxpool(bk_1)
        bk_1 = self.bk_conv5(bk_1)
        bk_1 = self.maxpool(bk_1)
        # print(bk_1.shape)

        bk_2 = self.bk_conv1(bk_2)
        bk_2 = self.maxpool(bk_2)
        bk_2 = self.bk_conv2(bk_2)
        bk_2 = self.maxpool(bk_2)
        bk_2 = self.bk_conv3(bk_2)
        bk_2 = self.maxpool(bk_2)
        bk_2 = self.bk_conv4(bk_2)
        bk_2 = self.maxpool(bk_2)
        bk_2 = self.bk_conv5(bk_2)
        bk_2 = self.maxpool(bk_2)
        # print(bk_2.shape)

        bk_3 = self.bk_conv1(bk_3)
        bk_3 = self.maxpool(bk_3)
        bk_3 = self.bk_conv2(bk_3)
        bk_3 = self.maxpool(bk_3)
        bk_3 = self.bk_conv3(bk_3)
        bk_3 = self.maxpool(bk_3)
        bk_3 = self.bk_conv4(bk_3)
        bk_3 = self.maxpool(bk_3)
        bk_3 = self.bk_conv5(bk_3)
        bk_3 = self.maxpool(bk_3)
        # print(bk_3.shape)

        bk_4 = self.bk_conv1(bk_4)
        bk_4 = self.maxpool(bk_4)
        bk_4 = self.bk_conv2(bk_4)
        bk_4 = self.maxpool(bk_4)
        bk_4 = self.bk_conv3(bk_4)
        bk_4 = self.maxpool(bk_4)
        bk_4 = self.bk_conv4(bk_4)
        bk_4 = self.maxpool(bk_4)
        bk_4 = self.bk_conv5(bk_4)
        bk_4 = self.maxpool(bk_4)
        # print(bk_4.shape)

        x = self.maxpool(x)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.conv_per_1(x)
        x = self.maxpool(x)

        x = self.conv_per_2(x)
        x = self.maxpool(x)

        x = self.conv_per_3(x)
        x = self.maxpool(x)
        # print(x.shape)

        x_bk = torch.cat((x, bk_1, bk_2, bk_3, bk_4), 1)
        x_bk = self.conv4(x_bk)
        # print(x_bk.shape)
        fea = x_bk.view(x_bk.size(0), -1)
        # print(fea.shape)
        x = self.linear(fea)
        x = self.sig(x)

        return x