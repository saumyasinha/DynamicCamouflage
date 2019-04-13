import numpy as np
import torch
from torch import nn
import torch.optim as optim
from Module_Discriminator.Model import discrimator_net
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

def train_model(PATH, bk_1_tensor, bk_2_tensor, bk_3_tensor, bk_4_tensor, n_epochs):
    train_on_gpu = torch.cuda.is_available()

    discriminator = discrimator_net()
    discriminator.load_state_dict(torch.load(PATH))
    discriminator.train()

    for param in discriminator.parameters():
        param.requires_grad_(False)

    if train_on_gpu:
        discriminator.cuda()

    print(discriminator)


    # iteration hyperparameters


    dataset = TensorDataset(bk_1_tensor.float(), bk_2_tensor.float(), bk_3_tensor.float(), bk_4_tensor.float())
    loader = DataLoader(
        dataset,
        batch_size = 1
    )

    # bk_1, bk_2, bk_3, bk_4, texture = bk_1_tensor.float(), bk_2_tensor.float(), bk_3_tensor.float(), bk_4_tensor.float(), texture.float()

    for single_bk_1, single_bk_2, single_bk_3, single_bk_4 in loader:
        texture_np = np.random.rand(1, 3, 16, 16)
        texture = Variable(torch.FloatTensor(texture_np), requires_grad=True)

        if train_on_gpu:
            texture = Variable(torch.FloatTensor(texture_np).cuda(), requires_grad=True)

        optimizer = optim.Adam([texture], lr=0.003)

        # print("texture")

        for epoch in range(1, n_epochs + 1):

            # keep track of training loss
            train_loss = 0.0

                # move tensors to GPU if CUDA is available
            if train_on_gpu:
                single_bk_1, single_bk_2, single_bk_3, single_bk_4 = single_bk_1.cuda(), single_bk_2.cuda(), single_bk_3.cuda(), single_bk_4.cuda()

            optimizer.zero_grad()

            output = discriminator(texture, single_bk_1, single_bk_2, single_bk_3, single_bk_4)
            # calculate the batch loss
            criterion = nn.BCELoss()
            loss = criterion(output, torch.zeros_like(output))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()


            train_loss = loss.data.cpu().numpy()

            # print training statistics
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, train_loss))


        # print("\n")






