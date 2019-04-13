import numpy as np
import torch
from torch import nn
import torch.optim as optim
import Module_Discriminator.Model as Model
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

def create_input_texture(local_images, texture_size):
    n = texture_size
    input_texture=[]
    for i in range(local_images.shape[0]):
        X = local_images[i].flatten()
        minX = np.min(X)
        maxX = np.max(X)
        # texture = np.random.choice(X, (n*n), replace=False)
        texture = np.linspace(minX, maxX, num=3*n*n)
        # print(texture.shape)
        input_texture.append(np.reshape(texture, (3,n,n)))


    return np.array(input_texture)



def train_model(local_images_dir, score_dir, batch_size, n_epochs):
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    # local_images = np.load(local_images_dir)
    local_images = np.random.uniform(low=0, high=255, size=(100, 4, 3, 64, 64))
    input_textures = create_input_texture(local_images, 64)
    print(local_images.shape)
    print(input_textures.shape)

    # scores = np.load(score_dir)
    scores = np.random.rand(100,1)


    texture_tensor = torch.tensor(input_textures)
    score_tensor = torch.tensor(scores)

    local_images_split  = np.split(local_images, indices_or_sections=4, axis=1)

    bk_1_tensor = torch.tensor(local_images_split[0][:, 0, :, :, :])
    bk_2_tensor = torch.tensor(local_images_split[1][:, 0, :, :, :])
    bk_3_tensor = torch.tensor(local_images_split[2][:, 0, :, :, :])
    bk_4_tensor = torch.tensor(local_images_split[3][:, 0, :, :, :])

    print(local_images_split[0][:, 0, :, :, :].shape)

    # specify loss function (categorical cross-entropy)
    criterion = nn.BCELoss()

    # create a complete CNN
    model = Model.discrimator_net()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.003)

    if train_on_gpu:
        model.cuda()

    dataset = TensorDataset(bk_1_tensor.float(), bk_2_tensor.float(), bk_3_tensor.float(), bk_4_tensor.float(), texture_tensor.float(), score_tensor.float())
    loader = DataLoader(
        dataset,
        batch_size=batch_size
    )



    for epoch in range(1, n_epochs + 1):

        # keep track of training loss
        train_loss = 0.0

        # num_batches = int(local_images.shape[0]/batch_size)

        ###################
        # train the model #
        ###################
        model.train()
        # for i in range(num_batches):
        for batch_idx, (bk_1_batch, bk_2_batch, bk_3_batch, bk_4_batch, texture_batch, score_batch) in enumerate(loader):
            # print(bk_1_batch.shape, texture_batch.shape, score_batch.shape)

            # batch_ind = np.random.choice(local_images.shape[0], batch_size, replace=True)
            # bk_1_batch = bk_1_tensor[batch_ind]
            # bk_2_batch = bk_2_tensor[batch_ind]
            # bk_3_batch = bk_3_tensor[batch_ind]
            # bk_4_batch = bk_4_tensor[batch_ind]
            #
            # texture_batch = texture_tensor[batch_ind]
            # score_batch = score_tensor[batch_ind]

            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                bk_1_batch, bk_1_batch, bk_1_batch, bk_1_batch, texture_batch, score_batch  = bk_1_batch.cuda(), bk_2_batch.cuda(), bk_3_batch.cuda(), bk_4_batch.cuda(), texture_batch.cuda(),score_batch.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(texture_batch, bk_1_batch, bk_2_batch, bk_3_batch, bk_4_batch)
            # calculate the batch loss
            loss = criterion(output, score_batch)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * bk_1_batch.size(0)

        # calculate average losses
        train_loss = train_loss / local_images.shape[0]

        # print training statistics
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss))


    torch.save(model.state_dict(), 'model_discriminatornet.pt')

    return('model_discriminatornet.pt', bk_1_tensor, bk_2_tensor, bk_3_tensor, bk_4_tensor)








