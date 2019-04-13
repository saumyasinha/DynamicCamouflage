import numpy as np
import torch
import torch.optim as optim
from Module_Discriminator.Model import discrimator_net
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch import nn
from torchvision import transforms, models

train_on_gpu = torch.cuda.is_available()

# get the "features" portion of VGG19 (we will not need the "classifier" portion)
vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)



# move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)


def get_features(image, model, layers=None):
      ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '28': 'conv5_1'}

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):

    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()

    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram



def train_model(PATH, bk_1_tensor, bk_2_tensor, bk_3_tensor, bk_4_tensor, n_epochs):

    discriminator = discrimator_net()
    discriminator.load_state_dict(torch.load(PATH))
    discriminator.train()

    for param in discriminator.parameters():
        param.requires_grad_(False)

    if train_on_gpu:
        discriminator.cuda()

    print(discriminator)

    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    score_weight = 1  # alpha
    style_weight = 1e6  # beta

    dataset = TensorDataset(bk_1_tensor.float(), bk_2_tensor.float(), bk_3_tensor.float(), bk_4_tensor.float())
    loader = DataLoader(
        dataset,
        batch_size=1
    )

    for single_bk_1, single_bk_2, single_bk_3, single_bk_4 in loader:
        texture_np = np.random.rand(1, 3, 64, 64)
        texture = Variable(torch.tensor(texture_np).float(), requires_grad=True)

        if train_on_gpu:
            texture = Variable(torch.tensor(texture_np).float().cuda(), requires_grad=True)

        style_features_1 = get_features(single_bk_1, vgg)
        style_features_2 = get_features(single_bk_2, vgg)
        style_features_3 = get_features(single_bk_3, vgg)
        style_features_4 = get_features(single_bk_4, vgg)

        style_grams = {}

        # calculate the gram matrices for each layer of our style representation
        style_grams_1 = {layer: gram_matrix(style_features_1[layer]) for layer in style_features_1}
        style_grams_2 = {layer: gram_matrix(style_features_2[layer]) for layer in style_features_2}
        style_grams_3 = {layer: gram_matrix(style_features_3[layer]) for layer in style_features_3}
        style_grams_4 = {layer: gram_matrix(style_features_4[layer]) for layer in style_features_4}

        for layer in style_grams_1.keys():
            list_across_images = [style_grams_1[layer], style_grams_2[layer], style_grams_3[layer], style_grams_4[layer]]
            print(list_across_images[0][0][0], list_across_images[1][0][0], list_across_images[2][0][0], list_across_images[3][0][0])
            style_grams[layer] = torch.mean(torch.stack(list_across_images), dim=0)
            print(style_grams[layer][0][0])

        optimizer = optim.Adam([texture], lr=0.003)

        for epoch in range(1, n_epochs + 1):

            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                single_bk_1, single_bk_2, single_bk_3, single_bk_4 = single_bk_1.cuda(), single_bk_2.cuda(), single_bk_3.cuda(), single_bk_4.cuda()

            optimizer.zero_grad()

            output = discriminator(texture, single_bk_1, single_bk_2, single_bk_3, single_bk_4)
            # calculate the batch loss
            criterion = nn.BCELoss()
            score_loss = criterion(output, torch.zeros_like(output))

            texture_features = get_features(texture, vgg)

            # the style loss
            # initialize the style loss to 0
            style_loss = 0

            for layer in style_weights:
                # get the "target" style representation for the layer
                texture_feature = texture_features[layer]
                texture_gram = gram_matrix(texture_feature)
                _, d, h, w = texture_feature.shape
                # get the "style" style representation
                style_gram = style_grams[layer]
                # the style loss for one layer, weighted appropriately
                layer_style_loss = style_weights[layer] * torch.mean((texture_gram - style_gram) ** 2)
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)

                # calculate the *total* loss
            total_loss = score_weight * score_loss + style_weight * style_loss

            total_loss.backward()
            optimizer.step()

            # print training statistics
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, total_loss.item()))




