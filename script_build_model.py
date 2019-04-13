import Module_Discriminator.train as MDL
import Module_Generator.baseline_train as MGL


if __name__ == '__main__':
    data_dir = ""

    score_dir = ""

    # how many samples per batch to load
    batch_size = 50

    n_epochs = 5

    PATH, bk_1_tensor, bk_2_tensor, bk_3_tensor, bk_4_tensor = MDL.train_model(data_dir, score_dir, batch_size, n_epochs)
    # MGL.train_model(PATH, bk_1_tensor, bk_2_tensor, bk_3_tensor, bk_4_tensor, batch_size, n_epochs)
    # MGL.train_model(PATH, bk_1_tensor, bk_2_tensor, bk_3_tensor, bk_4_tensor, n_epochs)