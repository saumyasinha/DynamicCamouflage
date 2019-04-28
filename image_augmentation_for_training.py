import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import numpy
from PIL import Image
ia.seed(4)


dir = "/Users/saumya/Downloads/720x360+ex+out/"
saving_dir="/Users/saumya/Desktop/texture_for_discriminator/"
image1 = imageio.imread(dir + "texture-0_camera-8_scene.jpg")
image2 = imageio.imread(dir + "texture-0_camera-9_scene.jpg")
image3 = imageio.imread(dir + "texture-0_camera-10_scene.jpg")
image4 = imageio.imread(dir + "texture-0_camera-11_scene.jpg")


seq = iaa.Sequential([
    iaa.Resize({"height": 128, "width": 128}),
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(30, 90))
], random_order=True)

images1_aug = [seq.augment_image(image1) for _ in range(5)]
images2_aug = [seq.augment_image(image1) for _ in range(5)]
images3_aug = [seq.augment_image(image1) for _ in range(5)]
images4_aug = [seq.augment_image(image1) for _ in range(5)]

images_aug_list = [images1_aug, images2_aug, images3_aug, images4_aug]
images_aug = [item for sublist in images_aug_list for item in sublist]


for i,img in enumerate(images_aug):
    imageio.imwrite(saving_dir+"aug_"+str(i)+".jpg",img)


for n in range(10):
    a = numpy.random.rand(128,128,3) * 255
    im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
    im_out.save(saving_dir+'out%000d.jpg' % n)



brick = imageio.imread(dir+"seamless_brick-512x512.png")
wave = imageio.imread(dir + "wave-400x400.jpg")
four_color = imageio.imread(dir + "Four-Color-Cubic-Patterns-512-512.jpg")

orig_textures = [image1, image2, image3, image4, brick, wave, four_color]

seq = iaa.Sequential([
    iaa.Resize({"height": 128, "width": 128})])

orig_textures = seq.augment_images(orig_textures)

for i,img in enumerate(orig_textures):
    im_out = Image.fromarray(img.astype('uint8')).convert('RGB')
    im_out.save(saving_dir+"orig_texture_"+str(i)+".jpg")

seq = iaa.Sequential([
    iaa.Resize({"height": 128, "width": 128}),
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(30, 90))
], random_order=True)

brick_aug = [seq.augment_image(brick) for _ in range(2)]
wave_aug = [seq.augment_image(wave) for _ in range(2)]
four_color_aug = [seq.augment_image(four_color) for _ in range(2)]

new_textures_aug_list = [brick_aug, wave_aug, four_color_aug]
new_textures_aug = [item for sublist in new_textures_aug_list for item in sublist]

for i,img in enumerate(new_textures_aug):
    im_out = Image.fromarray(img.astype('uint8')).convert('RGB')
    im_out.save(saving_dir + "new_texture_" + str(i) + ".jpg")


