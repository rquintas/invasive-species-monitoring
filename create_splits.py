import numpy as np
import os
import glob

from PIL import Image

from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics, model_selection

from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator

def ksplit(k, shuffle=True, balanced=False):
    kf = model_selection.StratifiedKFold(n_splits=k, shuffle=shuffle)

    data = np.loadtxt(data_dir+'/train_labels.csv', skiprows=1, dtype=int, delimiter=',')

    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    if balanced:
        ros = RandomOverSampler()

        X, y = ros.fit_sample(X, y)

        print('Resampled size: {} {}'.format(len(X), len(y)))
        print('Labels: 0: {}, 1: {}'.format(sum(y == 0), sum(y == 1)))

    print('Size: {} {}'.format(len(X), len(y)))
    print('Labels: 0: {}, 1: {}'.format(sum(y == 0), sum(y == 1)))

    for split in kf.split(X, y):
        train_idx, val_idx = split
        yield dict(
            train_data=X[train_idx],
            train_labels=y[train_idx],
            validation_data=X[val_idx],
            validation_labels=y[val_idx]
        )

def read_img(basepath, img_id, img_width, img_height):
    img = Image.open(basepath + '/%i.jpg' % (img_id))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((img_width, img_height))

    x = np.asarray(img, np.float32)

    # Normalize by maximum pixel value.
    x = x/255

    # Numpy array x has format (channel, height, width) (with channels first)
    # but original PIL image has format (width, height, channel)
    x = x.transpose((2, 0, 1))

    return x

def create_images(Xs, ys, gen, basepath):
    for x in tqdm(zip(Xs, ys)):
        img_id = x[0][0]
        y = x[1]

        orig_img = read_img(basepath, img_id, 256, 256)

        rescaled = (255.0 / orig_img.max() * (orig_img - orig_img.min())).astype(np.uint8)
        rescaled = rescaled.transpose((1, 2, 0))

        fpath = ('{}/{}.png').format(y,img_id)

        im = Image.fromarray(np.uint8(rescaled))
        im.save(fpath)

        for j in range(5):
            data = gen.random_transform(orig_img)

            rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
            rescaled = rescaled.transpose((1, 2, 0))

            fpath = ('{}/{}_{}.png').format(y,img_id,j)

            im = Image.fromarray(np.uint8(rescaled))
            im.save(fpath)

if __name__ == '__main__':


    os.chdir('../data')
    data_dir = os.getcwd()
    basepath = data_dir + '/train_orig'

    if not os.path.exists('data_5_splits_augmented_256_256'):

        os.mkdir('data_5_splits_augmented_256_256')
        os.chdir('data_5_splits_augmented_256_256')

        for i, split in enumerate(ksplit(5, True, True)):

            gen = ImageDataGenerator(
                        # featurewise_center = True,
                        rotation_range=30,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        # zca_whitening = True,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect')

            split_id = 'split_{}'.format(i)
            
            os.mkdir(split_id)
            os.chdir(split_id)

            os.mkdir('train')
            os.chdir('train')

            os.mkdir('0')
            os.mkdir('1')

            create_images(split['train_data'],split['train_labels'],gen,basepath)
            
            os.chdir('..')
            os.mkdir('validation')
            os.chdir('validation')
            os.mkdir('0')
            os.mkdir('1')

            create_images(split['validation_data'],split['validation_labels'],gen,basepath)

            os.chdir('..')
            os.chdir('..')
        
        os.chdir('..')

    if not os.path.exists('data_test_augmented_256_256'):

        os.mkdir('data_test_augmented_256_256')
        os.chdir('data_test_augmented_256_256')
        
        os.mkdir('0.5')

        gen = ImageDataGenerator(
                # featurewise_center = True,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                # zca_whitening = True,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='reflect')

        data = np.loadtxt(data_dir+'/sample_submission.csv', skiprows=1, dtype=float, delimiter=',')

        X = data[:, 0].reshape(-1, 1)
        y = data[:, 1]
        
        create_images(X,y,gen,data_dir+'/test/unknown')




# try:
#     os.mkdir('present')
#     os.mkdir('absent')
# except:
#     pass

# labels = np.loadtxt('../train_labels.csv', skiprows=1, dtype=int, delimiter=',')[:, 1]

# imagefiles = glob.glob('*.jpg')

# print(imagefiles)

# for image in imagefiles:
#     i = int(image.replace('.jpg',''))
#     if labels[i-1] == 1:
#         os.rename(image, 'present/' + image)
#     else:
#         os.rename(image, 'absent/' + image)

