import numpy as np
import scipy.io as sio
import scipy.misc
import numpy
import matplotlib.pyplot as plt
from scipy.misc import imread
import sys

# data preperation (loading, normalizing, reshaping)
def load():
    path = 'dog.jpeg'
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
    return X, img_size;


# Initializes K centroids that are to be used in K-Means on the dataset X.
def init_centroids(X, K):
    if K == 2:
        return np.asarray([[0.        , 0.        , 0.        ],
                            [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                            [0.49019608, 0.41960784, 0.33333333],
                            [0.02745098, 0.        , 0.        ],
                            [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                            [0.14509804, 0.12156863, 0.12941176],
                            [0.4745098 , 0.40784314, 0.32941176],
                            [0.00784314, 0.00392157, 0.02745098],
                            [0.50588235, 0.43529412, 0.34117647],
                            [0.09411765, 0.09019608, 0.11372549],
                            [0.54509804, 0.45882353, 0.36470588],
                            [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                            [0.4745098 , 0.38039216, 0.33333333],
                            [0.65882353, 0.57647059, 0.49411765],
                            [0.08235294, 0.07843137, 0.10196078],
                            [0.06666667, 0.03529412, 0.02352941],
                            [0.08235294, 0.07843137, 0.09803922],
                            [0.0745098 , 0.07058824, 0.09411765],
                            [0.01960784, 0.01960784, 0.02745098],
                            [0.00784314, 0.00784314, 0.01568627],
                            [0.8627451 , 0.78039216, 0.69803922],
                            [0.60784314, 0.52156863, 0.42745098],
                            [0.01960784, 0.01176471, 0.02352941],
                            [0.78431373, 0.69803922, 0.60392157],
                            [0.30196078, 0.21568627, 0.1254902 ],
                            [0.30588235, 0.2627451 , 0.24705882],
                            [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None


# return string that contain the centroid in specific iteration.
def print_cent(cent,t):
    print('iter ' + str(t) + ': ', end='');
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]', ']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]', ']').replace(' ', ', ')[1:-1]


# K means algorithm - calculate the centroid during 10 iteration and print
def k_means_algo(pixels, k,img_size):
    centroids_initialize = init_centroids(pixels, k);
    print('k=' + str(k)+':');
    loss_map = [];
    # run 11 iteration - from 0 to 10
    for iter in range(11):
        print(print_cent(centroids_initialize,iter));
        dict_centroids = {};
        loss = 0;
        # run over all the pixel in the picture
        for pix in pixels:
            minimum = float('inf');
            helper=0;
            # run over the centroid
            for index in range(k):
                # calculate the euclidean square distance between the pixel to the centroid
                min_dist = pow(np.linalg.norm(centroids_initialize[index] - pix), 2);
                # choose the minimum distance
                if min_dist < minimum:
                    minimum = min_dist;
                    index_min = index;
            loss += minimum;
            # add to dictionary the pixel with minimum distance with key= index of centroid
            try:
                dict_centroids[index_min].append(pix);
            except KeyError:
                dict_centroids[index_min] = [pix]
        # calculate the average of each centroid
        for key in dict_centroids.keys():
            centroids_initialize[key] = np.average(dict_centroids[key], axis=0);
        loss_avg = loss/float(len(pixels));
        loss_map.append(loss_avg);
    return loss_map;


# plot loss graph
def plot_loss(loss, k):
    y = list(range(11))
    plt.plot(y, loss);
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.title('The loss with K=' + str(k));
    plt.show()


# main function
def main():
    data, img_size = load();
    for k in [2, 4, 8, 16]:
        loss_map = k_means_algo(data, k,img_size);
        #plot_loss(loss_map,k);

main()

