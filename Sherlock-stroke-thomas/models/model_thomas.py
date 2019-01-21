# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:10:48 2018

@author: dumle
"""

import datetime
import glob
import os
from os import sys

import numpy as np
import tensorflow as tf
from scipy.cluster.hierarchy import linkage, fcluster
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Conv2DTranspose
from keras.layers import Reshape, MaxPooling2D, Input, Dense, UpSampling2D, Conv2D, LeakyReLU, TimeDistributed
from keras.layers.cudnn_recurrent import CuDNNLSTM
from keras.layers import Bidirectional
from keras.models import Model
# from keras.optimizers import Adam
# from keras.regularizers import l2
from keras.utils import multi_gpu_model

from base.base_model import BaseModel
from utils.config import process_config


# from data_loader import DataGenerator


class TemporalClustering(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.CNN_window_size = config.model.CNN_window_size
        # self.dense_units = {'DenseLayer1': 128,
        #                     'DenseLayer2': 128}  # {k: np.random.randint(v[0], v[1]) for k,v in config.dense_units.items()}
        self.dimensionality = config.model.dimensionality
        self.dropout_probability = config.model.dropout_probability
        self.input_shape = config.model.input_shape
        self.kernel_size = config.model.kernel_size
        self.channel_order = config.data_loader.channel_order
        self.initial_lr = config.model.initial_lr
        self.num_gpus = config.model.num_gpus
        self.num_LSTM_units = config.model.num_LSTM_units
        self.num_time_steps = config.model.num_time_steps
        self.padding = config.model.padding
        self.parallel_model = None
        self.scale_l2_regularization = config.model.scale_l2_regularization
        self.stride = config.model.stride
        self.pool_size = config.model.pool_size
        self.x = None
        self.y = None
        self.build_model()

    def build_model(self):
        self.x = Input(shape=self.config.model.input_shape)
        self.y = self.x
        # for layer in ['Layer1', 'Layer2']:
        #     self.y = Conv2D(filters=self.dimensionality[layer],
        #             kernel_size=(1,self.config.model.kernel_size[layer]),
        #             strides=(1),
        #             padding='same',
        #             data_format='channels_first'
        #             )(self.x)              # output shape = [dimensionality['Layer1'], 1, num_samples/2-kernel_length/2]
        # self.y = BatchNormalization(axis=1)(self.y)
        self.y = Conv2D(self.dimensionality['Layer1'],
                        kernel_size=(1,self.config.model.kernel_size['Layer1']),
                        padding='same',
                        data_format=self.channel_order)(self.x) # check for weight initialization and bias for conv2d
        self.y = (LeakyReLU())(self.y)
        self.y = MaxPooling2D(pool_size=(1, 4), data_format=self.channel_order, padding='same')(self.y)
        self.y = Reshape((int(self.y.shape[3]), int(self.y.shape[1])))(self.y)
        # self.y  = Permute((3,1,2))(self.y)
        # self.y = Reshape((int(self.y.shape[1]), int(self.y.shape[2])))(self.y)
        self.y = Bidirectional(CuDNNLSTM(self.num_LSTM_units['Layer1'], return_sequences=True))(self.y) #check weight and bias initialization
        self.y = (LeakyReLU())(self.y)
        self.y = Bidirectional(CuDNNLSTM(self.num_LSTM_units['Layer2'], return_sequences=True))(self.y) #check weight and bias initialization
        self.y = (LeakyReLU(name='LatentRep'))(self.y)
        self.y = (TimeDistributed(Dense(output_dim=50)))(self.y)  #check for weight and bias initialization
        self.y = (LeakyReLU())(self.y)
        self.y = Reshape((int(self.y.shape[2]), 1, int(self.y.shape[1])))(self.y)
        # self.y = Permute((2,1))(self.y)
        # self.y = Reshape(int(self.y.shape[1], int(self.y.shape[2])))(self.y)
        self.y = UpSampling2D(size=(1, 4), data_format=self.channel_order)(self.y)
        self.y = Conv2DTranspose(len(self.config.data_loader.channels_to_load),
                                 kernel_size=(1, self.config.model.kernel_size['Layer1']), name="Heat_map1",
                                 border_mode='same', data_format=self.channel_order)(self.y)
        # # softmax:
        # self.y = Dense(units=100, activation='sigmoid')(self.y)

        # compiling model
        self.model = Model(inputs=[self.x], outputs=[self.y])

        if self.num_gpus > 1:
            print('{} | [INFO] | Training with {} GPUs '.format(datetime.datetime.now(), self.num_gpus))
            parallel_model = multi_gpu_model(self.model, gpus=self.num_gpus, cpu_merge=False)
            parallel_model.compile(loss='mean_squared_error', metrics=['mse'],
                                        optimizer='Adam')
        else:
            self.model.compile(loss='mean_squared_error', metrics=['mse'],
                               optimizer='Adam')
        # load newest checkpoint weights if they exist from previous epochs
        # if os.listdir(self.config.callbacks.checkpoint_dir) != []:
        #     cps = [file for file in glob.glob( os.path.join(self.config.callbacks.checkpoint_dir, '*.hdf5'))]
        #     newest_cp = max(cps)
        #     self.model.load_weights(os.path.abspath(newest_cp))

        self.model.summary()
        return self.model

class ClusteringLayer_temporal(Layer):
    '''
        Temporal Clustering layer which converts latent space Z of input layer
        into a probability vector for each cluster defined by its centre in
        Z-space. Use Kullback-Leibler divergence as loss, with a probability
        target distribution.
        # Arguments
    0        input_dim: dimensionality of the input (integer).
                This argument (or alternatively, the keyword argument `input_shape`)
                is required when using this layer as the first layer in a model.
            weights: list of arrays to set as initial weights.
                The list should have 3 elements, of shape `(feature_length,timestep_length, output_dim)`
                and (output_dim,) for weights and biases respectively.
            alpha: parameter in Student's t-distribution. Default is 1.0.
        # Input shape
            3D tensor with shape: `(nb_features,nb_samples, input_dim)`.
        # Output shape
            2D tensor with shape: `(nb_samples, output_dim)`.
        '''
    def __init__(self, config, weights=None, **kwargs):
        self.output_dim = config.trainer.n_clusters
        self.input_dim = config.model.input_shape
        self.alpha = config.model.alpha
        self.initial_weights = weights
        self.input_spec = [InputSpec()]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusteringLayer_temporal, self).__init__(batch_size=config.data_loader.batch_size.train,
                                                       input_shape=config.model.input_shape)

    def build(self, input_shape):
        input_dim = input_shape[1:]
        self.input_spec = [InputSpec(dtype='float32',
                                     shape=(None,) + input_dim)]
        # self.W = K.variable(self.initial_weights)
        self.W = K.variable(np.random.normal(0, 0.01, (input_dim[1], input_dim[0], 1)))
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        """
        Main Forward pass call.
        1. Computes CID distances
        2. Uses Student t-distribution to get the probablities
        :return Probablities of input belongign to each of the cluster
        Note: Change the call here if you want to use another Metric!!
        """
        # Complexity of input variable x. Contains n_features * 1
        CE_X = tf.cast(K.expand_dims(K.sqrt(K.sum(K.square((x[:,1:,:]-x[:,:-1,:])), axis=1)),1),tf.float32) # eq. 3 in paper
        # Complexity of weight variable W contains total features*2 complexity terms. Because 2 centroids
        CE_W = K.sqrt(K.sum(K.square((self.W[:, 1:, :] - self.W[:, :-1, :])), axis=1))
        # Resultant Complexity Matrix
        CE_Mat = (K.maximum(CE_X, CE_W) / K.minimum(CE_X, CE_W)) # complexity factor (mentioned after eq. 2)
        # Multi variate Euclidian distance multiplied by the complexity term CID
        distance = K.sum(K.sqrt(K.sum(K.square(tf.cast(K.expand_dims(x, 1),tf.float32) - self.W), axis=2)))*CE_Mat  # eq. 2 ED(x,y)*CF(x,y)
        distance = K.sum(distance, axis=2)  # adding feature distances

        # calculating probability
        probability = 1.0 / (1.0 + distance ** 2 / self.alpha)
        probability = probability ** ((self.alpha + 1.0) / 2.0)
        probability = K.transpose(K.transpose(probability) / K.sum(probability, axis=1))
        return probability

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items))

    def cluster(self, X, model, y=None,
                tol=0.1, update_interval=None,
                test_x=None, test_y=None,
                save_interval=None,
                **kwargs):
        """
        The stop condition is removed here. This runs infinitely
        This is the Main Clustering routine. Does following things
        1. Initiaizes the main model i.e Autoencoder+ Clustering + HeatMap
        2. The jointly trains autoconder and clustering.
        3. Since there are no labels for clustering, we use KL loss to maximize confidence
        :param X: train X as a list of timeseries, example [TS1,TS2]
        :param model: Autoencoder Model
        :param y: Train_y labels as a 1D numpy array
        :param tol:
        :param update_interval:
        :param iter_max:
        :param test_x: Test X same structure as train X
        :param test_y: Test Y same structure as train Y
        :param save_interval:
        :param kwargs:
        :return:
        """
        batch_size = self.config.data_loader.batch_size
        iter_max = self.config.trainer.iter_max
        X_copy = np.copy(X[0])
        y_copy = np.copy(y)
        # X, y = unison_shuffled_copies(X, y)

        DEC, clust_layer = intialize_DEC_heat_map(self.config, model, X)

        train = True
        iteration, index = 0, 0
        accuracy = []
        y_pred_prev = None
        delta_label = 1

        while train:
            if iter_max < iteration:
                print('Reached maximum iteration limit. Stopping training.')
                return DEC.predict_generator(X, verbose=2)[0]  # , DEC.predict(test_x, verbose=0)[0]

            if True:  # iteration % update_interval == 0:
                # getting probabilities of cluster q and computing p(The true labels)
                q = DEC.predict_generator(X, verbose=2)[0]
                p = p_mat(q)
                y_pred = q[:, 1]

                # Test AUC routine
                if test_y is not None:
                    # test AUC routine
                    q_test = DEC.predict_generator(test_x, verbose=2)[0]
                    y_pred_test = q_test[:, 1]
                    if np.any(np.isnan(y_pred_test)):
                        d = 1
                    print(" Test AUC is " + str(roc_auc_score(test_y, y_pred_test)))

                # Stop Condition: Round to 3 digits the predictions and compare with prev predictions
                if y_pred_prev is not None:
                    delta_label = (
                            (np.asarray(np.round(y_pred, 2) != np.round(y_pred_prev, 2))).sum().astype(np.float32) /
                            y_pred.shape[0])
                    print("percent change is " + str(delta_label * 100))

                # Train AUC Routine
                if y is not None:
                    print(" Train AUC is " + str(roc_auc_score(y.astype(int), y_pred)))
                    acc = cluster_acc(y, q.argmax(1))[0]
                    accuracy.append(acc)
                    print('Iteration ' + str(iteration) + ', Accuracy ' + str(np.round(acc, 5)))
                else:
                    print(str(np.round(delta_label * 100, 5)) + '% change in label assignment')

                if delta_label + 1 < tol:  # (+1)
                    print('Reached tolerance threshold. Stopping training.')
                    train = False
                    continue

                y_pred_prev = y_pred
            for index, (data, _) in enumerate(X):
                # cutoff iteration
                # train on generator
                sys.stdout.write('Iteration %d, ' % iteration)
                loss = DEC.train_on_batch(data, [p[index * batch_size.train:(index + 1) * batch_size.train], data]) # error with more than 2 clusters
                # sys.stdout.write('Loss %f' % loss[0])

            iteration += 1
            sys.stdout.flush()
        return DEC.predict_generator(X, verbose=0)[0]

        # while train:
        #     for data, _ in X:
        #         # cutoff iteration
        #         if iter_max < iteration:
        #             print('Reached maximum iteration limit. Stopping training.')
        #             return DEC.predict_generator([X_copy], verbose=2)[0]  # , DEC.predict(test_x, verbose=0)[0]
        #
        #         if True:  # iteration % update_interval == 0:
        #             # getting probabilities of cluster q and computing p(The true labels)
        #             q = DEC.predict_generator(X, verbose=2)[0]
        #             p = p_mat(q)
        #             y_pred = q[:, 1]
        #
        #             # Test AUC routine
        #             if test_y is not None:
        #                 # test AUC routine
        #                 q_test = DEC.predict_generator(test_x, verbose=2)[0]
        #                 y_pred_test = q_test[:, 1]
        #                 if np.any(np.isnan(y_pred_test)):
        #                     d = 1
        #                 print(" Test AUC is " + str(roc_auc_score(test_y, y_pred_test)))
        #
        #             # Stop Condition: Round to 3 digits the predictions and compare with prev predictions
        #             if y_pred_prev is not None:
        #                 delta_label = (
        #                             (np.asarray(np.round(y_pred, 2) != np.round(y_pred_prev, 2))).sum().astype(np.float32) /
        #                             y_pred.shape[0])
        #                 print("percent change is " + str(delta_label * 100))
        #
        #             # Train AUC Routine
        #             if y is not None:
        #                 print(" Train AUC is " + str(roc_auc_score(y.astype(int), y_pred)))
        #                 acc = cluster_acc(y, q.argmax(1))[0]
        #                 accuracy.append(acc)
        #                 print('Iteration ' + str(iteration) + ', Accuracy ' + str(np.round(acc, 5)))
        #             else:
        #                 print(str(np.round(delta_label * 100, 5)) + '% change in label assignment')
        #
        #             if delta_label + 1 < tol:  # (+1)
        #                 print('Reached tolerance threshold. Stopping training.')
        #                 train = False
        #                 continue
        #
        #             y_pred_prev = y_pred
        #
        #         # train on batch
        #         sys.stdout.write('Iteration %d, ' % iteration)
        #         if (index + 1) * batch_size.train > sum(X.len_hypnograms):
        #             # loss = DEC.train_on_batch(
        #             #     [X[0][index * batch_size::]],
        #             #     [p[index * batch_size::], X[0][index * batch_size::]])
        #             index = 0
        #             # sys.stdout.write('Loss %f' % loss[0])
        #         else:
        #             loss = DEC.train_on_batch(data, [p[index * batch_size.train:(index + 1) * batch_size.train], data])
        #             sys.stdout.write('Loss %f' % loss[0])
        #             index += 1
        #
        #         iteration += 1
        #         sys.stdout.flush()
        # return DEC.predict_generator(X, verbose=0)[0]

def intialize_DEC_heat_map(config, model, X):
    """
    Initializes our Entire model including Autoencoder + (Clusteringlayer + HeatMap)
    The function does:
    1. Get centroids
    2. Init Cluster Layer
    3. Model Heat Map Layer
    4. Compile and return the model
    :param model: Auro Encoder Model
    :param X: The Training Data
    :return: Our Deep Temporal Clustering Model
    """
    n_clusters = config.trainer.n_clusters
    HeatMap = 0
    cluster_centres = None

    # We now Get Cluster centers. Give input X, get latent representation
    # Use CID + heirarchical clustering to get the cluster centroids
    temp_model = Model(inputs=model.input,outputs=model.get_layer("LatentRep").output)
    if cluster_centres is None:
        y_pred = temp_model.predict_generator(X, workers=config.trainer.num_workers, verbose=1)
        print('starting assigning labels to clusters')
        labels = myclust(y_pred)
        print('labels successfully assigned to clusters')
        cluster_centres = get_centroids(y_pred,labels,shape=y_pred.shape[1:])
        print('saving cluster parameters')
        np.save(os.path.join('/scratch/users/thomaslj/autogen_files', config.model.bash_name,
                             config.callbacks.checkpoint_dir, 'cluster_preds'), y_pred)
        np.save(os.path.join('/scratch/users/thomaslj/autogen_files', config.model.bash_name,
                             config.callbacks.checkpoint_dir, 'cluster_centres'), cluster_centres)
        np.save(os.path.join('/scratch/users/thomaslj/autogen_files', config.model.bash_name,
                             config.callbacks.checkpoint_dir, 'cluster_labels'), labels)

    # Use Centroids and create Clustering Layer
    clust_layer = ClusteringLayer_temporal(config, weights=cluster_centres,name='TemporalClustering')
    x0 = clust_layer(model.get_layer("LatentRep").output)

    if HeatMap is 1:
        # Up sample and decov and get heatMap
        x1 = Reshape((y_pred.shape[1],1,y_pred.shape[2]))(model.get_layer("LatentRep").output)
        x1 = UpSampling2D((5,1))(x1)
        x1 = Conv2DTranspose(1, (10,1), name="HeatMap", border_mode='same')(x1)
        x1 = (GlobalAveragePooling2D())(x1)
        x1 = Dense(2,activation='relu')(x1)
        x0 = Add()([x0, x1])

    DEC = Model(inputs=model.input,outputs=[x0,model.layers[-1].output])
    DEC.compile(loss=['kullback_leibler_divergence', 'mean_squared_error'], optimizer='adam', loss_weights=[1., 1.])
    return DEC,clust_layer

def my_multivariate_CID(Q, C,multi=1,shape=None):
    """
    Calcluates the temporal distance between two time series Q and C
    :param Q: Time series 1
    :param C: Time Series 2
    :param multi: 1 if multivariate and 0 if univariate
    :return: The Total Distance D
    """
    shape = CID_shape
    if multi:
        d=0
        Q = Q.reshape(shape)
        C = C.reshape(shape)
        for i in range(Q.shape[1]):
            d = d + my_multivariate_CID(Q[:,i], C[:,i],multi=0)
    else:
        CE_Q = np.sqrt(np.sum(np.square(np.diff(Q))))
        CE_C = np.sqrt(np.sum(np.square(np.diff(C))))
        d = np.sqrt(np.sum(np.square(Q - C))) *  (max(CE_Q, CE_C) / min(CE_Q, CE_C));
    return d

def myclust(latent,Y=None,clusters=2):
    """
    Given The latent representation, Clusters using CID- Complex invariant distance and cid
    :param latent: The latent representation
    :param Y: Labels
    :return: The predicted cluster labels
    """
    shape = latent.shape[1:]
    latent = latent.reshape(latent.shape[0],latent.shape[1]*latent.shape[2])
    global CID_shape
    CID_shape = shape
    Z = linkage(latent, 'complete', metric=my_multivariate_CID)
    k = clusters
    labels3 = fcluster(Z, k, criterion='maxclust')
    if Y:
        print(roc_auc_score(Y, labels3))

    return labels3
def get_centroids(latent,labels,shape):
    """
    Given the cluster labels and latent representation, gives the centroids of each cluster
    :param latent: The Latent Representation
    :param labels: Predicted Cluster Labels
    :return: The Centroids of each cluster
    """
    centroids=np.zeros((2,)+shape)
    idx = np.where(labels == 1)
    centroids[0,:,:] = np.mean(latent[idx], axis=0)
    idx = np.where(labels == 2)
    centroids[1,:,:] = np.mean(latent[idx], axis=0)
    return centroids

def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((int(D), int(D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w

def p_mat(q):
    weight = q**2/q.sum(0)
    return (weight.T / weight.sum(1)).T

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(a[0].shape[0])
    return [a[0][p], b[p]]




if __name__ == '__main__':
    config = process_config(r'C:\Users\dumle\OneDrive\Dokumenter\GitHub\stroke-thomas\config\param_configs.json')
    model = TemporalClustering(config)
    clust_layer = ClusteringLayer_temporal(config)
    # y = clust_layer.build(np.random.rand(2,1,37500).shape)
    y = clust_layer.build((2, 1, 37500))
    y2 = clust_layer.call(np.random.rand(2, 1, 37500))
    print(y2)

    print(clust_layer)
    # print_summary(model.model)