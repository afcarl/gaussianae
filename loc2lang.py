'''
Created on 21 Feb 2017
Given a training set of lat/lon as input and probability distribution over words as output,
train a model that can predict words based on location.
then try to visualise borders and regions (e.g. try many lat/lon as input and get the probability of word yinz
in the output and visualise that).
@author: af
'''
from __future__ import division
import matplotlib as mpl
import re
from itertools import product
from sklearn.model_selection._split import train_test_split
mpl.use('Agg')
import matplotlib.mlab as mlab
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.mlab import griddata
from matplotlib.patches import Polygon as MplPolygon
#import seaborn as sns
#sns.set(style="white")
import operator
from scipy.stats import multivariate_normal
import argparse
import sys
from scipy.spatial import ConvexHull
import os
import pdb
import random
import numpy as np
import sys
from os import path
import scipy as sp
import theano
import shutil
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer
import logging
import json
import codecs
import pickle
import gzip
from collections import OrderedDict, Counter
from sklearn.preprocessing import normalize
from haversine import haversine
from _collections import defaultdict
from scipy import stats
from mpl_toolkits.basemap import Basemap, cm, maskoceans
from scipy.interpolate import griddata as gd
from lasagne_layers import SparseInputDenseLayer, GaussianRBFLayer, DiagonalBivariateGaussianLayer, BivariateGaussianLayer,BivariateGaussianLayerWithPi,MDNSharedParams
from shapely.geometry import MultiPoint, Point, Polygon
import shapefile
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans, MiniBatchKMeans

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)   
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
np.random.seed(77)
random.seed(77)



def geo_latlon_eval(latlon_true, latlon_pred):
    distances = []
    for i in range(0, len(latlon_true)):
        lat_true, lon_true = latlon_true[i]
        lat_pred, lon_pred = latlon_pred[i]
        distance = haversine((lat_true, lon_true), (lat_pred, lon_pred))
        distances.append(distance)
    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
    return np.mean(distances), np.median(distances), acc_at_161




def inspect_inputs(i, node, fn):
    print(i, node, "input(s) shape(s):", [input[0].shape for input in fn.inputs])
    #print(i, node, "input(s) stride(s):", [input.strides for input in fn.inputs], end='')

def inspect_outputs(i, node, fn):
    print(" output(s) shape(s):", [output[0].shape for output in fn.outputs])
    #print(" output(s) stride(s):", [output.strides for output in fn.outputs])    

def softplus(x):
    return np.log(np.exp(x) + 1)
def softsign(x):
    return x / (1 + np.abs(x))

class NNModel():
    def __init__(self, 
                 n_epochs=10, 
                 batch_size=1000, 
                 input_size=None,
                 output_size = None, 
                 hid_size=500, 
                 drop_out=False, 
                 dropout_coef=0.5,
                 early_stopping_max_down=10,
                 dtype='float32',
                 n_gaus_comp=500,
                 mus=None,
                 sigmas=None,
                 corxy=None):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping_max_down = early_stopping_max_down
        self.dtype = dtype
        self.input_size = input_size
        self.output_size = output_size
        self.reload = reload
        self.n_gaus_comp = n_gaus_comp
        self.mus = mus
        self.sigmas = sigmas
        self.corxy = corxy

        
    def build(self):

        self.X_sym = T.matrix()
        self.Y_sym = T.matrix()

        l_in = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)
        logging.info('adding %d-comp bivariate gaussian layer...' %self.n_gaus_comp)
        self.l_gaus = MDNSharedParams(l_in, num_units=self.n_gaus_comp, mus=self.mus, sigmas=self.sigmas, corxy=self.corxy,
                                      nonlinearity=lasagne.nonlinearities.softmax,
                                      W=lasagne.init.GlorotUniform())

        
        pis = lasagne.layers.get_output(self.l_gaus, self.X_sym)
        mus, sigmas, corxy = self.l_gaus.mus, self.l_gaus.sigmas, self.l_gaus.corxy
        sigmas = T.nnet.softplus(sigmas) 
        corxy = T.nnet.nnet.softsign(corxy)
        nll_loss = self.nll_loss_sharedparams(mus, sigmas, corxy, pis, self.Y_sym)
        
        mus_pred = self.pred_sharedparams_sym(mus, sigmas, corxy, pis)
        sq_loss = lasagne.objectives.squared_error(mus_pred, self.Y_sym).mean()
        
        regul_loss = lasagne.regularization.regularize_network_params(self.l_gaus, penalty=lasagne.regularization.l2)
        
        #entropy_loss = lasagne.objectives.categorical_crossentropy(pis, pis).mean()
        
        sq_loss_coef = 0.0
        entropy_loss_coef = 1.0
        #enforce sigmas to be lower than k
        k = 4
        #enforce sigma to be smaller than k
        sigma_constrain_term = T.sum((((sigmas + k) + T.abs_(sigmas - k))/2.0) ** 2)
        #enforce sigma to be normal with mu=k
        #sigma_constrain_term = T.sum((sigmas - k)**2)
        
        loss = (1 - sq_loss_coef) * nll_loss + sq_loss * sq_loss_coef + entropy_loss_coef * sigma_constrain_term #+ 0.0001 * regul_loss
        
        parameters = lasagne.layers.get_all_params(self.l_gaus, trainable=True)
        updates = lasagne.updates.adam(loss, parameters, learning_rate=2e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.f_train = theano.function([self.X_sym, self.Y_sym], loss, updates=updates, on_unused_input='warn')#,  mode=theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs))
        self.f_val = theano.function([self.X_sym, self.Y_sym], loss, on_unused_input='warn')
        self.f_predict = theano.function([self.X_sym], [mus, sigmas, corxy, pis], on_unused_input='warn')


    def pred_sharedparams(self, mus, sigmas, corxy, pis, prediction_method='mixture'):
        '''
        select mus that maximize \sum_{pi_i * prob_i(mu)} if prediction_method is mixture
        else
        select the component with highest pi if prediction_method is pi.
        '''
        if prediction_method == 'mixture':
            X = mus[:, np.newaxis, :]
            diff = X - mus
            diffprod = np.prod(diff, axis=-1)
            sigmainvs = 1.0 / sigmas
            sigmainvprods = sigmainvs[:, 0] * sigmainvs[:, 1]
            sigmas2 = sigmas ** 2
            corxy2 = corxy **2
            diff2 = diff ** 2
            diffsigma = diff2 / sigmas2
            diffsigmanorm = np.sum(diffsigma, axis=-1)
            z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
            oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
            term = -0.5 * z * oneminuscorxy2inv
            expterm = np.exp(term)
            probs = (0.5 / np.pi) * sigmainvprods * np.sqrt(oneminuscorxy2inv) * expterm
            piprobs = pis[:, np.newaxis, :] * probs
            piprobsum = np.sum(piprobs, axis=-1)
            preds = np.argmax(piprobsum, axis=1)
            selected_mus = mus[preds, :]
     
            return selected_mus
        elif prediction_method == 'pi':
            logging.info('only pis are used for prediction')
            preds = np.argmax(pis, axis=1)
            selected_mus = mus[preds, :]      
            return selected_mus
        else:
            raise('%s is not a valid prediction method' %prediction_method)

    def pred_sharedparams_sym(self, mus, sigmas, corxy, pis, prediction_method='mixture'):
        '''
        select mus that maximize \sum_{pi_i * prob_i(mu)} if prediction_method is mixture
        else
        select the component with highest pi if prediction_method is pi.
        '''
        if prediction_method == 'mixture':
            X = mus[:, np.newaxis, :]
            diff = X - mus
            diffprod = T.prod(diff, axis=-1)
            sigmainvs = 1.0 / sigmas
            sigmainvprods = sigmainvs[:, 0] * sigmainvs[:, 1]
            sigmas2 = sigmas ** 2
            corxy2 = corxy **2
            diff2 = diff ** 2
            diffsigma = diff2 / sigmas2
            diffsigmanorm = T.sum(diffsigma, axis=-1)
            z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
            oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
            term = -0.5 * z * oneminuscorxy2inv
            expterm = T.exp(term)
            probs = (0.5 / np.pi) * sigmainvprods * T.sqrt(oneminuscorxy2inv) * expterm
            piprobs = pis[:, np.newaxis, :] * probs
            piprobsum = T.sum(piprobs, axis=-1)
            preds = T.argmax(piprobsum, axis=1)
            selected_mus = mus[preds, :]
     
            return selected_mus
        elif prediction_method == 'pi':
            logging.info('only pis are used for prediction')
            preds = T.argmax(pis, axis=1)
            selected_mus = mus[preds, :]      
            return selected_mus
        else:
            raise('%s is not a valid prediction method' %prediction_method)

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.l_gaus, params)

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]   

    def nll_loss_sharedparams(self, mus, sigmas, corxy, pis, y_true):
        mus_ex = mus[np.newaxis, :, :]
        X = y_true[:, np.newaxis, :]
        diff = X - mus_ex
        diffprod = T.prod(diff, axis=-1)
        corxy2 = corxy **2
        diff2 = diff ** 2
        sigmas2 = sigmas ** 2
        sigmainvs = 1.0 / sigmas
        sigmainvprods = sigmainvs[:, 0] * sigmainvs[:, 1]
        diffsigma = diff2 / sigmas2
        diffsigmanorm = T.sum(diffsigma, axis=-1)
        z = diffsigmanorm - 2 * corxy * diffprod * sigmainvprods
        oneminuscorxy2inv = 1.0 / (1.0 - corxy2)
        expterm = -0.5 * z * oneminuscorxy2inv
        new_exponent = T.log(0.5/np.pi) + T.log(sigmainvprods) + T.log(np.sqrt(oneminuscorxy2inv)) + expterm + T.log(pis)
        max_exponent = T.max(new_exponent ,axis=1, keepdims=True)
        mod_exponent = new_exponent - max_exponent
        gauss_mix = T.sum(T.exp(mod_exponent),axis=1)
        log_gauss = max_exponent + T.log(gauss_mix)
        loss = -T.mean(log_gauss)
        return loss    
    
    def fit(self, X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
        logging.info('training with %d n_epochs and  %d batch_size' %(self.n_epochs, self.batch_size))
        best_params = None
        best_val_loss = sys.maxint
        n_validation_down = 0
        
        for step in range(self.n_epochs):
            l_trains = []
            for batch in self.iterate_minibatches(X_train, Y_train, self.batch_size, shuffle=True):
                x_batch, y_batch = batch
                if sp.sparse.issparse(y_batch): y_batch = y_batch.todense().astype('float32')
                l_train = self.f_train(x_batch, y_batch)
                l_trains.append(l_train)
            l_train = np.mean(l_trains)
            l_val = self.f_val(X_dev, Y_dev)
            if l_val < best_val_loss:
                best_params = lasagne.layers.get_all_param_values(self.l_gaus)
                best_val_loss = l_val

                #logging.info('first mu (%f,%f) first covar (%f, %f, %f)' %(best_params[0][0, 0], best_params[0][0, 1], softplus(best_params[1][0, 0]), softplus(best_params[1][0, 1]), softsign(best_params[2][0])))
                #logging.info('second mu (%f,%f) second covar (%f, %f, %f)' %(best_params[0][1, 0], best_params[0][1, 1], softplus(best_params[1][1, 0]), softplus(best_params[1][1, 1]), softsign(best_params[2][1])))
                n_validation_down = 0
            else:
                n_validation_down += 1
                if n_validation_down > self.early_stopping_max_down:
                    logging.info('validation results went down. early stopping ...')
                    break

            logging.info('iter %d, train loss %f, dev loss %f, best dev loss %f, num_down %d' %(step, l_train, l_val, best_val_loss, n_validation_down))
            mus_eval, sigmas_eval, corxy_eval, pis_eval = self.f_predict(X_dev[0:1, :])

            draw_clusters(X_train, mus_eval, sigmas_eval, corxy_eval, iter=step)
            logging.info(np.asarray(mus_eval))
        
        lasagne.layers.set_all_param_values(self.l_gaus, best_params)    
        self.best_params = best_params   
        l_test = self.f_val(X_test, Y_test)
        l_dev = self.f_val(X_dev, Y_dev)
        logging.info('dev loss is %f and test loss is %f' %(l_dev, l_test))

                
    def predict(self, X):
        mus_eval, sigmas_eval, corxy_eval, pis_eval = self.f_predict(X)
        mus_eval, sigmas_eval, corxy_eval, pis_eval = np.asarray(mus_eval), np.asarray(sigmas_eval), np.asarray(corxy_eval), np.asarray(pis_eval)
        selected_mus = self.pred_sharedparams(mus_eval, sigmas_eval, corxy_eval, pis_eval)
        return selected_mus 

 
def draw_clusters(points, mus, sigmas, corxys, output_type='png', iter=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)

    numcols, numrows = 1000, 1000
    X = np.linspace(points[:, 0].min()-5, points[:, 0].max() + 5, numcols)
    Y = np.linspace(points[:, 1].min()-5, points[:, 1].max() + 5, numrows)

    X, Y = np.meshgrid(X, Y)

    Z = 0
    for k in xrange(mus.shape[0]):
        #here x is longitude and y is latitude
        #apply softplus to sigmas (to make them positive)
        sigmax=sigmas[k, 0]
        sigmay=sigmas[k, 1]
        mux=mus[k, 0]
        muy=mus[k, 1]
        corxy = corxys[k]
        #now given corxy find sigmaxy
        sigmaxy = corxy * sigmax * sigmay
        Z += mlab.bivariate_normal(X, Y, sigmax=sigmax, sigmay=sigmay, mux=mux, muy=muy, sigmaxy=sigmaxy)
    con = ax.contourf(X, Y, Z)

    for k in xrange(mus.shape[0]):
        #here x is longitude and y is latitude
        #apply softplus to sigmas (to make them positive)
        sigmax=sigmas[k, 0]
        sigmay=sigmas[k, 1]
        mux=mus[k, 0]
        muy=mus[k, 1]
        corxy = corxys[k]
        #now given corxy find sigmaxy
        sigmaxy = corxy * sigmax * sigmay
        Z = mlab.bivariate_normal(X, Y, sigmax=sigmax, sigmay=sigmay, mux=mux, muy=muy, sigmaxy=sigmaxy)
        con = ax.contour(X, Y, Z, levels=[0.001], linewidths=0.5, colors='darkorange', antialiased=True)

        contour_labels = False
        if contour_labels:
            plt.clabel(con, [con.levels[-1]], inline=True, fontsize=10)
        
    ax.scatter(points[:, 0], points[:, 1], s=0.2, c='black')
    ax.scatter(mus[:, 0], mus[:, 1], s=0.4, c='red')     
    
    iter = str(iter).zfill(4)

    plt.tight_layout()
    plt.savefig('./video/map_' + iter  + '.' + output_type, frameon=False, dpi=200)
    plt.close()
def toy_data(n_samples=1000):

    
    
    # generate spherical data centered on (20, 20)
    shifted_gaussian1 = np.random.randn(n_samples, 2) + np.array([10, 10])
    shifted_gaussian2 = np.random.randn(n_samples, 2) + np.array([-10, -10])
    shifted_gaussian3 = np.random.randn(n_samples, 2) + np.array([10, -10])
    shifted_gaussian4 = np.random.randn(n_samples, 2) + np.array([-10, 10])
    #shifted_gaussian5 = np.random.randn(n_samples, 2) + np.array([0, 0])
    
    # generate zero centered stretched Gaussian data
    #C = np.array([[0., -0.7], [3.5, .7]])
    #stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
    
    # concatenate the two datasets into the final training set
    X_train = np.vstack([shifted_gaussian1, shifted_gaussian2, shifted_gaussian3, shifted_gaussian4]).astype('float32')
    return X_train
     

def train(args, **kwargs):
    n_gaus_comp = args.ncomp
    kmeans_mu = kwargs.get('kmeans', False)
    X_train = toy_data(n_samples=10000)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, X_train, test_size=0.2, random_state=1)
    X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

    input_size = X_train.shape[1]
    output_size = X_train.shape[1]
    batch_size = 1000
    
    
    mus = np.random.randn(n_gaus_comp, 2).astype('float32')
    #mus = X_train[0:n_gaus_comp]
    raw_stds = None
    raw_cors = None
    
    model = NNModel(n_epochs=100000, batch_size=batch_size,
                    input_size=input_size, output_size=output_size,
                    early_stopping_max_down=10, 
                    n_gaus_comp=n_gaus_comp, mus=mus, 
                    sigmas=raw_stds, corxy=raw_cors)
    model.build()
    model.fit(X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
    mus_eval, sigmas_eval, corxy_eval, pis_eval = model.f_predict(X_dev)
    mus_eval, sigmas_eval, corxy_eval, pis_eval = np.asarray(mus_eval), np.asarray(sigmas_eval), np.asarray(corxy_eval), np.asarray(pis_eval)
    logging.info(mus_eval)
    logging.info(sigmas_eval)
    pdb.set_trace()

   



   
def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument( '-i','--dataset', metavar='str',  help='dataset for dialectology',  type=str, default='na')
    parser.add_argument( '-bucket','--bucket', metavar='int',  help='discretisation bucket size',  type=int, default=300)
    parser.add_argument( '-batch','--batch', metavar='int',  help='SGD batch size',  type=int, default=500)
    parser.add_argument( '-hid','--hidden', metavar='int',  help='Hidden layer size after bigaus layer',  type=int, default=500)
    parser.add_argument( '-mindf','--mindf', metavar='int',  help='minimum document frequency in BoW',  type=int, default=10)
    parser.add_argument( '-d','--dir', metavar='str',  help='home directory',  type=str, default='./data')
    parser.add_argument( '-enc','--encoding', metavar='str',  help='Data Encoding (e.g. latin1, utf-8)',  type=str, default='utf-8')
    parser.add_argument( '-reg','--regularization', metavar='float',  help='regularization coefficient)',  type=float, default=1e-6)
    parser.add_argument( '-drop','--dropout', metavar='float',  help='dropout coef default 0.5',  type=float, default=0.5)
    parser.add_argument( '-cel','--celebrity', metavar='int',  help='celebrity threshold',  type=int, default=10)
    parser.add_argument( '-conv', '--convolution', action='store_true',  help='if true do convolution')
    parser.add_argument( '-map', '--map', action='store_true',  help='if true just draw maps from pre-trained model')
    parser.add_argument( '-tune', '--tune', action='store_true',  help='if true tune the hyper-parameters') 
    parser.add_argument( '-tf', '--tensorflow', action='store_true',  help='if exists run with tensorflow') 
    parser.add_argument( '-autoencoder', '--autoencoder', type=int,  help='the number of autoencoder steps before training', default=0) 
    parser.add_argument( '-grid', '--grid', action='store_true',  help='if exists transforms the input from lat/lon to distance from grids on map')  
    parser.add_argument( '-ncomp', type=int,  help='the number of bivariate gaussians after the input layer', default=4) 
    parser.add_argument( '-m', '--message', type=str) 
    parser.add_argument( '-vbi', '--vbi', type=str,  help='if exists load params from vbi file and visualize bivariate gaussians on a map', default=None)
    parser.add_argument( '-nomdn', '--nomdn', action='store_true',  help='if true use tanh layer instead of MDN') 
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    #THEANO_FLAGS='device=cpu' nice -n 10 python loc2lang.py -d ~/datasets/na/processed_data/ -enc utf-8 -reg 0 -drop 0.0 -mindf 200 -hid 1000 -ncomp 100 -autoencoder 100 -map
    args = parse_args(sys.argv[1:])
    train(args)

