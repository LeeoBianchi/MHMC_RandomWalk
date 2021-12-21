import numpy as np
import pandas as pd
import time as t
import corner 
from sklearn import preprocessing
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy import special as sp

##########################################################
#############  MCMC  #####################################
##########################################################


#Gaussian with mu=0, sigma = 1
def phi(x):
    return 1/np.sqrt(2*np.pi) * np.exp(- x**2/(2))

#CDF of phi
def PHI(x):
    return 1/2*(1 + (sp.erf(x/np.sqrt(2))))

#Beta = vector of coeffs, Xs = design matrix
def log_like(Beta, ys, Xs):
    PHIs = PHI(Beta @ np.transpose(Xs))
    PHIs = np.where(PHIs > 1.0e-30, PHIs, 1.0e-30) #should avoid underflow problems in the log
    log_ps = np.log(PHIs) #probabilities P(y=0)
    log_l = np.sum(ys*log_ps) + np.sum((1-ys)*(1 - log_ps))
    return log_l

#Returns the sampling of the posterior of the parammeters 
def random_walk(Beta_init, ys, Xs):
    N_coeff = 6
    #Initialization
    Beta = Beta_init
    Beta_s = [] #Output matrix, will be N_it x N_coeff
    sigmas = np.array([1.9,1,1.1,1.1,1,10])*0.1 #std. dev. of proposal distribution
    acc = 0
    N_it = 100000
    for i in range(N_it):
        for j in range(6):
            Beta_st = np.array(Beta)
            Beta_st[j] = Beta[j] + np.random.randn(1)*sigmas[j] #proposal for Beta*
            R = np.exp(log_like(Beta_st, ys, Xs) - log_like(Beta, ys, Xs)) #acceptance ratio
            if np.random.uniform() < R:
                Beta = Beta_st
                acc += 1
        Beta_s.append(Beta)
    print('END, acceptance ratio = %.3f'%(acc/(N_it*6)))
    return np.array(Beta_s)

#G-R convergence analysis
def Gelman_Rubin(Xs, Ls, D):
    Rs = []
    for i,L in enumerate(Ls):
        Xs_cut = np.array(Xs[D:D+L,:])
        W = np.average(np.var(Xs_cut, axis=0))
        B = np.var(np.average(Xs_cut, axis=0))
        Rs.append(((L-1)/L*W + B/L)/W)
    return Rs

def Multiplot (dat, name = ''):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
    fig.subplots_adjust(hspace=0.8, wspace=0.5)
    fig.suptitle("Parameters' random walks traces: "+name, fontsize=20)
    axs[0,0].plot(range(len(dat[:,0])), dat[:,0],c='royalblue')
    axs[0,0].set_title('0 - age')
    axs[1,0].plot(range(len(dat[:,1])), dat[:,1],c='royalblue')
    axs[1,0].set_title('1 - netuse')
    axs[0,1].plot(range(len(dat[:,2])), dat[:,2],c='royalblue')
    axs[0,1].set_title('2 - treated')
    axs[1,1].plot(range(len(dat[:,3])), dat[:,3],c='royalblue')
    axs[1,1].set_title('3 - green')
    axs[0,2].plot(range(len(dat[:,4])), dat[:,4],c='royalblue')
    axs[0,2].set_title('4 - phc')
    axs[1,2].plot(range(len(dat[:,5])), dat[:,5],c='royalblue')
    axs[1,2].set_title('5 - const')
    
def Multiplot_acorr (dat, name = ''):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
    fig.subplots_adjust(hspace=0.8, wspace=0.5)
    fig.suptitle('Autocorrelations: '+name, fontsize=20)
    axs[0,0].acorr(dat[:,0], maxlags = 100)
    axs[0,0].set_title('0 - age')
    axs[1,0].acorr(dat[:,1], maxlags = 100)
    axs[1,0].set_title('1 - netuse')
    axs[0,1].acorr(dat[:,2], maxlags = 100)
    axs[0,1].set_title('2 - treated')
    axs[1,1].acorr(dat[:,3], maxlags = 100)
    axs[1,1].set_title('3 - green')
    axs[0,2].acorr(dat[:,4], maxlags = 100)
    axs[0,2].set_title('4 - phc')
    axs[1,2].acorr(dat[:,5], maxlags = 100)
    axs[1,2].set_title('5 - const')