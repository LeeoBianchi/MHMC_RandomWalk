import multiprocessing as mp
import time as t
import numpy as np
import pandas as pd
from sklearn import preprocessing
import MC

#load the data
Names = ["x", "y", "pos", "age", "netuse", "treated", "green", "phc"]
DF = pd.read_csv('gambia.ascii', sep = '\t', names = Names)
DF['const'] = np.repeat(1, len(DF)) #we introduce the ones-column to introduce the constant term.
ys = np.array(DF["pos"])
Xs = np.array(DF[["age", "netuse", "treated", "green", "phc", "const"]])
Xs = preprocessing.scale(Xs) #standardizing design matrix

#worker function recalling the algorithm
def worker (Ns):
    print('Chain %.f starting'%Ns)
    #random initialization around the starting point I liked
    Betas = MC.random_walk(np.array([48.99, -14.45, -15.08 ,  18.99, -3.14,
                                     -70.67])+np.random.randn(6)*np.array([1,1,1,1,1,5])*0.01, ys, Xs)
    np.savetxt('mhmc'+str(Ns)+'.gz', Betas)
    print('Chain %.f finished'%Ns)

if __name__ == '__main__':
    print('START')
    t1 = t.time()
    #N_p = mp.cpu_count() #as many processes as available on the machine
    N_p = 6 #arbitrary number
    Ns = np.array(range(N_p))
    pool = mp.Pool(processes=N_p) 
    res = pool.map(worker, Ns)
    t2 = t.time()
    print('END, time taken = '+str((t2-t1)/60)+' minutes')