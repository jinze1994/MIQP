import time
import pickle
import random
random.seed(1234)
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import matplotlib.pyplot as plt



if __name__ == '__main__':
    filename='new_ali_k=10_0to1.pkl'
    with open(filename, 'rb') as f:
        greedy_p = pickle.load(f)
        greedy_div=pickle.load(f)
        gurobi_p=pickle.load(f)
        gurobi_div=pickle.load(f)
    k=10
    print(greedy_p)
    print(greedy_div)
    print(gurobi_p)
    print(gurobi_div)
    #xa=np.linspace(-0.020,-0.0145,20)
    #xa=xa.tolist()

    new_greedy_div=[-1*i for i in greedy_div]
    new_gurobi_div=[-1*i for i in gurobi_div]
    #freq_p=[5.8306]*20
    freq_p=0.1920
    freq_div=-0.4
    rank_p=1.0495
    rank_div=-0.0181
    fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    l1,=plt.plot(new_greedy_div,greedy_p,label='greedy-p@'+str(k),color='r',marker='.')
    l2,=plt.plot(new_gurobi_div,gurobi_p,label='gurobi-p@'+str(k),color='g',marker='.') 
    #l3,=ax1.plot(freq_div, freq_p, marker = 'o', color = 'b', label='freq-p@'+str(k),markersize=8,markerfacecolor='none')
    l4,=plt.plot(rank_div, rank_p, marker = '^', color = 'k', label='rank-p@'+str(k),markersize=8,markerfacecolor='none')
    #l6,=ax1.plot(xa,freq_p,label='freq-p@'+str(k),color='r',linestyle='--')
    
    plt.ylabel('p@'+str(k))
    plt.xlabel('div')
    plt.legend(handles = [l1,l2,l4], loc = 'best')
    plt.show()

  
  


  