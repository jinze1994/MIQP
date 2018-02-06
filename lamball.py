import time
import pickle
import random
random.seed(1234)
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import matplotlib.pyplot as plt
from metrics import jaccard, precision, jaccard_sim, print_all_metrics
from basic import kMIQP as basic_kMIQP
from greedy import kMIQP as greedy_kMIQP
from gurobi import kMIQP as gurobi_kMIQP

 
min_max_scaler = preprocessing.MinMaxScaler()
def _preprocess(one_tuple):
  groundtruth, preds, scores = one_tuple
  # scores = np.exp(scores)
  # scores = min_max_scaler.fit_transform(np.reshape(scores, [-1, 1]))
  # scores = scores.flatten()
  M = []
  for i in range(len(preds)):
    for j in range(i+1, len(preds)):
      M.append((i, j, jaccard_sim(preds[i], preds[j])))
  return scores, M


def _watch_log(res):
  for one_tuple in res:
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=10, outputFlag=True)
    print(vx, max_res)
    break


def _watch_time_cost(res):
  start_time = time.time()
  T = 20
  for one_tuple in res[:T]:
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=10)
    print(vx, max_res)
  cost_time = time.time() - start_time
  print('User Per Second: %.4fs' % (cost_time / T))


def _watch_converge(res):
  random.shuffle(res)
  tqdmInput = tqdm(res, ncols=77, leave=True)
  prec, jacc = 0.0, 0.0
  for iter, one_tuple in enumerate(tqdmInput):
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=10)

    groundtruth, preds, scores = one_tuple
    preds = [preds[x] for x in vx]

    prec += precision(groundtruth, preds)
    jacc += jaccard(preds)

    tqdmInput.set_description('Prec@10: %.3f%% Div: %.3f'
        % (prec*100/(iter+1), jacc/(iter+1)))

def reduce_by_kMIQP(algoname,res, source_file, save_path=None):
    outputs = []
    k = 10
    pd=[]
    div=[]
    xa=[]
    for li in np.arange(0,1,0.05):
        lamb=li
        xa.append(lamb)
        for one_tuple in res:
            r, M = _preprocess(one_tuple)
            if algoname=='gurobi':
                vx, max_res = gurobi_kMIQP(r, M, lamb, k=k)
            elif algoname=='greedy':
                vx, max_res = greedy_kMIQP(r, M, lamb, k=k)
            else:
                vx, max_res = basic_kMIQP(r, M, lamb, k=k)


            groundtruth, preds, scores = one_tuple
            preds = [preds[x] for x in vx]
            outputs.append((groundtruth, preds, max_res))
        if save_path is None:
            prec, jacc = 0.0, 0.0
            for groundtruth, preds, scores in outputs:
                jacc += jaccard(preds)
                prec += precision(groundtruth, preds)
            pd.append(prec*100/len(outputs))
            div.append(jacc/len(outputs))
        else:
            with open(save_path, 'wb') as f:
                pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)
    return xa,pd,div
  



if __name__ == '__main__':
  # source_file = 'res_steam_50.pkl'
  #source_file = 'res_ele_50.pkl'
  source_file = 'res_clo_50.pkl'
  target_file = None
  # target_file = 'res_ele_05_MIQP.pkl'
  with open(source_file, 'rb') as f:
    res = pickle.load(f)
 
  # _watch_log(res)
  # _watch_time_cost(res)
  # _watch_converge(res)
  xa,greedypd,greedydiv=reduce_by_kMIQP('greedy',res, source_file, target_file)
  xa,gurobipd,gurobidiv=reduce_by_kMIQP('gurobi',res, source_file, target_file)
  xa,basicpd,basicdiv=reduce_by_kMIQP('basic',res, source_file, target_file)
  
  k=10


  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  freqpd=[0.0955]*20
  freqdiv=[0.0593]*20
  l1,=ax1.plot(xa,freqpd,label='freq-p@'+str(k),color='r',linestyle='--')
  l2,=ax1.plot(xa,basicpd,label='basic-p@'+str(k),color='r',marker='+') 
  l3,=ax1.plot(xa,greedypd,label='greedy-p@'+str(k),color='r',marker='o')
  l4,=ax1.plot(xa,gurobipd,label='gurobi-p@'+str(k),color='r',marker='*') 
  
   
  
  ax1.set_ylabel('p@'+str(k))
  ax2 = ax1.twinx()
  l5, =ax2.plot(xa, freqdiv, label = "freq-diversity",color='g',linestyle='--')
  l6, =ax2.plot(xa, basicdiv, label = "basic-diversity",color='g',marker='+')
  l7, =ax2.plot(xa, greedydiv, label = "greedy-diversity",color='g',marker='o')
  l8, =ax2.plot(xa, gurobidiv, label = "gurobi-diversity",color='g',marker='*')
  
  
  
  ax2.set_ylabel('diversity')
  ax1.set_xlabel('lamb(clo)')
  
  #my_xticks=np.arange(0,100,10)
  #plt.xticks(my_xticks)
  plt.legend(handles = [l1, l3,l4,l5,l7,l8,], loc = 'best')
  plt.show()
