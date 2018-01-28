import time
import pickle
import random
random.seed(1234)
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing

# from gurobi import kMIQP
from basic import kMIQP

def jaccard_sim(a, b):
  aa = set(a)
  bb = set(b)
  return len(aa & bb) / len(aa | bb)

with open('res_ele_50.pkl', 'rb') as f:
  res = pickle.load(f)
  

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


def _watch_log():
  for one_tuple in res:
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=10, outputFlag=True)
    print(vx, max_res)
    break


def _watch_time_cost():
  start_time = time.time()
  T = 20
  for one_tuple in res[:T]:
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=10)
    print(vx, max_res)
  cost_time = time.time() - start_time
  print('User Per Second: %.4fs' % (cost_time / T))


def _watch_converge():
  random.shuffle(res)
  tqdmInput = tqdm(res, ncols=77, leave=True)
  corrs, jaccs = 0.0, 0.0
  for iter, one_tuple in enumerate(tqdmInput):
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=10)

    groundtruth, preds, scores = one_tuple
    preds = [preds[x] for x in vx]

    corr = 0.0
    grou = set(groundtruth)
    for pred in preds:
      corr = max(corr, len(set(pred).intersection(grou)) / 2)
    assert(corr == 0.0 or corr == 0.5 or corr == 1.0)
    corrs += corr

    jacc = 0.0
    for i in range(len(preds)):
      for j in range(i+1, len(preds)):
        jacc += jaccard_sim(preds[i], preds[j])
    jacc /= len(preds) * (len(preds) - 1) / 2
    jaccs += jacc

    tqdmInput.set_description('Prec@10: %.3f%% Div: %.3f%%'
        % (corrs*100/(iter+1), jaccs*100/(iter+1)))


def reduce_by_kMIQP(save_path):
  outputs = []
  for one_tuple in tqdm(res, ncols=77):
    r, M = _preprocess(one_tuple)
    vx, max_res = kMIQP(r, M, lamb=1.0, k=10)
    groundtruth, preds, scores = one_tuple
    preds = [preds[x] for x in vx]
    outputs.append((groundtruth, preds, max_res))
  with open(save_path, 'wb') as f:
    pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  # _watch_log()
  # _watch_time_cost()
  _watch_converge()
  # reduce_by_kMIQP('res_ele_10_MIQP.pkl')
