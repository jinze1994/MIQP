import os
import sys
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def jaccard_sim(a, b):
  aa = set(a)
  bb = set(b)
  aa, bb = aa&bb, aa|bb
  if len(bb) == 0: return 0.0
  return len(aa & bb) / len(aa | bb)

def jaccard(preds):
  jac = 0.0
  for i in range(len(preds)):
    for j in range(i+1, len(preds)):
      jac += jaccard_sim(preds[i], preds[j])
  jac /= len(preds) * (len(preds) - 1) / 2
  return jac

def precision(groundtruth, preds):
  prec = 0.0
  if not isinstance(groundtruth[0], list):
    groundtruth = [groundtruth]
  for grou in groundtruth:
    prec_part = 0.0
    for pred in preds:
      # prec_part = max(prec_part, jaccard_sim(pred, grou))
      prec_part += jaccard_sim(pred, grou)
    prec_part /= len(preds)
    prec += prec_part
  prec /= len(groundtruth)
  return prec

def soft_sim(list1, list2, M):
  return cosine_similarity(M[list1], M[list2]).mean()

def soft_precision(groundtruth, preds, M):
  if M is None: return 0.0
  prec = 0.0
  if not isinstance(groundtruth[0], list):
    groundtruth = [groundtruth]
  for grou in groundtruth:
    prec_part = 0.0
    for pred in preds:
      prec_part += soft_sim(pred, grou, M)
    prec_part /= len(preds)
    prec += prec_part
  prec /= len(groundtruth)
  return prec

def fatch_M(flag):
  if flag=='clo':
    M_path = '../../data/node2vec/emb/' + 'clo_emb.pickle'
  elif flag=='ele':
    M_path = '../../data/node2vec/emb/' + 'ele_emb.pickle'
  else:
    assert(False)

  if not os.path.exists(M_path):
    return None
  with open(M_path, "r") as f:
    lines = f.readlines()[1:]
    lines = [[float(word) for word in line.strip().split(" ")[1:]] for line in lines]
    M = np.array(lines)
  return M

def print_all_metrics(flag, res, k=10, M=None):

  total = len(res)
  soft_prec, prec, jacc = 0.0, 0.0, 0.0
  for groundtruth, preds, scores in res:
    preds = preds[:k]

    prec += precision(groundtruth, preds)
    soft_prec += soft_precision(groundtruth, preds, M)
    jacc += jaccard(preds)

  if M is not None:
    print('%s\tP@%d: %.4f%%\tSP@%d: %.4f%%\tDiv: %.4f'
        % (flag, k, prec*100/total, k, soft_prec*100/total, jacc/total))
  else:
    print('%s\tP@%d: %.4f%%\tDiv: %.4f'
        % (flag, k, prec*100/total, jacc/total))


if __name__ == '__main__':
  # pkl_path = 'res_steam_50.pkl'
  # pkl_path = 'res_steam_10_MIQP.pkl'

  # pkl_path = 'res_ele_10_MIQP.pkl'
  # pkl_path = 'res_ele_50.pkl'

  # pkl_path = 'res_clo_50.pkl'
  # pkl_path = 'bpr_bundle_ele2.pkl'

  # pkl_path = 'res_steam_bpr.pkl'
  assert len(sys.argv) == 2
  pkl_path = sys.argv[1]

  with open(pkl_path, 'rb') as f:
    res = pickle.load(f)

  k = 10
  flag = "clo"
  if 'ele' in pkl_path:
    flag = 'ele'
  M = fatch_M(flag)
  print_all_metrics(flag, res, 10, M)
  print_all_metrics(flag, res, 5, M)
