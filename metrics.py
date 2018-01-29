import pickle

def jaccard_sim(a, b):
  aa = set(a)
  bb = set(b)
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


if __name__ == '__main__':
  # pkl_path = 'res_steam_50.pkl'
  # pkl_path = 'res_steam_10_MIQP.pkl'

  pkl_path = 'res_ele_10_MIQP.pkl'
  # pkl_path = 'res_ele_50.pkl'

  with open(pkl_path, 'rb') as f:
    res = pickle.load(f)

  limit = 10
  total = len(res)
  prec, jacc = 0.0, 0.0
  for groundtruth, preds, scores in res:
    preds = preds[:limit]
    jacc += jaccard(preds)
    prec += precision(groundtruth, preds)
  print('%s\tP@k: %.4f%%\tDiv: %.4f'
      % (pkl_path, prec*100/total, jacc/total))
