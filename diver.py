import pickle

with open('res_ele_50.pkl', 'rb') as f:
  res = pickle.load(f)

def jaccard_sim(a, b):
  aa = set(a)
  bb = set(b)
  return len(aa & bb) / len(aa | bb)

limit = 50
total = len(res)
cnt = 0.0
jacc = 0.0
for groundtruth, preds, scores in res:
  preds = preds[:limit]
  jac = 0.0
  for i in range(len(preds)):
    for j in range(i+1, len(preds)):
      jac += jaccard_sim(preds[i], preds[j])
  jac /= len(preds) * (len(preds) - 1) / 2
  jacc += jac

  grou = set(groundtruth)
  corr = 0.0
  for pred in preds:
    corr = max(corr, len(set(pred).intersection(grou))/2)
  assert(corr == 0.0 or corr == 0.5 or corr == 1.0)
  cnt += corr
print(cnt, total, cnt / total, jacc / total)
