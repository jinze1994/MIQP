import random
import numpy as np
from scipy import sparse
from gurobipy import *
import pickle
from MIQP import _preprocess


source_file = 'res_ali_50.pkl'
with open(source_file, 'rb') as f:
    res = pickle.load(f)

def kMIQP(r, M, lamb, k, mipGap=0.2, timeLimit=1.0, outputFlag=False):
  """ Sover function for k-MIQP.

  This code formulates and solves the following simple MIQP model:
    maximize
      r'.x-lambda * x'.Q.x
    subject to
      xi       BINARY
      x.sum() == k

  Args:
    r: `python list` contains the prob scores.
    M: `python list` contains sparse coords tuple (x,y,v)
    lamb: `python float`
    k: `python int`
    others: Parameters of gurobi model.

  Returns:
    A `python list` with size k contains indices been selected. 
    A `python float` as maximized result.
  """
  r = np.array(r)
  n = len(r)
  if isinstance(M, list):
    Q = np.zeros([n, n])
    for i, j, v in M:
      Q[i][j] = Q[j][i] = v
  else:
    Q = M
  assert n == Q.shape[0] and n == Q.shape[1]

  m = Model("qp")
  
  
  xx = m.addVars(n, vtype=GRB.BINARY)
  x = np.array(xx.values())
  y = np.dot(r, x) - lamb*(x.T.dot(Q).dot(x))
  m.setObjective(y, GRB.MAXIMIZE)
  m.addConstr(xx.sum() == k)

  m.setParam('OutputFlag', outputFlag)
  m.Params.tuneResults=1
  m.tune()
  if m.tuneResultCount>0:
    m.getTuneResult(0)
    m.write('tune.prm')
    m.optimize()

  vx = []
  for i, v in enumerate(m.getVars()):
    if v.x >= 0.9:
      vx.append(i)
  assert len(vx) == k
  return vx, y.getValue()

def _test():
  r = []
  for i in range(1000):
    r.append(random.uniform(-10, -5))
  Q = [(0,2,-9.8),(5,7,-5.6),(100,200,-6),(20,80,-7.9)]
  vx, max_res = kMIQP(r, Q, lamb=0.6, k=10)
  print(vx, max_res)

if __name__ == '__main__':
  r, M = _preprocess(res[0])
  vx, max_res = kMIQP(r, M, lamb=1.0, k=10)
  

  
