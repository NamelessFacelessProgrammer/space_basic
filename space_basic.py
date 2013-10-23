"""Determinant, vector, and matrix algebra using list comprehensions
and generator expressions: 80 lines of code, including doc strings,
but just 40 lines of actual code + 15 lines of module test code."""

from math import sqrt
from operator import __mul__ as mul, __add__ as add
from itertools import imap,izip

# functions related to computing a determinant

def minor(m,i,j):
  """Return the matrix with square dimension len(m)-1 equal
     to 'm' with the i-th row and j-th column struck. """
  enum=enumerate # speedup: local lookup
  return [[e for c,e in enum(v) if c!=j] for r,v in enum(m) if r!=i]

def alt(s,i):
  """Return -s if 'i' is odd or s if 'i' is even."""
  if i & 1: return -s # alt(s,i) call 2x faster than inline
  else: return s      # (cmp(1,i&1)-cmp(i&1,0))*s

def cof(m):
  """Return the cofactor matrix of square matrix 'm'."""
  a,dt,mnr,xr=alt,det,minor,xrange(len(m)) # speedup: local lookup
  return [[a(dt(mnr(m,i,j)),i-j) for j in xr] for i in xr]

def det(m):
  """Return the determinant of square matrix 'm'."""
  if len(m)==1: return m[0][0]
  a,mnr=alt,minor # speedup: local lookup
  return sum(a(e*det(mnr(m,0,c)),c) for c,e in enumerate(m[0]) if e!=0)

# vector functions

def v_metric(u,v):
  """Return the inner product of commensurate vectors 'u' and 'v'.
    'v_metric' is the Euclidean metric tensor, hence the name."""
  return sum(imap(mul,u,v))

def v_scale(s,v):
  """Return the vector 'v' scaled by scalar 's'."""
  return [s*a for a in v]

def v_sum(u,v):
  """Return the vector sum of commensurate vectors 'u' and 'v'."""
  return map(add,u,v)

# matrix functions

def m_flip(m):
  """Return the transpose of 'm'."""
  return zip(*m)

def m_scale(s,m):
  """Return the matrix 'm' scaled by scalar 's'."""
  v_sc=v_scale # speedup: local lookup
  return [v_sc(s,v) for v in m]

def m_inv(m):
  """Return the inverse of square matrix 'm'.
     Return 'None' if 'm' is singular."""
  c=cof(m); d=v_metric(m[0],c[0])
  if d == 0: return None
  else: return m_scale(1./d,m_flip(c))

def m_sum(m,n):
  """Return the matrix sum of commensurate matrices 'm' and 'n'."""
  v_sm=v_sum # speedup: local lookup
  return [v_sm(u,v) for u,v in izip(m,n)]

def m_mul(m,n):
  """Return m*n for compatible matrices 'm' and 'n'."""
  n=m_flip(n); v_mtrc=v_metric # speedup: local lookup
  return [[v_mtrc(r,c) for c in n] for r in m]

def m_volume(M):
  """Compute the m-volume in an n-dimensional space of the m-dimensional
     prism defined by m vectors of length n in the m by n matrix 'M'."""
  return sqrt(det(m_mul(M,m_flip(M))))

if __name__ == '__main__': # test suite
  M=[[3,-2,-1],[2,0,2],[1,2,-3]]
  #M=[ [-1,0, 2], [ 0,3, 0], [-2,0,-4] ]
  d=det(M)
  print d,'\n',M
  C=cof(M)
  print det(C),'\n',C
  T=m_flip(C)
  print T,'\n',det(T)
  dI=m_mul(M,T)
  print dI,'\n',det(dI),'\n',m_scale(1./d,dI)
  m=m_inv(m_scale(1./d,T))
  print m,'\n',m_sum(M,m_scale(-1,m))
  print m_volume(M) # 3-volume of the prism
  for v in M: print m_volume([v]) # length of each row
  # All possible cross product magnitudes of row pairs
  print [m_volume([u,v]) for u in M for v in M]
