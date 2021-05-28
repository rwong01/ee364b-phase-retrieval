# -*- coding: utf-8 -*-
"""

@author: Rebecca
"""
import numpy as np
import cvxpy as cp
from numpy import linalg as LA
from matplotlib import pyplot as plt
import scipy.fftpack
from scipy.linalg import dft

#%% Signal Construction
np.random.seed(0)

def A(x):
    # FFT Operator to map x from R^n to R^m
   xp = np.pad(x,(0,m-n))
   return (np.fft.fft(xp,m))

# Matrix form of fft, taking the first n columns of the m-point DFT
def A_matrix(m,n):
    return dft(m)[:,:n]

def A_inv(x):
   return (np.fft.ifft(x))[:n]

def construct_signal(m,n, noise = False, alpha = 0.1):
    np.random.seed(0)
    x0 = np.random.rand(n) + np.random.rand(n)*1j# original signal
    if(noise):
        x1 = x0 + (np.random.normal(0,alpha,x0.shape) + np.random.normal(0,alpha,x0.shape)*1j)
    else:
        x1 = x0
    b = np.abs(A(x1)) # measurement vector
    return x0, b

def reconstruct(b,p):
    r = b * np.cos(p)
    im = b * np.sin(p)
    return (r + 1j* im)

def validate_signal(A_n, x0, noise = False):
    # Check consistency of matrix A_n operator vs numpy fft function
    err = 1E-5
    ang = np.angle(A(x0))
    ag = (A_inv(reconstruct(b,ang)))
    X = np.expand_dims(x0,1) @ np.expand_dims(x0,0) # nxn symmetric
    b1 = np.zeros(m,)
    for i in range(m):
        z = np.expand_dims(A_n[i,:],0) @ X @ np.expand_dims(A_n[i,:],1)
        b1[i] = np.sqrt(np.abs(np.squeeze(z)))
    if(not noise):
        assert(LA.norm(ag - x0) < err), "Failed reconstructions check"
        assert(LA.norm(b - np.abs(A_n @ x0)) < err),"Matrix A DFT error 1"   
        assert(LA.norm(b - b1 ) < err ),"Matrix A DFT error 2" 
    return 

def norm_err(xhat):
    return LA.norm(b - np.abs(A_n @ xhat))/ LA.norm(b)

def fienup(b,A_n):
    xk = np.random.rand(n) # initial guess
    err = []
    for _ in range(10):
        Gk_p = np.angle(A(xk)) # take phase of FFT
        Gpk = reconstruct(b,Gk_p) # project given constraint
        gkp = (A_inv(Gpk)) # invert fft
        if gkp.any() < 0:
            xk = np.zeros(n,)
        else:
            xk = reconstruct(f,np.angle(gkp)) # project given constraint
        err.append(norm_err(xk))
    plt.figure()
    plt.plot(err)
    plt.title("Fienup algorithm")
    plt.ylabel("Error")
    r = min(err)
    print("Min Error: %0.4f" % r)
    return xk, err

def phasemax(b,A_n):
    xhat = np.random.rand(n)
    err = []
    for _ in range(10):
        z = cp.Variable(m,complex=True)
        B = np.diag(b)
        constr = [xhat == A_n.T @ LA.inv(B) @ z]
        obj = cp.norm(z,1)
        prob = cp.Problem(cp.Minimize(obj),constr)
        prob.solve()
        p = np.ones(m) 
        for i in range(m):
            if (np.abs(z.value[i]) > 1E-6):
                p[i] = z.value[i] / np.abs(z.value[i])
        xhat = A_inv(reconstruct(b,p))
        err.append(norm_err(xhat))
    plt.figure()
    plt.plot(err)
    plt.title("PhaseMax algorithm")
    plt.ylabel("Error")
    r = min(err)
    print("Min Error: %0.4f" % r)   
    return xhat, err

n = 50
m = 300
alpha = 0.2 # noise parameter
noise = False
x0, b = construct_signal(m,n,noise=noise,alpha=alpha)
f = np.abs(x0)
A_n = A_matrix(m,n)
if(not noise):
    validate_signal(A_n, x0)
#%%
x_f, _ = fienup(b, A_n)
x_m, _ = phasemax(b, A_n)

#%% PhaseLift
X = cp.Variable((n,n), PSD=True)
l = 1e-2
bb = np.square(b)
obj = 0
# Ax = []
# for i in range(m):
#     z = cp.reshape(cp.abs(np.expand_dims(A_n[i,:],0) @ X @ np.expand_dims(A_n[i,:],1)),(1,))
#     Ax.append(z)
    # obj += cp.square(z - bb[i])
obj += 0.5*cp.sum_squares(cp.diag(A_n@X@A_n.T) - bb)
# obj += cp.sum_squares(cp.reshape(A_n @ x,(300,)) - bb)
obj += cp.trace(X)
# constr = [A_n@X == b]
# prob = cp.Problem(cp.Minimize(obj), constr)
prob = cp.Problem(cp.Minimize(obj))
prob.solve(solver=cp.SCS, max_iters=10,verbose=True)

x_hat = np.sqrt(np.diag(X.value))
r = norm_err(x_hat)
print(r)
