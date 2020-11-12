# coding: utf-8
import numpy as np
from scipy.special import erf

Phi = lambda x: 0.5*(1 + erf(x/np.sqrt(2)))
    
def fourier_basis(dim, x, inverse_cdf=lambda x: x):
    assert x.ndim == 1
    x = inverse_cdf(x)
    assert np.all((0 <= x) & (x <= 1)), [x.min(), x.max()]
    z = np.ones((1,)+x.shape)
    nc = np.sqrt(2)                     # normalization constant
    fs = 2*np.pi*np.arange(1,dim//2+1)  # frequencies
    assert dim % 2 == 1
    if dim % 2 == 0:
        c = nc*np.cos(fs[:-1,None]*x[None,:])
    else:
        c = nc*np.cos(fs[:,None]*x[None,:])
    s = nc*np.sin(fs[:,None]*x[None,:])
    ret = np.concatenate([z,c,s], axis=0)
    assert ret.shape == (dim,) + x.shape
    return ret
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    xs = np.linspace(-10,10,1000)
    bs = fourier_basis(7, xs, Phi)
    for b in bs:
        plt.plot(xs, b)
    plt.show()
