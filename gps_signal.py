import numpy as np

def transient(n, size, duration, center):
    t = np.arange(0,n)
    d = size/(1 + np.exp(-(t - center)/(duration/10.0), dtype=np.float128))
    return d

def seasonal(n, an, sa, dt=1/365.25):
    t = dt*np.arange(n)
    c1 = an[0] + (an[1] - an[0])*np.random.rand()
    c2 = sa[0] + (sa[1] - sa[0])*np.random.rand()
    d = c1*np.sin(2*np.pi*(t + np.random.rand())) + c2*np.sin(4*np.pi*(t + np.random.rand()))
    return d

def colored_noise(n, wn, fn, rw, dt=1/365.25):
    c1 = wn[0] + (wn[1] - wn[0])*np.random.rand()
    c2 = fn[0] + (fn[1] - fn[0])*np.random.rand()
    c3 = rw[0] + (rw[1] - rw[0])*np.random.rand()
    d = c1*white_noise(n) + c2*flicker_noise(n,dt) + c3*random_walk(n,dt)
    return d

def white_noise(n):
    y = np.random.randn(n)
    return y

def flicker_noise(n,dt=1):

    if np.remainder(n,2):
        nt = n + 1
    else:
        nt = n
        
    x = np.random.randn(nt)
    X = np.fft.fft(x)
    
    nf = n/2 + 1
    k = np.arange(nf,dtype=np.int64)
    
    X = X[k]
    X = X/np.sqrt(k+1)

    X = np.concatenate((X,np.conj(X[-1:0:-1])))
    
    y = np.real(np.fft.ifft(X))
    y = y[:n]
    
    y = y - np.mean(y)
    y = y/np.std(y)
    
    return dt**0.25*y

def random_walk(n,dt=1):
    L = np.tril(np.ones((n,n)))
    w = np.random.randn(n)
    y = np.dot(L,w)
    
    return dt**0.5*y

def outliers(n, max_size, prob):
    c = np.random.rand(n) < prob
    d = c*max_size*(np.random.rand(n)-0.5)*2
    return d

