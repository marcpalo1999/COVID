import numpy as np

def SIRmodel(C, t, beta, r ):
    S,I,R=C
    dSdt=-beta*S*I #equation for S evolution. Beta is the rate at which population leaves this block
    dIdt=beta*S*I-r*I #equation for I evolution. r is the rate at which people recovers from being ill
    dRdt=r*I #equation for R evolution
    
    return np.array([dSdt, dIdt, dRdt]) #function returns an array with each of the 3 solutions


def SEIRmodel_q(C, r, q, N = 125*1e6, eta = 1/5.2, a = 1/2.3):
    S,E,I,R = C
    dSdt = - r * S * I/N
    dEdt = r * S * I/N - eta * E 
    dIdt = eta * E - a * I - q * I
    dRdt = a * I + q * I
    
    return dSdt, dEdt, dIdt, dRdt

