# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:22:53 2020

@author: JC
CID:01063446
cw number: 29
"""
import numpy as np
#import csv
#import requests
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

#removed the following since the csv files might not be available in future
#url = "http://www2.imperial.ac.uk/~eakc07/time_series/29.csv"
#df = pd.read_csv(url)
#df.to_csv('time_series_29.csv')
#X29 = np.array(df)

#test data arrays
#X29 is the array I have been assigned
X29 = np.array([-0.80943,0.89557,-0.60602,-0.79507,1.2194,-0.07982,0.62401,-1.6654,1.4422,-2.0845,1.6206,0.95431,-1.5015,0.37202,-0.48067,1.1997,-1.4485,1.7884,-3.281,3.4496,-2.2887,0.26445,1.3184,-1.4435,1.8865,-3.1346,4.013,-3.0548,2.7923,-2.4889,0.70116,0.40906,1.5118,-3.4642,3.6822,-2.5606,2.2182,-2.4785,2.8256,-2.7626,2.1844,-1.4922,0.46725,1.06,-1.0063,-0.77681,0.92075,1.2557,-2.5783,2.891,-3.1956,2.7608,-2.9898,3.6741,-3.1076,1.5424,0.33969,-2.8469,3.595,-1.5081,0.10545,0.215,-0.36186,1.6654,-0.90309,-0.83347,1.763,-1.5085,1.7631,-2.3265,3.4871,-3.2037,1.5275,-0.0084657,0.059112,0.64547,-2.8161,4.7899,-6.0862,5.714,-3.9207,1.9084,-0.1175,-1.5118,1.1149,0.078329,1.3752,-5.04,5.5181,-3.5078,3.7521,-4.8787,4.084,-2.088,0.83323,-0.066886,-0.30755,0.45992,-1.8095,3.5903,-2.9391,1.2529,-0.90301,0.44637,1.2656,-3.5147,4.7523,-5.4208,5.6231,-5.0141,4.7172,-3.8818,1.9821,-0.31023,-0.84693,1.6213,-3.1786,4.3151,-4.2072,3.9478,-3.2767,2.4216,-2.0798,1.2527,-1.3185,1.694,-1.0528,-0.68732])
X53 = np.array([2.4135,-1.368,0.27804,-0.61488,1.8702,-3.7477,3.5908,-3.408,4.5091,-4.6976,2.4978,-0.51101,-2.007,2.7725,-3.0852,4.6349,-5.119,3.0626,-1.2902,2.6869,-2.9881,0.35472,1.8755,-1.7187,-0.11214,0.72838,0.065904,0.2035,-0.91606,0.61434,0.24024,0.13759,-0.57877,0.81786,-1.6544,1.1779,-0.38896,1.0765,-0.27206,-1.4978,2.7694,-2.6828,1.7645,-1.8067,0.63173,1.6281,-3.0353,2.7271,-1.405,1.5798,-0.98134,1.4475,-0.97154,-0.75342,0.40423,0.46837,-0.4809,0.66586,-0.9947,2.8449,-5.136,5.1435,-3.8934,3.8021,-3.3199,0.83722,1.4342,-0.88647,-0.54928,0.040106,1.8328,-2.0187,2.4323,-2.7142,2.9007,-3.3,2.8899,-2.6265,3.4973,-3.1143,1.2989,-0.98082,1.2743,-1.6496,1.4874,-1.0538,0.63148,0.24511,-2.4744,3.3053,-1.9975,-0.21659,0.61639,-0.033603,1.6745,-3.9465,5.7519,-5.352,3.8863,-2.7475,1.6707,-0.36145,-0.13659,-0.43337,0.43704,0.072779,-0.46563,-0.30271,2.2473,-3.4746,2.2654,-1.5829,2.0392,-1.8993,0.84921,-0.46412,0.16198,-0.01874,-0.33196,0.96035,-1.0325,1.8647,-3.7513,3.8833,-1.853,0.81529,0.13165,-1.5716])


X=X29.copy()

def S_AR(f,phis,sigma2):
    """
    evaluates parametric form of the spectral density function for
    an AR(p) process on a designated set of frequencies.
    Inputs:
        f: vector of frequencies at which to evaluate the spectral density function
        phis: vector [phi_{1,p} ,..., phi_{p,p}]
        sigma2: variance of white noise
    output: S=vector of values of spectral density function evaluatet at elements of f
    """
    #len_f = len(f)
    len_phi = len(phis) #check length of phis
    
    one_to_p = np.arange(1,len_phi+1,1) #+1 since last index is not inclusive
    indices = np.outer(f,one_to_p) #generate matrix where each row element is f*n for n equals 1 to p
    poly_exp = np.exp(1j * 2 * np.pi * indices) #elementwise exponential of indices matrix multiplied by -i*2pi*f*n
    
    G_phi_f = np.dot(poly_exp,phis) #this multiplies each row of the poly_exp matrix with the phi coefficients
    
    sdf = sigma2/(np.abs(1-G_phi_f)**2)
    
    return sdf
    

def AR2_sim(phis,sigma2,N):
    """
    Parameters
    ----------
    phis : phi vector
    sigma2 : variance of white noise
    N : Length of sequence to create

    Returns
    -------
    burnin : burned in sequence

    """
    X_t= np.zeros(N+100) #preallocate space
    #X_t[0:2]=0 #initialise first two values to 0
    epsi = np.random.normal(0,sigma2,size=N-2+100) #generate white noise process
    
    for i in range(N-2+100):
        #X_t[i+2] = np.dot(phis,X_t[i:i+2]) +epsi[i] #AR2 sequence #need to reverse dot!!!!!! 
        X_t[i+2] =  phis[0]*X_t[i+1] +phis[1]*X_t[i] +epsi[i] #could vectorise this later, but only summing two terms
    burnin = X_t[100:] #extract the burned in sequence by excluding the first 100 values
    
    return burnin

def acvs_hat(X,tau):
    """
    estimates acvs for X at tau values
    Parameters
    ----------
    X : Time series vector
    tau : Vector of lags

    Returns
    -------
    s_hat : TYPE
        DESCRIPTION.

    """
    N = len(X)
    #xtlxb = X - xbar #'xt less xbar', not using because we assume 0 mean
    s_hat = np.zeros_like(tau,dtype=float) #dtype float, otherwise s_hat will be integer and not be able to append
    for i in range(len(tau)): #need an estimate for each tau value
        end_t = N - np.abs(tau[i]) #creating some indexes so we can vectorise the sum
        start_t = np.abs(tau[i])
        s_hat[i] = (1/N)* np.dot(X[:end_t],X[start_t:]) #we could replace X with xtlxb for a non zero mean
        
    return s_hat
        

def periodogram(X):
    """
    computes periodogram  of X at fourier frequencies

    Parameters
    ----------
    X : vector time series 

    Returns
    -------
    p : periodogram estimate of sdf
    """
    N = len(X)
    #fourier transform of Xts is periodogram
    p = (1/N)* (np.abs(np.fft.fft(X)))**2 
    return p

def direct(X):
    """
    computes direct spectral estimate of X at fourier freq.
    Parameters
    ----------
    X : vector time series
    
    Returns
    -------
    sdf_hat : estimate of sdf
    """
    N=len(X)
    t= np.arange(1,N+1,1)
    #create hanning taper
    h_t = 0.5*np.sqrt((8/(3*(N+1)))) *(1- np.cos(2*np.pi*t/(N+1))) 
    htxt = np.multiply(h_t,X) #apply taper
    Jf = np.fft.fft(htxt) #apply fft
    sdf_hat = (np.abs(Jf))**2 #estimate sdf
    return sdf_hat


def emp_bias(N,r=0.95,fdash=1/8,sigma2=1,sim=10000):
    
    bias_p = np.zeros((sim,3)) #preallocate space
    bias_d = np.zeros((sim,3)) #preallocate space

    one_i = int((1/8) * (N)) #index for 1/8 frequency
    two_i = int((2/8) * (N)) #2/8 index
    three_i=int((3/8) * (N)) #3/8 index
    #calculate phis using the given roots
    phis=[2*r*np.cos(2*np.pi*fdash),-r**2] 
    f=[1/8,2/8,3/8] #frequencies to evaluate
    real_sdf = S_AR(f, phis,sigma2) #true sdf values at the given frequencies
            
    for i in range(sim): #10000 simulations
        X_sim = AR2_sim(phis,sigma2,N) #simulate one run of length N 
        
        prd = periodogram(X_sim) #periodogram estimate
        dse = direct(X_sim) #direct spectral estimate
        
        """#slow to run... will try and average parameters before taking bias
        bias_p[i,0] = real_sdf[0]- prd[one_i] #vectorise later, need to check we only have arrays and not lists
        bias_p[i,1] = real_sdf[1]- prd[two_i]
        bias_p[i,2] = real_sdf[2]- prd[three_i]
        
        bias_d[i,0] = real_sdf[0]- dse[one_i]
        bias_d[i,1] = real_sdf[1]- dse[two_i]
        bias_d[i,2] = real_sdf[2]- dse[three_i]
        """
        bias_p[i,0] = prd[one_i] #appending sdf values to an array
        bias_p[i,1] = prd[two_i] 
        bias_p[i,2] = prd[three_i]
        #appending for the direct method
        bias_d[i,0] = dse[one_i]
        bias_d[i,1] = dse[two_i]
        bias_d[i,2] = dse[three_i]
    #taking the mean of the sdf values and then the difference
    mean_bias_p = np.mean(bias_p,axis=0) -real_sdf 
    mean_bias_d = np.mean(bias_d,axis=0) -real_sdf
            
    return mean_bias_p, mean_bias_d

#plot true sdf 
#p51
N=16
r=0.95
fdash=1/8
sigma2=1
f = np.linspace(0,1,1000) 

plot_sdf = sigma2/((1-2*r*np.cos(2*np.pi*(fdash+f))+r**2)*(1-2*r*np.cos(2*np.pi*(fdash-f))+r**2))
plt.figure()
plt.plot(plot_sdf)    
plt.title('true spectral density function')
plt.ylabel('S(f)')
plt.xlabel('frequency f')    
plt.xticks(np.linspace(0,1000,8), (1/8) *np.arange(0,8,1))
#preallocate space for bias
plot1 = np.zeros((9,2)) #make 7 9
plot2 = np.zeros((9,2))
plot3 = np.zeros((9,2))
#loop from 16 to 4096
for i in np.arange(4,13,1):#make 11 13
    print(i) #progress
    (p_bias,d_bias) = emp_bias(2**i) #calculate bias
    plot1[i-4,:] = np.array([p_bias[0],d_bias[0]])
    plot2[i-4,:] = np.array([p_bias[1],d_bias[1]])
    plot3[i-4,:] = np.array([p_bias[2],d_bias[2]])
    
def bias_plot(plot,freq):
    plt.figure()
    lineObjects = plt.plot(plot)
    plt.title('empirical bias against sequence length at f=%f' %freq)
    plt.ylabel('empirical bias')
    plt.xlabel('length of simulated sequence (log2 scale)')
    plt.xticks(np.arange(plot.shape[0]), 2**np.arange(4, plot.shape[0]+4))
    plt.legend(lineObjects, ('Periodogram','Direct'))

bias_plot(plot1,1/8)
bias_plot(plot2,2/8)
bias_plot(plot3,3/8)

plt.figure()
lineObjects = plt.plot(plot1)
plt.title('empirical bias against sequence length at f=1/8')
plt.ylabel('empirical bias')
plt.xlabel('length of simulated sequence (log2 scale)')
plt.xticks(np.arange(plot1.shape[0]), 2**np.arange(4, plot1.shape[0]+4))
plt.legend(lineObjects, ('Periodogram','Direct'))

plt.figure()
lineObjects = plt.plot(plot2)
plt.title('empirical bias against sequence length at f=2/8')
plt.ylabel('empirical bias')
plt.xlabel('length of simulated sequence (log2 scale)')
plt.xticks(np.arange(plot2.shape[0]), 2**np.arange(4, plot2.shape[0]+4))
plt.legend(lineObjects, ('Periodogram','Direct'))

plt.figure()
lineObjects = plt.plot(plot3)
plt.title('empirical bias against sequence length at 3/8')
plt.ylabel('empirical bias')
plt.xlabel('length of simulated sequence (log2 scale)')
plt.xticks(np.arange(plot3.shape[0]), 2**np.arange(4, plot3.shape[0]+4))
plt.legend(lineObjects, ('Periodogram','Direct'))

def periodogramshift(X):
    N = len(X)
    #fourier transform of Xts is periodogram
    p = (1/N)* (np.abs(np.fft.fft(X)))**2 
    p = np.fft.fftshift(p) #shift
    x = np.linspace(-0.5,0.5,num=N)
    plt.plot(x,p)
    plt.title('periodogram spectral density estimate of 29.csv')
    plt.ylabel('S(f)')
    plt.xlabel('Frequency')
    return p

def directshift(X):
    N=len(X)
    x = np.linspace(-0.5,0.5,num=N)  
    t= np.arange(1,N+1,1)
    #create hanning taper
    h_t = 0.5*np.sqrt((8/(3*(N+1)))) *(1- np.cos(2*np.pi*t/(N+1))) 
    htxt = np.multiply(h_t,X) #apply taper
    Jf = np.fft.fft(htxt) #estimate sdf
    Jf = np.fft.fftshift(Jf) #shift
    sdf_hat = (np.abs(Jf))**2
    plt.plot(x, sdf_hat)
    plt.title('direct spectral density estimate of 29.csv')
    plt.ylabel('S(f)')
    plt.xlabel('Frequency')
    return sdf_hat


periodogramshift(X29)
directshift(X29)

def yw(p,X):
    """
    yule walker estimate for phi and sigma2 parameters
    Parameters
    ----------
    p : degree of AR(p) model you want to fit
    X : time series data
    Returns
    -------
    phis : vector containing estimates of the phis 
        for phi_1,p ... to... phi_p,p
    sigma2 : estimate of variance for white 0mean noise process
    """
    s_hat = acvs_hat(X,np.arange(0,p+1,1))
    #big_gamma_p = linalg.toeplitz(s_hat[:-1],s_hat[:-1]) #toeplitz matrix with acvs estimators
    #solve Ax=Id, where A is the toeplitz matrix, so x is inverse
    big_gam_inv = linalg.solve_toeplitz((s_hat[:-1],s_hat[:-1]), np.eye(p)) 
    small_gamma = s_hat[1:] #ignoring 0 entry
    phis = np.dot(big_gam_inv,small_gamma)
    sigma2 = s_hat[0] - np.dot(phis,s_hat[1:])
    
    return phis,sigma2

def fls(p,X):
    """
    forward least squares estimator for phi and sigma2 parameters
    Parameters
    ----------
    p : degree of AR(p) model you want to fit
    X : time series data
    Returns
    -------
    phis : vector containing estimates of the phis 
        for phi_1,p ... to... phi_p,p
    sigma2 : estimate of variance for white 0mean noise process
    """
    #we want X_p term on top left corner, but it is 0index
    Fmat = linalg.toeplitz(X[p-1:-1],X[p-1::-1]) 
    ftf_inv_ft = linalg.inv(np.transpose(Fmat)@Fmat) @ np.transpose(Fmat) 
    phi_hat = np.dot(ftf_inv_ft,X[p:])
    N=len(X)
    bottom = N-2*p
    top1 = np.transpose(X[p:] - Fmat@phi_hat)
    top2 = X[p:]- Fmat@phi_hat
    sigma2 = top1@top2/bottom
    
    return phi_hat, sigma2


def maxlh(p,X):
    """
    max likelihood estimator for phi and sigma2 parameters
    Parameters
    ----------
    p : degree of AR(p) model you want to fit
    X : time series data
    Returns
    -------
    phis : vector containing estimates of the phis 
        for phi_1,p ... to... phi_p,p
    sigma2 : estimate of variance for white 0mean noise process

    """
    Fmat = linalg.toeplitz(X[p-1:-1],X[p-1::-1]) 
    ftf_inv_ft = linalg.inv(np.transpose(Fmat)@Fmat) @ np.transpose(Fmat) 
    phi_hat = np.dot(ftf_inv_ft,X[p:])
    N=len(X)
    sigma2 = (np.transpose(X[p:] - Fmat@phi_hat) @ (X[p:] - Fmat@phi_hat))/(N-p) 
    #only difference to fls
    
    return phi_hat, sigma2    


def aic_table(X,pcount=20):
    N=len(X)
    AIC = np.zeros((20,3))
    for i in range (pcount):
        p=i+1 
        yw_sig = yw(p,X)[1]
        ls_sig = fls(p,X)[1]
        ml_sig = maxlh(p,X)[1]
        #calculate aic values
        AIC_yw = 2*p + (N*np.log(yw_sig))
        AIC_ls = 2*p +N*np.log(ls_sig)
        AIC_ml = 2*p +N*np.log(ml_sig)
        #appendaic values, for this p value, to the corresponding row of aic
        AIC[i,:]=[AIC_yw,AIC_ls,AIC_ml] 
    #plot figure
    plt.figure()    
    lineObjects = plt.plot(AIC)
    plt.legend(lineObjects, ('Yule-walker','Forward least squares','Maximum Likelihood'))
    plt.xticks(np.arange(AIC.shape[0]), np.arange(1, AIC.shape[0]+1))
    plt.ylabel('AIC')
    plt.xlabel('p')
    plt.title('AIC for different methods and different order p')
    
    return AIC
    
f=np.linspace(-0.5,0.5,512) #x axis
yplot = S_AR(f,yw(4,X29)[0],yw(4,X29)[1]) #y axis
lplot = S_AR(f,fls(6,X29)[0],fls(6,X29)[1])
mplot = S_AR(f,maxlh(13,X29)[0],maxlh(13,X29)[1])
plt.figure #plot
plt.plot(f,yplot,label='Yule-Walker')
plt.plot(f,lplot,label='Least Squares')
plt.plot(f,mplot,label='Maximum Likelihood')
plt.xlabel('frequency f')
plt.ylabel('S(f)')
plt.legend()
plt.title('spectral density function for optimum order p')
plt.show


(yphis,ysig) = yw(4,X29)
(lphis,lsig) = fls(6,X29)
(mphis,msig) = maxlh(13,X29)

past = X29[0:118]
ywforecast = np.concatenate((past,np.zeros(10)))
lsforecast = np.concatenate((past,np.zeros(10)))
mlforecast = np.concatenate((past,np.zeros(10)))

#forecast
for i in np.arange(118,128,1):
    ywforecast[i] = np.dot(yphis,ywforecast[i-1:i-1-4:-1])
    lsforecast[i] = np.dot(lphis,ywforecast[i-1:i-1-6:-1])
    mlforecast[i] = np.dot(mphis,ywforecast[i-1:i-1-13:-1])

t= np.arange(110,129,1)

def forecast(X29,yw_plot,fls_plot,ml_plot): 
    (yphis,ysig) = yw(4,X29) #return and save the parameters
    (lphis,lsig) = fls(6,X29)
    (mphis,msig) = maxlh(13,X29)
    
    past = X29[0:118] #preallocate space to put new forecasts
    ywforecast = np.concatenate((past,np.zeros(10)))
    lsforecast = np.concatenate((past,np.zeros(10)))
    mlforecast = np.concatenate((past,np.zeros(10)))
    
    for i in np.arange(118,128,1): #forecast for t=119 to 128
        ywforecast[i] = np.dot(yphis,ywforecast[i-1:i-1-4:-1])
        lsforecast[i] = np.dot(lphis,ywforecast[i-1:i-1-6:-1])
        mlforecast[i] = np.dot(mphis,ywforecast[i-1:i-1-13:-1])
    
    t= np.arange(110,129,1)
    plt.figure #plot
    if yw_plot==1:
       plt.plot(t,ywforecast[109:],label='Yule-Walker')
    if fls_plot==1:
        plt.plot(t,lsforecast[109:],label='least Squares')
    if ml_plot==1:
        plt.plot(t,mlforecast[109:],label='Maximum Likelihood')
    plt.plot(t,X29[109:],label='real data') 
    plt.title('forecasted time series vs real time series')
    plt.xlabel('time')
    plt.legend()
    plt.show
    
    return ywforecast[118:],lsforecast[118:],mlforecast[118:]
        
def forecastplot(yw,fls,ml):
    
    plt.figure
    if yw==1:
       plt.plot(t,ywforecast[109:],label='Yule-Walker')
    if fls==1:
        plt.plot(t,lsforecast[109:],label='least Squares')
    if ml==1:
        plt.plot(t,mlforecast[109:],label='Maximum Likelihood')
    plt.plot(t,X29[109:],label='real data') 
    plt.title('forecasted time series vs real time series')
    plt.xlabel('time')
    plt.legend()
    plt.show
    
        