"""
functions.py

Functions needed by pyTirTran

"""

import numpy as np
import scipy.optimize as opt

def fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(-np.inf, np.inf), method=None, jac=None, **kwargs):
  """
    See scipy.optimize.curve_fit for explanation
    The only difference is that this returns also 
      the variance "chi" 
      and the coefficient of determination "r2". 
      In particular:
      ss_res = sum( (yi - fi)**2 )
      ss_tot = sum( (yi - <yi>)**2 )
      chi = sqrt( ss_res/N )
      r2  = 1 - ss_res/ss_tot
  """
  popt, pcov = opt.curve_fit(f, xdata, ydata, p0, sigma, absolute_sigma, check_finite, bounds, method, jac, **kwargs)
  ss_res = np.sum((ydata-f(xdata, *popt))**2)
  ss_tot = np.sum((ydata-np.average(ydata))**2)
  chi = np.sqrt(ss_res/len(ydata.flatten()))
  r2 = 1. - ss_res/ss_tot
  return popt, pcov, chi, r2


def ps(temp):
    """
    This function provides the saturation pressure of water
    Temperature "temp" in Celsius.
    """
    # Pascal
    a = 611.21
    # non-dim
    b = 17.966
    # Celsius
    c= 247.15
    # J/(kg K)
    Rw = 462.
    return a*np.exp(b*temp/(c+temp))

def rhow(hum, T):
    """
    Water vapour bulk density
    """
    Rw = 462.
    return hum/100.*ps(T-273.15)/(Rw*T)

def rhowThC(hum, T):
    """
    Water vapour bulk density in g/m3 from the ThC empirical model
    """
    h1 = 1.5587
    h2 = 6.939e-02
    h3 = -2.7816e-04
    h4 = 6.8455e-07
    return hum*1e-02*np.exp(h1+h2*(T-273.15)+h3*(T-273.15)**2+h4*(T-273.15)**3)

def Boltz(x, T):
    """
    Stefan-Boltzmann law
    """
    hC = 0.6626068e-33
    cC = 299792458.
    k_B = 0.13806503e-22
    CT = hC*cC/(x*k_B)
    CI = (2.*hC*cC**2)/((x)**5)
    return CI/(np.exp(CT/T) - 1.)

def Step(x, lambda_min, lambda_max, len_gd):
    """
    Define spectral response as step function 
    """
    indx_SpRS = np.where((x >= lambda_min) & (x <= lambda_max))
    SpRS_eq_1 = np.ones(len(indx_SpRS[0]))
    SpRS_eq_0i = np.zeros(indx_SpRS[0][0]-1)
    SpRS_eq_0f = np.zeros(len_gd - indx_SpRS[0][-1])
    SpRS_i = np.append(SpRS_eq_0i,SpRS_eq_1)
    return np.append(SpRS_i,SpRS_eq_0f)


### arrived here. I need to try to call this RIR
def TtoR(xx, T, SR):
    """
    Given temperature T and wavelength x,
    beside spectral response SR,
    the function convets into radiance
    """
    return np.trapz(SR*Boltz(xx, T), x=xx, axis=0)

def T_func(r, a, b, c, d):
    """
    This definition provides the approximation of the brightness temperature 
    as combination of a power law and a linear function 
    """
    return a*np.power(r, b) + c + d*r

#    Transmission and conversion definitions 
def tauThC(hum, T, dist):
    """
    Atmospheric transmittance as given by the ThC model
    """
    # ThC coefs of the transmittance formula given in Eq. (B1)
    K_atm = 1.9
    alpha1 = 0.0066
    alpha2 = 0.0126
    beta1 = -0.0023
    beta2 = -0.0067
    tau_ThC1 = K_atm*np.exp(-np.sqrt(dist)*(alpha1+beta1*np.sqrt(rhowThC(hum, T))))
    tau_ThC2 = (1.-K_atm)*np.exp(-np.sqrt(dist)*(alpha2+beta2*np.sqrt(rhowThC(hum, T))))
    return tau_ThC1 + tau_ThC2

def tauA(rhoi, Ai, dist, xx, T, SR):
    """
    rhoi         - array of components bulk densities
    Ai           - array of components specific absorption coefficient
    dist         - camera-to-object distance [m]
    xx           - wavelength
    T            - temperature [K]
    SR           - spectral response
    """
    len_i = len(rhoi)
    Ktot = 0.
    for i in range(len_i):
        Ktot += rhoi[i]*Ai[i]
    B = Boltz(xx, T)
    return np.trapz(SR*B*np.exp(-Ktot*dist), x=xx, axis=0)/np.trapz(SR*B, x=xx, axis=0)

def Rtot(Robj, Ratm, taua, tau_ext, eps):
    return tau_ext*(eps*taua*Robj + (1. - eps)*taua*Ratm + (1. - taua)*Ratm) + (1.- tau_ext)*Ratm
