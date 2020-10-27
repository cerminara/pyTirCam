"""
functions.py

Functions needed by pyTirTran

"""

import numpy as np
import scipy.optimize as opt

def fit2(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(-np.inf, np.inf), method=None, jac=None, **kwargs):
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
