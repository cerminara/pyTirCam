#! /usr/bin/env python
########################################################
### copiato da tech_note3.py (exp_calib versione 03) ###
########################################################
"""
pyTran - Python script for transmission and conversion algorithm used by commercial thermal camera 

Copyright (C) 2020-#### 
               by    Matteo Cerminara matteo.cerminara@ingv.it
               and   Benedetta Calusi benedetta.calusi@ingv.it

SCRIVERE QUALCOSA DEL GENERE E ADD LA LICENZA 

This program is free software; you can redistribute it and/or 
modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation; either version 
2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
GNU General Public License for more details.

You should have received a copy of the GNU General Public License 
along with this program; if not, write to the Free Software 
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import numpy.polynomial.polynomial as poly
from fitting import fit

#    Flags
# Spectral respose as step spectral response yes (True) or not (False)
step = False
# Select the results provided in the technical note 
#results = "tau_SR_ThC" # save the transmittance data of Figs5,6; 
#results = "Fig.7" # show the plot given in Fig.7; 
results = "Tables2-4" # compute the data reported in Tables 2-4

#    Inputs

# Temperaure range K
len_T = 1000 
Tmin = 273.15 # -10. + 273.15
Tmax = 500. + 273.15
T = np.linspace(Tmin,Tmax,len_T)
TT = np.reshape(T, (1, len_T))
Tmax_fit = Tmax
# 400 ppm (molar fraction)
rhoCO2 = 0.775e-3
# max of the spectral response 
SpR_max = 0.82
# Define domain of the spectral spectral respose as step function  
lambda_min = 7.5*1e-6
lambda_max = 13.*1e-6
# Camera-object distance (in the technical note is named d)
L = 3.e+03
# External optics transmittance
if (results == "Tables2-4") or (results == "tau_SR_ThC"):
    tau_ext = 1.0
elif (results == "Fig.7"):
    tau_ext = np.array([1.0,0.86])
# Load the H2O and CO2 absorption coefficients 
# wavelength data is based on H2O wavelength data 
fileIn_gasdataH2O = '../spectralData/h2o_conv.dat'
data_gd_H2O = np.genfromtxt(fileIn_gasdataH2O)
x = data_gd_H2O[:,0]*1e-6
len_gd = len(x)
xx = np.reshape(x, (len_gd,1))
Ah2o = np.reshape(data_gd_H2O[:,1], (len_gd, 1))

fileIn_gasdataCO2 = '../spectralData/co2_conv.dat'
data_gd_CO2 = np.genfromtxt(fileIn_gasdataCO2)
Aco2 = np.reshape(data_gd_CO2[:,1], (len_gd, 1))

# Define of saturation pressure of water
# Pascal
a = 611.21
# non-dim
b = 17.966
# Celsius
c= 247.15
# J/(kg K)
Rw = 462.
def ps(temp):
    """
    This definition provides the saturation pressure of water
    """
    return a*np.exp(b*temp/(c+temp))

# ThC coefs of the transmittance formula given in Eq. (B1)
K_atm = 1.9
alpha1 = 0.0066
alpha2 = 0.0126
beta1 = -0.0023
beta2 = -0.0067
h1 = 1.5587
h2 = 6.939e-02
h3 = -2.7816e-04
h4 = 6.8455e-07

# Check Stefan-Boltzmann law 
hC = 0.6626068e-33
cC = 299792458.
k_B = 0.13806503e-22
sigma = 5.6704e-8
CT = hC*cC/(xx*k_B)
CI = (2.*hC*cC**2)/((xx)**5)
B = CI/(np.exp(CT/TT) - 1.)
# R_BB = np.trapz(B, x=xx2, axis=0)
# R_StBo = sigma*T**4/np.pi 

# Load spectral response 
fileIn_SpR = '../spectralData/sr_calibrated.dat'
data_SpR = np.genfromtxt(fileIn_SpR)
XSpR = data_SpR[:,0]*1e-6
YSpR = SpR_max*data_SpR[:,1]
len_SpR = len(XSpR)
IX = PchipInterpolator(XSpR, YSpR)
SpR = IX(x)
#plt.plot(x,SpR)
#plt.show()

# Define spectral response as step function 
indx_SpRS = np.where((x >= lambda_min) & (x <= lambda_max))
SpRS_eq_1 = np.ones(len(indx_SpRS[0]))
SpRS_eq_0i = np.zeros(indx_SpRS[0][0]-1)
SpRS_eq_0f = np.zeros(len_gd - indx_SpRS[0][-1])
SpRS = np.append(SpRS_eq_0i,SpRS_eq_1)
SpRS = np.append(SpRS,SpRS_eq_0f)
#plt.plot(x,SpRS)
#plt.show()

# SpR/SpRS 
if step:
    SR = np.reshape(SpRS, (len_gd, 1)) 
else:
    SR = np.reshape(SpR, (len_gd, 1))

#    Fitting procedure to approximate radiance and brightness temperature 
# Radiance based on black-body theory
R = np.trapz(SR*B, x=xx, axis=0)

ii = np.where(T < Tmax_fit)
coefs, stats = poly.polyfit(T[ii], R[ii], 4, full=True) # coefs begin from index zero 
R_fit = poly.polyval(T,coefs)

def T_func(r, a, b, c, d):
    """
    This definition provides the approximation of the brightness temperature 
    as combination of a power law and a linear function 
    """
    return a*np.power(r, b) + c + d*r

popt, pcov, sigma, r2 = fit(T_func, R[ii], T[ii], p0=[67.7,0.28,94.0,0.196])
T_fit = T_func(R, *popt)


#    Transmission and conversion definitions 
def TAU_R(Tobj,Tatm, dist, eps, hum):
    """
    This definition from the input:
    
    Tobj, Tatm   - object and atmospheric temperature [Celsius]
    dist         - distance from object [meters]
    eps          - emissivity [-]
    hum          - humidity [-]
    
    computes: 
    
    taua         - transmittance given by Eq. (10), SpR/SpRS [-]
    tau_ThC      - transmittance given by Eq. (B1), ThC [-]
    Robj, Ratm   - radiance of the object and atmosphere by using Stefan-Boltzmann law [Watt/meters^2]
    """
    Tobj += 273.15
    Tatm += 273.15
    lenT = np.size(Tobj)
    # to plot atmospheric transmittance with respect to distance
    if lenT == 1:
        lenT = np.size(dist)
        dist = np.reshape(dist, (1, lenT))
    # to write when Tobj is an array and the other fields are
    # scalars or fields with the same length of Tobj
    else:
        Tobj = np.reshape(Tobj, (1, lenT))
        if np.size(Tatm) > 1:
            Tatm = np.reshape(Tatm, (1, lenT))
        if np.size(dist) > 1:
            dist = np.reshape(dist, (1, lenT))
        if np.size(eps) > 1:
            eps = np.reshape(eps, (1, lenT))
        if np.size(hum) > 1:
            hum = np.reshape(hum, (1, lenT))
    Bobj = CI/(np.exp(CT/Tobj) - 1.)
    Robj = np.trapz(SR*Bobj, x=xx, axis=0)
    Batm = CI/(np.exp(CT/Tatm) - 1.)
    Ratm = np.trapz(SR*Batm, x=xx, axis=0)
    PS = ps(Tatm-273.15)
    dens = hum/100.*PS/(Rw*Tatm)
    Ktot = dens*Ah2o + rhoCO2*Aco2
    taua = np.trapz(SR*Batm*np.exp(-(dens*Ah2o + rhoCO2*Aco2)*dist), x=xx, axis=0)/(np.trapz(SR*Batm, x=xx, axis=0))
    ww = hum*1e-02*np.exp(h1+h2*(Tatm-273.15)+h3*(Tatm-273.15)**2+h4*(Tatm-273.15)**3) # here Tatm in Celsius degree
    tau_ThC1 = K_atm*np.exp(-np.sqrt(dist)*(alpha1+beta1*np.sqrt(ww)))
    tau_ThC2 = (1.-K_atm)*np.exp(-np.sqrt(dist)*(alpha2+beta2*np.sqrt(ww)))
    tau_ThC = tau_ThC1 + tau_ThC2
    return [taua,tau_ThC,Robj,Ratm,Tobj,Tatm,eps]

def Ttot(out_TAU_R):
    """
    The input of this definition is input the output of def TAU_R to compute the observed radiance 

    Robs        - [Watt/meters^2]
    Robs_ThC    - [Watt/meters^2]
    
    by using Eq. (16) with taua and tau_ThC, respectively, 
    and the corresponding temperature object by means of def T_func [Celsius]. 
    """
    taua = out_TAU_R[0]
    tau_ThC = out_TAU_R[1]
    Robj = out_TAU_R[2]
    Ratm = out_TAU_R[3]
    Tobj = out_TAU_R[4]
    Tatm = out_TAU_R[5]
    eps = out_TAU_R[6]
    Robs = tau_ext*(eps*taua*Robj + (1. - eps)*taua*Ratm + (1. - taua)*Ratm) + (1.- tau_ext)*Ratm
    Robs_ThC = tau_ext*(eps*tau_ThC*Robj + (1. - eps)*tau_ThC*Ratm + (1. - tau_ThC)*Ratm) + (1.- tau_ext)*Ratm
    return [T_func(Robs, *popt) - 273.15, taua, T_func(Robs_ThC, *popt) - 273.15, tau_ThC, Robs, Robs_ThC]


def TAUS(Tobj,Tatm, dist, eps, hum):
    """
    Check the atmospheric transmission approximation given in Eq. (10)
    [tau(Tobj) approx as tau(Tatm)]
    This definition from the input:

    Tobj, Tatm   - object and atmospheric temperature [Celsius]
    dist         - distance from object [meters]
    eps          - emissivity [-]
    hum          - relative humidity [-]

    computes:

    taua         - tau(Tatm), atmosphere transmittance, SpR/SpRS [-]
    tauo         - tau(Tobj), object transmittance, SpR/SpRS [-]
    ratio        - tauo/taua [-]
    """
    Tobj += 273.15
    Tatm += 273.15
    lenT = np.size(Tobj)
    # to plot atmospheric transmittance with respect to distance
    if lenT == 1:
        lenT = np.size(dist)
        dist = np.reshape(dist, (1, lenT))
    # to write when Tobj is an array and the other fields are
    # scalars or fields with the same length of Tobj
    else:
        Tobj = np.reshape(Tobj, (1, lenT))
        if np.size(Tatm) > 1:
            Tatm = np.reshape(Tatm, (1, lenT))
        if np.size(dist) > 1:
            dist = np.reshape(dist, (1, lenT))
        if np.size(eps) > 1:
            eps = np.reshape(eps, (1, lenT))
        if np.size(hum) > 1:
            hum = np.reshape(hum, (1, lenT))
    Bobj = CI/(np.exp(CT/Tobj) - 1.)
    Robj = np.trapz(SR*Bobj, x=xx, axis=0)
    Batm = CI/(np.exp(CT/Tatm) - 1.)
    Ratm = np.trapz(SR*Batm, x=xx, axis=0)
    PS = ps(Tatm-273.15)
    dens = hum/100.*PS/(Rw*Tatm)
    Ktot = dens*Ah2o + rhoCO2*Aco2
    taua = np.trapz(SR*Batm*np.exp(-(dens*Ah2o + rhoCO2*Aco2)*dist), x=xx, axis=0)/(np.trapz(SR*Batm, x=xx, axis=0))
    tauo = np.trapz(SR*Bobj*np.exp(-(dens*Ah2o + rhoCO2*Aco2)*dist), x=xx, axis=0)/(np.trapz(SR*Bobj, x=xx, axis=0))
    ratio = tauo/taua
    return [taua,tauo,ratio]

if (results=="tau_SR_ThC"):
    #   Save data of tau_SpR/SpRS and tau_ThC
    D = np.linspace(0., 10000., 2000.) 
    TATM=np.array([0., 5., 10., 15.])
    for tt in TATM: # TEXP=15. , EPS =1. , HUM = 50.
        [T_SR,tau_SR,T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(15.,tt, D, 1., 50.))
        if step:
            np.savetxt("FigureData/tau_50_%d_step.dat"%tt, np.c_[D, tau_SR], fmt="%.8f")
            np.savetxt("FigureData/tau_50_%d_ThC_step.dat"%tt, np.c_[D, tau_ThC[0]], fmt="%.8f")
        else:
            np.savetxt("FigureData/tau_50_%d_ThC.dat"%tt, np.c_[D, tau_ThC[0]], fmt="%.8f")
            np.savetxt("FigureData/tau_50_%d.dat"%tt, np.c_[D, tau_SR], fmt="%.8f")
    
    HUM=np.array([25., 50., 75., 100.])
    for hh in HUM: # TEXP = 15. , TATM = 15. , EPS = 1. 
        [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(15.,15., D, 1., hh))
        if step:
            np.savetxt("FigureData/tau_%d_15_ThC_step.dat"%hh, np.c_[D, tau_ThC[0]], fmt="%.8f")
            np.savetxt("FigureData/tau_%d_15_step.dat"%hh, np.c_[D, tau_SR], fmt="%.8f")
        else:
            np.savetxt("FigureData/tau_%d_15_ThC.dat"%hh, np.c_[D, tau_ThC[0]], fmt="%.8f")
            np.savetxt("FigureData/tau_%d_15.dat"%hh, np.c_[D, tau_SR], fmt="%.8f")
elif (results == "Fig.7"):
    #    High temperature contrast
    # comparison of atmosphere and external optics
    tau_ext_aus = np.copy(tau_ext)
    tau_ext = tau_ext[0]
    [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(T-273.15,20., 1e3, 0.98, 40.))
    T3 = np.copy(T_SR)
    t3 = np.copy(tau_SR)
    [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(T-273.15,20., 1e4, 0.98, 40.))
    T4 = np.copy(T_SR)
    t4 = np.copy(tau_SR)
    tau_ext = tau_ext_aus[1] #0.86
    [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(T-273.15,20., 1e3, 0.98, 40.))
    T5 = np.copy(T_SR)
    t5 = np.copy(tau_SR)
    [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(T-273.15,20., 1e4, 0.98, 40.))
    T6 = np.copy(T_SR)
    t6 = np.copy(tau_SR)
    # L = 0
    [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(T-273.15,20., 0., 0.98, 40.))
    T2 = np.copy(T_SR)
    t2 = np.copy(tau_SR)
    print "observed temperatures at 500 Celsius: ", T2[-1], T3[-1], T4[-1], T5[-1], T6[-1]
    
    plt.plot(T-273.15,T-273.15,'-k', label='object temperature')
    plt.plot(T-273.15,T2, '-g', label=r"$\tau_{ext} = %.2f$"%tau_ext)
    plt.plot(T-273.15,T3, '-b', label=r"$d =$1 km, $\tau_{atm} = %.2f$"%t3)
    plt.plot(T-273.15,T4, '--b', label=r"$d =$10 km, $\tau_{atm} = %.2f$"%t4)
    plt.plot(T-273.15,T5, '-r', label=r"$d =$1 km, $\tau_{atm} = %.2f$, $\tau_{ext} = %.2f$"%(t5, tau_ext))
    plt.plot(T-273.15,T6, '--r', label=r"$d =$10 km, $\tau_{atm} = %.2f$, $\tau_{ext} = %.2f$"%(t6, tau_ext))
    plt.legend(loc='best')
    plt.ylabel(r'$T_{obs}$ [$^\circ\mathrm{C}$]',fontsize=14)#, fontproperties=font)
    plt.xlabel(r'$T_{obj}$ [$^\circ\mathrm{C}$]',fontsize=14)
    plt.savefig("atmoAndOptics.png")
    plt.close()
    
    
    # tau_atm approximation
    [taua1, tauo1, ratio1] = TAUS(T-273.15,20., 1e3, 0.98, 50.)
    [taua2, tauo2, ratio2] = TAUS(T-273.15,20., 1e4, 0.98, 50.)
    plt.plot(T-273.15,ratio1,'-k', label=r"$d = 1$ km")
    plt.plot(T-273.15,ratio2,'--k', label=r"$d = 10$ km")
    plt.legend(loc='best')
    plt.ylabel(r"$\tau_{obj}/\tau_{atm}$",fontsize=14)#, fontproperties=font)
    plt.xlabel(r'$T_{obj}$ [$^\circ\mathrm{C}$]',fontsize=14)
    plt.savefig("tauObjOnTauAtm.png")
    plt.close()
    print "max and min relative error at 1 km: ", (np.max(ratio1)-1.)*100., "%, ", (np.min(ratio1)-1.)*100., "%"
    print "max and min relative error at 10 km: ", (np.max(ratio2)-1.)*100., "%, ", (np.min(ratio2)-1.)*100., "%"
elif (results == "Tables2-4"):
    #    Brightness temperature data calculated by using tau_SpR and tau_ThC
    # ENT configuration camera settings
    
    print 'Table2, ENT configuration'
    TEXP=np.array([49.7, 39., 47.3])
    TATM=np.array([20., 40., 20.])
    DIST=np.array([3.047e3, 0., 3.047e3])
    EPS=np.array([0.98, 1., 0.98])
    HUM=np.array([40., 0., 40.])
    print 'Texp' , TEXP
    print 'Tatm' , TATM
    print 'Dist' , DIST
    print 'Eps' , EPS
    print 'Hum' , HUM
    [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(TEXP,TATM, DIST, EPS, HUM))
    print 'T_SR', T_SR
    print 'T_ThC', T_ThC
    print 'tau_SR', tau_SR
    print 'tau_ThC', tau_ThC
    print 'Robs' , Robs
    print 'Robs_ThC', Robs_ThC
    
    print'####'
    
    print 'Table3, ENT configuration'
    TEXP=np.array([-4.5, -10, -6, 4., 4.5, -4.])
    TATM=20.
    DIST=np.array([0., 3.047e3, 3.047e3, 0., 0., 3.047e3])
    EPS = 0.98
    HUM = 40.
    print 'Texp' , TEXP
    print 'Tatm' , TATM
    print 'Dist' , DIST
    print 'Eps' , EPS
    print 'hum', HUM
    [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(TEXP,TATM, DIST, EPS, HUM))
    print 'T_SR', T_SR
    print 'T_ThC', T_ThC
    print 'tau_SR', tau_SR
    print 'tau_ThC', tau_ThC
    print 'Robs' , Robs
    print 'Robs_ThC', Robs_ThC
    
    print'####'
    
    print 'Table4, ENT configuration'
    TEXP=np.array([-5.3, 4.3, 4.7, 5., -0.3, -0.3, -3.8, -13., -0.2])
    TATM = 20.
    DIST=np.array([3.047e3, 0., 0., 0., 0., 0., 3.047e3, 3.047e3, 0.])
    EPS=np.array([0.98, 0.98, 0.98, 1., 0.98, 0.98, 0.98, 0.98, 1.])
    HUM=np.array([40., 40., 40., 40., 40., 0., 0., 40., 40.])
    print 'Texp' , TEXP
    print 'Tatm', TATM
    print 'Dist' , DIST
    print 'Eps' , EPS
    print 'Hum' , HUM
    [T_SR,tau_SR,T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(TEXP,TATM, DIST, EPS, HUM))
    print 'T_SR', T_SR
    print 'T_ThC', T_ThC
    print 'tau_SR', tau_SR
    print 'tau_ThC', tau_ThC
    print 'Robs' , Robs
    print 'Robs_ThC', Robs_ThC
    
    print '####'
