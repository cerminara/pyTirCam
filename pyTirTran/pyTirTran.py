#! /usr/bin/env python3
########################################################
########################################################
"""
pyTran - Python script for transmission and conversion algorithm used by commercial thermal camera 

Copyright (C) 2020-#### 
               by    Matteo Cerminara matteo.cerminara@ingv.it
               and   Benedetta Calusi benedetta.calusi@ingv.it

This program is free software; you can redistribute it and/or 
modify it under the terms of the GNU General Public License 
as published by the Free Software Foundation; either version 
3 of the License, or (at your option) any later version.

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
from functions import *

### --- Select the results provided in the technical note  --- ###
## Spectral respose as step spectral response yes (True) or not (False)
step = False

### --- Flags --- ###
#results = "tau_SR_ThC" # save the transmittance data of Figs5,6; 
#results = "Fig.8" # show the plot given in Fig.8; 
results = "Tables2-4" # compute the data reported in Tables 2-4

###--- Inputs

exec(open("input.py").read())

## External optics transmittance
if (results == "Tables2-4") or (results == "tau_SR_ThC"):
    tau_ext = 1.0
elif (results == "Fig.8"):
    tau_ext = np.array([1.0,tauext])

## Load the H2O and CO2 absorption coefficients 
## wavelength space is based on H2O wavelength space
data_gd_H2O = np.genfromtxt(fileIn_gasdataH2O)
x = data_gd_H2O[:,0]*1e-6
len_gd = len(x)
xx = np.reshape(x, (len_gd,1))
Ah2o = np.reshape(data_gd_H2O[:,1], (len_gd, 1))

data_gd_CO2 = np.genfromtxt(fileIn_gasdataCO2)
Aco2 = np.reshape(data_gd_CO2[:,1], (len_gd, 1))


## Contruct temperature
T = np.linspace(Tmin,Tmax,len_T)
TT = np.reshape(T, (1, len_T))

## Check Stefan-Boltzmann law 
#sigma = 5.6704e-8
#x2 = np.linspace(0.01*lambda_min, 1000.*lambda_max, len_gd*3)
#xx2 = np.reshape(x2, (len_gd*3,1))
#B = Planck(xx2, TT)
#R_BB = np.trapz(B, x=xx2, axis=0)
#R_StBo = sigma*T**4/np.pi 
#plt.plot(T, R_BB/R_StBo - 1.)
#plt.show()
#exit()

## Load spectral response 
data_SpR = np.genfromtxt(fileIn_SpR)
XSpR = data_SpR[:,0]*1e-6
YSpR = SpR_max*data_SpR[:,1]
len_SpR = len(XSpR)
IX = PchipInterpolator(XSpR, YSpR)
SpR = IX(x)
#plt.plot(x,SpR)
#plt.show()

## Define spectral response as step function 
SpRS = Step(x, lambda_min, lambda_max, len_gd)
#plt.plot(x,SpRS)
#plt.show()

## SpR/SpRS 
if step:
    SR = np.reshape(SpRS, (len_gd, 1)) 
else:
    SR = np.reshape(SpR, (len_gd, 1))


## Fitting procedure to approximate radiance and brightness temperature 
## Radiance based on black-body theory
Rir = lambda Tx: RIR(xx, Tx, SR)
R = Rir(TT)
## Fit of radiance as a function of temperature
ii = np.where(T < Tmax_fit)
coefs, stats = poly.polyfit(T[ii], R[ii], 4, full=True) # coefs begin from index zero 
R_fit = lambda Tx: poly.polyval(Tx,coefs)
## Here choose Rir for better accuracy, or R_fit for better efficiency
TtoR = Rir  # R_fit
## Fit of temperature as a function of radiance
popt, pcov, sigma, r2 = fit(T_func, R[ii], T[ii], p0=[67.7,0.28,94.0,0.196])
T_fit = lambda Rx: T_func(Rx, *popt)
RtoT = T_fit

### --- Functions to process data --- ###

def TAU_R(Tobj,Tatm, dist, eps, hum):
    """
    Input:

    Tobj, Tatm   - object and atmospheric temperature [Celsius]
    dist         - distance from object [meters]
    eps          - emissivity [-]
    hum          - humidity [-]
    
    Output: 
    
    taua         - atmospheric transmittance using camera spectral response
                   and the HITRAN database: Eq. (10), SpR/SpRS [-]
    tau_ThC      - atmospheric transmittance given by the thermal camera 
                   empirical model: Eq. (B1), ThC [-]
    Robj, Ratm   - radiance emitted by the object and by the atmosphere [Watt/meters^2]

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
    Robj = TtoR(Tobj)
    Ratm = TtoR(Tatm)
    dens = rhow(hum, Tatm)
    taua = tauA([dens, rhoCO2], [Ah2o, Aco2], dist, xx, Tatm, SR)
    tau_ThC = tauThC(hum, Tatm, dist)
    return [taua,tau_ThC,Robj,Ratm,Tobj,Tatm,eps]

def Ttot(out_TAU_R):
    """
    The input of this function is the output of TAU_R, to compute the observed radiance 
    by using Eq. (16) with taua and tau_ThC, respectively, 
    and the corresponding temperature object by means of RtoT [Celsius]. 

    Output:

    Tobs        - Observed temperature with SpR/SpRS atmospheric models [Celsius]
    Tobs_ThC    - Observed temperature with ThC atmospheric model [Celsius]
    Robs        - Observed radiance with SpR/SpRS atmospheric models [Watt/meters^2]
    Robs_ThC    - Observed radiance with ThC atmospheric model [Watt/meters^2]
    
    """
    taua = out_TAU_R[0]
    tau_ThC = out_TAU_R[1]
    Robj = out_TAU_R[2]
    Ratm = out_TAU_R[3]
    Tobj = out_TAU_R[4]
    Tatm = out_TAU_R[5]
    eps = out_TAU_R[6]
    Robs = Rtot(Robj, Ratm, taua, tau_ext, eps)
    Robs_ThC = Rtot(Robj, Ratm, tau_ThC, tau_ext, eps)
    Tobs = RtoT(Robs)
    Tobs_ThC = RtoT(Robs_ThC)
    return [ Tobs - 273.15, taua, Tobs_ThC - 273.15, tau_ThC, Robs, Robs_ThC]


def TAUS(Tobj,Tatm, dist, eps, hum):
    """
    This function checks the atmospheric transmission approximation given in Eq. (10)
    [tau(Tobj) approx as tau(Tatm)].

    Input:

    Tobj, Tatm   - object and atmospheric temperature [Celsius]
    dist         - distance from object [meters]
    eps          - emissivity [-]
    hum          - relative humidity [-]

    Output:

    taua         - tau(Tatm), atmospheric transmittance at Tatm, SpR/SpRS [-]
    tauo         - tau(Tobj), atmospheric transmittance at Tobj, SpR/SpRS [-]
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
    dens = rhow(hum, Tatm)
    taua = tauA([dens, rhoCO2], [Ah2o, Aco2], dist, xx, Tatm, SR)
    tauo = tauA([dens, rhoCO2], [Ah2o, Aco2], dist, xx, Tobj, SR)
    ratio = tauo/taua
    return [taua,tauo,ratio]


### --- Main --- ###

# Save data to plot tau_SpR/SpRS and tau_ThC (Figs5,6)
if (results=="tau_SR_ThC"):
    D = np.linspace(0., 10000., 2000) 
    TATM=np.array([0., 5., 10., 15.])
    for tt in TATM: # TEXP=15. , EPS =1. , HUM = 50.
        [T_SR,tau_SR,T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(15.,tt, D, 1., 50.))
        np.savetxt("FigureData/tau_50_%d_ThC.dat"%tt, np.c_[D, tau_ThC[0]], fmt="%.8f")
        if step:
            np.savetxt("FigureData/tau_50_%d_step.dat"%tt, np.c_[D, tau_SR], fmt="%.8f")
        else:
            np.savetxt("FigureData/tau_50_%d.dat"%tt, np.c_[D, tau_SR], fmt="%.8f")
    
    HUM=np.array([25., 50., 75., 100.])
    for hh in HUM: # TEXP = 15. , TATM = 15. , EPS = 1. 
        [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(15.,15., D, 1., hh))
        np.savetxt("FigureData/tau_%d_15_ThC.dat"%hh, np.c_[D, tau_ThC[0]], fmt="%.8f")
        if step:
            np.savetxt("FigureData/tau_%d_15_step.dat"%hh, np.c_[D, tau_SR], fmt="%.8f")
        else:
            np.savetxt("FigureData/tau_%d_15.dat"%hh, np.c_[D, tau_SR], fmt="%.8f")
#
# Variation of the observed temperature due to atmosphere and ext. optics
#
elif (results == "Fig.8"):
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
    print("observed temperatures when object is at 500 Celsius: ", T2[-1], T3[-1], T4[-1], T5[-1], T6[-1])
    
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
    
    
    # tau_obj ~ tau_atm approximation
    [taua1, tauo1, ratio1] = TAUS(T-273.15,20., 1e3, 0.98, 50.)
    [taua2, tauo2, ratio2] = TAUS(T-273.15,20., 1e4, 0.98, 50.)
    plt.plot(T-273.15,ratio1,'-k', label=r"$d = 1$ km")
    plt.plot(T-273.15,ratio2,'--k', label=r"$d = 10$ km")
    plt.legend(loc='best')
    plt.ylabel(r"$\tau_{obj}/\tau_{atm}$",fontsize=14)#, fontproperties=font)
    plt.xlabel(r'$T_{obj}$ [$^\circ\mathrm{C}$]',fontsize=14)
    plt.savefig("tauObjOnTauAtm.png")
    plt.close()
    print("tauObj max and min relative error at 1 km: ", (np.max(ratio1)-1.)*100., "%, ", (np.min(ratio1)-1.)*100., "%")
    print("tauObj max and min relative error at 10 km: ", (np.max(ratio2)-1.)*100., "%, ", (np.min(ratio2)-1.)*100., "%")
#
# Experimental results
#
elif (results == "Tables2-4"):
    # Brightness temperature data calculated by using tau_SpR and tau_ThC
    # ENT configuration camera settings
    np.set_printoptions(precision=2, floatmode='fixed')
    
    print('Table2, ENT configuration')
    print('---------')
    TEXP=np.array([49.7, 39., 47.3])
    TATM=np.array([20., 40., 20.])
    DIST=np.array([3.047e3, 0., 3.047e3])
    EPS=np.array([0.98, 1., 0.98])
    HUM=np.array([40., 0., 40.])
    print('Texp' , TEXP)
    print('Tatm' , TATM)
    print('Dist' , DIST)
    print('Eps' , EPS)
    print('Hum' , HUM)
    print('---------')
    [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(TEXP,TATM, DIST, EPS, HUM))
    print('T_SR', T_SR)
    print('T_ThC', T_ThC)
    print('---------')
    print('tau_SR', tau_SR)
    print('tau_ThC', tau_ThC)
    print('Robs' , Robs)
    print('Robs_ThC', Robs_ThC)
    
    print('####\n')
    
    print('Table3, ENT configuration')
    print('---------')
    TEXP=np.array([-4.5, -10, -6, 4., 4.5, -4.])
    TATM=20.
    DIST=np.array([0., 3.047e3, 3.047e3, 0., 0., 3.047e3])
    EPS = 0.98
    HUM = 40.
    print('Texp' , TEXP)
    print('Tatm' , TATM)
    print('Dist' , DIST)
    print('Eps' , EPS)
    print('hum', HUM)
    print('---------')
    [T_SR, tau_SR, T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(TEXP,TATM, DIST, EPS, HUM))
    print('T_SR', T_SR)
    print('T_ThC', T_ThC)
    print('---------')
    print('tau_SR', tau_SR)
    print('tau_ThC', tau_ThC)
    print('Robs' , Robs)
    print('Robs_ThC', Robs_ThC)
    
    print('####\n')
    
    print('Table4, ENT configuration')
    print('---------')
    TEXP=np.array([-5.3, 4.3, 4.7, 5., -0.3, -0.3, -3.8, -13., -0.2])
    TATM = 20.
    DIST=np.array([3.047e3, 0., 0., 0., 0., 0., 3.047e3, 3.047e3, 0.])
    EPS=np.array([0.98, 0.98, 0.98, 1., 0.98, 0.98, 0.98, 0.98, 1.])
    HUM=np.array([40., 40., 40., 40., 40., 0., 0., 40., 40.])
    print('Texp' , TEXP)
    print('Tatm', TATM)
    print('Dist' , DIST)
    print('Eps' , EPS)
    print('Hum' , HUM)
    print('---------')
    [T_SR,tau_SR,T_ThC,tau_ThC,Robs,Robs_ThC] = Ttot(TAU_R(TEXP,TATM, DIST, EPS, HUM))
    print('T_SR', T_SR)
    print('T_ThC', T_ThC)
    print('---------')
    print('tau_SR', tau_SR)
    print('tau_ThC', tau_ThC)
    print('Robs' , Robs)
    print('Robs_ThC', Robs_ThC)
    
    print('####\n')
