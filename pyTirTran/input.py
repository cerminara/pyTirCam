# Temperaure range K
Tmin = 273.15
Tmax = 500. + 273.15
len_T = 1000 
Tmax_fit = Tmax
# 400 ppmv (molar fraction)
rhoCO2 = 0.775e-3
# max of the spectral response 
SpR_max = 0.82
# Define domain of the spectral spectral respose as step function  
lambda_min = 7.5*1e-6
lambda_max = 13.*1e-6
# Camera-object distance (in the paper, it is named d)
L = 3.e+03
# External optics transmittance
tauext = 0.86
# Load the H2O and CO2 absorption coefficients 
# wavelength data is based on H2O wavelength data 
fileIn_gasdataH2O = 'spectralData/h2o_conv.dat'
fileIn_gasdataCO2 = 'spectralData/co2_conv.dat'
# Spectral response file
fileIn_SpR = 'spectralData/sr_calibrated.dat'
fileIn_Temp = 'TEMP.dat'


