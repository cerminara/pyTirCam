#! /usr/bin/env python3

"""
pyTirConv - Python script to recover the radiometric thermal data 
            from compressed thermal images 

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

from PIL import Image, ImageChops 
import numpy as np
#from matplotlib.image import imread
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from PIL import Image, ImageChops
from matplotlib.colors import LogNorm
from functions import *


###---   Flags
analytic = False
evalConv = True
evalRadio = False
load = False

exec(open("input.py").read())


###--- Execute jobs

# Open image with colorbar in RGB mode 
im_tot = Image.open(FILE)
imarr_tot = np.array(im_tot)
#w, h = im_tot.size
# print 'w , h image size ' , w, h
imarr_cut = imarr_tot[hi:hf, wi:wf, :] # selected zone 


# Define colors from image colorbar
temp, COLORBAR = fromJpgToBar(FILE, [Tmin, Tmax], [bar_wi, bar_wf, bar_hi, bar_hf])


#    Find the closest color for the whole image or selected image zone

if load:
  Tarr = np.genfromtxt('Temperature.dat')
  im_out = Image.open('OutputImage.jpg')
  imarr_out = np.array(im_out)
else:
  [Tarr, imarr_out] = fromJpgToArray(FILE, [wi, wf, hi, hf], temp, COLORBAR)
  np.savetxt('Temperature.dat', Tarr, fmt="%.8f")
  im_out = Image.fromarray(imarr_out)
  im_out.save('OutputImage.jpg')

if evalConv:
  #    Plot diff image 
  difference = ImageChops.difference(Image.fromarray(imarr_cut), im_out)
  difference.save('difference.jpg')

###---    Compute the absolute error between the radiometric and the recovered data
if evalRadio:
  # Load the radiometric data  
  data = np.genfromtxt('InputData.dat')
  # Image data dimensions: w = width (column), h = height (raw)
  w_data = data[:,1].size
  h_data = data[1,:].size

  data_cut = data[hi:hf, wi:wf]

  # error in the selected area
  AbsErr = np.abs(Tarr - data_cut)
  print('#### Image selected area: Absolute error = (max, avg, min) Celsius')
  print('With image colorbar: ' , (np.max(AbsErr) , np.average(AbsErr), np.min(AbsErr)))

  #    Plot absolute error 
  # Absolute error of the recovered data with image colorbar
  col_map = 'RdBu_r'
  plt.imshow(AbsErr, norm=LogNorm(vmin=0.1, vmax=5.), cmap=col_map)
  clb1 = plt.colorbar()
  clb1.ax.get_yaxis().labelpad = 15
  clb1.ax.set_ylabel('Absolute Error', rotation=270)
  plt.title(r"Image colorbar; avg error = %.2f $^\circ$C"%(np.average(AbsErr)), fontsize=12)
  plt.show()




if analytic:
  # Define colors from analytic colorbar 
  avga = analyticBar(temp)[1]
  COLORANALYTIC = analyticBar(temp)[1]

  # Find the closest color for a single pixel
  pixel = imarr_tot[28,34,:]
  cc = closest_color(pixel, COLORBAR)
  TEMP_cc = temp[cc] 

  ###--- Check and compare the retrieved colorbar and the analytical colorbar
  # Plot selected pixel, image colorbar, and analytic colorabar RGB
  fig, ax = plt.subplots()
  ax.plot(temp, COLORBAR[:,0], '-r', label='R')
  ax.plot(TEMP_cc, pixel[0], '*r', markersize=10, markeredgewidth=1.0, markeredgecolor='black')
  ax.plot(temp, COLORANALYTIC[:,0], '--r')
  ax.plot(temp, COLORBAR[:,1], '-g', label='G')
  ax.plot(TEMP_cc, pixel[1], '*g', markersize=10, markeredgewidth=1.0, markeredgecolor='black')
  ax.plot(temp, COLORANALYTIC[:,1], '--g')
  ax.plot(temp, COLORBAR[:,2], 'b', label='B')
  ax.plot(TEMP_cc, pixel[2], '*b', markersize=10, markeredgewidth=1.0, markeredgecolor='black')
  ax.plot(temp, COLORANALYTIC[:,2], '--b')
  leg = ax.legend()
  plt.show()

  if load:
    Tarr_analytic = np.genfromtxt('Temperature_analytic.dat')
  else:
    [Tarr_analytic, imarr_analytic] = fromJpgToArray("InputImage.jpg", [wi, wf, hi, hf], temp, COLORANALYTIC)
    np.savetxt('Temperature_analytic.dat', Tarr_analytic, fmt="%.8f")
    im_analytic = Image.fromarray(imarr_analytic)
    im_analytic.save('OutputImage_analytic.jpg')

  if evalRadio:

    AbsErr_analytic = np.abs(Tarr_analytic - data_cut)
    print('#### Image selected area: Absolute error = (max, avg, min) Celsius')
    print('With analytical colorbar: ' , (np.max(AbsErr_analytic) , np.average(AbsErr_analytic), np.min(AbsErr_analytic)))
    # Comparison of the absolute errors
    col_map = 'RdBu_r'
    fig, axs = plt.subplots(1, 2)
    im1 = axs[0].imshow(AbsErr, norm=LogNorm(vmin=0.1, vmax=5.), cmap=col_map)
    axs[0].set_title(r"Image colorbar; avg error = %.2f $^\circ$C"%(np.average(AbsErr)), fontsize=8)
    im2 = axs[1].imshow(AbsErr_analytic, norm=LogNorm(vmin=0.1, vmax=5.), cmap=col_map)
    axs[1].set_title(r"Analytical colorbar; avg error = %.2f $^\circ$C"%(np.average(AbsErr_analytic)), fontsize=8)
    fig.suptitle('Absolute Error', y=0.85, fontsize=16)
    fig.colorbar(im1, ax=axs, shrink=0.6, location='bottom')
    #fig.colorbar(im1, label='Absolute Error', ax=axs, shrink=0.6, location='bottom')
    plt.show()
