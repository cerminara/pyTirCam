#! /usr/bin/env python

"""
pyTirConv - Python script to recover the radiometric thermal data 
            from compressed thermal images 

Copyright (C) 2020-#### 
               by    Matteo Cerminara EMAIL  
               and   Benedetta Calusi EMAIL

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

from PIL import Image, ImageChops 
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageChops
from matplotlib.colors import LogNorm
import time


#   flags

# load = 0 retrieve, save and load the recover radiometric data
# load = 1 load the recovered radiometric temperature data
load = 1 # 0 

# "Y" cut and select a zone of the image 
# "N" mantain the original image size
cut = "N"  

def closest_color(rgb, colors):
    """
    Definition: Closest color
    """
    r, g, b = rgb
    nColors = len(colors[:, 0])
    color_diffs = np.ones(nColors)*1e10
    for i in range(nColors):
        cr, cg, cb = colors[i, :]
        color_diffs[i] = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
    return np.argmin(color_diffs)

def analyticBar(xT):
    """
    Definition: Analytical colorbar
    """
    T1 = -10.
    T2 = 0.
    T3 = 10.
    T4 = 18.
    T5 = 28.
    T6 = 38.
    T7 = 48.
    T8 = 60.
    conds = [xT < T2,
             (xT >= T2) & (xT < T3),
             (xT >= T3) & (xT < T4),
             (xT >= T4) & (xT < T5),
             (xT >= T5) & (xT < T6),
             (xT >= T6) & (xT < T7),
             xT >= T7]
    funcr = [lambda xT: 255./(T2-T1)*xT + 255.,
             lambda xT: -255./(T3-T2)*xT + 255.,
             lambda xT: 0.,
             lambda xT: 0.,
             lambda xT: 255./(T6-T5)*(xT-T5),
             lambda xT: -55./(T7-T6)*(xT-T6) + 255.,
             lambda xT: 55./(T8-T7)*(xT-T7) + 200.]
    funcg = [lambda xT: 0.,
             lambda xT: 5.0*xT,
             lambda xT: 195./(T4-T3)*(xT-T3) + 50.,
             lambda xT: -140/(T5-T4)*(xT-T4) + 245.,
             lambda xT: 150./(T6-T5)*(xT-T5) + 105.,
             lambda xT: -255./(T7-T6)*(xT-T6) + 255.,
             lambda xT: 240./(T8-T7)*(xT-T7)]
    funcb = [lambda xT: 250./(T2-T1)*(xT-T1),
             lambda xT: -55./(T3-T2)*(xT-T2) + 250.,
             lambda xT: 45./(T4-T3)*(xT-T3) + 195.,
             lambda xT: -240./(T5-T4)*(xT-T4) + 240.,
             lambda xT: 0.,
             lambda xT: 0.,
             lambda xT: 240./(T8-T7)*(xT-T7)]
    r = np.piecewise(xT, conds, funcr)
    g = np.piecewise(xT, conds, funcg)
    b = np.piecewise(xT, conds, funcb)
    return r, g, b

#    Inputs 

# Open image with colorbar in RGB mode 
im_cbar = Image.open("-10_60_Leg.jpg")
imarr_cbar = np.array(im_cbar)
# Open image without colorbar 
im_tot = Image.open("-10_60Jpeg.jpg")
# w = image width, h = image height
w, h = im_tot.size
# print 'w , h image size ' , w, h
imarr_tot = np.array(im_tot)

# Select the dimension of the colorbar (image dimension 320x240)
bar_wi=330
bar_wf=337
bar_hi=6
bar_hf=234
n_h = bar_hf - bar_hi

# Insert temperature range of the image colorbar 
Tmin = -10.
Tmax = 60.0
temp = np.linspace(Tmin, Tmax, n_h)

# Select the colorbar and image zone
bar = imarr_cbar[bar_hi:bar_hf, bar_wi:bar_wf, :]
if (cut=="Y"):
    # Select dimension of the zone of the image
    wi = 140
    wf = 200
    hi = 140
    hf = 188
    Simarr = imarr_tot[hi:hf, wi:wf, :] # selected zone 
else: 
    Simarr = imarr_tot
Sim = Image.fromarray(Simarr)
Sim.save('Color_ORIG.jpg')
w_Sim = Sim.size[0] 
h_Sim = Sim.size[1]
# print 'w_Sim, h_Sim ', w_Sim, h_Sim

# Check bar and image 
plt.imshow(bar)
#plt.show()
plt.imshow(Simarr)
#plt.show()

#    Check and compare the retrieved colorbar and the analytical colorbar
# Define colors from image colorbar
avg_aus = np.average(bar, axis=1)
avg = avg_aus[::-1]
COLORS = avg

# Find the closest color for a single pixel
pixel = Simarr[28,34,:]
cc = closest_color(pixel, COLORS)
TEMP_cc = temp[cc] 

# Define colors from analytic colorbar 
ra, ga, ba = analyticBar(temp)
avga = np.c_[ra, ga, ba]
COLORS_anal = avga

# Plot selected pixel, image colorbar, and analytic colorabar RGB
fig, ax = plt.subplots()
ax.plot(temp, avg[:,0], '-r', label='R')
ax.plot(TEMP_cc, pixel[0], '.r')
ax.plot(temp, ra, '--r')
ax.plot(temp, avg[:,1], '-g', label='G')
ax.plot(TEMP_cc, pixel[1], '.g')
ax.plot(temp, ga, '--g')
ax.plot(temp, avg[:,2], 'b', label='B')
ax.plot(TEMP_cc, pixel[2], '.b')
ax.plot(temp, ba, '--b')
leg = ax.legend()
#plt.show()

#    Find the closest color for the whole image or selected image zone
start_time = time.time()
Simarr2 = Simarr.copy()
Simarr2_anal = Simarr.copy()
if not load:
    TEMP_cc_ij = np.zeros((h_Sim,w_Sim))
    TEMP_cc_ij_anal = np.zeros((h_Sim,w_Sim))
    for ii in range(h_Sim):
        for jj in range(w_Sim):
            pixel_ij = Simarr[ii,jj,:]
            cc_ij = closest_color(pixel_ij, COLORS)
            cc_ij_anal = closest_color(pixel_ij, COLORS_anal) 
            TEMP_cc_ij[ii,jj] = temp[cc_ij]  
            TEMP_cc_ij_anal[ii,jj] = temp[cc_ij_anal] 
            Simarr2[ii,jj,:] = COLORS[cc_ij, :]
            Simarr2_anal[ii,jj,:] = COLORS_anal[cc_ij, :]
    np.savetxt('DataTemp_ETNA.dat', TEMP_cc_ij, fmt="%.8f")
    np.savetxt('DataTemp_ETNA_A.dat', TEMP_cc_ij_anal, fmt="%.8f")
    Sim2 = Image.fromarray(Simarr2)
    Sim2_anal = Image.fromarray(Simarr2_anal)
    Sim2.save('Color_ETNA.jpg')
    Sim2_anal.save('Color_ETNA_A.jpg')
    # load_recdata = "Y"
    data_cbar = np.genfromtxt('DataTemp_ETNA.dat') # recovered data with image colorbar
    data_anal = np.genfromtxt('DataTemp_ETNA_A.dat') # recovered data with analytical colorbar
    print("--- %s seconds ---" % (time.time() - start_time))
else:
    data_cbar = np.genfromtxt('DataTemp_ETNA.dat') # recovered data with image colorbar
    data_anal = np.genfromtxt('DataTemp_ETNA_A.dat') # recovered data with analytical colorbar
    Sim2 = Image.open('Color_ETNA.jpg')
    Sim2_anal = Image.open('Color_ETNA_A.jpg')
    Simarr2 = np.array(Sim2)
    Simarr2_anal = np.array(Sim2_anal)
    # load_recdata = "N"

##########################################
#         TOGLIEREI
# ## Load temperature data of the image as a matrix
# if (load_recdata=="N"):
#     data = np.genfromtxt('Jpeg_csv.dat') # exported radiometric temperature data
# else:
#     data = np.genfromtxt('Jpeg_csv.dat') # exported radiometric temperature data
#     data_cbar = np.genfromtxt('DataTemp_ETNA.dat') # recovered data with image colorbar
#     data_anal = np.genfromtxt('DataTemp_ETNA_A.dat') # recovered data with analytical colorbar
##########################################

#    Compute the absolute error between the radiometric and the recovered data
# Load the radiometric data  
data = np.genfromtxt('Jpeg_csv.dat') # exported radiometric temperature data
# Image data dimensions: w = width (column), h = height (raw)
w_data = data[:,1].size
h_data = data[1,:].size

if  (cut=="Y"):
    Sdata = data[hi:hf, wi:wf]
    if data_cbar.size > Sdata.size: 
        print "WARNING: the loaded data files have different size. Check the load and cut flags. "
        print "Size radiometric data: " ,  Sdata.size
        print "Size recovered data: " , data_cbar.size 
        print "UPDATE flags with (load = 0 and cut = ""Y"") or (load = 1 and cut = ""N"")."
        exit()
    else: 
        Sdata_cbar = data_cbar
        Sdata_anal = data_anal
        # selected area
        AbsErr_cbar = np.abs(Sdata_cbar-Sdata)
        AbsErr_anal = np.abs(Sdata_anal-Sdata)
        print '#### Image selected area: Absolute error = (max, avg, min) '
        print 'Image colorbar data ' , (np.max(AbsErr_cbar) , np.average(AbsErr_cbar), np.min(AbsErr_cbar))
        print 'Analytical colorbar data ' , (np.max(AbsErr_anal) , np.average(AbsErr_anal), np.min(AbsErr_anal))
else:
    if data_cbar.size<data.size:
        print "WARNING: the loaded data files have different size. Check the load and cut flags."
        print "Size radiometric data: " ,  data.size
        print "Size recovered data: " , data_cbar.size
        print "UPDATE flags with (load = 0 and cut = ""N"") or (load = 1 and cut = ""Y"")."
        exit()
    else:
        # whole image
        AbsErr_cbar = np.abs(data_cbar-data)
        AbsErr_anal = np.abs(data_anal-data)
        print '#### Whole image: Absolute error = (max, avg, min) '
        print 'Image colorbar data ' , (np.max(AbsErr_cbar) , np.average(AbsErr_cbar), np.min(AbsErr_cbar))
        print 'Analytical colorbar data ' , (np.max(AbsErr_anal) , np.average(AbsErr_anal), np.min(AbsErr_anal))

#    Plot absolute error 
# Image from absolute error data
im_AbsErr_cbar = Image.fromarray(AbsErr_cbar)
im_AbsErr_anal = Image.fromarray(AbsErr_anal)
# Comparison of the absolute errors
col_map = 'RdBu_r'
fig, axs = plt.subplots(1, 2)
im1 = axs[0].imshow(im_AbsErr_cbar, norm=LogNorm(), vmin=0.1, vmax=5., cmap=col_map)
axs[0].set_title('Image colorbar')
im2 = axs[1].imshow(im_AbsErr_anal, norm=LogNorm(), vmin=0.1, vmax=5., cmap=col_map)
axs[1].set_title('Analytical colorbar')
fig.suptitle('Absolute Error', y=0.85, fontsize=16)
fig.colorbar(im1, ax=axs, shrink=0.6, location='bottom')
#fig.colorbar(im1, label='Absolute Error', ax=axs, shrink=0.6, location='bottom')
plt.show()
# Absolute error of the recovered data with image colorbar
col_map = 'RdBu_r'
plt.imshow(im_AbsErr_cbar, norm=LogNorm(), vmin=0.1, vmax=5., cmap=col_map)
clb1 = plt.colorbar()
clb1.ax.get_yaxis().labelpad = 15
clb1.ax.set_ylabel('Absolute Error', rotation=270)
plt.show()

