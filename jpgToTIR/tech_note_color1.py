#! /usr/bin/env python

from PIL import Image, ImageChops 
import numpy as np
from matplotlib.image import imread
from matplotlib.colors import LinearSegmentedColormap  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


##--- flags
# load = 1 load the recovered temperature data and execute tech_note_temp1.py (load_recdata="N") to compute the absoulte error
load = 0 
#"Y" # to cut and select a zone of the image
cut = "Y"  

##--- Definition: Closest color
def closest_color(rgb, colors):
    r, g, b = rgb
    nColors = len(colors[:, 0])
    color_diffs = np.ones(nColors)*1e10
    for i in range(nColors):
        cr, cg, cb = colors[i, :]
        color_diffs[i] = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
    return np.argmin(color_diffs)

##--- Definition: Analytical colorbar
def analyticBar(xT):
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

#### Open image with colorbar in RGB mode 
im_cbar = Image.open("-10_60_Leg.jpg")
imarr_cbar = np.array(im_cbar)

#### Open image without colorbar 
im_tot = Image.open("-10_60Jpeg.jpg")
# w = image width, h = image height
w, h = im_tot.size
print 'w , h image size ' , w, h
imarr_tot = np.array(im_tot)

##-- Insert temperature range of the image colorbar 
Tmin = -10.
Tmax = 60.0

##-- Select the dimension of the colorbar (image dimension 320x240)
bar_wi=330
bar_wf=337
bar_hi=6
bar_hf=234
n_h = bar_hf - bar_hi

##-- Select dimension of the zone of the image
wi = 140
wf = 200
hi = 140
hf = 188

##-- Insert temperature range
Tmin = -10.
Tmax = 60.0
temp = np.linspace(Tmin, Tmax, n_h)

#### Select the colorbar and image zone
bar = imarr_cbar[bar_hi:bar_hf, bar_wi:bar_wf, :]
if (cut=="Y"):
  Simarr = imarr_tot[hi:hf, wi:wf, :] # selected zone 
else: 
  Simarr = imarr_tot
Sim = Image.fromarray(Simarr)
Sim.save('Color_ORIG.jpg')
w_Sim = Sim.size[0] 
h_Sim = Sim.size[1]
print 'w_Sim, h_Sim ', w_Sim, h_Sim

##-- bar e plume control
plt.imshow(bar)
#plt.show()
plt.imshow(Simarr)
#plt.show()

#### Define colors from image colorbar
avg_aus = np.average(bar, axis=1)
avg = avg_aus[::-1]
COLORS = avg

##--find the closest color for a single pixel
pixel = Simarr[28,34,:]
cc = closest_color(pixel, COLORS)
TEMP_cc = temp[cc] 

#### Define colors from analytic colorbar 
ra, ga, ba = analyticBar(temp)
avga = np.c_[ra, ga, ba]
COLORS_anal = avga

##-- Comparison between image colorbar and analytic colorabar colors
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

#### Find the closest color for all the image

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
  print("--- %s seconds ---" % (time.time() - start_time))
else:
  data_cbar = np.genfromtxt('DataTemp_ETNA.dat') # recovered data with image colorbar
  data_anal = np.genfromtxt('DataTemp_ETNA_A.dat') # recovered data with analytical colorbar
  Sim2 = Image.open('Color_ETNA.jpg')
  Sim2_anal = Image.open('Color_ETNA_A.jpg')
  Simarr2 = np.array(Sim2)
  Simarr2_anal = np.array(Sim2_anal)
  #### Absolute error of the temperature data 
  #### recovered with image colorbar and with analytical colorbar
  load_recdata = "N"
  execfile('tech_note_temp1.py')

