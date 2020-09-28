#! /usr/bin/env python

import numpy as np
from PIL import Image, ImageChops 
from matplotlib.image import imread
from matplotlib.colors import LinearSegmentedColormap  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from matplotlib.colors import LogNorm

plume_x0 = 140
plume_x1 = 200
plume_y0 = 140
plume_y1 = 188

data1 = np.genfromtxt('Jpeg_csv.dat')
#data1 = np.genfromtxt('Temperatura_Jpeg_2006.dat')
data2 = np.genfromtxt('/home/utente/infrared/jpgToTIR/TEMP_ETNA.dat')
data3 = np.genfromtxt('/home/utente/infrared/jpgToTIR/TEMP_ETNA_A.dat')
#data2 = np.genfromtxt('cTEMP_ETNA.dat')

data11 = data1#[plume_y0:plume_y1, plume_x0:plume_x1]
data22 = data2#[plume_y0:plume_y1, plume_x0:plume_x1]
data33 = data3#[plume_y0:plume_y1, plume_x0:plume_x1]
print 'dimens[r,c] data11 experim temp ' ,  data11[:,1].size, data11[1,:].size
print 'dimens[r,c] data2 simul temp ' ,  data2[:,1].size, data2[1,:].size
print 'dimens[r,c] data3 simul temp ' ,  data3[:,1].size, data3[1,:].size

#DIFF =  (data11-data2)/data2 #relative error
#print 'relative error ' , DIFF #np.where(DIFF == 0.)

#DIFF =  np.abs(data22-data11)/(273.15+data11) #relative error
DIFF_2 = np.abs(data22-data11)
DIFF_3 = np.abs(data33-data11)
print 'absolute error avg ' , np.max(DIFF_2) , np.average(DIFF_2), np.min(DIFF_2) #np.where(DIFF_2 == 0.)
print 'absolute error ana ' , np.max(DIFF_3) , np.average(DIFF_3), np.min(DIFF_3) #np.where(DIFF_3 == 0.)
#print np.size(DIFF)#data2-data11
#im_DIFF = np.array(DIFF)

# Opens a image in RGB mode 
im = Image.open("-10_60_Leg.jpg") 
im2 = Image.open("-10_60Jpeg.jpg") 
width, height = im.size 
imarr=np.array(im)
imarr2 = np.array(im2)

plt.imshow(imarr2[plume_y0:plume_y1, plume_x0:plume_x1])
#plt.show()
#exit()

bar_x0=330
bar_x1=337
bar_y0=6
bar_y1=234
n_y = bar_y1 - bar_y0
n_bin = n_y
bar = imarr[bar_y0:bar_y1, bar_x0:bar_x1, :]
avg_aus = np.average(bar, axis=1)
avg = avg_aus[::-1]
COLORS = avg

cmap_name = 'my_list'
# convert from integers to floats
COLORS2 = COLORS.astype('float32')
#print 'COLORS ' , COLORS
# normalize to the range 0-1
COLORS2 /= 255.0
#print 'COLORS2 ' , COLORS2
cm = LinearSegmentedColormap.from_list(cmap_name, COLORS2, N=n_bin)

col_map = 'nipy_spectral'
fig, (ax1, ax2) = plt.subplots(1, 2)
im_DIFF_2 = Image.fromarray(DIFF_2)
im_DIFF_3 = Image.fromarray(DIFF_3)
im1 = ax1.imshow(im_DIFF_2, norm=LogNorm(), vmin=0.1, vmax=5., cmap=col_map)
im2 = ax2.imshow(im_DIFF_3, norm=LogNorm(), vmin=0.1, vmax=5., cmap=col_map)
clb1 = fig.colorbar(im1, ax=ax1)
clb1.ax.get_yaxis().labelpad = 15
clb1.ax.set_ylabel('Absolute Error', rotation=270)
clb2 = fig.colorbar(im2, ax=ax2)
clb2.ax.get_yaxis().labelpad = 15
clb2.ax.set_ylabel('Absolute Error', rotation=270)
plt.show()
exit()
#col_map = 'Greys_r'
col_map = 'Greys'
im_DIFF = Image.fromarray(DIFF)
plt.imshow(im_DIFF, vmin=np.min(im_DIFF), vmax=np.max(im_DIFF), cmap=col_map)#, interpolation='nearest', cmap=cm) #,vmin=0.05, vmax=1
#plt.colorbar() 
clb = plt.colorbar()
clb.ax.get_yaxis().labelpad = 15
clb.ax.set_ylabel('Relative Error', rotation=270)
plt.show()

#col_map2 = 'gist_rainbow_r'nipy_spectral
col_map2 = 'nipy_spectral'
#col_map2 = 'gnuplot'
im_Data1 = Image.fromarray(data11)
im_Data2 = Image.fromarray(data22)
plt.imshow(im_Data1, vmin=-10.0, vmax=60.0, cmap=col_map2)#, interpolation='nearest', cmap=cm) #,vmin=0.05, vmax=1
plt.colorbar() 
#clb = plt.colorbar()
#clb.ax.set_ylabel('# of contacts', rotation=270)#.ax.set_title(r'$^\circ$C')
plt.show()

plt.imshow(im_Data2, vmin=-10.0, vmax=60.0, cmap=col_map2)#, interpolation='nearest', cmap=cm) #,vmin=0.05, vmax=1
plt.colorbar() 
plt.show()


print 'TempMax sperimentale e ricavata ' , np.max(data1), np.max(data2)
print 'where TempMax sperimentale e ricavata ' , np.where(data1 == np.max(data1)), np.where(data2 == np.max(data2))
print 'TempMin sperimentale e ricavata ' , np.min(data1), np.min(data2)
print 'where TempMin sperimentale e ricavata ' , np.where(data1 == np.min(data1)), np.where(data2 == np.min(data2))

exit()

levels = [-10.0, 2.0, 12.1, 22.0, 31.8, 42.0, 51.8, 60.0]
#levels = [7.0, 11.1, 14.1, 17.0, 19.8, 22.5, 25.1, 62.5]
cm = col_map2
cmap_nonlin = nlcmap(cm, levels)

fig, (ax2, ax3) = plt.subplots(1, 2)
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#ax1.imshow(cTEMP_cc_ij, interpolation='nearest')#, cmap=cm)
ax2.imshow(im_Data1, interpolation='nearest')#, cmap=cm)
ax3.imshow(im_Data2)#, vmin=Tmin, vmax=Tmax, cmap=cm)
fig.subplots_adjust(left=.28)
plt.subplots_adjust(wspace=0.5)
ccbar_ax = fig.add_axes([0.10, 0.15, 0.05, 0.7])
#for the colorbar we map the original colormap, not the nonlinear one:
sm = plt.cm.ScalarMappable(cmap=cm, 
                norm=plt.Normalize(vmin=-10.0, vmax=60.0))

