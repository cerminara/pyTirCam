#! /usr/bin/env python

import numpy as np
from PIL import Image, ImageChops 
from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm

##--- flags
# insert "N" only if you run the tech_note_color1.py with load=1
load_recdata = "N" 

## Load temperature data of the image as a matrix  
if (load_recdata=="N"):
  data = np.genfromtxt('Jpeg_csv.dat') # exported temperature data
else:
  data = np.genfromtxt('Jpeg_csv.dat') # exported temperature data
  data_cbar = np.genfromtxt('DataTemp_ETNA.dat') # recovered data with image colorbar 
  data_anal = np.genfromtxt('DataTemp_ETNA_A.dat') # recovered data with analytical colorbar 

## Image data dimensions: w = width (column), h = height (raw)
w_data = data[:,1].size
h_data = data[1,:].size

## Select a zone of the image
# w from 0 to w_data (from left to right) - wi<wf
wi = 140
wf = 200
# h from 0 to h_data (from top to bottom) - hi<hf
hi = 140
hf = 188
# data[hi:hf,wi:wf]
if data_cbar.size<data.size:
  Sdata = data[hi:hf, wi:wf]
  Sdata_cbar = data_cbar
  Sdata_anal = data_anal
  # selected area
  SAbsErr_cbar = np.abs(Sdata_cbar-Sdata)
  SAbsErr_anal = np.abs(Sdata_anal-Sdata)
  print '#### Image selected area: Absolute error = (max, avg, min) '
  print 'Image colorbar data ' , (np.max(SAbsErr_cbar) , np.average(SAbsErr_cbar), np.min(SAbsErr_cbar))
  print 'Analytical colorbar data ' , (np.max(SAbsErr_anal) , np.average(SAbsErr_anal), np.min(SAbsErr_anal))
  # selected area
  AbsErr_cbar = np.abs(Sdata_cbar-Sdata)
  AbsErr_anal = np.abs(Sdata_anal-Sdata)
  print '#### Image selected area: Absolute error = (max, avg, min) '
  print 'Image colorbar data ' , (np.max(AbsErr_cbar) , np.average(AbsErr_cbar), np.min(AbsErr_cbar))
  print 'Analytical colorbar data ' , (np.max(AbsErr_anal) , np.average(AbsErr_anal), np.min(AbsErr_anal))
else: 
  Sdata = data[hi:hf, wi:wf]
  Sdata_cbar = data_cbar[hi:hf, wi:wf]
  Sdata_anal = data_anal[hi:hf, wi:wf]

  ## Check selected data dimensions 
  print '#### Image selected area: Dimension = (height,width) '
  print 'Exported data ' ,  (Sdata[:,1].size, Sdata[1,:].size)
  print 'Image colorbar data ' ,  (Sdata_cbar[:,1].size, Sdata_cbar[1,:].size)
  print 'Analytical colorbar data ' ,  (Sdata_anal[:,1].size, Sdata_anal[1,:].size)
  
  #### Absolute error 
  # whole image
  AbsErr_cbar = np.abs(data_cbar-data)
  AbsErr_anal = np.abs(data_anal-data)
  print '#### Whole image: Absolute error = (max, avg, min) '
  print 'Image colorbar data ' , (np.max(AbsErr_cbar) , np.average(AbsErr_cbar), np.min(AbsErr_cbar))
  print 'Analytical colorbar data ' , (np.max(AbsErr_anal) , np.average(AbsErr_anal), np.min(AbsErr_anal))
  # selected area
  SAbsErr_cbar = np.abs(Sdata_cbar-Sdata)
  SAbsErr_anal = np.abs(Sdata_anal-Sdata)
  print '#### Image selected area: Absolute error = (max, avg, min) '
  print 'Image colorbar data ' , (np.max(SAbsErr_cbar) , np.average(SAbsErr_cbar), np.min(SAbsErr_cbar))
  print 'Analytical colorbar data ' , (np.max(SAbsErr_anal) , np.average(SAbsErr_anal), np.min(SAbsErr_anal))

# Opens a image in RGB mode 
im = Image.open("-10_60_Leg.jpg") 
im2 = Image.open("-10_60Jpeg.jpg") 
width, height = im.size 
imarr=np.array(im)
imarr2 = np.array(im2)
#### Check image 
#plt.imshow(imarr2[plume_y0:plume_y1, plume_x0:plume_x1])
#plt.show()

#### Plot absolute errors 
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


