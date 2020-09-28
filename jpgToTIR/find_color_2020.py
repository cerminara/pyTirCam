#! /usr/bin/env python

from PIL import Image, ImageChops 
import numpy as np
from matplotlib.image import imread
from matplotlib.colors import LinearSegmentedColormap  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


# flags
load =0 


##--- definition 1
def closest_color(rgb, colors):
    r, g, b = rgb
    nColors = len(colors[:, 0])
    color_diffs = np.ones(nColors)*1e10
    for i in range(nColors):
        cr, cg, cb = colors[i, :]
        color_diffs[i] = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
    return np.argmin(color_diffs)

##--- class definition 2
class nlcmap(object):
    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels
        self.levmin = self.levels.min()
        self.levmax = self.levels.max()
        self.transformed_levels = np.linspace(self.levmin, self.levmax, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self.transformed_levels)
        return self.cmap(yi / self.levmax, alpha)

##--- analytical_bar
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
             lambda xT: -140/(T5-T4)*(xT-T4)+245.,
             lambda xT: 150./(T6-T5)*(xT-T5)+105.,
             lambda xT: -255./(T7-T6)*(xT-T6) + 255.,
             lambda xT: 240./(T8-T7)*(xT-T7)]
    funcb = [lambda xT: 250./(T2-T1)*(xT-T1),
             lambda xT: -55./(T3-T2)*(xT-T2) + 250.,
             lambda xT: 45./(T4-T3)*(xT-T3) + 195.,
             lambda xT: -240./(T5-T4)*(xT-T4)+240.,
             lambda xT: 0.,
             lambda xT: 0.,
             lambda xT: 240./(T8-T7)*(xT-T7)]
    r = np.piecewise(xT, conds, funcr)
    g = np.piecewise(xT, conds, funcg)
    b = np.piecewise(xT, conds, funcb)
    return r, g, b


##---
Tmin = -10.#7
Tmax = 60.0#62.5

## Legenda immagine 320x240
bar_x0=330
bar_x1=337
bar_y0=6
bar_y1=234

## Legenda immagine 776x520
#bar_x0=701
#bar_x1=710
#bar_y0=7
#bar_y1=513

#cbar_x0=330
#cbar_x1=369
#cbar_y0=33
#cbar_y1=206

plume_x0 = 140#265
plume_x1 = 200#380
plume_y0 = 140#280
plume_y1 = 188#400


#plume_x0 = 98
#plume_x1 = 210
#plume_y0 = 14
#plume_y1 = 204

# Opens a image in RGB mode 
#im = Image.open("-10_60_Legbig.jpg") 
im = Image.open("-10_60_Leg.jpg") 
width, height = im.size 
print 'w , h image size ' , width, height   
imarr=np.array(im)

# open second image 
im11 = Image.open("-10_60Jpeg.jpg")
width11, height11 = im11.size 
print 'w11 , h11 image size ' , width11, height11   
imarr11=np.array(im11)

n_y = bar_y1 - bar_y0
temp = np.linspace(Tmin, Tmax, n_y)

bar = imarr[bar_y0:bar_y1, bar_x0:bar_x1, :]
#cbar = imarr[cbar_y0:cbar_y1, cbar_x0:cbar_x1, :]
plume = imarr11#[plume_y0:plume_y1, plume_x0:plume_x1, :] # 
imP = Image.fromarray(plume)
imP.save('COLR_ORIG.jpg')
width_p = imP.size[0] #.shape[1]
height_p = imP.size[1] #.shape[2]
print 'w_p, h_p ', width_p, height_p
#exit()
## -- bar e plume control
#plt.imshow(bar)#cbar)
plt.imshow(plume)#cbar)
plt.show()
#exit()

#### questa parte serve per colorbar non lineare
#cn_y_top = cbar_y0 - bar_y0
#cn_y_middle = cbar_y1 - cbar_y0
#cn_y_bottom = bar_y1 - cbar_y1
#cTmin = 11.1
#cTmax = 25.1
#ctemp_top = np.linspace(cTmax, Tmax, cn_y_top)
#ctemp_middle = np.linspace(cTmin, cTmax, cn_y_middle)
#ctemp_bottom = np.linspace(Tmin, cTmin, cn_y_bottom)
#ctemp = np.append(ctemp_bottom,ctemp_middle)
#ctemp = np.append(ctemp,ctemp_top)
#print ctemp.size, n_y
#exit()#.shape[0]


avg_aus = np.average(bar, axis=1)
avg = avg_aus[::-1]
COLORS = avg

###---find the closest color for a single pixel
pixel = plume[28,34,:]#imarr[30,30,:]
#print pixel  
cc = closest_color(pixel, COLORS)
TEMP_cc = temp[cc] 
#TEMP_cc = ctemp[cc]
#print cc, temp[cc]

ra, ga, ba = analyticBar(temp)
avga = np.c_[ra, ga, ba]
COLORS = avga

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
leg = ax.legend();
plt.show()
#exit()

###---find the closest color for all the image

start_time = time.time()
plume2 = plume.copy()
if not load:
  TEMP_cc_ij = np.zeros((height_p,width_p))
  #cTEMP_cc_ij = np.zeros((height_p,width_p))
  for ii in range(height_p):
    for jj in range(width_p):
      pixel_ij = plume[ii,jj,:]#imarr[ii,jj,:]
      cc_ij = closest_color(pixel_ij, COLORS)#imarr[ii,jj,:])
      TEMP_cc_ij[ii,jj] = temp[cc_ij] # fornisce i colori giusti 
      #cTEMP_cc_ij[ii,jj] = ctemp[cc_ij] #fornisce i valori giusti di temp seguendo la scala di colmap
      plume2[ii,jj,:] = COLORS[cc_ij, :]
  #np.savetxt('TEMP_ETNA.dat', TEMP_cc_ij, fmt="%.8f")
  np.savetxt('TEMP_ETNA_A.dat', TEMP_cc_ij, fmt="%.8f")
  #np.savetxt('cTEMP_ETNA.dat', cTEMP_cc_ij, fmt="%.8f")
  im2 = Image.fromarray(plume2)
  im2.save('COLR_ETNA.jpg')
  #im2.save('COLR_ETNA_A.jpg')
  print("--- %s seconds ---" % (time.time() - start_time))
else:
  TEMP_cc_ij = np.genfromtxt('TEMP_ETNA_A.dat')
  im2 = Image.open('COLR_ETNA_A.jpg')
  plume2 = np.array(im2)


exit()
cmap_name = 'my_list'
n_bin = n_y 
# convert from integers to floats
COLORS2 = COLORS.astype('float32')
#print 'COLORS ' , COLORS
# normalize to the range 0-1
COLORS2 /= 255.0
#print 'COLORS2 ' , COLORS2

#newcolors = np.vstack(COLORScn_y_bottom,
                       #cn_y_middle, cn_y_top)
#cm = mpl.colors.ListedColormap(newcolors, name='GreyBlue')

cm = LinearSegmentedColormap.from_list(cmap_name, COLORS2, N=n_bin) # plt.cm.nipy_spectral

plt.imshow(TEMP_cc_ij, interpolation='nearest', cmap=cm)
#plt.imshow(cTEMP_cc_ij, interpolation='nearest', cmap=cm)
plt.colorbar() # Create the colorbar
plt.show()

#exit()


plt.subplot(1,2,1)
plt.imshow(TEMP_cc_ij, interpolation='nearest', cmap=cm)
plt.colorbar() # Create the colorbar
plt.subplot(1,2,2)
plt.imshow(plume, vmin=Tmin, vmax=Tmax, cmap=cm)
plt.colorbar()
plt.subplots_adjust(wspace=0.5)
plt.savefig('out.png')
plt.close()

#exit()
#levels = ctemp
levels = [-10.0, 2.0, 12.1, 22.0, 31.8, 42.0, 51.8, 60.0]
#levels = [7.0, 11.1, 14.1, 17.0, 19.8, 22.5, 25.1, 62.5]
cmap_nonlin = nlcmap(cm, levels)

fig, (ax2, ax3) = plt.subplots(1, 2)
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#ax1.imshow(cTEMP_cc_ij, interpolation='nearest')#, cmap=cm)
ax2.imshow(TEMP_cc_ij, interpolation='nearest')#, cmap=cm)
ax3.imshow(plume)#, vmin=Tmin, vmax=Tmax, cmap=cm)
fig.subplots_adjust(left=.28)
plt.subplots_adjust(wspace=0.5)
ccbar_ax = fig.add_axes([0.10, 0.15, 0.05, 0.7])
#for the colorbar we map the original colormap, not the nonlinear one:
sm = plt.cm.ScalarMappable(cmap=cm, 
                norm=plt.Normalize(vmin=np.min(temp), vmax=np.max(temp))) # 
#sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, 
#                norm=plt.Normalize(vmin=0, vmax=tmax))
sm._A = []
ccbar = fig.colorbar(sm, cax=ccbar_ax)
#here we are relabel the linear colorbar ticks to match the nonlinear ticks
ccbar.set_ticks(cmap_nonlin.transformed_levels)
ccbar.set_ticklabels(["%.1f" % lev for lev in levels])
plt.show()


#---
diff = ImageChops.difference(imP, im2)
print np.min(diff), np.max(diff) , np.size(diff)
#plt.colorbar()
plt.imshow(diff,cmap=cm) #,vmin=0, vmax=0.05 'nipy_spectral'
plt.colorbar() 
#plt.show()
diff.save('COLR_DIFF.jpg')

im_diff = Image.open("COLR_DIFF.jpg") # e' in rgb mode
im_diff2 = np.array(im_diff)
print np.min(im_diff2) , np.max(im_diff2) , np.size(im_diff2)
im_DIFF = Image.fromarray(im_diff2)
plt.imshow(im_DIFF,cmap=cm) #,vmin=0.05, vmax=1
plt.colorbar() 
#plt.show()

#diff2 = ImageChops.subtract(imP, im2, scale=1.0, offset=0)
#diff2.save('COLR_DIFF2.jpg')





