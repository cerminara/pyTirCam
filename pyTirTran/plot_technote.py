#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

fileIn_1 = 'FigureData/tau_25_15.dat'
fileIn_2 = 'FigureData/tau_50_15.dat'
fileIn_3 = 'FigureData/tau_75_15.dat'
fileIn_4 = 'FigureData/tau_100_15.dat'

fileIn_5 = 'FigureData/tau_25_15_step.dat'
fileIn_6 = 'FigureData/tau_50_15_step.dat'
fileIn_7 = 'FigureData/tau_75_15_step.dat'
fileIn_8 = 'FigureData/tau_100_15_step.dat'

fileIn_1s = 'FigureData/tau_25_15_ThC.dat'
fileIn_2s = 'FigureData/tau_50_15_ThC.dat'
fileIn_3s = 'FigureData/tau_75_15_ThC.dat'
fileIn_4s = 'FigureData/tau_100_15_ThC.dat'

fileIn_1T = 'FigureData/tau_50_0.dat'
fileIn_2T = 'FigureData/tau_50_5.dat'
fileIn_3T = 'FigureData/tau_50_10.dat'
fileIn_4T = 'FigureData/tau_50_15.dat'

fileIn_5T = 'FigureData/tau_50_0_step.dat'
fileIn_6T = 'FigureData/tau_50_5_step.dat'
fileIn_7T = 'FigureData/tau_50_10_step.dat'
fileIn_8T = 'FigureData/tau_50_15_step.dat'


fileIn_1Ts = 'FigureData/tau_50_0_ThC.dat'
fileIn_2Ts = 'FigureData/tau_50_5_ThC.dat'
fileIn_3Ts = 'FigureData/tau_50_10_ThC.dat'
fileIn_4Ts = 'FigureData/tau_50_15_ThC.dat'

data_1 = np.genfromtxt(fileIn_1)
data_2 = np.genfromtxt(fileIn_2)
data_3 = np.genfromtxt(fileIn_3)
data_4 = np.genfromtxt(fileIn_4)
data_5 = np.genfromtxt(fileIn_5)
data_6 = np.genfromtxt(fileIn_6)
data_7 = np.genfromtxt(fileIn_7)
data_8 = np.genfromtxt(fileIn_8)

data_1s = np.genfromtxt(fileIn_1s)
data_2s = np.genfromtxt(fileIn_2s)
data_3s = np.genfromtxt(fileIn_3s)
data_4s = np.genfromtxt(fileIn_4s)

data_1T = np.genfromtxt(fileIn_1T)
data_2T = np.genfromtxt(fileIn_2T)
data_3T = np.genfromtxt(fileIn_3T)
data_4T = np.genfromtxt(fileIn_4T)
data_5T = np.genfromtxt(fileIn_5T)
data_6T = np.genfromtxt(fileIn_6T)
data_7T = np.genfromtxt(fileIn_7T)
data_8T = np.genfromtxt(fileIn_8T)

data_1Ts = np.genfromtxt(fileIn_1Ts)
data_2Ts = np.genfromtxt(fileIn_2Ts)
data_3Ts = np.genfromtxt(fileIn_3Ts)
data_4Ts = np.genfromtxt(fileIn_4Ts)
#print np.shape(data_1T)
#plt.plot(data_1T[:,0],data_1T[:,1])
# plt.show()

#plot tau SpR hum=50
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.15, left=0.2)
ax.plot(data_1Ts[:,0],data_1Ts[:,1], 'b',linewidth=2,label=r'0$^\circ\mathrm{C}$')
ax.plot(data_2Ts[:,0],data_2Ts[:,1], 'g',linewidth=2,label=r'5$^\circ\mathrm{C}$')
ax.plot(data_3Ts[:,0],data_3Ts[:,1], 'r',linewidth=2,label=r'10$^\circ\mathrm{C}$')
ax.plot(data_4Ts[:,0],data_4Ts[:,1], 'c',linewidth=2,label=r'15$^\circ\mathrm{C}$')
ax.plot(data_1T[:,0],data_1T[:,1], 'b--',linewidth=2)
ax.plot(data_2T[:,0],data_2T[:,1], 'g--',linewidth=2)
ax.plot(data_3T[:,0],data_3T[:,1], 'r--',linewidth=2)
ax.plot(data_4T[:,0],data_4T[:,1], 'c--',linewidth=2)
ax.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax.set_xlabel('d [m]',fontsize=14)
ax.set_xlim(left = 0 , right = 1000)
ax.set_ylim(bottom = 0.8 , top = 1)
ax.grid()
ax.legend()
plt.title('ThC vs SpR model, humidity 50%')
plt.show()

#plot tau SpR temp=15
fig4, ax4 = plt.subplots()
fig4.subplots_adjust(bottom=0.15, left=0.2)
ax4.plot(data_1s[:,0],data_1s[:,1], 'b',linewidth=2,label=r'$\omega_\% =0.25$')
ax4.plot(data_2s[:,0],data_2s[:,1], 'g',linewidth=2,label=r'$\omega_\% =0.5$')
ax4.plot(data_3s[:,0],data_3s[:,1], 'r',linewidth=2,label=r'$\omega_\% =0.75$')
ax4.plot(data_4s[:,0],data_4s[:,1], 'c',linewidth=2,label=r'$\omega_\% =1$')
ax4.plot(data_1[:,0],data_1[:,1], 'b--',linewidth=2)
ax4.plot(data_2[:,0],data_2[:,1], 'g--',linewidth=2)
ax4.plot(data_3[:,0],data_3[:,1], 'r--',linewidth=2)
ax4.plot(data_4[:,0],data_4[:,1], 'c--',linewidth=2)
ax4.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax4.set_xlabel('d [m]',fontsize=14)
ax4.set_xlim(left = 0 , right = 1000)
ax4.set_ylim(bottom = 0.7 , top = 1)
ax4.grid()
ax4.legend()
plt.title('ThC vs SpR model, temperature 15 C')
plt.show()

#plot tau SpRS hum=50
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.15, left=0.2)
ax.plot(data_1Ts[:,0],data_1Ts[:,1], 'b',linewidth=2,label=r'0$^\circ\mathrm{C}$')
ax.plot(data_2Ts[:,0],data_2Ts[:,1], 'g',linewidth=2,label=r'5$^\circ\mathrm{C}$')
ax.plot(data_3Ts[:,0],data_3Ts[:,1], 'r',linewidth=2,label=r'10$^\circ\mathrm{C}$')
ax.plot(data_4Ts[:,0],data_4Ts[:,1], 'c',linewidth=2,label=r'15$^\circ\mathrm{C}$')
ax.plot(data_5T[:,0],data_5T[:,1], 'b--',linewidth=2)
ax.plot(data_6T[:,0],data_6T[:,1], 'g--',linewidth=2)
ax.plot(data_7T[:,0],data_7T[:,1], 'r--',linewidth=2)
ax.plot(data_8T[:,0],data_8T[:,1], 'c--',linewidth=2)
ax.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax.set_xlabel('d [m]',fontsize=14)
ax.set_xlim(left = 0 , right = 1000)
ax.set_ylim(bottom = 0.8 , top = 1)
ax.grid()
ax.legend()
plt.title('ThC vs SpRS model, humidity 50%')
plt.show()

#plot tau SpRS temp=15
fig4, ax4 = plt.subplots()
fig4.subplots_adjust(bottom=0.15, left=0.2)
ax4.plot(data_1s[:,0],data_1s[:,1], 'b',linewidth=2,label=r'$\omega_\% =0.25$')
ax4.plot(data_2s[:,0],data_2s[:,1], 'g',linewidth=2,label=r'$\omega_\% =0.5$')
ax4.plot(data_3s[:,0],data_3s[:,1], 'r',linewidth=2,label=r'$\omega_\% =0.75$')
ax4.plot(data_4s[:,0],data_4s[:,1], 'c',linewidth=2,label=r'$\omega_\% =1$')
ax4.plot(data_5[:,0],data_5[:,1], 'b--',linewidth=2)
ax4.plot(data_6[:,0],data_6[:,1], 'g--',linewidth=2)
ax4.plot(data_7[:,0],data_7[:,1], 'r--',linewidth=2)
ax4.plot(data_8[:,0],data_8[:,1], 'c--',linewidth=2)
ax4.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax4.set_xlabel('d [m]',fontsize=14)
ax4.set_xlim(left = 0 , right = 1000)
ax4.set_ylim(bottom = 0.7 , top = 1)
ax4.grid()
ax4.legend()
plt.title('ThC vs SpRS model, temperature 15 C')
plt.show()

#plot comparison until d=10000, temp=0, hum=50
fig6, ax6 = plt.subplots()
fig6.subplots_adjust(bottom=0.15, left=0.2)
ax6.plot(data_1Ts[:,0],data_1Ts[:,1], 'b',linewidth=2,label='ThC')
ax6.plot(data_1T[:,0],data_1T[:,1], 'b--',linewidth=2,label='SpR')
ax6.plot(data_5T[:,0],data_5T[:,1], 'b-.',linewidth=2,label='SpRS')
ax6.grid()
ax6.legend()
ax6.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax6.set_xlabel('d [m]',fontsize=14)
ax6.set_xlim(left = 0 , right = 10000)
plt.title('ThC vs SpR and SpRS model, temperature 0 C, humidity 50%')
plt.show()

#plot comparison until d=10000, temp=15, hum=50
fig2, ax2 = plt.subplots()
fig2.subplots_adjust(bottom=0.15, left=0.2)
ax2.plot(data_4Ts[:,0],data_4Ts[:,1], 'c',linewidth=2,label='ThC')
ax2.plot(data_4T[:,0],data_4T[:,1], 'c--',linewidth=2,label='SpR')
ax2.plot(data_8T[:,0],data_8T[:,1], 'c-.',linewidth=2,label='SpRS')
ax2.grid()
ax2.legend()
ax2.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax2.set_xlabel('d [m]',fontsize=14)
ax2.set_xlim(left = 0 , right = 10000)
plt.title('ThC vs SpR and SpRS model, temperature 15 C, humidity 50%')
plt.show()

