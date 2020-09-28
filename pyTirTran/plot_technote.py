#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

fileIn_1 = 'Fig_26062020/tau_25_15.dat'
fileIn_2 = 'Fig_26062020/tau_50_15.dat'
fileIn_3 = 'Fig_26062020/tau_75_15.dat'
fileIn_4 = 'Fig_26062020/tau_100_15.dat'

fileIn_1s = 'Fig_26062020/tau_25_15_sensor.dat'
fileIn_2s = 'Fig_26062020/tau_50_15_sensor.dat'
fileIn_3s = 'Fig_26062020/tau_75_15_sensor.dat'
fileIn_4s = 'Fig_26062020/tau_100_15_sensor.dat'

fileIn_1T = 'Fig_26062020/tau_50_0.dat'
fileIn_2T = 'Fig_26062020/tau_50_5.dat'
fileIn_3T = 'Fig_26062020/tau_50_10.dat'
fileIn_4T = 'Fig_26062020/tau_50_15.dat'


fileIn_1Ts = 'Fig_26062020/tau_50_0_sensor.dat'
fileIn_2Ts = 'Fig_26062020/tau_50_5_sensor.dat'
fileIn_3Ts = 'Fig_26062020/tau_50_10_sensor.dat'
fileIn_4Ts = 'Fig_26062020/tau_50_15_sensor.dat'

data_1 = np.genfromtxt(fileIn_1)
data_2 = np.genfromtxt(fileIn_2)
data_3 = np.genfromtxt(fileIn_3)
data_4 = np.genfromtxt(fileIn_4)

data_1s = np.genfromtxt(fileIn_1s)
data_2s = np.genfromtxt(fileIn_2s)
data_3s = np.genfromtxt(fileIn_3s)
data_4s = np.genfromtxt(fileIn_4s)

data_1T = np.genfromtxt(fileIn_1T)
data_2T = np.genfromtxt(fileIn_2T)
data_3T = np.genfromtxt(fileIn_3T)
data_4T = np.genfromtxt(fileIn_4T)

data_1Ts = np.genfromtxt(fileIn_1Ts)
data_2Ts = np.genfromtxt(fileIn_2Ts)
data_3Ts = np.genfromtxt(fileIn_3Ts)
data_4Ts = np.genfromtxt(fileIn_4Ts)
#print np.shape(data_1T)
#plt.plot(data_1T[:,0],data_1T[:,1])
# plt.show()

#plot tau eq. (8) hum=50
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.15, left=0.2)
ax.plot(data_1T[:,0],data_1T[:,1], 'b',linewidth=2,label=r'0$^\circ\mathrm{C}$')
ax.plot(data_2T[:,0],data_2T[:,1], 'g',linewidth=2,label=r'5$^\circ\mathrm{C}$')
ax.plot(data_3T[:,0],data_3T[:,1], 'r',linewidth=2,label=r'10$^\circ\mathrm{C}$')
ax.plot(data_4T[:,0],data_4T[:,1], 'c',linewidth=2,label=r'15$^\circ\mathrm{C}$')
ax.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax.set_xlabel('d [m]',fontsize=14)
ax.set_xlim(left = 0 , right = 1000)
ax.set_ylim(bottom = 0.8 , top = 1)
ax.grid()
ax.legend()
plt.show()
#plot comparison until d=10000, temp=15, hum=50
fig2, ax2 = plt.subplots()
fig2.subplots_adjust(bottom=0.15, left=0.2)
ax2.plot(data_4T[:,0],data_4T[:,1], 'c',linewidth=2,label='Eq.(8)')
ax2.plot(data_4Ts[:,0],data_4Ts[:,1], 'c--',linewidth=2,label='ThermaCAM 595 LW')
ax2.grid()
ax2.legend()
ax2.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax2.set_xlabel('d [m]',fontsize=14)
ax2.set_xlim(left = 0 , right = 10000)
plt.show()

#plot tau sensor hum=50
fig3, ax3 = plt.subplots()
fig3.subplots_adjust(bottom=0.15, left=0.2)
ax3.plot(data_1Ts[:,0],data_1Ts[:,1], 'b--',linewidth=2,label=r'0$^\circ\mathrm{C}$')
ax3.plot(data_2Ts[:,0],data_2Ts[:,1], 'g--',linewidth=2,label=r'5$^\circ\mathrm{C}$')
ax3.plot(data_3Ts[:,0],data_3Ts[:,1], 'r--',linewidth=2,label=r'10$^\circ\mathrm{C}$')
ax3.plot(data_4Ts[:,0],data_4Ts[:,1], 'c--',linewidth=2,label=r'15$^\circ\mathrm{C}$')
ax3.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax3.set_xlabel('d [m]',fontsize=14)
ax3.set_xlim(left = 0 , right = 1000)
ax3.set_ylim(bottom = 0.8 , top = 1)
ax3.grid()
ax3.legend()
plt.show()



#plot tau eq. (8) temp=15
fig4, ax4 = plt.subplots()
fig4.subplots_adjust(bottom=0.15, left=0.2)
ax4.plot(data_1[:,0],data_1[:,1], 'b',linewidth=2,label=r'$\omega_\% =0.25$')
ax4.plot(data_2[:,0],data_2[:,1], 'g',linewidth=2,label=r'$\omega_\% =0.5$')
ax4.plot(data_3[:,0],data_3[:,1], 'r',linewidth=2,label=r'$\omega_\% =0.75$')
ax4.plot(data_4[:,0],data_4[:,1], 'c',linewidth=2,label=r'$\omega_\% =1$')
ax4.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax4.set_xlabel('d [m]',fontsize=14)
ax4.set_xlim(left = 0 , right = 1000)
ax4.set_ylim(bottom = 0.7 , top = 1)
ax4.grid()
ax4.legend()
plt.show()


#plot tau sensor temp=15
fig5, ax5 = plt.subplots()
fig5.subplots_adjust(bottom=0.15, left=0.2)
ax5.plot(data_1s[:,0],data_1s[:,1], 'b--',linewidth=2,label=r'$\omega_\% =0.25$')
ax5.plot(data_2s[:,0],data_2s[:,1], 'g--',linewidth=2,label=r'$\omega_\% =0.5$')
ax5.plot(data_3s[:,0],data_3s[:,1], 'r--',linewidth=2,label=r'$\omega_\% =0.75$')
ax5.plot(data_4s[:,0],data_4s[:,1], 'c--',linewidth=2,label=r'$\omega_\% =1$')
ax5.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax5.set_xlabel('d [m]',fontsize=14)
ax5.set_xlim(left = 0 , right = 1000)
ax5.set_ylim(bottom = 0.7 , top = 1)
ax5.grid()
ax5.legend()
plt.show()


#plot comparison until d=10000, temp=0, hum=50
fig6, ax6 = plt.subplots()
fig6.subplots_adjust(bottom=0.15, left=0.2)
ax6.plot(data_1T[:,0],data_1T[:,1], 'b',linewidth=2,label='Eq.(8)')
ax6.plot(data_1Ts[:,0],data_1Ts[:,1], 'b--',linewidth=2,label='ThermaCAM 595 LW')
ax6.grid()
ax6.legend()
ax6.set_ylabel(r'$\tau$ [-]',fontsize=14)#, fontproperties=font)
ax6.set_xlabel('d [m]',fontsize=14)
ax6.set_xlim(left = 0 , right = 10000)
plt.show()
