"""
Created on Sat Oct 10 16:04:28 2020

@author: manuelmelon
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from numpy import linalg as LA
from scipy.fftpack import ifft, fftshift, fft
#plt.close('all');

c=340 # Speed of sound
rho=1.2 # Air density

#  Source geometry
N=8 # Number of sources
# rs=2  # Radius (m)
# dtheta=2*np.pi/N # Angular spacing
# theta=np.linspace(0,2*np.pi-dtheta,N); # Angular vector
# xs=rs*np.cos(theta) # Coordinate vector of the sources along the x direction
#ys=rs*np.sin(theta); # Coordinate vector of the sources along the y direction
if N==4:
    xs=[ -.505, -.225, .225, .505]
elif N==6:
    xs=[-.505, -.225, -.15, .15, .225, .505]
elif N==8:
    xs=[-.58, -.505, -.225, -.15, .15, .225, .505, .58]

ys=np.zeros(N)+1e-2;

# Zone Geometries
Mx=3
My=4 # Square root of number of control points per zones
dmx=2e-2# Microphone spacing
dmy=5e-2

# Dark zone
# Coordinates of the zone center
xcd1=.235
xcd2=.235+.22
ycd1=0;


# Coordinates of the control points
xd1=np.arange(Mx)*dmx+xcd1
xd1=np.matlib.repmat((xd1),My,1).transpose().reshape(1,Mx*My)
xd2=np.arange(Mx)*dmx+xcd2
xd2=np.matlib.repmat((xd2),My,1).transpose().reshape(1,Mx*My)
xd=np.concatenate((xd1,xd2), axis=1)
yd=-np.arange(My)*dmy+ycd1
yd=np.matlib.repmat(yd,1,Mx)
yd=np.concatenate((yd,yd), axis=1)


# Bright zone
#  Coordinates of the zone center
xcb1=-.235
xcb2=-.235-.22
ycb1=0
# Coordinates of the control points
xb1=-np.arange(Mx)*dmx+xcb1
xb1=np.matlib.repmat((xb1),My,1).transpose().reshape(1,Mx*My)
xb2=-np.arange(Mx)*dmx+xcb2
xb2=np.matlib.repmat((xb2),My,1).transpose().reshape(1,Mx*My)
xb=np.concatenate((xb1,xb2), axis=1)
yb=-np.arange(My)*dmy+ycb1
yb=np.matlib.repmat(yb,1,Mx)
yb=np.concatenate((yb,yb), axis=1)

# Plot of the problem geometry
plotgeom=1 # 1 = plot
if plotgeom==1:
    plt.figure(1)
    plt.plot(xs,ys,'rs',linewidth=2.0,markersize=6.0,fillstyle='none')
    plt.plot(xd,yd,'co',linewidth=2.0,markersize=6.0,fillstyle='none');
    plt.plot(xb,yb,'yo',linewidth=2.0,markersize=6.0,fillstyle='none');
    plt.xlabel('x-axis (m)',fontsize=14)
    plt.ylabel('y-axis (m)',fontsize=14)
    plt.axis('equal')
    plt.grid()


# Calculation of the distances of interest
(Xs,Xd)=np.meshgrid(xs,xd)
(Xs,Xb)=np.meshgrid(xs,xb)
(Ys,Yd)=np.meshgrid(ys,yd)
(Ys,Yb)=np.meshgrid(ys,yb)
rd=np.sqrt((Xd-Xs)**2+(Yd-Ys)**2)
rb=np.sqrt((Xb-Xs)**2+(Yb-Ys)**2)


Fs=3000;
Np=196;
Fv=np.linspace(0,Fs/2,Np+1)
Id=np.eye(N);
#Pt=np.ones((Mx*My*2,1))
Qr=1e-3;
Qt=np.zeros([N,1])

# Speaker used for the target pressure
if N==8:
    Qt[1:3]=Qr
else:
    Qt[0:2]=Qr
    
Qnf=np.ones((N,1))*Qr
Qpmf=np.zeros([N,Np+1],dtype=complex)
Pb=np.zeros([Mx*My*2,Np+1],dtype=complex)
Pd=np.zeros([Mx*My*2,Np+1],dtype=complex)
Contrast=np.zeros([Np+1])
Err=np.zeros([Np+1])
Cond=np.zeros([Np+1])
Eff=np.zeros([Np+1])


for it in range(Np+1):
    f=Fv[it] # Frequency
    k=2*np.pi*f/c # Wave vector
    # Calculation of the transfer functions betweenn sources and dark and bright zones.
    Gd=1j*k*rho*c*np.exp(-1j*k*rd)/(4*np.pi*rd)
    Gb=1j*k*rho*c*np.exp(-1j*k*rb)/(4*np.pi*rb)
    Pt=Gb@Qt
    Qpm=LA.inv((np.conjugate(np.transpose(Gb))@Gb)+np.conjugate(np.transpose(Gd))@Gd+1e-12*Id)@(np.conjugate(np.transpose(Gb))@Pt)
    Cond[it]=LA.cond(np.conjugate(np.transpose(Gb))@Gb+np.conjugate(np.transpose(Gd))@Gd)
    Qpmf[:,it]=np.squeeze(Qpm)
    Pbpm=Gb@Qpm
    Pdpm=Gd@Qpm
    Pb[:,it]=np.squeeze(Pbpm)
    Pd[:,it]=np.squeeze(Pdpm)
    Cont=10*np.log10(sum(abs(np.power(Pbpm,2)))/sum(abs(np.power(Pdpm,2))))
    Contrast[it]=float(Cont)
    err=np.conjugate(np.transpose(Pt-Pbpm))@(Pt-Pbpm)/(np.conjugate(np.transpose(Pt))@(Pt))
    Err[it]=np.squeeze(np.abs(err))
    Eff[it]=10*np.log(np.real(np.conjugate(np.transpose(Qpm))@Qpm)/(2*Qr**2))


Qpmfn=np.fliplr(Qpmf).conj()
Qpmt=np.concatenate((Qpmf[:,:-1], Qpmfn[:,:-1]), axis=1)
Qpmri=fftshift(np.real(ifft(Qpmt)),axes=1)
Qpm_ver=fft(Qpmri)

# Plot of the impulses responses
plotIRs=0 # 1 = plot
if plotIRs==1:
    plt.figure(2)
    for fig in range(N):
        plt.subplot(3,3,fig+1)
        plt.plot(np.arange(0,70)/Fs,Qpmri[fig,Np-25:Np+45]/Qr)
        plt.xlabel('Time [s]')
        plt.suptitle('Filter impulse responses')

# Plot of the filter spectrum ampliturdes
plotfiltamp=0 # 1 = plot
if plotfiltamp==1:
    plt.figure(3)
    for fig in range(N):
        plt.subplot(3,3,fig+1) 
        plt.semilogx(20*np.log10(np.abs(Qpm_ver[fig,0:Np])/Qr))
        plt.semilogx((20*np.log10(np.abs(Qpmf[fig,:])/Qr)))
        plt.grid(True)
        plt.ylim(-60, 10)
        plt.xlabel('Frequency [Hz]')
        plt.suptitle('Filter amplitudes')
    
# Plot of the filter spectrum phases
plotfiltpha=0 # 1 = plot
if plotfiltpha==1:
    plt.figure(4)
    for fig in range(N):
        plt.subplot(3,3,fig+1)
        plt.semilogx((np.angle(Qpmf[fig,:])))
        plt.grid(True)
        plt.ylim(-np.pi, np.pi)
        plt.xlabel('Frequency [Hz]')
        plt.suptitle('Filter phases [rad]')

 
# Plot of Contrast
plotcont=1 # 1 = plot
if plotcont==1:
    plt.figure(5)  
    plt.semilogx(Fv,Contrast)
    plt.grid(b=True,which='both',axis='both')
    plt.axis([50, 1500, 0, 60])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Contrast [dB]')

# Plot of error
ploterr=1 # 1 = plot
if ploterr==1:
    plt.figure(6)  
    plt.loglog(Fv,Err)
    plt.grid(b=True,which='both',axis='both')
    plt.axis([50, 1500, 1e-7, 1e1])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Error')
    
# Plot of efffort
ploteff=1 # 1 = plot
if ploteff==1:
    plt.figure(7)  
    plt.semilogx(Fv,Eff)
    plt.grid(b=True,which='both',axis='both')
    plt.axis([50, 1500, -5, 20])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Effort [dB]')

# Plot of condition number
plotcond=0 # 1 = plot
if plotcond==1:
    plt.figure(8)  
    plt.semilogx(Fv,Cond)
    plt.grid(b=True,which='both',axis='both')
    plt.axis([50, 1500, 0, 4500])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Conditionnement [dB]')
    
    
#plot of pressure phase  in the bright zone
plotears=0 # 1 = plot 
if plotears==1:
    plt.figure(9)
    for it in range(12):
        plt.semilogx(Fv,np.angle(Pb[it,:]),'r')
        plt.semilogx(Fv,np.angle(Pb[it+12,:]),'b')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase in the bright zone [rad]')
    plt.grid(b=True,which='both',axis='both')
    plt.axis([50, 1500, -np.pi, np.pi])
    plt.legend(['Right', 'Left'])


#plots for a given frequency
plotf=0
if plotf==1:
    F=1400 # chosen frequency
    ind1=np.argmin(abs(Fv-F))
    k=2*np.pi*Fv[ind1]/c 
    Gd=1j*k*rho*c*np.exp(-1j*k*rd)/(4*np.pi*rd)
    Gb=1j*k*rho*c*np.exp(-1j*k*rb)/(4*np.pi*rb)
    
    do=2e-2
    Span=1.6
    xo=np.arange(-Span/2,Span/2+do,do)
    sm=len(xo);
    yo=xo;
    ff=sm*sm
    xo1=np.matlib.repmat(xo,sm,1).transpose()
    xo2=np.reshape(xo1,(1,ff))
    yo1=np.matlib.repmat(yo,1,sm)
    
    # Distances
    [Xs1,Xo]=np.meshgrid(xs,xo2)
    [Ys1,Yo]=np.meshgrid(ys,yo1)
    ro=np.sqrt((Xo-Xs1)**2+(Yo-Ys1)**2)
    
    
    ##Presssure Matching
    Pnf=1j*k*rho*c*np.exp(-1j*k*ro)/(4*np.pi*ro)@Qnf
    Ppm=1j*k*rho*c*np.exp(-1j*k*ro)/(4*np.pi*ro)@Qpmf[:,ind1]
    
    
    Pbpm=1j*k*rho*c*np.exp(-1j*k*rb)/(4*np.pi*rb)@Qpmf[:,ind1]
    Pdpm=1j*k*rho*c*np.exp(-1j*k*rd)/(4*np.pi*rd)@Qpmf[:,ind1]
    
    
    Lpnf=20*np.log10(abs(Pnf)/2e-5)
    Lpnf=np.reshape(Lpnf,(sm,sm))
    Lpnf = np.squeeze(np.asarray(Lpnf)).transpose()
    
    Lppm=20*np.log10(abs(Ppm)/2e-5)
    Lppm=np.reshape(Lppm,(sm,sm))
    Lppm = np.squeeze(np.asarray(Lppm)).transpose()
    
    plt.figure(10)
    Lmm = np.max(Lpnf)
    plt.pcolormesh(xo,yo,Lpnf,vmin=Lmm-40,vmax=Lmm-10,shading='gouraud')
    plt.colorbar()
    plt.plot(xd,yd,'ko',markersize=6,fillstyle='none')
    plt.plot(xb,yb,'wo',markersize=6,fillstyle='none')
    plt.xlabel('x-axis (m)',fontsize=14)
    plt.ylabel('y-axis (m)',fontsize=14)
    str1='No filters '+str(np.round(Fv[ind1],0))+ ' Hz'
    plt.title(str1)
    
    
    plt.figure(11)
    Lmm = np.max(Lppm)
    plt.pcolormesh(xo,yo,Lppm,vmin=Lmm-40,vmax=Lmm+1,shading='gouraud')
    plt.colorbar()
    plt.plot(xd,yd,'ko',markersize=6,fillstyle='none')
    plt.plot(xb,yb,'wo',markersize=6,fillstyle='none')
    plt.xlabel('x-axis (m)',fontsize=14)
    plt.ylabel('y-axis (m)',fontsize=14)
    str2=str(np.round(Fv[ind1],0)) + ' Hz, Contrast: '+str(np.round(Contrast[ind1],1))+' dB, error : ' + str(np.round(Err[ind1],4))+ ' %'
    plt.title(str2)


