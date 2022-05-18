"""
16/05/2022
Author: Virginia Listanti

This script generates fake kiwi syllables for experimental purposes from a known IF law
Syllables are 1 second longer

For each syllables:

- create if and analytical form of syllables
- stores if into a .csv file
- stores .jpg image of IF
- save non-padded and padded sound file

"""

import SignalProc
import IF as IFreq
import numpy as np
import os
import wavio
import matplotlib.pyplot as plt
import csv

def Syllable_IF_fun1(t,t0,t1,t2,f0,f1,f2):
#2 quadratic chirps 1 concave one convess
    IF=np.zeros(np.shape(t))
    for i in range(len(t)):
        if t[i]<t1:
            a = (f0 - f1) / (t0 - t1) ** 2
            b = -2 * a * t1
            c = f1 + a* t1 ** 2
        else:
            a = (f1 - f2) / (t1 - t2) ** 2
            b = -2 * a * t2
            c = f2 + a * t2 ** 2

        IF[i]=a*t[i]**2+b*t[i]+c
    return IF

def Syllable_IF_fun2(t,t0,t1,t2,f0,f1,f2):
    #2 quadratic chirps, both convess
    IF=np.zeros(np.shape(t))
    for i in range(len(t)):
        if t[i]<t1:
            a = (f0 - f1) / (t0 - t1) ** 2
            b = -2 * a * t1
            c = f1 + a * t1 ** 2
            IF[i] = a * t[i] ** 2 + b * t[i] + c
        else:
            a = -(f1 - f2) / (t1 - t2) ** 2
            b = -2 * a * t1
            c = f1+ a * t1 ** 2

            IF[i]=a*t[i]**2+b*t[i]+c
    return IF

def Syllable_IF_fun3(t,t0,t1,t2,t3,t4,t5,f0,f1,f2,f3,f4,f5):
    # trill syllabe
    IF = np.zeros(np.shape(t))
    for i in range(len(t)):
        if t[i] < t1:
            a = (f0 - f1) / (t0 - t1) ** 2
            b = -2 * a * t1
            c = f1 + a * t1 ** 2
            IF[i] = a * t[i] ** 2 + b * t[i] + c
        elif t[i]>=t1 and t[i]<t2:
            C1 = (f2 - f1) / (t2 - t1)
            IF[i]=f1 + C1* t[i]
        elif t[i]>=t2 and t[i]<t3:
            C2 = (f3 - f2) / (t3 - t2)
            IF[i]=f2+C2*t[i]
        elif t[i]>=t3 and t[i]<t4:
            C3 = (f4 - f3)/(t4 - t3)
            IF[i]=f3+C3*t[i]
        else:
            a = -(f5 - f4) / (t5 - t4) ** 2
            b = -2 * a * t4
            c = f4 + a * t4 ** 2

            IF[i] = a * t[i] ** 2 + b * t[i] + c
    return IF


##################################### MAIN ########################################################

test_dir = "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\fake_kiwi_syllables\\New"


samplerate = 16000
T=1
A=1
phi=0
#A= np.iinfo(np.int16).max
t = np.linspace(0., T, samplerate*T,endpoint=False)

syl_fold = "syllable_19"
file_id = syl_fold + ".wav"
save_dir = test_dir+"\\"+syl_fold
if not syl_fold in os.listdir(test_dir):
    os.mkdir(save_dir)

# ## SYLLABLE 1
# f0 = 1500
# f1 = 2000
# f2 = 1800
# t0 = 0
# t1 = 0.25
# t2 = 1
# a1 = (f0-f1)/(t0-t1)**2
# b1 = -2*a1*t1
# c1 = f1+a1*t1**2
# a2 = (f1-f2)/(t1-t2)**2
# b2 = -2*a2*t2
# c2 = f2+a2*t2**2
# # s1=np.zeros((np.shape(t)))
# phi_t = np.zeros((np.shape(t)))
# phi_t[int(t0*samplerate):int(t1*samplerate)] = phi + 2*np.pi*((a1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1/2) * t[int(t0*samplerate):int(t1*samplerate)]**2+c1*t[int(t0*samplerate):int(t1*samplerate)])
# phi_t[int(t1*samplerate):]=phi+ 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])
# s1 = A*np.sin(phi+phi_t)
# if1=Syllable_IF_fun1(t,t0,t1,t2,f0,f1,f2)

# ## SYLLABLE 2
# f0=1500
# f1=2000
# f2=1800
# t0=0
# t1=0.5
# t2=1
# a1=(f0-f1)/(t0-t1)**2
# b1=-2*a1*t1
# c1=f1+a1*t1**2
# a2=(f1-f2)/(t1-t2)**2
# b2=-2*a2*t2
# c2=f2+a2*t2**2
# phi_t=np.zeros((np.shape(t)))
# phi_t[int(t0*samplerate):int(t1*samplerate)]=phi+ 2*np.pi*((a1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1/2)*t[int(t0*samplerate):int(t1*samplerate)]**2+c1*t[int(t0*samplerate):int(t1*samplerate)])
# phi_t[int(t1*samplerate):]=phi+ 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])
# s1 = A*np.sin(phi+phi_t)
# if1 =Syllable_IF_fun1(t,t0,t1,t2,f0,f1,f2)


# ## SYLLABLE 3
# f0=1500
# f1=2000
# f2=1800
# t0=0
# t1=0.75
# t2=1
# a1=(f0-f1)/(t0-t1)**2
# b1=-2*a1*t1
# c1=f1+a1*t1**2
# a2=(f1-f2)/(t1-t2)**2
# b2=-2*a2*t2
# c2=f2+a2*t2**2
# # s1=np.zeros((np.shape(t)))
# phi_t=np.zeros((np.shape(t)))
# phi_t[int(t0*samplerate):int(t1*samplerate)]=phi+ 2*np.pi*((a1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1/2)*t[int(t0*samplerate):int(t1*samplerate)]**2+c1*t[int(t0*samplerate):int(t1*samplerate)])
# phi_t[int(t1*samplerate):]=phi+ 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])
# s1=A*np.sin(phi+phi_t)
# if1=Syllable_IF_fun1(t,t0,t1,t2,f0,f1,f2)


# ## SYLLABLE 4
# f0=1500
# f1=2000
# f2=1800
# t0=0
# t1=0.25
# t2=1
# a1=(f0-f1)/(t0-t1)**2
# b1=-2*a1*t1
# c1=f1+a1*t1**2
# a2=-(f1-f2)/(t1-t2)**2
# b2=-2*a2*t1
# c2=f1+a2*t1*2
# # s1=np.zeros((np.shape(t)))
# phi_t=np.zeros((np.shape(t)))
# phi_t[int(t0*samplerate):int(t1*samplerate)]=phi+ 2*np.pi*((a1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1/2)*t[int(t0*samplerate):int(t1*samplerate)]**2+c1*t[int(t0*samplerate):int(t1*samplerate)])
# phi_t[int(t1*samplerate):]=phi+ 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])
# s1=A*np.sin(phi+phi_t)
# if1=Syllable_IF_fun2(t,t0,t1,t2,f0,f1,f2)

# ## SYLLABLE 5
# f0=1500
# f1=2000
# f2=1800
# t0=0
# t1=0.5
# t2=1
# a1=(f0-f1)/(t0-t1)**2
# b1=-2*a1*t1
# c1=f1+a1*t1**2
# a2=-(f1-f2)/(t1-t2)**2
# b2=-2*a2*t1
# c2=f1+a2*t1**2
# # s1=np.zeros((np.shape(t)))
# phi_t=np.zeros((np.shape(t)))
# phi_t[int(t0*samplerate):int(t1*samplerate)]=phi+ 2*np.pi*((a1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1/2)*t[int(t0*samplerate):int(t1*samplerate)]**2+c1*t[int(t0*samplerate):int(t1*samplerate)])
# phi_t[int(t1*samplerate):]=phi+ 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])
# s1=A*np.sin(phi+phi_t)
# if1=Syllable_IF_fun2(t,t0,t1,t2,f0,f1,f2)

# ## SYLLABLE 6
# f0=1500
# f1=2000
# f2=1800
# t0=0
# t1=0.75
# t2=1
# a1=(f0-f1)/(t0-t1)**2
# b1=-2*a1*t1
# c1=f1+a1*t1**2
# a2=-(f1-f2)/(t1-t2)**2
# b2=-2*a2*t1
# c2=f1+a2*t1**2
# # s1=np.zeros((np.shape(t)))
# phi_t=np.zeros((np.shape(t)))
# phi_t[int(t0*samplerate):int(t1*samplerate)]=phi+ 2*np.pi*((a1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1/2)*t[int(t0*samplerate):int(t1*samplerate)]**2+c1*t[int(t0*samplerate):int(t1*samplerate)])
# phi_t[int(t1*samplerate):]=phi+ 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])
# s1=A*np.sin(phi+phi_t)
# if1=Syllable_IF_fun2(t,t0,t1,t2,f0,f1,f2)

# ## SYLLABLE 6
# f0=1500
# f1=2000
# f2=1800
# t0=0
# t1=0.25
# t2=1
# a1=(f0-f1)/(t0-t1)**2
# b1=-2*a1*t1
# c1=f1+a1*t1**2
# a2=-(f1-f2)/(t1-t2)**2
# b2=-2*a2*t1
# c2=f1+a2*t1**2
# # s1=np.zeros((np.shape(t)))
# phi_t=np.zeros((np.shape(t)))
# phi_t[int(t0*samplerate):int(t1*samplerate)]=phi+ 2*np.pi*((a1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1/2)*t[int(t0*samplerate):int(t1*samplerate)]**2+c1*t[int(t0*samplerate):int(t1*samplerate)])
# phi_t[int(t1*samplerate):]=phi+ 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])
# s1=A*np.sin(phi+phi_t)
# if1=Syllable_IF_fun2(t,t0,t1,t2,f0,f1,f2)

# # SYLLABLE 7-8-9-10-11-12-13-14
# #chancing alpha and eps
# f0=1500
# f1=2000
# f2=1800
# t0=0
# t1=0.25
# t2=1
# a1=(f0-f1)/((t0-t1)**2)
# b1=-2*a1*t1
# c1=f1+a1*t1**2
# a2=-(f1-f2)/(t1-t2)**2
# b2=-2*a2*t1
# c2=f1+a2*(t1**2)
# alpha=50
# eps=300
# phi_t=np.zeros((np.shape(t)))
# phi_t[int(t0*samplerate):int(t1*samplerate)]=phi+ 2*np.pi*((a1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1/2)*t[int(t0*samplerate):int(t1*samplerate)]**2+c1*t[int(t0*samplerate):int(t1*samplerate)])
# phi_t[int(t1*samplerate):]=phi+ 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])
# trill=2*np.pi*((eps/alpha)*(1-np.cos(alpha*t)))
# s1=A*np.sin(phi+phi_t+trill)
# if_base=Syllable_IF_fun2(t,t0,t1,t2,f0,f1,f2)
# if1=if_base+eps*np.sin(alpha*t)


# #SYLLABLE  15-16-17
# #MID-TRILL SYLLABLE
# #playing with parameters alpha and eps
#
# #base syllable points
# f0=1500
# f1=2000
# f2=1800
# t0=0
# t1=0.25
# t2=1
#
# #trill boundaries
# t3 = 0.2
# t4= 0.4
# t_trill = np.linspace(t3, t4, int(samplerate*(t4-t3)),endpoint=False)
#
# #IF1 parameters
# a1=(f0-f1)/((t0-t1)**2)
# b1=-2*a1*t1
# c1=f1+a1*t1**2
#
# #IF2 parameters
# a2=-(f1-f2)/(t1-t2)**2
# b2=-2*a2*t1
# c2=f1+a2*(t1**2)
#
# #trill parameters
# alpha=5 #number of trills
# eps=400
#
# #phase
# phi_t=np.zeros((np.shape(t)))
# phi_t[int(t0*samplerate):int(t1*samplerate)]=phi+ 2*np.pi*((a1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1/2)*t[int(t0*samplerate):int(t1*samplerate)]**2+c1*t[int(t0*samplerate):int(t1*samplerate)])
# phi_t[int(t1*samplerate):]=phi+ 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])
# trill_phase=(eps/alpha)*(t4-t3)*(-np.cos(((2*np.pi*alpha)/(t4-t3))*t_trill))
# phi_t[int(t3*samplerate):int(t4*samplerate)]+=trill_phase
# s1=A*np.sin(phi+phi_t)
# if_base=Syllable_IF_fun2(t,t0,t1,t2,f0,f1,f2)
# if1=if_base
# trill = eps *np.sin((2*np.pi*alpha*(t_trill-t3))/(t4-t3))
# if1[int(t3*samplerate):int(t4*samplerate)] +=  trill

# SYLLABLES 18-19
# Syllable 1 + 1 harmonic

t0 = 0
t1 = 0.25
t2 = 1
# harmonic 1
A1=1
f0_1 = 1500
f1_1 = 2000
f2_1 = 1800
a1_1 = (f0_1-f1_1)/(t0-t1)**2
b1_1 = -2*a1_1*t1
c1_1 = f1_1+a1_1*t1**2
a2_1 = (f1_1-f2_1)/(t1-t2)**2
b2_1 = -2*a2_1*t2
c2_1 = f2_1+a2_1*t2**2
# s1=np.zeros((np.shape(t)))
phi_t_1 = np.zeros((np.shape(t)))
phi_t_1[int(t0*samplerate):int(t1*samplerate)] = phi + 2*np.pi*((a1_1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1_1/2) * t[int(t0*samplerate):int(t1*samplerate)]**2+c1_1*t[int(t0*samplerate):int(t1*samplerate)])
phi_t_1[int(t1*samplerate):]=phi+ 2*np.pi*((a2_1/3)*t[int(t1*samplerate):]**3+(b2_1/2)*t[int(t1*samplerate):]**2+c2_1*t[int(t1*samplerate):])
s1_1 = A1*np.sin(phi+phi_t_1)
if1=Syllable_IF_fun1(t,t0,t1,t2,f0_1,f1_1,f2_1)

# harmonic 2
A2=0.1
df1= 2000
f0_2 = f0_1 + df1
f1_2 = f1_1 + df1
f2_2 = f2_1 + df1
a1_2 = (f0_2-f1_2)/(t0-t1)**2
b1_2 = -2*a1_2*t1
c1_2 = f1_2+a1_2*t1**2
a2_2 = (f1_2-f2_2)/(t1-t2)**2
b2_2 = -2*a2_2*t2
c2_2 = f2_2+a2_2*t2**2
# s1=np.zeros((np.shape(t)))
phi_t_2 = np.zeros((np.shape(t)))
phi_t_2[int(t0*samplerate):int(t1*samplerate)] = phi + 2*np.pi*((a1_2/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1_2/2) * t[int(t0*samplerate):int(t1*samplerate)]**2+c1_2*t[int(t0*samplerate):int(t1*samplerate)])
phi_t_2[int(t1*samplerate):]=phi+ 2*np.pi*((a2_2/3)*t[int(t1*samplerate):]**3+(b2_2/2)*t[int(t1*samplerate):]**2+c2_2*t[int(t1*samplerate):])
s1_2 = A2*np.sin(phi+phi_t_2)
if2=Syllable_IF_fun1(t,t0,t1,t2,f0_2,f1_2,f2_2)

# total
s1 = s1_1 + s1_2


#
# # No Harmonics
#
# #save plot IF
# fig_name = save_dir + "\\"+file_id[:-4] + ".jpg"
# fig = plt.figure()
# plt.plot(if1)
# fig.suptitle(file_id[:-4])
# plt.savefig(fig_name)
#
# #save IF
# csvfilename  = save_dir + "\\"+file_id[:-4]+"_IF.csv"
# fieldnames=["IF"]
# with open(csvfilename, 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for i in range(len(if1)):
#         writer.writerow({"IF":if1[i]})


#  With Harmonics

#save plot IF
fig_name = save_dir + "\\"+file_id[:-4] + ".jpg"
fig = plt.figure()
plt.plot(if1, 'r')
plt.plot(if2, 'b')
fig.suptitle(file_id[:-4])
plt.savefig(fig_name)

#save IF
#first harmonic
csvfilename  = save_dir + "\\"+file_id[:-4]+"_IF_Harmonic1.csv"
fieldnames=["IF"]
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(if1)):
        writer.writerow({"IF":if1[i]})


#first harmonic
csvfilename  = save_dir + "\\"+file_id[:-4]+"_IF_Harmonic2.csv"
fieldnames=["IF"]
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(if2)):
        writer.writerow({"IF":if2[i]})

#save file
file_name = save_dir+"\\"+file_id
wavio.write(file_name, s1, samplerate,sampwidth=2)

#save padded version
s2=np.concatenate((np.zeros(int(14.5*samplerate)), s1, np.zeros(int(14.5*samplerate))))
file_name2 = save_dir+"\\"+file_id[:-4]+'_padded.wav'
wavio.write(file_name2, s2, samplerate,sampwidth=2)




