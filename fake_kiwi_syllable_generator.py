"""
17/05/2022
Author: Virginia Listanti

This script generates a fake kiwi syllables for experimental purposes from parameters inserted by the user

Parameters hard coded (need to change script):
- test_dir where to store syllables
- sample_rate

Parameters inserted by user:
- syllable_id (string)
- syllable type: A or B
- trill: yes or no
- syllabe parameters:
                        - f0
                        - (t1, f1)
                        - f2
                        [if trill]
                        - t3
                        - t4
                        - k (number of trills)
                        - delta (amplitude oscillationss)
                        - n (number of harmonics) manages also the case with only fundamental frequency

For each syllables:

- create a folder in test_dir
- create if and analytical form of syllables
- stores if into a .csv file for each harmonics
- stores .jpg image of IF with all harmonics
- save non-padded and padded sound file

"""

import numpy as np
import os
import wavio
import matplotlib.pyplot as plt
import csv


def Syllable_IF_fun1(t, t0, t1, t2, f0, f1, f2):
    """
    Instantaneous frequency function for syllable type A:
    2 quadratic chirps 1 concave one convess
    """

    IF = np.zeros(np.shape(t))
    for index in range(len(t)):
        if t[index] < t1:
            a = (f0 - f1) / (t0 - t1) ** 2
            b = -2 * a * t1
            c = f1 + a * t1 ** 2
        else:
            a = (f1 - f2) / (t1 - t2) ** 2
            b = -2 * a * t2
            c = f2 + a * t2 ** 2

        IF[index] = a*t[index]**2+b*t[index]+c
    return IF


def Syllable_IF_fun2(t, t0, t1, t2, f0, f1, f2):
    """
    Instantaneous frequency function for syllable type B
     2 quadratic chirps, both convess
    """

    IF = np.zeros(np.shape(t))
    for index in range(len(t)):
        if t[index] < t1:
            a = (f0 - f1) / (t0 - t1) ** 2
            b = -2 * a * t1
            c = f1 + a * t1 ** 2
            IF[index] = a * t[index] ** 2 + b * t[index] + c
        else:
            a = -(f1 - f2) / (t1 - t2) ** 2
            b = -2 * a * t1
            c = f1 + a * t1 ** 2

            IF[index] = a*t[index]**2+b*t[index]+c
    return IF


def Syllable_phase_fun1(t, t0, t1, t2, f0, f1, f2, phi, samplerate):
    """
    Phase function fro syllable type A
    """

    Ph_fun = np.zeros(np.shape(t))
    a1 = (f0-f1)/(t0-t1)**2
    b1 = -2*a1*t1
    c1 = f1+a1*t1**2
    a2 = (f1-f2)/(t1-t2)**2
    b2 = -2*a2*t2
    c2 = f2+a2*t2**2
    Ph_fun[int(t0 * samplerate):int(t1 * samplerate)] = \
        phi + 2 * np.pi * ((a1 / 3) * t[int(t0 * samplerate): int(t1 * samplerate)] ** 3 + (b1 / 2) *
                           t[int(t0 * samplerate):int(t1 * samplerate)] ** 2 + c1 *
                           t[int(t0 * samplerate):int(t1 * samplerate)])
    Ph_fun[int(t1*samplerate):] = \
        phi + 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])

    return Ph_fun


def Syllable_phase_fun2(t, t0, t1, t2, f0, f1, f2, phi, samplerate):
    """
    Phase function fro syllable type B
    """

    Ph_fun = np.zeros(np.shape(t))
    a1 = (f0-f1)/(t0-t1)**2
    b1 = -2*a1*t1
    c1 = f1+a1*t1**2
    a2 = -(f1-f2)/(t1-t2)**2
    b2 = -2*a2*t1
    c2 = f1+a2*t1**2

    Ph_fun[int(t0*samplerate):int(t1*samplerate)] =\
        phi + 2*np.pi*((a1/3)*t[int(t0*samplerate):int(t1*samplerate)]**3+(b1/2) *
                       t[int(t0*samplerate):int(t1*samplerate)]**2+c1*t[int(t0*samplerate):int(t1*samplerate)])
    Ph_fun[int(t1*samplerate):] = \
        phi + 2*np.pi*((a2/3)*t[int(t1*samplerate):]**3+(b2/2)*t[int(t1*samplerate):]**2+c2*t[int(t1*samplerate):])
    return Ph_fun


##################################### MAIN ########################################################

# hardcoded variables
test_dir = "/home/listanvirg/Documents/Individual_identification/Fake_kiwi_syllables/"
samplerate = 16000
Amplitude_base = 1
phi = 0  # initial phase

# user input variables
print("\n Welcome to fake kiwi syllable generator \n")
print("Please, enter the parameters we need to generate your fake kiwi syllable\n")

syllable_id = input("\n Enter the syllable name:    ")

print("\n Which syllable type you want? \n There are 2 options: A or B.")
print("\n A: 2 quadratic chirps: 1 concave-down, 1 concave up")
print("\n B: 2 quadratic chirps: both concave-down \n")
syllable_type = input("\n Type A or B :  ")
while syllable_type != 'A' and syllable_type != 'B':
    syllable_type = input("\n Which syllable type you want? Type A or B :")

# syllable parameters
print("\n Please insert syllable parameters \n")
T = float(input("\n Enter syllable length (in seconds):  "))
t0 = 0.0
t2 = T
f0 = float(input("\n Enter initial frequency f0 (in Hz):   "))
t1 = float(input("\n Enter t1, time maximum frequency, (in seconds):    "))
while t1 > t2:
    print(f"\n ERROR: we need t1<{t2}")
    t1 = float(input("\n Enter t1, time maximum frequency, (in seconds):    "))
f1 = float(input("\n Enter maximum frequency f1 (in Hz):   "))
f2 = float(input("\n Enter final frequency f2 (in Hz):   "))

# do we have trills?
trill_flag = input("\n\n Do you want a trill in your syllable? Type yes or no:  ")
while trill_flag != 'yes' and trill_flag != 'no':
    trill_flag = input("\n Do you want a trill in your syllable? Type yes or no:  ")

delta = 0
# trill parameters
if trill_flag == 'yes':
    print("\n Please insert trill parameters ")
    k = int(input("\n How many trills you want? Enter an integer:  "))
    t3 = float(input("\n Enter t3, the starting time of the trill in seconds:   "))
    while t3 > t2:
        print(f"\n ERROR: we need t3<{t2}")
        t3 = float(input("\n Enter t3, the starting time of the trill in seconds:  "))
    t4 = float(input("\n Enter t4, the final time of the trill in seconds:  "))
    while t4 > t2 or t4 < t3:
        print(f"\n ERROR: we need {t3}<t4<{t2}")
        t4 = float(input("\n Enter t4, the final time of the trill in seconds:  "))

    delta = float(input("\n Enter the amplitude of the trill:  "))

# harmonics
n_max = np.floor((samplerate/2)/(f1+delta))
print("\n How many harmonics do you want?")
n = int(input("\n Enter an integer. Enter 1 if you want only the fundamental frequency. n =  "))
while n < 1 or n > n_max:
    print(f"\n ERROR: n must be an integer between 1 and {n_max}")
    n = int(input("\n Enter an integer. Enter 1 if you want only the fundamental frequency. n =  "))

t = np.linspace(0., T, int(samplerate*T), endpoint=False)


file_id = syllable_id + ".wav"
save_dir = os.path.join(test_dir, syllable_id)
if syllable_id not in os.listdir(test_dir):
    os.mkdir(save_dir)

# deal with harmonics

signal = np.zeros((np.shape(t)))
fig_name = os.path.join(save_dir, file_id[:-4] + ".jpg")
fig = plt.figure()
for j in range(1, n+1):
    f0_h = j*f0
    f1_h = j*f1
    f2_h = j*f2
    A = Amplitude_base/10**(j-1)
    if_t = np.zeros((np.shape(t)))
    phi_t = np.zeros((np.shape(t)))
    # create base syllable
    if syllable_type == 'A':
        if_t = Syllable_IF_fun1(t, t0, t1, t2, f0_h, f1_h, f2_h)
        phi_t = Syllable_phase_fun1(t, t0, t1, t2, f0_h, f1_h, f2_h, phi, samplerate)
    else:
        if_t = Syllable_IF_fun2(t, t0, t1, t2, f0_h, f1_h, f2_h)
        phi_t = Syllable_phase_fun2(t, t0, t1, t2, f0_h, f1_h, f2_h, phi, samplerate)

    # add trills

    if trill_flag == 'yes':
        t_trill = np.linspace(t3, t4, int(np.round(samplerate * (t4 - t3))), endpoint=False)
        trill_phase = (delta / k) * (t4 - t3) * (-np.cos(((2 * np.pi * k) / (t4 - t3)) * t_trill))
        phi_t[int(t3 * samplerate):int(t4 * samplerate)] += trill_phase
        trill = delta * np.sin((2 * np.pi * k * (t_trill - t3)) / (t4 - t3))
        if_t[int(t3 * samplerate):int(t4 * samplerate)] += trill

    signal += A*np.sin(phi+phi_t)

    # plot If harmonics
    plt.plot(if_t)

    # save IF harmonics
    csvfilename = os.path.join(save_dir, file_id[:-4] + "_IF_harmonic_"+str(j)+".csv")
    fieldnames = ["IF"]
    with open(csvfilename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(if_t)):
            writer.writerow({"IF": if_t[i]})

fig.suptitle(file_id[:-4])
plt.savefig(fig_name)

# save file
file_name = os.path.join(save_dir, file_id)
wavio.write(file_name, signal, samplerate, sampwidth=2)

# save padded version
s2 = np.concatenate((np.zeros(int(14.5*samplerate)), signal, np.zeros(int(14.5*samplerate))))
file_name2 = file_name[:-4]+'_padded.wav'
wavio.write(file_name2, s2, samplerate, sampwidth=2)
