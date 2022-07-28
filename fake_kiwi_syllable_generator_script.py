"""
14/07/2022
Author: Virginia Listanti

This script generates a fake kiwi syllable for experimental purposes with hardcoded parameters.py

Sylllables are built combining piecewise defined instantaneous frequency and instantaneous phase.

Parameters hard coded (need to change script):
- test_dir where to store syllables
- sample_rate
- syllable_id (string)
- syllable type: A, B, C, D, E, F, G, H, I, J, K, L, ,
- trill: yes or no
- harmonics: yes or no
- syllabe parameters:
                        - f0
                        - (t1, f1)
                        - f2
                        [if trill]
                        - t3
                        - t4
                        - k (number of trills)
                        - delta (amplitude oscillation
                        [if harmonics]
                        - n (number of harmonics)

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


def Syllable_IF_fun1(t, p1, p2):
    """
    Instantaneous frequency function for piece-type A:
    quadratic chirps between P1 and P2, where P1 is vertex
    """

    IF = np.zeros(np.shape(t))

    #quadratic chirp parameters
    t1 = p1[0]
    t2 = p2[0]
    f1 = p1[1]
    f2 = p2[1]
    a = (f2-f1)/((t2-t1)**2)
    b = -2*t1*a
    c = f2 - a*(t2**2 -2*t1*t2)

    # IF law
    IF = a*t**2+b*t+c
    return IF

def Syllable_IF_fun2(t, p1, p2):
    """
    Instantaneous frequency function for piece-type B:
    quadratic chirps between P1 and P2, where P2 is vertex
    """

    IF = np.zeros(np.shape(t))

    #quadratic chirp parameters
    t1 = p1[0]
    t2 = p2[0]
    f1 = p1[1]
    f2 = p2[1]
    a = -(f2-f1)/((t2-t1)**2)
    b = -2*t2*a
    c = f2 + a*(t2**2)

    # IF law
    IF = a*t**2+b*t+c
    return IF

def Syllable_IF_fun3(t, p1, p2):
    """
    Instantaneous frequency function for piece-type C:
    linear chirps between P1 and P2
    note: it covers pure tone as well
    """

    IF = np.zeros(np.shape(t))

    #quadratic chirp parameters
    t1 = p1[0]
    t2 = p2[0]
    f1 = p1[1]
    f2 = p2[1]
    c = (f2-f1)/(t2-t1)

    # IF law
    IF = f1 +c*t
    return IF

def Syllable_Phase_fun1(t, p1, p2):
    """
    Instantaneous phase function for piece-type A:
    quadratic chirps between P1 and P2, where P1 is vertex
    """

    Phase = np.zeros(np.shape(t))

    #quadratic chirp parameters
    t1 = p1[0]
    t2 = p2[0]
    f1 = p1[1]
    f2 = p2[1]
    a = (f2-f1)/((t2-t1)**2)
    b = -2*t1*a
    c = f2 - a*(t2**2 -2*t1*t2)

    # Phase law
    Phase = 2*np.pi*((a/3)*t**3 + (b/2)*t**2 + c*t + (a/3)*t1**3 + (b/2)*t1**2 + c*t1)
    return Phase

def Syllable_Phase_fun2(t, p1, p2):
    """
    Instantaneous phase function for piece-type B:
    quadratic chirps between P1 and P2, where P2 is vertex
    """

    Phase = np.zeros(np.shape(t))

    #quadratic chirp parameters
    t1 = p1[0]
    t2 = p2[0]
    f1 = p1[1]
    f2 = p2[1]
    a = -(f2-f1)/((t2-t1)**2)
    b = -2*t2*a
    c = f2 + a*(t2**2)

    # Phase law
    Phase = 2*np.pi*((a / 3) * t ** 3 + (b / 2) * t ** 2 + c * t + (a / 3) * t1 ** 3 + (b / 2) * t1 ** 2 + c * t1)
    return Phase

def Syllable_Phase_fun3(t, p1, p2):
    """
    Instantaneous phase function for piece-type C:
    linear chirps between P1 and P2
    note: it covers pure tone as well
    """

    Phase = np.zeros(np.shape(t))

    #quadratic chirp parameters
    t1 = p1[0]
    t2 = p2[0]
    f1 = p1[1]
    f2 = p2[1]
    c = (f2-f1)/(t2-t1)

    # IF law
    Phase = 2*np.pi*(f1*t +(c/2)*t**2 - f1*t1 + (c/2)*t1**2)
    return Phase





##################################### MAIN ########################################################

# hardcoded variables
save_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
           "exemplars\\Models"
samplerate = 32000
Amplitude_base = 1
phi = 0  # initial phase

syllable_id = "Y_fake"

syllable_type = "O"

# # syllable A
# P1 = np.array([0, 1098.4])
# P2 = np.array([0.08, 1269.9])
# P3 = np.array([0.56, 1375.4])
# P4 = np.array([0.66, 1217.1])

# # syllable B
# P1 = np.array([0, 1074.1])
# P2 = np.array([0.08, 1451.3])
# P3 = np.array([0.49, 1548.3])
# P4 = np.array([0.49, 1300.4])
# P5 = np.array([0.74, 1254.8])

# # syllable C
# P1 = np.array([0, 1353.7])
# P2 = np.array([0.34, 1597.7])
# P3 = np.array([0.34, 1393.3])
# P4 = np.array([0.63, 1202.1])


# # syllable D
# P1 = np.array([0, 1208.9])
# P2 = np.array([0.1, 1413.1])
# P3 = np.array([0.6, 1393.3])


# # syllable E
# P1 = np.array([0, 1220.8])
# P2 = np.array([0.53, 1602.2])
# P3 = np.array([0.56, 1090.9])

# # syllable F
# P1 = np.array([0, 970.3])
# P2 = np.array([0.79, 1613])
# P3 = np.array([0.82, 1278.1])


# # syllable G
# P1 = np.array([0, 626.9])
# P2 = np.array([0.08, 1721.9])
# P3 = np.array([0.46, 1721.9])
# P4 = np.array([0.5 , 487.7])


# # syllable H
# P1 = np.array([0, 664])
# P2 = np.array([0.14, 1833.3])
# P3 = np.array([0.52, 1740.5])


# # syllable I
# P1 = np.array([0, 580.5])
# P2 = np.array([0.15, 1731.2])
# P3 = np.array([0.52, 1406.4])

# # syllable J
# P1 = np.array([0, 599.1])
# P2 = np.array([0.14, 1592])
# P3 = np.array([0.56, 1508.5])
# P4 = np.array([0.62 , 571.2])

# # syllable K
# P1 = np.array([0, 1425])
# P2 = np.array([0.48, 1944.6])
# P3 = np.array([0.52, 1592])

# # syllable L
# P1 = np.array([0, 1415.7])
# P2 = np.array([0.43, 2102.4])
# P3 = np.array([0.43, 1703.3])
# P4 = np.array([0.54, 1425])

# # syllable M
# P1 = np.array([0, 1675.5])
# P2 = np.array([0.16, 2130.2])
# P3 = np.array([0.55, 1684.8])

# # syllable N
# P1 = np.array([0, 1434.2])
# P2 = np.array([0.27, 2455])
# P3 = np.array([0.27, 1851.8])
# P4 = np.array([0.51, 1573.4])

# # syllable O
# P1 = np.array([0, 1851.8])
# P2 = np.array([0.57, 1851.8])

# # syllable P
# P1 = np.array([0, 2226.6])
# P2 = np.array([1.09, 2583.3])
# P3 = np.array([1.09, 2246.5])
# P4 = np.array([1.31, 2259.7])

# # syllable Q
# P1 = np.array([0, 1883.2])
# P2 = np.array([0.1, 2629.5])
# P3 = np.array([0.55, 2662.6])
# P4 = np.array([0.55, 2834.3])
# P5 = np.array([0.65, 2794.6])
# P6 = np.array([0.65, 2451.2])
# P7 = np.array([0.75, 1803.9])

# # syllable R
# P1 = np.array([0, 1871.6])
# P2 = np.array([0.08, 2733.7])
# P3 = np.array([0.49, 2733.7])
# P4 = np.array([0.49, 2475.1])
# P5 = np.array([0.63, 2475.1])
# P6 = np.array([0.66, 1656.1])

# # syllable S
# P1 = np.array([0, 1516])
# P2 = np.array([0.09, 2561.3])
# P3 = np.array([0.26, 2561.3])
# P4 = np.array([0.26, 2248.8])
# P5 = np.array([0.54, 2248.8])
# P6 = np.array([0.54, 1710])
# P7 = np.array([0.67,1322])

# # syllable T
# P1 = np.array([0, 1354.3])
# P2 = np.array([0.4, 2130.2])
# P3 = np.array([0.4, 1645.3])
# P4 = np.array([0.56, 1289.7])

# # syllable U
# P1 = np.array([0, 1462.1])
# P2 = np.array([0.11, 1871.6])
# P3 = np.array([0.51, 1257.3])

# # syllable V
# P1 = np.array([0, 1397.4])
# P2 = np.array([0.15, 2108.7])
# P3 = np.array([0.15, 2863.1])
# P4 = np.array([0.57, 2529])
# P5 = np.array([0.57, 1990.1])
# P6 = np.array([0.67, 1257.3])

# # syllable W
# P1 = np.array([0, 1979.4])
# P2 = np.array([0.14, 2604.4])
# P3 = np.array([0.64, 2938.5])
# P4 = np.array([0.64, 2572.1])
# P5 = np.array([1.15, 2345.8])
# P6 = np.array([1.15, 1850.1])
# P7 = np.array([1.21, 1591.4])

# # syllable X
# P1 = np.array([0, 1656.1])
# P2 = np.array([0.45, 1968.6])
# P3 = np.array([0.45, 1548.3])
# P4 = np.array([0.59, 1235.8])

# syllable Y
P1 = np.array([0, 1656.1])
P2 = np.array([0.25, 1839.3])
P3 = np.array([0.25, 1537.5])
P4 = np.array([0.57, 1257.3])

# # syllable Z
# P1 = np.array([0, 1505.2])
# P2 = np.array([0.06, 2011.7])
# P3 = np.array([0.14, 1990.1])
# P4 = np.array([0.14, 1677.6])
# P5 = np.array([0.81, 1192.7])
# P6 = np.array([0,0])
# P7 = np.array([0,0])
# P8 = np.array([0,0])

#syllable length
T = P4[0]

# do we have trills?
# trill_flag = "no"
trill_flag = "yes"

delta = 33
k = 8 # number of trills
trill_t1 = 0.25
trill_t2 = 0.57

# harmonics
n_limit = np.floor((samplerate/2)/(P1[1]+delta))
n = 5
n_max = np.int(np.min([n,n_limit]))

t = np.linspace(0., T, int(samplerate*T), endpoint=False)


file_id = syllable_id + ".wav"

signal = np.zeros((np.shape(t)))
fig_name = os.path.join(save_dir, file_id[:-4] + ".jpg")
fig = plt.figure()
for j in range(1, n_max+1):
    A = Amplitude_base/10**(j-1)
    if_t = np.zeros((np.shape(t)))
    phi_t = np.zeros((np.shape(t)))

    # create base syllable
    if syllable_type == 'A':
        P1_h = np.array([P1[0], j*P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        t2 = int(P2[0]*samplerate)
        t3 = int(P3[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun1(t[:t2], P1_h, P2_h)
        if_t[t2:t3] = Syllable_IF_fun1(t[t2:t3], P2_h, P3_h)
        if_t[t3:] = Syllable_IF_fun2(t[t3:], P3_h, P4_h)
        phi_t[:t2] = Syllable_Phase_fun1(t[:t2], P1_h, P2_h)
        phi_t[t2:t3] = Syllable_Phase_fun1(t[t2:t3], P2_h, P3_h)
        phi_t[t3:] = Syllable_Phase_fun2(t[t3:], P3_h, P4_h)

    elif syllable_type == 'B':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        P5_h = np.array([P5[0], j * P5[1]])
        t2 = int(P2[0] * samplerate)
        t3 = int(P3[0] * samplerate)
        t4 = int(P4[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun1(t[:t2], P1_h, P2_h)
        if_t[t2:t3] = Syllable_IF_fun3(t[t2:t3], P2_h, P3_h)
        if_t[t4:] = Syllable_IF_fun1(t[t4:], P4_h, P5_h)
        phi_t[:t2] = Syllable_Phase_fun1(t[:t2], P1_h, P2_h)
        phi_t[t2:t3] = Syllable_Phase_fun3(t[t2:t3], P2_h, P3_h)
        phi_t[t4:] = Syllable_Phase_fun1(t[t4:], P4_h, P5_h)

    elif syllable_type == 'C':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        t2 = int(P2[0] * samplerate)
        t3 = int(P3[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun3(t[:t2], P1_h, P2_h)
        if_t[t3:] = Syllable_IF_fun1(t[t3:], P3_h, P4_h)
        phi_t[:t2] = Syllable_Phase_fun3(t[:t2], P1_h, P2_h)
        phi_t[t3:] = Syllable_Phase_fun1(t[t3:], P3_h, P4_h)

    elif syllable_type == 'D':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        t2 = int(P2[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun2(t[:t2], P1_h, P2_h)
        if_t[t2:] = Syllable_IF_fun1(t[t2:], P2_h, P3_h)
        phi_t[:t2] = Syllable_Phase_fun2(t[:t2], P1_h, P2_h)
        phi_t[t2:] = Syllable_Phase_fun1(t[t2:], P2_h, P3_h)

    elif syllable_type == 'E':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        t2 = int(P2[0] * samplerate)
        t3 = int(P3[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun2(t[:t2], P1_h, P2_h)
        if_t[t2:t3] = Syllable_IF_fun3(t[t2:t3], P2_h, P3_h)
        if_t[t3:] = Syllable_IF_fun1(t[t3:], P3_h, P4_h)
        phi_t[:t2] = Syllable_Phase_fun2(t[:t2], P1_h, P2_h)
        phi_t[t2:t3] = Syllable_Phase_fun3(t[t2:t3], P2_h, P3_h)
        phi_t[t3:] = Syllable_Phase_fun1(t[t3:], P3_h, P4_h)

    elif syllable_type == 'F':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        t2 = int(P2[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun2(t[:t2], P1_h, P2_h)
        if_t[t2:] = Syllable_IF_fun2(t[t2:], P2_h, P3_h)
        phi_t[:t2] = Syllable_Phase_fun2(t[:t2], P1_h, P2_h)
        phi_t[t2:] = Syllable_Phase_fun2(t[t2:], P2_h, P3_h)

    elif syllable_type == 'G':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        t2 = int(P2[0] * samplerate)
        t3 = int(P3[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun2(t[:t2], P1_h, P2_h)
        if_t[t2:t3] = Syllable_IF_fun2(t[t2:t3], P2_h, P3_h)
        if_t[t3:] = Syllable_IF_fun1(t[t3:], P3_h, P4_h)
        phi_t[:t2] = Syllable_Phase_fun2(t[:t2], P1_h, P2_h)
        phi_t[t2:t3] = Syllable_Phase_fun2(t[t2:t3], P2_h, P3_h)
        phi_t[t3:] = Syllable_Phase_fun1(t[t3:], P3_h, P4_h)

    elif syllable_type == 'H':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        t2 = int(P2[0] * samplerate)
        t3 = int(P3[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun2(t[:t2], P1_h, P2_h)
        if_t[t3:] = Syllable_IF_fun1(t[t3:], P3_h, P4_h)
        phi_t[:t2] = Syllable_Phase_fun2(t[:t2], P1_h, P2_h)
        phi_t[t3:] = Syllable_Phase_fun1(t[t3:], P3_h, P4_h)

    elif syllable_type == 'I':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        if_t = Syllable_IF_fun3(t, P1_h, P2_h)
        phi_t = Syllable_Phase_fun3(t, P1_h, P2_h)

    elif syllable_type == 'J':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        t2 = int(P2[0] * samplerate)
        t3 = int(P3[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun3(t[:t2], P1_h, P2_h)
        if_t[t3:] = Syllable_IF_fun3(t[t3:], P3_h, P4_h)
        phi_t[:t2] = Syllable_Phase_fun3(t[:t2], P1_h, P2_h)
        phi_t[t3:] = Syllable_Phase_fun3(t[t3:], P3_h, P4_h)

    elif syllable_type == 'K':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        P5_h = np.array([P5[0], j * P5[1]])
        P6_h = np.array([P6[0], j * P6[1]])
        P7_h = np.array([P7[0], j * P7[1]])
        t2 = int(P2[0] * samplerate)
        t3 = int(P3[0] * samplerate)
        t4 = int(P4[0] * samplerate)
        t5 = int(P5[0] * samplerate)
        t6 = int(P6[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun2(t[:t2], P1_h, P2_h)
        if_t[t2:t3] = Syllable_IF_fun3(t[t2:t3], P2_h, P3_h)
        if_t[t4:t5] = Syllable_IF_fun3(t[t4:t5], P4_h, P5_h)
        if_t[t6:] = Syllable_IF_fun1(t[t6:], P6_h, P7_h)
        phi_t[:t2] = Syllable_Phase_fun2(t[:t2], P1_h, P2_h)
        phi_t[t2:t3] = Syllable_Phase_fun3(t[t2:t3], P2_h, P3_h)
        phi_t[t4:t5] = Syllable_Phase_fun3(t[t4:t5], P4_h, P5_h)
        phi_t[t6:] = Syllable_Phase_fun1(t[t6:], P6_h, P7_h)

    elif syllable_type == 'L':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        P5_h = np.array([P5[0], j * P5[1]])
        P6_h = np.array([P6[0], j * P6[1]])
        t2 = int(P2[0] * samplerate)
        t3 = int(P3[0] * samplerate)
        t4 = int(P4[0] * samplerate)
        t5 = int(P5[0] * samplerate)
        t6 = int(P6[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun1(t[:t2], P1_h, P2_h)
        if_t[t2:t3] = Syllable_IF_fun3(t[t2:t3], P2_h, P3_h)
        if_t[t4:t5] = Syllable_IF_fun3(t[t4:t5], P4_h, P5_h)
        if_t[t5:] = Syllable_IF_fun2(t[t5:], P5_h, P6_h)
        phi_t[:t2] = Syllable_Phase_fun1(t[:t2], P1_h, P2_h)
        phi_t[t2:t3] = Syllable_Phase_fun3(t[t2:t3], P2_h, P3_h)
        phi_t[t4:t5] = Syllable_Phase_fun3(t[t4:t5], P4_h, P5_h)
        phi_t[t5:] = Syllable_Phase_fun2(t[t5:], P5_h, P6_h)

    elif syllable_type == 'N':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        P5_h = np.array([P5[0], j * P5[1]])
        P6_h = np.array([P6[0], j * P6[1]])
        t2 = int(P2[0] * samplerate)
        t3 = int(P3[0] * samplerate)
        t4 = int(P4[0] * samplerate)
        t5 = int(P5[0] * samplerate)
        t6 = int(P6[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun2(t[:t2], P1_h, P2_h)
        if_t[t3:t4] = Syllable_IF_fun3(t[t3:t4], P3_h, P4_h)
        if_t[t5:] = Syllable_IF_fun1(t[t5:], P5_h, P6_h)
        phi_t[:t2] = Syllable_Phase_fun2(t[:t2], P1_h, P2_h)
        phi_t[t3:t4] = Syllable_Phase_fun3(t[t3:t4], P3_h, P4_h)
        phi_t[t5:] = Syllable_Phase_fun1(t[t5:], P5_h, P6_h)

    elif syllable_type == 'O':
        P1_h = np.array([P1[0], j * P1[1]])
        P2_h = np.array([P2[0], j * P2[1]])
        P3_h = np.array([P3[0], j * P3[1]])
        P4_h = np.array([P4[0], j * P4[1]])
        t2 = int(P2[0] * samplerate)
        t3 = int(P3[0] * samplerate)
        if_t[:t2] = Syllable_IF_fun1(t[:t2], P1_h, P2_h)
        if_t[t3:] = Syllable_IF_fun1(t[t3:], P3_h, P4_h)
        phi_t[:t2] = Syllable_Phase_fun1(t[:t2], P1_h, P2_h)
        phi_t[t3:] = Syllable_Phase_fun1(t[t3:], P3_h, P4_h)

    else:
        print('Model not supported')
        break

    # add trills

    if trill_flag == 'yes':
        t_trill = np.linspace(trill_t1, trill_t2, int(np.round(samplerate * (trill_t2 - trill_t1))), endpoint=False)
        trill_phase = (delta / k) * (trill_t2 - trill_t1) * (-np.cos(((2 * np.pi * k) / (trill_t2 - trill_t1)) * t_trill))
        phi_t[int(trill_t1 * samplerate):int(trill_t2 * samplerate)] += trill_phase
        trill = delta * np.sin((2 * np.pi * k * (t_trill - trill_t1)) / (trill_t2 - trill_t1))
        if_t[int(trill_t1 * samplerate):int(trill_t2 * samplerate)] += trill

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
s2 = np.concatenate((np.zeros(int(5*samplerate)), signal, np.zeros(int(5*samplerate))))
file_name2 = file_name[:-4]+'_padded.wav'
wavio.write(file_name2, s2, samplerate, sampwidth=2)
