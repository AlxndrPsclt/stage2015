#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import numpy as np
import os
os.chdir('/home/alex/Documents')


sample_rate = 1000
nyq= sample_rate/2.0

#Cuttof and order frequency of the position filter (Hz)
cutoff = 12
order = 15


#Reading the CSV
def read_array(filename, dtype, separator=','):
    """ Read a file with an arbitrary number of columns.
    The type of data in each column is arbitrary
    It will be cast to the given dtype at runtime
    """
    cast = np.cast
    data = [[] for dummy in range(len(dtype))]
    for line in open(filename, 'r'):
        fields = line.strip().split(separator)
        for i, number in enumerate(fields):
            data[i].append(number)
    for i in range(len(dtype)):
        data[i] = cast[dtype[i]](data[i])
    return np.rec.array(data, dtype=dtype)

mydescr = np.dtype([('czas', 'float32'), ('sterowanie', 'float32'), ('pozycja', 'float32'), ('generatorPRBS', 'float32'), ('V1', 'float32'), ('V2', 'float32') , ('Vm', 'float32') ])


myrecarray = read_array('Pomiar1_02.csv', mydescr, ";")



#tool for convolutions
def convo_cut(sig, convolutor, shift=-1):
    "Computes the convolution and removes edge effects"
    l= len(convolutor)
    with_edge = np.convolve(sig, convolutor)[l:-l]
    first = [with_edge[0]]
    last = [with_edge[-1]]
##    print(first)
##    print(first*l)
    res = np.concatenate([first*l, with_edge, last*l])
    #res = res[math.ceil(l/2):]
    if shift==-1:
        shift=math.ceil(l/2)
    res = res[shift:]
    return res






#Tools for differentiation

def numericalDiff_coeff(derivative=1, order=4):
    "Returns an array of the coeffs used to compute a central numerical differentiation"
    coeffs = [[[-1/2,0,1/2],
               [1/12,-2/3,0,2/3,-1/12],
               [-1/60,3/20,-3/4,0,3/4,-3/20,1/60],
               [1/280,-4/105,1/5,-4/5,0,4/5,-1/5,4/105,-1/280]],
              [[2,2,1,-2,1],
               [-1/12,4/3,-5/2,4/3,-1/12],
               [1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90],
               [-1/560,8/315,-1/5,8/5,-205/72,8/5,-1/5,8/315,-1/560]]]
    return np.array(coeffs[derivative-1][order -1])


def backwardDiff_coeff(derivative=1, order=4):
    "Returns an array of the coeffs used to compute a backward numerical differentiation."
    coeffs = [[[-1,1],
               [1/2,-2,3/2],
               [-1/3,3/2,-3,11/6],
               [1/4,-4/3,3,-4,25/12],
               [-1/5,5/4,-10/3,5,-5,137/60],
               [1/6,-6/5,15/4,-20/3,15/2,-6,49/20]],
              [[1,-2,1],
               [-1,4,-5,2],
               [11/12,-14/3,19/2,-26/3,35/12],
               [-5/6,61/12,-13,107/6,-77/6,15/4],
               [137/180,-27/5,33/2,-254/9,117/4,-87/5,203/45],
               [-7/10,1019/180,-201/10,41,-949/18,879/20,-223/10,469/90]]]
    return np.array(coeffs[derivative-1][order -1])
              





def v_numericalDiff(pos, derivative=1, order =4):
    "Computes the speed using a centered differentiation formula"
    coeff = numericalDiff_coeff(derivative, order)
    return convo_cut(pos , coeff)


def v_backwardDiff(pos, derivative=1, order =4):
    coeff = backwardDiff_coeff(derivative, order)
    return convo_cut(pos , coeff, order)


#Plot the fft

V1_fft = np.fft.fft(myrecarray['V1'])

pozycja_fft = np.fft.fft(myrecarray['pozycja'])




#Synthetize the position filter, and filter the signal

taps = signal.firwin(order, cutoff/nyq)

filteredPos = convo_cut(myrecarray['pozycja'], taps)

filteredPos_fft = np.fft.fft(filteredPos)




#Get speed using different methods

v_raw = v_numericalDiff(myrecarray['pozycja'])
#V_filteredPos = v_numericalDiff(filteredPos)
#V_backard = v_backwardDiff(myrecarray['pozycja'])
#V_backard_filtered = v_backwardDiff(filteredPos)


v_numericalDiffs = []
for i in range(1,5):
    v_numericalDiffs.append(v_numericalDiff(filteredPos, 1, i))



v_backwardDiffs=[]
for i in range(1,7):
    v_backwardDiffs.append(v_backwardDiff(filteredPos, 1, i))



#-----Integrate-----------
integrated_numerical_speeds = []
for sig in v_numericalDiffs:
    integrated_speed= np.cumsum(sig)
    integrated_speed*=-1
    integrated_speed+=filteredPos[0]
    integrated_speed=np.concatenate([[filteredPos[0]],integrated_speed])
    integrated_numerical_speeds.append(integrated_speed)



integrated_backward_speeds = []
for sig in v_backwardDiffs:
    integrated_speed= np.cumsum(sig)
    integrated_speed*=-1
    integrated_speed+=filteredPos[0]
    integrated_speed=np.concatenate([[filteredPos[0]],integrated_speed])
    integrated_backward_speeds.append(integrated_speed)





# Plots

##plt.figure(1)
##plt.title('Position FFT')
##
##
##plt.plot(abs(pozycja_fft))
##plt.plot(abs(filteredPos_fft))


#------Plot 2------------
plt.figure(2)
plt.title('Position')

plt.plot(myrecarray['pozycja'], label="pos")
plt.plot(filteredPos, label="filtered_pos")

plt.legend()




#------Plot 3------------
plt.figure(3)
plt.title('Central Speed using different precisions')

#raw
v_plot, = plt.plot(v_raw, label="V raw")

#filtered central
v_numericalDiffsPlots=[]
for i in range(len(v_numericalDiffs)):
    v_filteredPos_plot, = plt.plot(v_numericalDiffs[i], label="Order="+str(i+1))
    v_numericalDiffsPlots.append(v_filteredPos_plot)
plt.legend()




#------Plot 4------------
plt.figure(4)
plt.title('Backward Speed using different precisions')

#raw
v_plot, = plt.plot(v_raw, label="V raw")

#filtered backward
v_backwardDiffsPlots=[]
for i in range(len(v_backwardDiffs)):
    v_filteredPos_plot, = plt.plot(v_backwardDiffs[i], label="Order="+str(i+1))
    v_backwardDiffsPlots.append(v_filteredPos_plot)


plt.legend()




#------Plot 5------------
plt.figure(5)
integrated_numerical_speeds_plots = []
plt.title('Real Position and position integrated from central derivative')

#data postion and smoothed position
plt.plot(myrecarray['pozycja'], label="pos")
plt.plot(filteredPos, label="filtered_pos")

#Integrated positions
for i in range(len(integrated_numerical_speeds)):
    integrated_numerical_speeds_plot, = plt.plot(integrated_numerical_speeds[i], label="Order="+str(i+1))
    integrated_numerical_speeds_plots.append(integrated_numerical_speeds_plot)
plt.legend()



#------Plot 6------------
plt.figure(6)
integrated_backward_speeds_plots = []
plt.title('Real Position and position integrated from backward derivative')

#data postion and smoothed position
plt.plot(myrecarray['pozycja'], label="pos")
plt.plot(filteredPos, label="filtered_pos")

#Integrated positions
for i in range(len(integrated_backward_speeds)):
    integrated_backward_speeds_plot, = plt.plot(integrated_backward_speeds[i], label="Order="+str(i+1))
    integrated_backward_speeds_plots.append(integrated_backward_speeds_plot)
plt.legend()





#V_filteredPos_plot, = plt.plot(V_filteredPos, label="V_filteredPos")
#V_backard_plot, = plt.plot(V_backard, label="V_backard")
#V_backard_filtered_plot, = plt.plot(V_backard_filtered, label="V_backard_filtered")

#plt.legend([V_plot, V_filteredPos_plot, V_backard_filtered_plot ], ['V raw', 'V_filteredPos', 'V_backard_filtered' ])



plt.show()
