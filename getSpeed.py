#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
def convo_cut(sig, convolutor, shift=0):
    "Computes the convolution and removes edge effects"
    l= len(convolutor)
    with_edge = np.convolve(sig, convolutor)[l:-l]
    first = [with_edge[0]]
    last = [with_edge[-1]]
##    print(first)
##    print(first*l)
    res = np.concatenate([first*l, with_edge, last*l])
    res = res[shift:]
    return res






#Tools for differentiqtion

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
    return convo_cut(pos , coeff)


#Plot the fft

#plt.plot(myrecarray['czas'], myrecarray['V1'])
#plt.plot(myrecarray['czas'], myrecarray['Vm'])

V1_fft = np.fft.fft(myrecarray['V1'])

pozycja_fft = np.fft.fft(myrecarray['pozycja'])




#Synthetise the position filter

taps = signal.firwin(order, cutoff/nyq)

filteredPos = convo_cut(myrecarray['pozycja'], taps, len(taps)/2)

filteredPos_fft = np.fft.fft(filteredPos)




#Get speed using different methods


V_raw = v_numericalDiff(myrecarray['pozycja'])
V_filteredPos = v_numericalDiff(filteredPos)
V_backard = v_backwardDiff(myrecarray['pozycja'])
V_backard_filtered = v_backwardDiff(filteredPos)







# Plots
##plt.figure(1)
##plt.title('Position FFT')
##
##
##plt.plot(abs(pozycja_fft))
##plt.plot(abs(filteredPos_fft))



plt.figure(2)
plt.title('Position')

plt.plot(myrecarray['pozycja'], label="pos")
plt.plot(filteredPos, label="filtered_pos")



plt.figure(3)
plt.title('Speed')

V_plot, = plt.plot(V_raw, label="V")
V_filteredPos_plot, = plt.plot(V_filteredPos, label="V_filteredPos")
#V_backard_plot, = plt.plot(V_backard, label="V_backard")
V_backard_filtered_plot, = plt.plot(V_backard_filtered, label="V_backard_filtered")

plt.legend([V_plot, V_filteredPos_plot, V_backard_filtered_plot ], ['V', 'V_filteredPos', 'V_backard_filtered' ])



plt.show()
