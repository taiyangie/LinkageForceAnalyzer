# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:47:29 2023
@author: sunny_li

Notes on naming conventions:
F(axis)(fore/aft)(right/left)
Fyfr would be Force in the y direction through the front right assembly
"""



## IMPORTS ##
import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd
## INPUTS ##
W = 530 #weight
F = 40 #front weight distribution
R = 60 #rear weight dist
L = 61 #wheelbase
h = 12 #CoG Height, inch
r = 10 #wheel rad, inch
Wfs = 212 #static front weight
Wrs = 318 #static rear weight
c = 24.4 #28 #CG to rear axle
b = 36.6 #32.125 #front axle to CG
mu = 1.34 #peak coef of friction
G = 2.7 #(in-lb/psi) Rear Break gain (in-lb/psi)
Pa = 586.71 # psi Rear Break application pressure


#-------------------------------------------------------
## LINEAR ACCELERATION
#
# #Y
# LinAccel_Fyfr = 0 #forces in y, front right
# LinAccel_Fyfl = 0
# LinAccel_Fyrr = 0
# LinAccel_Fyrl = 0
# #X
# mu=.714
# LinAccel_Fxrr = ((mu*W*b)/L) / (1-(h/L)*mu) / 2
# LinAccel_Fxrl = ((mu*W*b)/L) / (1-(h/L)*mu) / 2
# LinAccel_Fxfr = 0
# LinAccel_Fxfl = 0
# #Z
# g = .9945
# LinAccel_Fzfr = (W* ((c/L) - g*(h/L))) / 2
# LinAccel_Fzfl = (W* ((c/L) - g*(h/L))) / 2
# LinAccel_Fzrr = (W* ((b/L) + g*(h/L))) / 2 #div by 2 for ea wheel
# LinAccel_Fzrl = (W* ((b/L) + g*(h/L))) / 2
#-------------------------------------------------------


## BRAKING PERFORMANCE

#Y
Brake_Fyfr = 0
Brake_Fyfl = 0

Brake_Fyrr = 0
Brake_Fyrl = 0
#X
Fb = G*(Pa/r) #breaking force per wheel (Rear)
Fxr = 2 * Fb #Linear breaking means FB is on two rear wheels
Fxmf = (mu * (Wfs + (h/L)*Fxr)) / (1 - mu*(h/L))#6.97*(407.37/r)
Dx = (Fxmf + Fxr)/W #1.7, this is off by .1

Brake_Fxfr = (mu* (Wfs + ((W*Dx*h)/L))) / 2
Brake_Fxfl = (mu* (Wfs + ((W*Dx*h)/L))) / 2

Brake_Fxrr = (mu * (Wrs - ((W*Dx*h)/L))) / 2 #this is off by 2lb
Brake_Fxrl = (mu * (Wrs - ((W*Dx*h)/L))) / 2
#Z
Brake_Fzfr =( Wfs + ((W*Dx*h)/L)) /2
Brake_Fzfl =( Wfs + ((W*Dx*h)/L)) /2

Brake_Fzrr = (Wrs -((W*Dx*h)/L)) / 2 #these are all slightly off from the paper :(
Brake_Fzrl = (Wrs -((W*Dx*h)/L)) / 2

#-------------------------------------------------------

# ## STEADY STATE CORNERING
# h1 = 12 #height of spring mass center of grav above roll axis
# hf = 3.03 #dist front acle and roll axis
# Kphif = 3219
# Kphir = 3663
# phi = (W*h1* (1)) / (Kphif+Kphir-W*h1) #replace 1 w number of gs the vehicle undergoes while cornering
