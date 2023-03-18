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

#-------------------------------------------------------
# ## LINEAR ACCELERATION
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
# #-------------------------------------------------------


# ## BRAKING PERFORMANCE
#
# #Y
# Fy = 0
# #X
# Fxr = 2* 2.7*(586.71/r)
# Fxmf = (1.34 * (Wfs + (h/L)*Fxr)) / (1 - 1.34*(h/L))#6.97*(407.37/r)
# Dx = (Fxmf + Fxr)/W #1.7, this is off by .1
# Fxfront = 1.34* (Wfs + ((W*Dx*h)/L))
#
# Fxrear = 1.34 * (Wrs - ((W*Dx*h)/L)) #this is off by 2lb
# #Z
# Fzfront = Wfs + ((W*Dx*h)/L)
# Fzrear = Wrs -((W*Dx*h)/L) #these are all slightly off from the paper :(

#-------------------------------------------------------

## STEADY STATE CORNERING
h1 = 12 #height of spring mass center of grav above roll axis
hf = 3.03 #dist front acle and roll axis
Kphif = 3219
Kphir = 3663
phi = (W*h1* (1)) / (Kphif+Kphir-W*h1) #replace 1 w number of gs the vehicle undergoes while cornering
