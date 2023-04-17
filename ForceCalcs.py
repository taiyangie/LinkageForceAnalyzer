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
track_f = 47.244 # in, from optimum
track_r = 45.67 #in, from optimum
gravity = 9.81

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


# ## BRAKING PERFORMANCE
#
# #Y
# Brake_Fyfr = 0
# Brake_Fyfl = 0
#
# Brake_Fyrr = 0
# Brake_Fyrl = 0
# #X
# Fb = G*(Pa/r) #breaking force per wheel (Rear)
# Fxr = 2 * Fb #Linear breaking means FB is on two rear wheels
# Fxmf = (mu * (Wfs + (h/L)*Fxr)) / (1 - mu*(h/L))
# Dx = (Fxmf + Fxr)/W #1.7, this is off by .1
#
# Brake_Fxfr = (mu* (Wfs + ((W*Dx*h)/L))) / 2
# Brake_Fxfl = (mu* (Wfs + ((W*Dx*h)/L))) / 2
#
# Brake_Fxrr = (mu * (Wrs - ((W*Dx*h)/L))) / 2 #this is off by 2lb
# Brake_Fxrl = (mu * (Wrs - ((W*Dx*h)/L))) / 2
# #Z
# Brake_Fzfr =(Wfs + ((W*Dx*h)/L)) /2
# Brake_Fzfl =( Wfs + ((W*Dx*h)/L)) /2
#
# Brake_Fzrr = (Wrs -((W*Dx*h)/L)) / 2 #these are all slightly off from the paper :(
# Brake_Fzrl = (Wrs -((W*Dx*h)/L)) / 2

#-------------------------------------------------------

# ## STEADY STATE CORNERING

# Lateral Acceleration


cg_to_f = Wfs * L / W # cg distance to front axle, m
cg_to_r = Wrs * L / W # cg distance to rear axle, m

corner_v = 10.5 # m/s, from past skidpad data THIS IS AN INPUT
corner_r = 18.25 / 2 # skidpad radius
lateral_accel_horizontal = (corner_v**2) / (corner_r * gravity) # THIS IS Gs OF CORNERING, should we make this the input instead?

bank_angle = 0 # Flat circuit, so no bank
lateral_accel = lateral_accel_horizontal * np.cos(bank_angle) - np.sin(bank_angle) # in g

effective_weight = W * (lateral_accel_horizontal * np.sin(bank_angle) + np.cos(bank_angle)) # in kg

# Effective Weights from Cornering

effective_axle_mass_f = effective_weight * cg_to_r / L # kg
effective_axle_mass_r = effective_weight * cg_to_f / L # kg
# Roll Gradient


f_rollcenter_z = 3.578  # inches in optimum
r_rollcenter_z = 1.867 #  inches in optimum

slope = (r_rollcenter_z - f_rollcenter_z) / L
height_under_cg = f_rollcenter_z + (slope * cg_to_f)

cg_to_rollaxis = h - height_under_cg

r_rollrate = 0.4 * 32375 # N/m / rad - from suspension secrets graphic interpolation split f/r
f_rollrate = 0.6 * 32375 # N/m / rad - from suspension secrets graphic interpolation split f/r
roll_gradient = -W * gravity * cg_to_rollaxis / (f_rollrate + r_rollrate) # radians/g
# Lateral Load Transfer Rates

f_lateral_load_transfer = lateral_accel * (W / track_f) * ((cg_to_rollaxis * f_rollrate) / (f_rollrate + r_rollrate)) + (cg_to_r / L) * f_rollcenter_z

r_lateral_load_transfer = lateral_accel * (W / track_r) * ((cg_to_rollaxis * r_rollrate) / (f_rollrate + r_rollrate)) + (cg_to_f / L) * r_rollcenter_z


# Individual Tire Weights

front_outside_wheel_static_wt = Wfs/2
front_inside_wheel_static_wt = Wfs/2
rear_outside_wheel_static_wt = Wrs/2
rear_inside_wheel_static_wt = Wrs/2

Corner_Fzfl_front_outside_wheel_wt = front_outside_wheel_static_wt + f_lateral_load_transfer
Corner_Fzfr_front_inside_wheel_wt = front_inside_wheel_static_wt - f_lateral_load_transfer
Corner_Fzrl_rear_outside_wheel_wt = rear_outside_wheel_static_wt + r_lateral_load_transfer
Corner_Fzrr_rear_inside_wheel_wt = rear_inside_wheel_static_wt - r_lateral_load_transfer


#-------------------------------------------------------

# ## STEADY STATE CORNERING