from scipy.integrate import odeint
from scipy.optimize import root_scalar

import numpy as np

#----------------------------------------
# calibration curve
#----------------------------------------

def CC (R, start=0.7951563356960366, end=1.6384548724324521, k=7.867450337976098): # calibration curve
    return start + (end - start) * R / (k + R) # F500 / F430

#----------------------------------------
# initial conditions
#----------------------------------------

def IC (alpha): # initial conditions

    ADP_tot = A_tot / (1 + alpha) # ADP + MgADP
    ATP_tot = alpha * A_tot / (1 + alpha) # ATP + MgATP

    def cubic (M): # cubic equation for y0_Mg_in

        a = K_d_MgADP + K_d_MgATP + A_tot - Mg_tot # auxiliar parameter [M]
        b = (K_d_MgADP * K_d_MgATP + \
            (A_tot / (1 + alpha)) * (K_d_MgATP + alpha * K_d_MgADP) - \
            Mg_tot * (K_d_MgADP + K_d_MgATP)) # auxiliar parameter [M^2]
        c = - Mg_tot * K_d_MgADP * K_d_MgATP # auxiliar parameter [M^3]

        return M**3 + a * M**2 + b * M + c

    sol = root_scalar (cubic, bracket=[1e-12, Mg_tot-1e-12], xtol=1e-12) # solution of the cubic equation for y0_Mg_in

    if not sol.converged:
        raise ValueError ("Solver did not converge")

    Mg = sol.root
    ADP = ADP_tot * K_d_MgADP / (K_d_MgADP + Mg)
#    MgADP = ADP_tot * Mg / (K_d_MgADP + Mg)
#    ATP = ATP_tot * K_d_MgATP / (K_d_MgATP + Mg)
    MgATP = ATP_tot * Mg / (K_d_MgATP + Mg)

    return [ADP, MgATP]

#----------------------------------------
# enzyme velocities
#----------------------------------------

def ArcD_vel (y, nArcD): # ArcD velocity [mol/s]

    arg_out, orn_out, arg_in, orn_in = y

    f_arg_out = (arg_out / KmDarg) / (1 + arg_out / KmDarg + orn_out / KmDorn) # fractional concentration
    f_orn_out = (orn_out / KmDorn) / (1 + arg_out / KmDarg + orn_out / KmDorn) # fractional concentration
    f_arg_in = (arg_in / KmDarg) / (1 + arg_in / KmDarg + orn_in / KmDorn) # fractional concentration
    f_orn_in = (orn_in / KmDorn) / (1 + arg_in / KmDarg + orn_in / KmDorn) # fractional concentration

    V_ArcD = nArcD * kDp * kDm * (f_arg_out * f_orn_in - f_arg_in * f_orn_out) / \
    (kDp * (f_arg_out + f_orn_out) + kDm * (f_orn_in + f_arg_in)) # ArcD velocity [mol/s]

    return V_ArcD

def ArcA_vel (arg, ArcA): # ArcA velocity [M/s]
    return 4 * ArcA * kA * (arg / KmAarg) / (1 + arg / KmAarg + arg**2 / (KmAarg * KiAarg))

def ArcB_vel (y, ArcB):  # ArcB velocity [M/s]

    cit, CP, HPO4, orn = y

    N = kBunfav / (KiBhpo4 * KmBcit) * HPO4 * cit - kBfav / (KmBorn * KiBcp) * orn * CP # numerator

    D = 1 + \
        cit * orn * CP / (KiBcit * KmBorn * KiBcp) + \
        orn * CP / (KmBorn * KiBcp) + \
        CP / KiBcp + \
        orn * KmBcp / (KmBorn * KiBcp) # denominator

    V_ArcB = 6 * ArcB * N / D

    return V_ArcB

def ArcC_vel (y, ArcC): # ArcC velocity [M/s]

    carb, CP, MgADP, MgATP = y

    alp = MgADP / KmCadp # auxiliar parameter
    bta = CP / KmCcp # auxiliar parameter
    pi = MgATP / KmCatp # auxiliar parameter
    rho = carb / KmCcarb # auxiliar parameter

    V_ArcC = 2 * ArcC * (kCp * alp * bta - kCm * pi * rho) * \
        (alp + pi) * (1 + bta + rho) / ((1 + (alp + pi)**2) * \
        (1 + (bta + rho) * (2 + bta + rho))) # ArcC velocity [M/s]

    return V_ArcC

def AAC_vel (y, nT): # AAC velocity [mol/s]

    ADP_out, ATP_out, ADP_in, ATP_in = y

    f_adp_out = (ADP_out / KmTadp) / (1 + ADP_out / KmTadp + ATP_out / KmTatp) # fractional concentration
    f_atp_out = (ATP_out / KmTatp) / (1 + ADP_out / KmTadp + ATP_out / KmTatp) # fractional concentration
    f_adp_in = (ADP_in / KmTadp) / (1 + ADP_in / KmTadp + ATP_in / KmTatp) # fractional concentration
    f_atp_in = (ATP_in / KmTatp) / (1 + ADP_in / KmTadp + ATP_in / KmTatp) # fractional concentration

    V_AAC = nT * kTp * kTm * \
        (f_adp_out * f_atp_in - f_adp_in * f_atp_out) / \
        (kTp * (f_adp_out + f_atp_out) + kTm * (f_atp_in + f_adp_in)) # AAC velocity [mol/s]

    return V_AAC

#----------------------------------------
# system of mass-balance equations
#----------------------------------------

def SDE_woT (y, t, y0_orn_in): # system of differential equations without AAC

    arg_out, carb_out, HCO3_out, NH3_out, orn_out = y[:5] # outside concentrations [M]
    ADP_in, carb_in, cit_in, CP_in, HCO3_in, HPO4_in, MgATP_in, NH3_in, OCN_in, orn_in = y[5:15] # inside concentrations [M]

# inside concentrations [M]

    arg_in = y0_orn_in - orn_in - cit_in
    Mg_in = (Mg_tot - MgATP_in) / (1 + ADP_in / K_d_MgADP)
    ATP_in = K_d_MgATP * MgATP_in / Mg_in
    MgADP_in = Mg_in * ADP_in / K_d_MgADP

# ArcD, ArcA, ArcB, and ArcC velocities

    V_ArcD = ArcD_vel([arg_out, orn_out, arg_in, orn_in], n_ArcD) # ArcD velocity [mol/s]
    V_ArcA = ArcA_vel(arg_in, ArcA) # ArcA velocity [M/s]
    V_ArcB = ArcB_vel([cit_in, CP_in, HPO4_in, orn_in], ArcB) # ArcB velocity [M/s]
    V_ArcC = ArcC_vel([carb_in, CP_in, MgADP_in, MgATP_in], ArcC) # ArcC velocity [M/s]

# passive diffusion

    Jco2 = 1e-3 * Pco2 * (HCO3_out - HCO3_in) * H / K_a_co2 # CO2 flux [mol/m2]
    uco2 = A * Jco2 # CO2 diffusion velocity [mol/s]

    Jnh3 = 1e-3 * Pnh3 * (NH3_out - NH3_in) # NH3 flux [mol/m2]
    unh3 = A * Jnh3 # NH3 diffusion velocity [mol/s]

# CP degradation

    vCPh = kCPh * CP_in # CP hydrolysis velocity [M/s]

# buffer

    vbuff = (V_ArcB - vCPh) / (1 + K_a_buff / H) # buffer velocity [M/s]

# carb, CO2, NH4, OCN

    v_carb_out = kpcarb * (carb_out - NH3_out * HCO3_out / K_eq_carb) # carb [M/s]
    v_co2_out = -uco2 / Vo - (v_carb_out - uco2 / Vo) / (1 + K_a_co2 / H) # CO2 [M/s]
    v_nh4_out = (unh3 / Vo - v_carb_out) / (1 + K_a_nh4 / H) # NH4 [M/s]

    v_carb_in = kpcarb * (carb_in - NH3_in * HCO3_in / K_eq_carb) # carb [M/s]
    v_OCN_in = kOCNh * OCN_in # OCN [M/s]
    v_co2_in = uco2 / Vi - (v_carb_in + v_OCN_in + uco2 / Vi) / (1 + K_a_co2 / H) # CO2 [M/s]
    v_nh4_in = V_ArcA - (V_ArcA + unh3 / Vi + v_carb_in + v_OCN_in) / (1 + K_a_nh4 / H) # NH4 [M/s]

# ADP, MgATP

    a11 = K_d_MgADP + Mg_in + ADP_in # auxiliar parameter [M]
    a12 = ADP_in # auxiliar parameter [M]
    a21 = ATP_in # auxiliar parameter [M]
    a22 = K_d_MgATP + Mg_in + ATP_in # auxiliar parameter [M]

    b1 = - K_d_MgADP * V_ArcC # auxiliar parameter [M^2/s]
    b2 = K_d_MgATP * V_ArcC # auxiliar parameter [M^2/s]

    v_MgADP, v_MgATP = (a22 * b1 - a12 * b2) / (a11 * a22 - a12 * a21), (a21 * b1 - a11 * b2) / (a12 * a21 - a11 * a22) # [M/s]

# differential equations

    d_arg_out = - V_ArcD / Vo
    d_carb_out = - v_carb_out
    d_HCO3_out = v_co2_out + v_carb_out
    d_NH3_out = - unh3 / Vo + v_nh4_out + v_carb_out
    d_orn_out = V_ArcD / Vo

    d_ADP_in = v_MgADP
    d_carb_in = V_ArcC - v_carb_in + vCPh / 2
    d_cit_in = V_ArcA - V_ArcB
    d_CP_in = V_ArcB - V_ArcC - vCPh
    d_HCO3_in = v_co2_in + v_carb_in + v_OCN_in
    d_HPO4_in = - V_ArcB + vbuff + vCPh
    d_MgATP_in = V_ArcC - v_MgATP
    d_NH3_in = unh3 / Vi + v_nh4_in + v_carb_in + v_OCN_in
    d_OCN_in = vCPh / 2 - v_OCN_in
    d_orn_in = - V_ArcD / Vi + V_ArcB

    return [d_arg_out, d_carb_out, d_HCO3_out, d_NH3_out, d_orn_out, d_ADP_in, d_carb_in, d_cit_in, d_CP_in, d_HCO3_in, d_HPO4_in, d_MgATP_in, d_NH3_in, d_OCN_in, d_orn_in]

def SDE_wT (y, t, y0_orn_in, y0_ADP_out): # system of differential equations with AAC

    ADP_out, arg_out, carb_out, HCO3_out, NH3_out, orn_out = y[:6] # outside concentrations [M]
    ADP_in, carb_in, cit_in, CP_in, HCO3_in, HPO4_in, MgATP_in, NH3_in, OCN_in, orn_in = y[6:16] # inside concentrations [M]

# concentrations [M]

    ATP_out = y0_ADP_out - ADP_out
    arg_in = y0_orn_in - orn_in - cit_in
    Mg_in = (Mg_tot - MgATP_in) / (1 + ADP_in / K_d_MgADP)
    ATP_in = K_d_MgATP * MgATP_in / Mg_in
    MgADP_in = Mg_in * ADP_in / K_d_MgADP

# ArcD, ArcA, ArcB, ArcC, and AAC velocities

    V_ArcD = ArcD_vel([arg_out, orn_out, arg_in, orn_in], n_ArcD) # ArcD velocity [mol/s]
    V_ArcA = ArcA_vel(arg_in, ArcA) # ArcA velocity [M/s]
    V_ArcB = ArcB_vel([cit_in, CP_in, HPO4_in, orn_in], ArcB) # ArcB velocity [M/s]
    V_ArcC = ArcC_vel([carb_in, CP_in, MgADP_in, MgATP_in], ArcC) # ArcC velocity [M/s]
    V_AAC = AAC_vel([ADP_out, ATP_out, ADP_in, ATP_in], n_AAC) # AAC velocity [mol/s]

# passive diffusion

    Jco2 = 1e-3 * Pco2 * (HCO3_out - HCO3_in) * H / K_a_co2 # CO2 flux [mol/m2]
    uco2 = A * Jco2 # CO2 diffusion velocity [mol/s]

    Jnh3 = 1e-3 * Pnh3 * (NH3_out - NH3_in) # NH3 flux [mol/m2]
    unh3 = A * Jnh3 # NH3 diffusion velocity [mol/s]

# CP degradation

    vCPh = kCPh * CP_in # CP hydrolysis velocity [M/s]

# buffer

    vbuff = (V_ArcB - vCPh) / (1 + K_a_buff / H) # buffer velocity [M/s]

# carb, CO2, NH4, OCN

    v_carb_out = kpcarb * (carb_out - NH3_out * HCO3_out / K_eq_carb) # carb [M/s]
    v_co2_out = - uco2/Vo - (v_carb_out - uco2 / Vo) / (1 + K_a_co2 / H) # CO2 [M/s]
    v_nh4_out = (unh3 / Vo - v_carb_out) / (1 + K_a_nh4 / H) # NH4 [M/s]

    v_carb_in = kpcarb * (carb_in - NH3_in * HCO3_in / K_eq_carb) # carb [M/s]
    v_OCN_in = kOCNh * OCN_in # OCN [M/s]
    v_co2_in = uco2 / Vi - (v_carb_in + v_OCN_in + uco2 / Vi) / (1 + K_a_co2 / H) # CO2 [M/s]
    v_nh4_in = V_ArcA - (V_ArcA + unh3 / Vi + v_carb_in + v_OCN_in) / (1 + K_a_nh4 / H) # NH4 [M/s]

# ADP, MgATP

    a11 = K_d_MgADP + Mg_in + ADP_in # auxiliar parameter [M]
    a12 = ADP_in # auxiliar parameter [M]
    a21 = ATP_in # auxiliar parameter [M]
    a22 = K_d_MgATP + Mg_in + ATP_in # auxiliar parameter [M]

    b1 = - K_d_MgADP * V_ArcC - Mg_in * V_AAC / Vi # auxiliar parameter [M^2/s]
    b2 = K_d_MgATP * V_ArcC + Mg_in * V_AAC / Vi # auxiliar parameter [M^2/s]

    v_MgADP, v_MgATP = (a22 * b1 - a12 * b2) / (a11 * a22 - a12 * a21), (a21 * b1 - a11 * b2) / (a12 * a21 - a11 * a22) # [M/s]

# differential equations

    d_ADP_out = - V_AAC / Vo
    d_arg_out = - V_ArcD / Vo
    d_carb_out = - v_carb_out
    d_HCO3_out = v_co2_out + v_carb_out
    d_NH3_out = - unh3 / Vo + v_nh4_out + v_carb_out
    d_orn_out = V_ArcD / Vo

    d_ADP_in = V_AAC / Vi + v_MgADP
    d_carb_in = V_ArcC - v_carb_in + vCPh / 2
    d_cit_in = V_ArcA - V_ArcB
    d_CP_in = V_ArcB - V_ArcC - vCPh
    d_HCO3_in = v_co2_in + v_carb_in + v_OCN_in
    d_HPO4_in = - V_ArcB + vbuff + vCPh
    d_MgATP_in = V_ArcC - v_MgATP
    d_NH3_in = unh3 / Vi + v_nh4_in + v_carb_in + v_OCN_in
    d_OCN_in = vCPh / 2 - v_OCN_in
    d_orn_in = - V_ArcD / Vi + V_ArcB

    return [d_ADP_out, d_arg_out, d_carb_out, d_HCO3_out, d_NH3_out, d_orn_out, d_ADP_in, d_carb_in, d_cit_in, d_CP_in, d_HCO3_in, d_HPO4_in, d_MgATP_in, d_NH3_in, d_OCN_in, d_orn_in]

def SSDE_woT (t, y0, y0_orn_in): # solution of the system of differential equations without AAC
    return odeint (SDE_woT, y0, t, args=(y0_orn_in,), rtol=1e-10, atol=1e-10)

def SSDE_wT (t, y0, y0_orn_in, y0_ADP_out): # solution of the system of differential equations with AAC
    return odeint (SDE_wT, y0, t, args=(y0_orn_in, y0_ADP_out), rtol=1e-10, atol=1e-10)

y0_orn_in = [0.5e-3, 10e-3] # initial ornithine concentration inside [M]
y0_arg_out = 10e-3 # initial arginine concentration outside [M]
y0_ADP_out = [0.1e-3, 0.5e-3] # initial ADP concentration outside [M]

n_ArcD = 31.4e-12 # ArcD amount [mol]
n_AAC = 51.0e-12 # AAC amount [mol]

ArcA = 1e-6 # ArcA concentration [M]
ArcB = 2e-6 # ArcB concentration [M]
ArcC = 5e-6 # ArcC concentration [M]

K_a_buff = 10**(-6.87) # acid dissociation constant of the reaction H2PO4- = HPO4-2 + H+ [M]
K_a_nh4 = 10**(-9.084) # acid dissociation constant of the reaction NH4+ = NH3 + H+ [M]
K_a_co2 = 10**(-6.33) # acid dissociation constant of the reaction CO2 + H2O = HCO3- + H+ [M]

K_eq_carb=0.53 # equilibrium constant of the reaction carb = NH3 + HCO3- [M]

K_d_MgADP = 0.6e-3 # dissociation constant of the reaction MgADP = Mg+2 + ADP [M]
K_d_MgATP = 0.1e-3 # dissociation constant of the reaction MgATP = Mg+2 + ATP [M]

A_tot = 10e-3 # total nucleotide concentration inside [M]
Mg_tot = 10e-3 # total Mg+2 concentration [M]
KPi_tot = 50e-3 # total buffer concentration [M]

H = 10**(-7) # H+ concentration [M]
y0_HPO4_in = KPi_tot / (1 + H / K_a_buff) # initial HPO4 concentration [M]

kDm, KmDarg, KmDorn = 87, 28.7e-6, 1.402e-3 # ArcD parameters
kA, KmAarg, KiAarg = 6.93, 5.1e-6, 3.2e-3 # ArcA parameters
kBunfav, KmBcp, KiBcp, KmBorn, KmBcit, KiBcit, KiBhpo4, KBeq = 89/6, 0.898e-3, 1.37e-6, 1.095e-3, 0.255e-3, 3.168e-3, 109.00e-3, 13.00e-6 # ArcB parameters
VfvC, kCm, KmCadp, KmCcp, KmCatp, KCeq = 1.50, 573, 2.54e-3, 0.618e-3, 9.5e-3, 6 # ArcC parameters
kTm, KmTatp = 8.5, 4.05e-6 # AAC parameters

kDp = 6.0 * kDm / (kDm - 6.0) # ArcD
kBfav = kBunfav * KmBorn * KiBcp / (KBeq * KiBhpo4 * KmBcit) # Haldane relationship (ArcB)
kCp = 7.3e7 * VfvC / (2 * 60e3) # ArcC
KmCcarb = kCp * KmCcp * KmCadp / (KCeq * kCm * KmCatp) # Haldane relationship (ArcC)
kTp = 1.9 * kTm / (kTm - 1.9) # AAC
KmTadp = 2.2e-6 * (kTp + kTm) / kTm # AAC

Pco2 = 0.16e-2 # CO2 permeability coefficient [m/s]
Pnh3 = 0.13e-2 # NH3 permeability coefficient [m/s]

kCPh = 1.5e-4 # CP hydrolysis rate [1/s]
kOCNh = 5.341e-4 # cyanate -> NH3 + HCO3- [1/s]
kpcarb = 124 # carbamate hydrolysis rate [1/s]

Vt = 125e-6 # total volume [l]
Vi = 0.893e-6 # internal volume [l]
Vo = Vt - Vi # external volume [l]

vr = 226e-9 # vesicle radius [m]
A = 3e-3 * Vi / vr # surface area of the membrane [m^2]
