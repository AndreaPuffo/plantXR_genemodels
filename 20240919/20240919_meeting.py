# Model organs: hypocotyl+leaf, root.
# Functions: Respiration (light,T), StomataOpen (T,D,light)
# Need to include: leaf angle, petiole length, photosynthetic activity based on leaf area,
# sink strenghts (fixed then dynamic), carbon consumption (growth constrained by its availability)
# and storage in starch. Water should be further considered in Farquhar

# Future steps: 1. how this partitionaning is affected by TxD (HY5->SWEET)?
# 2. light-dark cycle differences in C allocation (into or out of starch)
# 3. stomata opening model should be more elaborate, i.e. exploit efforts from Reka Albert and add Bergmann dev.

# FSPM ref
# 2019 https://academic.oup.com/jxb/article/70/9/2463/5336616
# 2022 https://academic.oup.com/insilicoplants/article/4/2/diac010/6594971

import numpy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle


# Notation
# Production pVariable
# Decay decayVariable - not dVariable because that is the ODE
# Half-rate hrVariable
# Inactivation iVariable
# Activation aVariable
# Association ass
# Dissociation diss
# Growth gOrgan

# Hypocotyl model - just need to include sat regulation in PIF4 sequestration
def hypocotyl(a, t, ABA, Temp, lighttime, light):
    hrCOP1 = 25
    hrABA = 0.5
    hrPhyB = 25
    hrPIF4 = 0.4
    basalgrowth = 0.1
    decayPhyB = 0.01
    pPhyB = 100 * decayPhyB
    aLPhyB = 0.5
    iTPhyB = 0.5
    iPhyB = 0.2  # Revisar porque PhyB no se esta regulando...
    decayELF3 = 0.001
    pELF3 = 100 * decayELF3
    decayELF3 = 0.001 * 500  # decay has to be huge to recover nice COP1 pattern
    iELF3 = 0.4
    aELF3 = 0.1
    decayPIF4 = 0.001
    pPIF4 = 100 * decayPIF4 / 2
    pPIF4T = 100 * decayPIF4 / 2
    decayPIF4PhyB = 0.001 * 2
    seqPIF4 = 0.4
    seqinvPIF4 = 0.2  # This needs improvement- should be modelled as protein complexes...
    decayCOP1 = 0.1
    pCOP1 = 100 * decayCOP1
    decayCOP1nucleus = 0.5  # COP1 needs improvement - should be modelled as a cytoplasmic and a nuclear pool
    decayHY5 = 0.001
    pHY5 = 100 * decayHY5
    decayHY5COP1 = 0.01 * 20
    growthrate = 5
    PhyBi = a[0]
    PhyBa = a[1]
    ELF3a = a[2]
    ELF3i = a[3]
    PIF4 = a[4]
    PIF4seq = a[5]
    COP1 = a[6]
    HY5 = a[7]
    Carbon = a[10]
    PhyBregulation = PhyBa * PhyBa / (PhyBa * PhyBa + hrPhyB * hrPhyB)
    dPhyBi = pPhyB - decayPhyB * PhyBi - light * aLPhyB * PhyBi + Temp * iTPhyB * PhyBa + iPhyB * PhyBa
    dPhyBa = light * aLPhyB * PhyBi - Temp * iTPhyB * PhyBa - iPhyB * PhyBa - decayPhyB * PhyBa
    dELF3a = abs(1 - light) * pELF3 - decayELF3 * ELF3a - Temp * iELF3 * ELF3a + aELF3 * ELF3i
    dELF3i = Temp * iELF3 * ELF3a - aELF3 * ELF3i - decayELF3 * ELF3i
    dPIF4 = pPIF4 + pPIF4T * Temp - decayPIF4 * PIF4 - decayPIF4PhyB * PhyBa * PIF4 - seqPIF4 * PIF4 * (
                HY5 + PhyBa + ELF3a) + seqinvPIF4 * PIF4seq
    dPIF4seq = seqPIF4 * PIF4 * (
                HY5 + PhyBa + ELF3a) - seqinvPIF4 * PIF4seq - decayPIF4 * PIF4seq - decayPIF4PhyB * PhyBa * PIF4seq
    #    dCOP1=pCOP1-decayCOP1*COP1-decayCOP1nucleus*COP1*(ABA*ABA/(ABA*ABA+hrABA*hrABA)+PhyBa*PhyBa/(PhyBa*PhyBa+hrPhyB*hrPhyB))
    dCOP1 = pCOP1 - decayCOP1 * COP1 - decayCOP1nucleus * COP1 * (ABA * ABA / (ABA * ABA + hrABA * hrABA) + (
                -0.7 * Temp + 1) * light)  # light nuclear regulation was poorly modelled
    # COP1 decay looks slower in lightxT Nieto 22 (light*(Temp-22)/6) - they modelled it as an increas light production! mech is via decay! this is my px
    # Equation so that 22 is 1 and 28 is lower (0.5) -0.5*Temp+1
    # Nieto paper claims COP1 increased activity in light period is what matter for therm
    dHY5 = pHY5 - decayHY5 * HY5 - decayHY5COP1 * HY5 * COP1 * COP1 / (
                COP1 * COP1 + hrCOP1 * hrCOP1)  # Improves HY5 in control
    dGrowth = basalgrowth + (1 - basalgrowth) * PIF4 * PIF4 / (PIF4 * PIF4 + hrPIF4 * hrPIF4)
    dLeaf = basalgrowth + (1 - basalgrowth) * hrPIF4 * hrPIF4 / (
                PIF4 * PIF4 + hrPIF4 / 2 * hrPIF4 / 2)  # should be via HY5+
    # HY5 promotes cotyledon expansion https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004416
    # Thermom involves reduction in leaf blade size https://www.sciencedirect.com/science/article/pii/S1369526622000607
    dCarbon1 = 0
    return (dPhyBi, dPhyBa, dELF3a, dELF3i, dPIF4, dPIF4seq, dCOP1, dHY5, dGrowth, dLeaf, dCarbon1)


# Saturation in pif4 seq to recover TxD in hypocotyl

# root model - with saturating functions for regulation (think of meaning for T sensing)
def root(a, t, ABA, Temp, lighttime, light):
    hrABA = 0.5
    hrTemp = 0.5
    hrCOP1 = 20
    hrHY5 = 2
    basalgrowth = 0.1
    decayCOP1 = 0.1
    pCOP1 = 100 * decayCOP1
    decayCOP1nucleus = 0.5
    decayHY5 = 0.001
    pHY5 = 100 * decayHY5
    decayHY5COP1 = 0.01 * 10
    growthrate = 5
    COP1 = a[0]
    HY5 = a[1]
    Carbon = a[3]
    dCOP1 = pCOP1 - decayCOP1 * COP1 - decayCOP1nucleus * COP1 * (
                ABA * ABA / (ABA * ABA + hrABA * hrABA) + Temp * Temp / (Temp * Temp + hrTemp * hrTemp))
    dHY5 = pHY5 - decayHY5 * HY5 - decayHY5COP1 * HY5 * COP1 * COP1 / (COP1 * COP1 + hrCOP1 * hrCOP1)
    dGrowth = basalgrowth + (1 - basalgrowth) * HY5 * HY5 / (HY5 * HY5 + hrHY5 * hrHY5)  # updated this one
    dCarbon = 0
    return (dCOP1, dHY5, dGrowth, dCarbon)


# Respiration model
# simple explanation of Farquhar model: https://biocycle.atmos.colostate.edu/shiny/photosynthesis/
# Code from the site is in R, here I translated it to Python

def farquhar(Vmax=50, Jmax=100, APAR=500, ci=30, tempC=26):
    # Model inputs:
    # V.max = maximum rubisco-limited rate in micromoles per (m^2 sec) - Stomata should come here?
    # J.max = maximum light-limited rate in micromoles per (m^2 sec)
    # APAR = absorbed photosynthetically-active radiation in micromoles per (m^2 sec)
    # c.i = intercellular CO2 partial pressure in Pascals (roughly ppm/10)
    # temp.C = leaf temperature (Celsius)

    # Return value = net carbon assimilation (micromoles CO2 per m^2 of leaf area per second)

    # Some local parameters we need (adjusted for temperature according to Collatz 91)
    psfc = 101325  # surface air pressure (Pascals)
    Oi = 0.209 * psfc  # oxygen partial pressure in chloroplast

    # Adjust some parameters for temperature according to Collatz et al (1991) Table A1
    tau = 2600 * 0.57 ** ((tempC - 25) / 10)  # CO2/O2 specificity ratio for rubisco
    gamma = Oi / (2 * tau)  # CO2 compensation point (Pascals, Collatz A3)
    Kc = 30 * 2.1 ** ((tempC - 25) / 10)  # Michaelis constant for carboxylation (Pascals)
    Ko = 30000 * 1.2 ** ((tempC - 25) / 10)  # Michaelis constant for oxidation (Pascals)

    # Temp-adjusted maximum carboxylation rate
    cold = 10.
    hot = 40.
    slopecold = .25
    slopeheat = .4
    coldinhibition = 1 + math.exp(slopecold * (cold - tempC))
    heatinhibition = 1 + math.exp(slopeheat * (tempC - hot))
    Vm = Vmax * 2.1 ** ((tempC - 25) / 10) / (coldinhibition * heatinhibition)

    # Temp-adjusted leaf respiration
    Rd = 0.015 * Vm * 2.4 ** ((tempC - 25) / 10) / (1. + math.exp((1.3 * (tempC - 55))))

    # Solution of quadratic (Bonan 17.8)
    a = 0.7
    b = -(Jmax + 0.385 * APAR)
    c = 0.385 * Jmax * APAR
    J1 = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    J2 = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    J = min(J1, J2)

    # Rubisco-limited rate of photosynthesis
    wc = Vm * (ci - gamma) / (ci + Kc * (1 + Oi / Ko))  # Collatz A5 or Bonan 17.6

    # Light-limited rate of photosynthesis
    wj = J * (ci - gamma) / (4 * (ci + 2 * gamma))  # Bonan 17.7 or Collatz A2

    # Sink-limited rate of photosynthesis     # Collatz A7
    ws = Vm / 2

    # Net assimilation
    An = min(wc, wj, ws) - Rd

    return (An)


# Need to check references and parameters for At

def StomataOpen(Temp, ABA, light):  # Temp: 0 is 21, 1 is 27
    hrTemp = 0.5
    hrABA = 0.5
    basalOpening = 0.5
    Stomata = light * (basalOpening + (1 - basalOpening) * Temp * Temp / (Temp * Temp + hrTemp * hrTemp)) * (
                hrABA * hrABA / (ABA * ABA + hrABA * hrABA))
    return (Stomata)


# This is assimilated CO2 per unit of area in leaves. Then multiply this by leaf area variable and get total C budget
farquhar(Vmax=100 * StomataOpen(0, 1, 1), Jmax=100, APAR=500, ci=30, tempC=27)

ai = numpy.zeros(4)
ai[0] = StomataOpen(0, 0, 1)  # Optimal
ai[1] = StomataOpen(1, 0, 1)  # 28
ai[2] = StomataOpen(0, 1, 1)  # D
ai[3] = StomataOpen(1, 1, 1)  # 28xD

print(ai)


# Simulating all TxD conditions

# We assume growth only occurs at night. We assume partition is 0.5 shoot, 0.5 root
shootAllocation = 0.5
rootAllocation = 1 - shootAllocation

# here you save final "size" after n days
growth_saving_shoot = numpy.zeros(4)
growth_saving_leaf = numpy.zeros(4)
growth_saving_root = numpy.zeros(4)

days = 3
variablesModel = 9 + 1 + 1
variablesModel_r = 3 + 1
ABA = 1
Temp = 1
lighttime = 10
light = 0

tosave = -1

# here you save every light-dark period for n days, 4 TxD conditions
Stomata = numpy.zeros((days * 2, 4))
Carbon = numpy.zeros((days * 2, 4))

fig, mon = plt.subplots(7)
fig.suptitle('Shoot model')
colores = ("black", "red", "blue", "chartreuse")

fig, mon1 = plt.subplots(3)
fig.suptitle('Root model')

for ABA in range(0, 2, 1):
    for Temp in range(0, 2, 1):
        if (Temp == 0):
            tempfarq = 21
        else:
            tempfarq = 26
            # Shoot - IC for "pre-trained" system (bcs dark and light cycle)
        a = (7.26269075e+01, 2.73423925e+01, 1.79060814e-01, 0.00000000e+00,
             8.69945605e-03, 5.27419825e-01, 7.15477948e-01, 1.33320414e+00,
             0, 0, 0)
        b = (ABA, Temp, lighttime, light)  # inputs for ODE
        helper = 24 - lighttime  # start with dark
        t = numpy.linspace(0, helper, 1001 * helper)  # 11 points across 1-10
        plantState = [0] * variablesModel
        plantState = odeint(hypocotyl, a, t, b)
        a = plantState[1000 * helper][:]  # Saving final state for next light cycle

        # Root
        a_r = (0, 0, 0, 0)
        plantState_r = [0] * variablesModel_r
        plantState_r = odeint(root, a_r, t, b)
        a_r = plantState_r[1000 * helper][:]

        daystosave = -1
        tosave += 1

        for day in range(days):
            light = 1
            b = (ABA, Temp, lighttime, light)
            t = numpy.linspace(0, lighttime, 1001 * lighttime)  # 11 points across 1-10
            # Shoot
            out = [0] * variablesModel
            out = odeint(hypocotyl, a, t, b)
            plantState = numpy.concatenate((plantState, out))
            a = out[1000 * lighttime][:]
            # Root
            out_r = [0] * variablesModel_r
            out_r = odeint(root, a_r, t, b)
            plantState_r = numpy.concatenate((plantState_r, out_r))
            a_r = out_r[1000 * lighttime][:]

            daystosave += 1
            # p4help=out[1000*lighttime][4]
            Stomata[daystosave, tosave] = StomataOpen(Temp, ABA, light)
            Carbon[daystosave, tosave] = max(0, farquhar(Vmax=100 * StomataOpen(Temp, ABA, light), Jmax=100 * light,
                                                         APAR=500, ci=30, tempC=tempfarq) * out[1000 * lighttime][9], 0)

            light = 0
            b = (ABA, Temp, lighttime, light)
            helper = 24 - lighttime
            t = numpy.linspace(0, helper, 1001 * helper)  # 11 points across 1-10
            # Shoot
            out = [0] * variablesModel
            out = odeint(hypocotyl, a, t, b)
            plantState = numpy.concatenate((plantState, out))
            # here i am using leaf area to define carbon equation... very strange
            #            out[1000*helper][:]=shootAllocation*max(0,farquhar(Vmax=100*StomataOpen(Temp,ABA),Jmax=100*light,APAR=500,ci=30,tempC=tempfarq)*out[1000*helper][9])
            a = out[1000 * helper][:]
            # Root
            out_r = [0] * variablesModel_r
            out_r = odeint(root, a_r, t, b)
            plantState_r = numpy.concatenate((plantState_r, out_r))
            #            out_r[1000*helper][3]=rootAllocation*max(0,farquhar(Vmax=100*StomataOpen(Temp,ABA),Jmax=100*light,APAR=500,ci=30,tempC=tempfarq)*out[1000*helper][9])
            a_r = out_r[1000 * helper][:]

            daystosave += 1
            # p4help=out[1000*helper][4]
            Stomata[daystosave, tosave] = StomataOpen(Temp, ABA, light)
            Carbon[daystosave, tosave] = max(0, farquhar(Vmax=100 * StomataOpen(Temp, ABA, light), Jmax=100 * light,
                                                         APAR=500, ci=30, tempC=tempfarq) * out[1000 * helper][9])

        ai = numpy.shape(plantState[:, [8]])
        growth_saving_shoot[tosave] = plantState[ai[0] - 1, [8]]
        growth_saving_leaf[tosave] = plantState[ai[0] - 1, [9]]
        growth_saving_root[tosave] = plantState_r[ai[0] - 1, [2]]

        # Plot
        mon[0].plot(plantState[:, [1]], colores[tosave], lw=8 - tosave * 1.5, label="PhyB")
        mon[1].plot(plantState[:, [2]], colores[tosave], lw=8 - tosave * 1.5, label="ELF3")
        mon[2].plot(plantState[:, [6]], colores[tosave], lw=8 - tosave * 1.5, label="COP1")
        mon[3].plot(plantState[:, [4]], colores[tosave], lw=8 - tosave * 1.5, label="PIF4")
        mon[4].plot(plantState[:, [7]], colores[tosave], lw=8 - tosave * 1.5, label="HY5")
        mon[5].plot(plantState[:, [8]], colores[tosave], lw=8 - tosave * 1.5, label="Hypocotyl growth")
        mon[6].plot(plantState[:, [9]], colores[tosave], lw=8 - tosave * 1.5, label="Leaf growth")

        mon1[0].plot(plantState_r[:, [0]], colores[tosave], lw=8 - tosave * 1.5, label="COP1")
        mon1[1].plot(plantState_r[:, [1]], colores[tosave], lw=8 - tosave * 1.5, label="HY5")
        mon1[2].plot(plantState_r[:, [2]], colores[tosave], lw=8 - tosave * 1.5, label="Growth")

names = ("PhyB", "ELF3", "COP1", "PIF4", "HY5", "Hypocotyl", "Leaf")
namescounter = 0
for mo in mon.flat:
    mo.set(ylabel=names[namescounter])
    namescounter += 1
    mo.grid()
    mo.set_xticks(numpy.arange(0, days * 24000, 24000))
    for k in range(days + 2):  # number of days
        mo.add_patch(Rectangle((-100 + (k - 1) * 24000, -19), 14000, 9999, facecolor='0.9'))

names = ("COP1", "HY5", "Growth")
namescounter = 0
for mo in mon1.flat:
    mo.set(ylabel=names[namescounter])
    namescounter += 1
    mo.grid()
    mo.set_xticks(numpy.arange(0, days * 24000, 24000))
    for k in range(days + 2):  # number of days
        mo.add_patch(Rectangle((-100 + (k - 1) * 24000, -19), 14000, 9999, facecolor='0.9'))

plt.savefig("output.pdf", format="pdf", bbox_inches="tight")
plt.show();
growth_saving_shoot



####### plots
fig = plt.figure(figsize = (5, 3))
values=("Control","T","D","TxD")
plt.bar(values,growth_saving_leaf, color ='darkgreen',
        width = 0.4)
#plt.xlabel("Conditions")
plt.ylabel("Leaf area")
plt.title("Growth in 3 days")

plt.show();

fig = plt.figure(figsize = (5, 3))
values=("Control","T","D","TxD")
plt.bar(values,growth_saving_shoot, color ='green',
        width = 0.4)
#plt.xlabel("Conditions")
plt.ylabel("Hypocotyl length")
#plt.title("Growth in 3 days")

plt.show();

fig = plt.figure(figsize = (5, 3))
plt.bar(values,growth_saving_root, color ='maroon',
        width = 0.4)

plt.xlabel("Conditions")
plt.ylabel("Root length")
#plt.title("Growth in 3 days")

plt.show();

#Carbon

fig = plt.figure(figsize = (3, 3))
values=("Control","T","D","TxD")
plt.bar(values,Stomata[0][:], color ='grey', width = 0.4)
plt.ylabel("Stomata aperture")

plt.show();



