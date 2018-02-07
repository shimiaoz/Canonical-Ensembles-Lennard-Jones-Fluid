# Lennard-Jones Fluid
# In this version, uVT ensemble is added.

import numpy as np
import scipy.stats as stats
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import HelperFunc as LJFF
from time import time


#====================Initialization===================
t0 = time()

#=====CHANGE Configuration parameters=====
mass = 1.0; seed = 10
nPart = 100; density = 1.0; temp = 1.0; nStep = 100; nDim = 3
InitCoordGen = 'Lattice'          # 'Random' or 'Random_Cond' or 'Lattice'
SimAlgorithm = 'MD'               # 'MD' or 'MC'
Ensemble = 'NVT'                  # 'NVE' or 'NVT' or 'uVT'
MP = None                         # None or 'MPPoolPro' or 'MPPro'
rc = 4.0                          # Cutoff distance: None or number
initnPart = nPart
flag = 0                          # For NVT turn off

#========CHANGE Plotting Settings=========
Animation = False; show_save = 'Show'           # 'Show' or 'Save' animation, only for MD
PlotVelProfile = False; VelSampleFreq = 5000    # Velocity profile, only for MD
PlotTemp = True
PlotEnergy = True
PlotnPart = False
PlotPressure = True
#=====================================================

# Initial Coordinates of particles
if InitCoordGen == 'Random':
  (coord, L) = LJFF.Init_CoordPart_Rand(nPart, density, nDim, seed)
elif InitCoordGen == 'Random_Cond':
  (coord, L) = LJFF.Init_CoordPart_Rand_Cond(nPart, density, nDim, seed, disttouch=1.0)
elif InitCoordGen == 'Lattice':
  (coord, L) = LJFF.Init_CoordPart_Lattice(nPart, density, nDim)
print('%s initial coordinates in %d dimensions for %d atoms generated' %(InitCoordGen, nDim, nPart))

#=============== MD Simulation Settings and Runnings===============
if SimAlgorithm == 'MD':
  print('MD simulation running...')

  # MD Simulation Settings
  dt = 5e-3
  CollisionFreq = 0.10                  # Frequency of collisions for Andersen thermostat
  CollisionCount = 0
  CollisionCount_list = [0]

  # Initial Velocities of particles
  velMean = 0                           # Mean of velocity
  velSTD = math.sqrt(temp/mass)         # Standard deviation of velocity
  vel = LJFF.Init_Vel(velMean, velSTD, nPart, nDim, seed)   # Random Gaussian distribution

  # Set Initial Momentum to 0
  CoMvel = 1.0*np.sum(vel, axis=0)/nPart            # Center of mass velocity
  for i in range(nDim):
    vel[:,i] = vel[:,i] - CoMvel[i]

  # Set initial Kinetic Energy to 3/2*N*kbT
  MSvel = 1.0*np.sum(vel*vel)/nPart             # Mean squared velocity
  SFvel = math.sqrt(nDim*temp/MSvel)            # Scaling factor of velocity
  for i in range(nDim):
    vel[:,i] = vel[:,i]*SFvel
  init_vel = np.copy(vel)

  # Calculate Initial Force/Potential Energy with LJ 12/6 Potential
  [force, PE, pPart2] = LJFF.LJ_Force_PE(coord, L, nDim, rc, MP)
  init_PE = PE
  init_KE = 0.5*mass*np.sum(vel*vel)
  init_T = np.sum(vel*vel)/(nDim*nPart)
  init_p = density*temp + pPart2/(nDim*nPart/density)

  #=========Miscellaneous==========
  #coord_step = [np.zeros((nDim, nStep), dtype=float) for i in range(nPart)]
  PE_list = [init_PE]           # Record potential energy
  KE_list = [init_KE]           # Record kinetic energy
  TE_list = [init_PE+init_KE]       # Record total energy
  Temp_list = [init_T]          # Record temperature
  press_list = [init_p]         # Record pressure
  velSamples = [init_vel]       # Record sampled velocities
  PBC_FailStep = []         # Record steps when PBC fails

  # Sample two particles for verification purpose
  Part01_DistPE = [[], []]
  dr = coord[0] - coord[1]
  dr2 = np.dot(dr,dr)
  dPart01 = dr2**0.5
  Part01_DistPE[0].append(dPart01)
  PE01= 4.0*((1.0/dr2)**6 - (1.0/dr2)**3)
  Part01_DistPE[1].append(PE01)

  #t2 = time()
  #print 'Time elasped for MD settings and initialization: {:.3f}'.format(t2-t1)

  #=========MD Simulation Running==========
  np.random.seed(seed)
  MDtime = 0
  for step in range(1, nStep+1):
    #tp = time()
    MDtime += dt

    # Velocity Verlet Algorithm
    coord = coord + vel*dt + 0.5*(dt**2)*force;

    for i in range(nPart):          # Update coordinates
      coord[i] = LJFF.PBC(coord[i], L, nDim)    # Apply PBC

      # PBC Safety Check
      for j in range(nDim):
        if coord[i][j] > L/2.0 or coord[i][j] < -L/2.0:
          print("PBC fails at step %d" %step)
          PBC_FailStep.append(step)
          print(coord)

      # Record coordinates for animation
      if Animation:
        coord_step[i][0][step-1] = coord[i][0]      # x coordinate
        coord_step[i][1][step-1] = coord[i][1]      # y coordinate
        if nDim == 3:
          coord_step[i][2][step-1] = coord[i][2]    # z coordinate

    #tp1 = time()
    #print 'Time elasped for coordinate update: {:.3f}'.format(tp1-tp)

    # Update velocities
    vel = vel + 0.5*dt*force;       # v(t+dt) = v(t) + 0.5*(a(t)+a(t+dt))*dt
    [force, PE, pPart2] = LJFF.LJ_Force_PE(coord, L, nDim, rc, MP)
    vel = vel + 0.5*dt*force;

    #tp2 = time()
    #print 'Time elasped for velocity update: {:.3f}'.format(tp2-tp1)

    # =========================
    # Sample two particles and record distance and energy
    dr = coord[0] - coord[1]
    dr = LJFF.PBC(dr, L, nDim)
    dr2 = np.dot(dr,dr)
    dPart01 = dr2**0.5
    Part01_DistPE[0].append(dPart01)
    PE01= 4.0*((1.0/dr2)**6 - (1.0/dr2)**3)
    Part01_DistPE[1].append(PE01)
    # =========================

    InstantTemp = np.sum(vel*vel)/(nDim*nPart)
    Temp_list.append(InstantTemp)
    KE = 0.5*mass*np.sum(vel*vel)
    PE_list.append(PE); KE_list.append(KE)
    TE_list.append(PE + KE)
    InstantPress = density*InstantTemp + pPart2/(nDim*nPart/density)
    press_list.append(InstantPress)

    # Ensemble
    if flag == 0 and step >= 80000 and InstantTemp-temp < -0.01 and InstantTemp-temp > -0.02:
      flag = 1
      print('NVT Turned off at %d' %step)
    if Ensemble == 'NVT' and flag == 0:     # Andersen thermostat
      for i in range(nPart):
        if np.random.random() < CollisionFreq*dt:
          vel[i] = np.random.normal(velMean, velSTD, nDim)
          CollisionCount += 1
      CollisionCount_list.append(CollisionCount)

    if step % VelSampleFreq == 0:
      velSamples.append(np.copy(vel))

    #tp3 = time()
    #print 'Time elasped for the rest: {:.3f}'.format(tp3-tp2)
    print('{:<6} {:<8.4f} {:<8.4f} {:>15.6f} {:>15.6f} {:>15.6f} {:>10.2f}'.format(step,\
          InstantTemp, InstantPress, KE, PE, KE+PE, L**3))

  print("Percent of energy change: %f%%" %(100*(PE+KE-init_PE-init_KE)/(init_PE+init_KE)))
  MDt_list = [i*dt for i in range(nStep+1)]


#=============== MC Simulation Settings and Runnings===============
if SimAlgorithm == 'MC':
  print('MC simulation running...')
  vol = L**nDim
  AddRemProb = 0.5
  deBroWave3 = 0.185**3/(2*math.pi*temp)**1.5       # Cubic of de Broglie wavelength
  densityId = 1.0                   # Density of ideal gass
  ChemPotId = temp*math.log(deBroWave3*densityId)   # Chemical potential of ideal gas
  zz = math.exp(ChemPotId/temp)/deBroWave3

  [force, PE, pPart2] = LJFF.LJ_Force_PE(coord, L, nDim, rc, MP)
  InstantPress = density*temp + pPart2/(nDim*vol)
  PE_list = [PE]
  nPart_list = [nPart]
  press_list = [InstantPress]

  #CyclePartMoved = nPart           # Cycle of MC moves in one step
  MaxMovedDist = 0.5*L/(nPart**(1.0/nDim)+1)

  np.random.seed(seed)
  for step in range(1, nStep+1):

    if Ensemble == 'uVT':
      # Add or Remove particles
      if np.random.random() < AddRemProb:
        if np.random.random() < 0.5:        # Select a particle to be removed
          PartID = np.random.randint(0, nPart)
          TrialPartCoord = coord[PartID]
          (PE_part, pPart2_part) = LJFF.LJ_Energy_Part(TrialPartCoord, coord, L, nDim, rc)
          Prob_Rem = nPart*math.exp(PE_part/temp)/(vol*zz)
          #print('Removing Particel Prob: %f' %Prob_Rem)
          if np.random.random() < Prob_Rem: # Removal accepted
            PE -= PE_part
            nPart -= 1
            InstantPress -= temp/vol + pPart2_part/(nDim*vol)
            coord = np.delete(coord, PartID, 0)

        else:                   # Add a particle
          TrialPartCoord = [np.random.uniform(-L/2.0, L/2.0) for tpc in range(nDim)] 
          (PE_part, pPart2_part) = LJFF.LJ_Energy_Part(TrialPartCoord, coord, L, nDim, rc)
          Prob_Add = vol*zz*math.exp(-PE_part/temp)/(nPart+1)
          #print('Adding Particel Prob: %f' %Prob_Add)
          if np.random.random() < Prob_Add: # Addition accepted
            PE += PE_part
            nPart += 1
            InstantPress += temp/vol + pPart2_part/(nDim*vol)
            coord = np.append(coord, np.array([TrialPartCoord]), axis=0)            
      nPart_list.append(nPart)

    # Random Displacements of Randomly Selected Particles
    CyclePartMoved = nPart
    for i in range(CyclePartMoved):
      PartID = np.random.randint(0, nPart)  # Select a random particle

      TrialPartCoord = coord[PartID] + MaxMovedDist*np.array([np.random.random()-0.5 for p in range(nDim)])
      TrialPartCoord = LJFF.PBC(TrialPartCoord, L, nDim)

      (dE, pPart2_diff) = LJFF.LJ_Energy_Diff(coord, PartID, TrialPartCoord, L, nDim, rc)
      if np.random.random() < math.exp(-dE/temp):
        coord[PartID] = TrialPartCoord      # MC move accepted
        PE += dE                # Update energy
        InstantPress += pPart2_diff/(nDim*vol)
    PE_list.append(PE)
    press_list.append(InstantPress)
    [force1, PE1, pPart21] = LJFF.LJ_Force_PE(coord, L, nDim, rc, MP)
    InstantPress1 = nPart*temp/vol + pPart21/(nDim*vol)

    print('{:<6d} {:<5d} {:>8.4f} {:>8.4f} {:>15.6f} {:>15.6f} {:>10.2f}'.format(step, nPart, InstantPress, InstantPress1, PE, PE1, vol))
  print("Percent of energy change: %f%%" %(100*(PE_list[-1]-PE_list[0])/PE_list[0]))


#==========nPart/Energy/Temp/Velocity Plottings==========
# Number of particles plotting
if PlotnPart:
  fig, axN = plt.subplots()
  axN.set_title('Number of Particles Profile', fontweight='bold', size=28)
  axN.set_xlabel('MC Step', fontsize = 24)
  axN.set_ylabel('Number of Particles', fontsize = 24)
  #axN.set_ylim([0, max(nPart_list)])
  axN.plot(range(nStep+1), nPart_list, 'k-')
  plt.savefig('%sTemp%.2f_s%d_d%.2f_dId%.2f.png' %(SimAlgorithm, temp, nStep, density, densityId))
  plt.show()

  with open('%s_uVT_nPart_T%.2f_s%d_d%.2f_dId%.2f.text' %(SimAlgorithm, temp, nStep, density, densityId), 'w') as f:
    k = 0
    for i in nPart_list:
      f.write('{:<6d} {:>d}\n'.format(k, i))
      k += 1

# Temperature plotting
if PlotTemp:
  fig, axT = plt.subplots()
  axT.set_title('Temperature Profile', fontweight='bold', size=28)
  axT.set_xlabel('Time', fontsize = 24)
  axT.set_ylabel('Temperature', fontsize = 24)
  axT.plot(MDt_list, Temp_list, 'k-')
  axT.plot(MDt_list, temp*np.ones(len(Temp_list)), 'r--')
  plt.savefig('%sTemp_cf%.2f_n%d_s%d_d%.2f.png' %(SimAlgorithm, CollisionFreq, nPart, nStep, density))
  plt.show()

  with open('NVT_temp_cf%.2f_n%d_s%d_d%.2f.text' %(CollisionFreq, nPart, nStep, density), 'w') as f:
    k = 0
    for i, j in zip(Temp_list, CollisionCount_list):
      f.write('{:<6d} {:>f} {:>6d}\n'.format(k, i, j))
      k += 1

# Pressure plotting
if PlotPressure:
  fig, axP = plt.subplots()
  axP.set_title('Pressure Profile', fontweight='bold', size=28)
  axP.set_ylabel('Pressure', fontsize = 24)
  if SimAlgorithm == 'MD':
    axP.set_xlabel('Time', fontsize = 24)
    axP.plot(MDt_list, press_list, 'k-')
  else:
    axP.set_xlabel('MC Step', fontsize = 24)
    axP.plot(range(nStep+1), press_list, 'k-')
  plt.savefig('%sPressure_n%d_T%.2f_s%d_d%.2f.png' %(SimAlgorithm, initnPart, temp, nStep, density))
  plt.show()

  with open('%sPressure_n%d_T%.2f_s%d_d%.2f.text' %(SimAlgorithm, nPart, temp, nStep, density), 'w') as f:
    k = 0
    for i in press_list:
      f.write('{:<6d} {:>f}\n'.format(k, i))
      k += 1

# Energy plotting
if PlotEnergy:
  # Per particle energy
  PE_list = [item/nPart for item in PE_list]
  if SimAlgorithm == 'MD':
    KE_list = [item/nPart for item in KE_list]
    TE_list = [item/nPart for item in TE_list]

  fig, axE = plt.subplots()
  axE.set_title('Energy Profile', fontweight='bold', size=28)
  axE.set_ylabel('Energy Per Particle', fontsize=24)

  if SimAlgorithm == 'MD':
    axE.set_xlabel('Time', fontsize=24)
    axE.plot(MDt_list, PE_list, 'k:', label='Potential Energy')
    axE.plot(MDt_list, KE_list, 'k--', label='Kinetic Energy')
    axE.plot(MDt_list, TE_list, 'k', label='Total Energy')
  elif SimAlgorithm == 'MC':
    axE.set_xlabel('MC Step', fontsize=24)
    axE.plot(range(nStep+1), PE_list, 'k:', label='Potential Energy')

  legend = axE.legend(loc='best', shadow=None, frameon=None)
  plt.savefig('%sEnergy_n%d_s%d_d%.2f_T%.2f.png' %(SimAlgorithm, nPart, nStep, density, temp))
  plt.show()

# Velocity profiles
if PlotVelProfile:
  VelDim = 0            # 0 for x, 1 for y, 2 for z, 3 for overall
  def v4hist(vsam, VelDim):
    if VelDim == 3:
      Vel2 = np.sum(vsam*vsam, axis=1)
      v4hist = Vel2**0.5
    else:
      v4hist = vsam[:,VelDim]
    return v4hist

  lowv = 0 if VelDim == 3 else -5*temp
  for i, vsam in enumerate(velSamples):
    v = v4hist(vsam, VelDim)

    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(v, bins=50, range=(lowv, 5*temp), normed=True, color='lightgray')

    x = np.linspace(lowv, 5*temp, 100)
    if VelDim == 3:
      params = stats.maxwell.fit(v, loc=0, scale=1)
      ax.plot(x, stats.maxwell.pdf(x, *params), 'k--', lw=3)
    else:
      ax.plot(x, mlab.normpdf(x, np.mean(v), np.std(v)), 'k--', lw=3)
    ax.set_title('Velocity Profile', fontweight='bold', size=28)
    ax.set_xlabel('Velocity', fontsize=24)
    ax.set_ylabel('Probability Density', fontsize=24)
    plt.savefig('VelDstrib_%d_n%d_step%d_T%.2f.png' %(VelDim, nPart, i*VelSampleFreq, temp))
    plt.show()

#==========Animation==========
if Animation:
  if nDim == 3:
    def animation_coord(step, coord_step, part_dots):
      for dotPart, coord in zip(part_dots, coord_step):
        dotPart.set_data(coord[0:2, step-1:step])
        dotPart.set_3d_properties(coord[2, step-1:step])
      return part_dots

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    part_dots = [ax.plot(p[0, 0:1], p[1, 0:1], p[2, 0:1], 'o')[0] for p in coord_step]

    ax.set_xlim3d([-L/2.0, L/2.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([-L/2.0, L/2.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-L/2.0, L/2.0])
    ax.set_zlabel('Z')
    ax.set_title('LJ Fluid', fontsize=28)

    # Creating the Animation object
    LJ_ani = animation.FuncAnimation(fig, animation_coord, nStep, init_func=None, \
                            fargs=(coord_step, part_dots), interval=1, blit=False)

  elif nDim == 2:
    def init():
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_title('LJ Fluid', fontsize=28)
      for dotPart in part_dots:
        dotPart.set_data([],[])
      return part_dots

    def animation_coord(step, coord_step, part_dots):
      for coord, dotPart in zip(coord_step, part_dots):
        dotPart.set_data(coord[0:2, step-1:step])
      return part_dots

    fig = plt.figure()
    ax = plt.axes(xlim=([-L/2.0, L/2.0]), ylim=([-L/2.0, L/2.0]))
    part_dots = [ax.plot([], [], 'o')[0] for i in range(nPart)]

    # Creating the Animation object
    LJ_ani = animation.FuncAnimation(fig, animation_coord, nStep, init_func=init, \
                            fargs=(coord_step, part_dots), interval=1, blit=False)

  if show_save == 'Save':
    #LJ_ani.save('./LJ_Fluid_Animation_%dD_n%d.gif' %(nDim, nPart), writer='imagemagick', fps=10)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=None)
    LJ_ani.save('./LJ_Fluid_Animation_%dD.mp4' %nDim, writer=writer)
  elif show_save == 'Show':
    try: plt.show()
    except AttributeError: pass


t_final = time()
runtime = t_final - t0
mm, ss = divmod(runtime, 60)
hh, mm = divmod(mm, 60)
print('Total Simulation Time: {:d}h {:d}min {:.2f}s'.format(int(hh), int(mm), ss))

"""
#=========================================
fig, ax1 = plt.subplots()

ax1.plot(MDt_list, Part01_DistPE[0], 'b-')
ax1.set_xlabel('Time', fontsize=24)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Distance', color='b', fontsize=24)
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(MDt_list, Part01_DistPE[1], 'r.')
ax2.set_ylabel('PE', color='r', fontsize=24)
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.savefig('Step_distPE.png')
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel('Distance', fontsize=24)
ax.set_ylabel('PE', fontsize=24)
ax.set_xlim([min(Part01_DistPE[0])-0.01, max(Part01_DistPE[0])+0.01])
ax.set_ylim([min(Part01_DistPE[1])-0.1, max(Part01_DistPE[1])+0.1])
ax.plot(Part01_DistPE[0], Part01_DistPE[1], 'k-', lw=2)
x = np.linspace(min(Part01_DistPE[0])-0.01, max(Part01_DistPE[0])+0.01, 50)
ax.plot(x, np.zeros(50), 'r--')
plt.savefig('dist_vs_PE.png')
plt.show()
"""