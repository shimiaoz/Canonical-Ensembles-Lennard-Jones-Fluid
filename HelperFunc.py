import numpy as np
import math
from time import time
from multiprocessing import Process, Pool, cpu_count  
from functools import partial

# Set Initial Coordinates of Particles
def Init_CoordPart_Rand(nPart, density, nDim, seed):  # Pseudo-random
  LBox = (1.0*nPart/density)**(1.0/nDim)
  coord = np.zeros((nPart, nDim), dtype=float, order='C')
  for i in range(0, nPart):
    np.random.seed(seed)
    coord[i][0] = np.random.uniform(-LBox/2.0, LBox/2.0)
    coord[i][1] = np.random.uniform(-LBox/2.0, LBox/2.0)
    if nDim == 3:
      coord[i][2] = np.random.uniform(-LBox/2.0, LBox/2.0)
    seed += 1
  return (coord, LBox)

def Init_CoordPart_Rand_Cond(nPart, density, nDim, seed, disttouch=1.0):  # Pseudo-random
  LBox = (1.0*nPart/density)**(1.0/nDim)
  coord = np.zeros((nPart, nDim), dtype=float, order='C')
  np.random.seed(seed)
  for i in range(nDim):
    coord[0][i] = np.random.uniform(-LBox/2.0, LBox/2.0)    # First atom
  for i in range(1, nPart):
    seed += 1
    coord[i] = Insert_Part(coord[:i], LBox, nDim, seed, disttouch)
  return (coord, LBox)

def Insert_Part(coord, LBox, nDim, seed, disttouch):
  np.random.seed(seed)
  r2 = disttouch**2
  while True:
    trial_coordpart = np.zeros(nDim)
    for i in range(nDim):
      trial_coordpart[i] = np.random.uniform(-LBox/2.0, LBox/2.0)
    for p in coord:
      dr = trial_coordpart - p
      dr = PBC(dr, LBox, nDim)
      dr2 = np.dot(dr, dr)
      if dr2 < r2:
        break
    else:
      new_coord = trial_coordpart
      break
  return new_coord

def Init_CoordPart_Lattice(nPart, density, nDim): # Lattice
  LBox = (1.0*nPart/density)**(1.0/nDim)
  coord = np.zeros((nPart, nDim), dtype=float, order='C')
  nCube = math.ceil(nPart**(1.0/nDim))

  intervalPosi = np.zeros(nDim)
  for i in range(nPart):
    coord[i] = (intervalPosi + np.ones(nDim)*(0.5-nCube/2.0))*(1.0*LBox/nCube)
    intervalPosi[0] += 1
    if intervalPosi[0] == nCube:
      intervalPosi[0] = 0
      intervalPosi[1] += 1
      if nDim == 3:
        if intervalPosi[1] == nCube:
          intervalPosi[1] = 0
          intervalPosi[2] += 1
  return (coord, LBox)

# Set Initial Velocities of Particles
def Init_Vel(velMean, velSTD, nPart, nDim, seed):
  np.random.seed(seed)
  vel = np.random.normal(velMean, velSTD, size=(nPart, nDim))
  return vel

"""
def Init_Vel(velMean, velSTD, nPart, nDim, seed):
  np.random.seed(seed)
  vel_x = np.random.normal(velMean, velSTD, size=nPart)
  vel_y = np.random.normal(velMean, velSTD, size=nPart)
  if nDim == 2:
    vel = np.vstack((vel_x,vel_y))
  elif nDim == 3:
    vel_z = np.random.normal(velMean, velSTD, size=nPart)
    vel = np.vstack((vel_x,vel_y,vel_z))
  vel = np.transpose(vel)
  return vel
"""

# LJ Force and Energy formula
def LJ_FE_Form(dr, rc):         # Only for single distance vector
  # LJ(r) = 4 * epsilon * [(sigma/r)**12 - (sigma/r)**6]
  # sigma = 1, epsilon = 1
  # LJ(r) = 4 * [(1/r)**12 - (1/r)**6]
  # Fx(r) = 4 * (x/r) * [12*(1/r)**13 - 6*(1/r)**7]
  #       = 48 * x * (1/r)**8 * [(1/r)**6 - 0.5]
  # Similar for y and z components

  dr2 = np.dot(dr,dr)

  if rc == None:
    PE_dr = (1.0/dr2)**6 - (1.0/dr2)**3
    force_r = (1.0/dr2)**4 * ((1.0/dr2)**3 - 0.5)
  else:
    if dr2 <= rc**2:
      PE_dr = (1.0/dr2)**6 - (1.0/dr2)**3 - ((1.0/rc**2)**6 - (1.0/rc**2)**3)
      force_r = (1.0/dr2)**4 * ((1.0/dr2)**3 - 0.5)
    else:
      PE_dr = 0
      force_r = 0

  PE_dr = 4.0*PE_dr
  force_r = 48*force_r

  return (force_r, PE_dr)

# Calculate Force and Energy
def LJ_Force_PE_partial(ID_list, Coord, LBox, nDim, rc = None):
  force = np.zeros(np.shape(Coord), dtype=float, order='C')
  PE = 0
  pPart2 = 0
  nPart = len(Coord)
  
  # Loop over all pairs of particles
  for i in ID_list:
    for j in range(i+1, nPart):
      dr = Coord[i] - Coord[j]
      dr = PBC(dr, LBox, nDim)
      dr2 = np.dot(dr,dr)

      (force_r, PE_dr) = LJ_FE_Form(dr, rc)

      PE += PE_dr
      force[i] = force[i] + dr*force_r
      force[j] = force[j] - dr*force_r
      pPart2 += force_r*dr2

  return (force, PE, pPart2)

# Periodic Boundary Condition
def PBC(coord_dist, LBox, nDim):   # coord_dist can be coordinate or distance vector
  for i in range(nDim):
    coord_dist[i] -= LBox*round(1.0*coord_dist[i]/LBox)
  return coord_dist

# Calculate LJ Energy Difference After MC Move
def LJ_Energy_Diff(Coord, PartID, TrialPartCoord, LBox, nDim, rc = None):
  nPart = len(Coord)
  dE = 0
  pPart2_diff = 0

  for i in range(nPart):
    if i == PartID:   # Particle ID ranges from 0 to nPart-1
      continue      # Avoid calculating self interaction
    
    drOld = Coord[i] - Coord[PartID]
    drNew = Coord[i] - TrialPartCoord

    drOld = PBC(drOld, LBox, nDim)
    drOld2 = np.dot(drOld, drOld)
    drNew = PBC(drNew, LBox, nDim)
    drNew2 = np.dot(drNew, drNew)

    (force_r_Old, PE_drOld) = LJ_FE_Form(drOld, rc)
    (force_r_New, PE_drNew) = LJ_FE_Form(drNew, rc)

    pPart2_diff += force_r_New*drNew2 - force_r_Old*drOld2
    dE += PE_drNew - PE_drOld

  return (dE, pPart2_diff)

def LJ_Energy_Part(TrialPartCoord, Coord, LBox, nDim, rc = None):
  nPart = len(Coord)
  PE_part = 0
  pPart2_part = 0

  for i in range(nPart):
    if  np.array_equal(Coord[i], TrialPartCoord): # Particle ID ranges from 0 to nPart-1
      continue            # Avoid calculating self interaction
    
    dr = Coord[i] - TrialPartCoord
    dr = PBC(dr, LBox, nDim)
    dr2 = np.dot(dr, dr)

    (force_r, PE_dr) = LJ_FE_Form(dr, rc)

    pPart2_part += force_r*dr2
    PE_part += PE_dr

  return (PE_part, pPart2_part)

# Speed-up Calculation of LJ Force and Energy Using Multiprocessing
class MP_Force_Energy(Process):
  def __init__(self, func, args, name=None):
    Process.__init__(self)
    self.func = func
    self.args = args
    self.name = name

  def getresults(self):
    return self.res

  def run(self):
    #t0 = time()
    self.res = self.func(*self.args)
    #print '{} takes {:.5f} seconds; energy: {}'.format(self.name, time()-t0, self.res[1])

def LJ_Force_PE(Coord, LBox, nDim, rc = None, MultiPro=None):
  nPart = len(Coord)
  if MultiPro == None:
    (force, energy, pPart2) = LJ_Force_PE_partial(range(0, nPart-1), Coord, LBox, nDim, rc)
    return (force, energy, pPart2)

  elif MultiPro == 'MPPoolPro':
    #t0 = time()
    nProcess = cpu_count()
    pool = Pool(processes=4 if nProcess<4 else nProcess)
    res = pool.map(partial(LJ_Force_PE_partial, Coord=Coord, LBox=LBox, nDim=nDim, rc=rc), \
                   [[j] for j in range(nPart-1)])
    #print 'Time for Pool {:d}: {:.5f}'.format(i, time()-t0)
    pool.close()
    pool.join()
    force = res[0][0]; energy = res[0][1]; pPart2 = res[0][2]
    for FEpair in res[1:]:
      force = np.add(force, FEpair[0])
      energy += FEpair[1]
      pPart2 += FEpair[2]
    return (force, energy, pPart2)

  elif MultiPro == 'MPPro':   # ONLY for large nPart, e.g. >100
    ID_arr = []
    id_init_end = np.arange(0, 1.0, 0.04)
    for i in range(len(id_init_end)-1):
      idsublist = range(int(id_init_end[i]*nPart), int(id_init_end[i+1]*nPart))
      ID_arr.append(idsublist)
    ID_arr.append(range(int(id_init_end[-1]*nPart), nPart-1))

    #print 'Starting Multiprocessing Calculation...'
    nProcess = len(ID_arr)
    Process_list=[]
    for i in range(nProcess):
      t = MP_Force_Energy(LJ_Force_PE_partial, (ID_arr[i], Coord, LBox, nDim, rc), \
                          name='Part {:d}'.format(i))
      Process_list.append(t)

    for i in range(nProcess):
      Process_list[i].start()

    for i in range(nProcess):
      Process_list[i].join()
    return None

"""
# Distance of a particle pair with PBC applied
def Distance_PBC_3D(dr, LBox):
  for i in range(3):
    if dr[i] > LBox/2.0:
      dr[i] -= LBox
    elif dr[i] < -LBox/2.0:
      dr[i] += LBox
    # Safety Check to be added
  return dr

def Distance_PBC_3D(dr, LBox):
  for i in range(3):
    dr[i] -= LBox*round(1.0*dr[i]/LBox)
  return dr
"""