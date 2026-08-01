import copy
import rebound
import numpy as np
import pandas as pd
from tqdm import tqdm
from rebound import hash as h
from collections import defaultdict

# Functionalities written myself.
import general as gen

simelements = np.array(['P', 'm', 'e', 'inc', 'Omega', 'omega', 'M'])
cosimelements = np.array(['P', 'm', 'e', 'inc', 'pomega', 'M'])
saveorbs = np.array(['a', 'P', 'e', 'inc', 'omega', 'Omega', 'l', 'pomega'])

def checkOrbNaN(row, orbels):
    '''
    Checking the orbital element array which is used
    to input parameters into the rebound simulation!
    We remove the nan values. 

    df (DataFrame): the dataframe where we take inputs.
    orbels (np array): the array of orbital elements.
    '''

    existsm = row[row.index[np.isin(row.index, orbels)]].notna()
    if not np.all(existsm):
        invalidelements = existsm[~existsm].index
        print(f'Invalid values for {invalidelements}; removing.')
        orbels = np.setdiff1d(orbels, invalidelements)
    
    return orbels

def configSim(df, orbel, returnhash=True, verbose=True, units=('AU', 'yr', 'Msun')):
    '''
    Setting up our simulation and filtering for the correct
    rows within our dataframe (df) by sysname (str).
    Note: we use the period to initiate our simulation because
    it tends to be more precisely determined.

    df (DataFrame):
    orbel (np array): 
    returnhash (bool):
    verbose (bool):
    units (tuple): 
    '''

    sim = rebound.Simulation()
    sim.units = units
    hashmap = {}
    if verbose:
        print(f'Our units are {units} and G = {sim.G}.')

    df = df.copy() # So the original dataframe is not modified
    if len(orbel) != len(df):
        orbel = np.tile(orbel, (len(df), 1))
    df = df.sort_values(by='P')

    # Loop through to add the particles (planets) to the simulation.
    for i, (idx, row) in enumerate(df.iterrows()):        
        if len(sim.particles) < 1: # Adding the star
            
            if verbose: # The only parameter considered is mass.
                print(f'Added star with m = {row.mstar} Msun.')
            sim.add(m=row.mstar, hash='star')
            # hashmap['star'] = h('star')
        checkedarr = checkOrbNaN(row, orbel[i])
        orbdict = row[checkedarr].to_dict()
        
        if verbose: 
            printstr = f'Added {idx} with:'
            for key, val in orbdict.items():
                if val == 0: decpoint = 1
                else: decpoint = -1 * (np.floor(np.log10(np.abs(val)))) + 2
                printstr += f'\n {key} = {np.round(val, int(decpoint))}'
            print(printstr) 

        orbdict['hash'] = idx
        hashmap[idx] = h(idx)
        sim.add(**orbdict) # Adding planet based on orbital elements
  
    sim.move_to_com() 
    if returnhash: return sim, hashmap
        
    return sim

def getbounds(sim, buff=1.1):
    '''
    Using a and the eccentricity to get apoapsis! And then use a 
    buffer factor to get the limits of the plots.
    '''
    outeridx = np.argmax([porbit.a for porbit in sim.orbits()])
    maxdis = sim.orbits()[outeridx].a*(1 + sim.orbits()[outeridx].e)    
    return -buff*maxdis, buff*maxdis

def orbelementdf(sim, orbels, hashmap=None, indexstr=''):
    '''
    Can account for the case where we do not have particle names.
    Otherwise, we need a dictionary or list/array/tuple of names.
    Getting the orbital elements specified by orbels from the sim.

    sim: rebound simulation
    hashmap (list, np array, dict): names of particles or dictionary
    orbels (list or np array): the names of the particles. 
    indexstr (str): to distingish dataframes if necessary
    '''

    angmom = sim.angular_momentum()

    if hashmap is None:
        particles = sim.particles[1:]
        pnames = np.arange(1, len(particles)+1)
    else:
        if isinstance(hashmap, dict):
            pnames = list(hashmap.keys())
        elif isinstance(x, (list, tuple, np.ndarray)):
            pnames = hashmap
        particles = [sim.particles[name] for name in pnames]        

    for i, particle in enumerate(particles):
        porb = particle.orbit() # default is Jacobi!
        partdict = {ele: getattr(porb, ele) for ele in orbels}
        pdf = pd.DataFrame(partdict, index=[f'{pnames[i]}'])
        pdf['angmom'] = [(angmom.x, angmom.y, angmom.z)]

        if i == 0: df = pdf
        else: df = pd.concat([df, pdf])
        
    return df

def rotate2invariant(sim):
    '''
    Rotating so that the invariant plane is now the x-y plane
    by angular momentum as the z-axis.

    sim: rebound simulation    
    '''
    rotmatrix = rebound.Rotation.to_new_axes(newz=sim.angular_momentum())
    sim.rotate(rotmatrix)

def formatsave(savearr, savevars, pns):
    '''
    savearr[pnum, :, i] = np.array([getattr(particles[pn], e) for e in savevars])
    '''
    savedict = {}
    for i, pn in enumerate(pns):
        savedict[pn] = dict(zip(savevars, savearr[i]))
    return savedict

def rotatesimdf(sim, simhash, df, rotatedfelements=saveorbs, \
                boundd0=None, boundd1=None, plot=False):
    '''
    '''
    ogsim = sim.copy()    
    ogorbdf = orbelementdf(sim, rotatedfelements, simhash)
    rotate2invariant(sim)
    rotorbdf = orbelementdf(sim, rotatedfelements, simhash)
    
    if plot:
        fig, axs = gen.setupPlots((15, 10), row=2, col=3, logscale=0)
        if boundd0 == None:
            boundd0 = {'x': getbounds(ogsim), 'y': (-0.05, 0.05), 'z': getbounds(ogsim)}
        elif isinstance(boundd0, float):
            boundd0 = {'x': getbounds(ogsim), 'y': getbounds(ogsim), 'z': (-boundd0, boundd0)}
        axs[0, :] = orbitplot(ogsim, fig, axs[0, :], boundd=boundd0)
        if boundd1 == None:
            boundd1 = {'x': getbounds(sim), 'y': getbounds(sim), 'z': (-0.05, 0.05)}
        elif isinstance(boundd1, float):
            boundd1 = {'x': getbounds(sim), 'y': getbounds(sim), 'z': (-boundd1, boundd1)}
        axs[1, :] = orbitplot(sim, fig, axs[1, :], boundd=boundd1)
        fig.tight_layout()
        fig.show()

    compdf = (rotorbdf.round(3)).compare(ogorbdf.round(3), \
              align_axis='columns', result_names=('Rot', 'OG'))
    compcols = np.unique([pair[0] for pair in compdf.columns])
    rotated = df.copy()
    rotated.update(rotorbdf[compcols])

    return sim, rotated, compdf

def nbodyint_whfast(sys, savevars, pns, nsteps=10**4, tmax=1000, orbdex=(0,1), expectedstart=0, verbose=True):
    '''
    To get the list of the orbital element names:
    [name for name, dtype in sys.orbits()[0]._fields_]

    The varnames must match the rebound notation!
    '''

    skipsteps = int(np.ceil(tmax/(sys.dt*nsteps)))
    times = np.zeros((nsteps, ))
    savearr = np.zeros((len(pns), len(savevars), nsteps))

    if sys.t != expectedstart:
        print('Returning zero arrays because not at expected start-time.')
        return times, savearr

    for i in tqdm(range(nsteps)):
        particles = sys.particles
        times[i] = sys.t
        for pnum, pn in enumerate(pns):    
            savearr[pnum, :, i] = np.array([getattr(particles[pn], e) for e in savevars])

        try: sys.steps(skipsteps)
        except rebound.Encounter as error:
            print(f'{error} at {np.round(sys.t, 2)} years!')
            return times[:i], formatsave(savearr[:, :, :i], savevars, pns)
            
        if i < nsteps-1 and sys.t >= tmax:
            return times[:i+1], formatsave(savearr[:, :, :i+1], savevars, pns)
            
    return times, formatsave(savearr, savevars, pns)

def nbodyint_ias15(sys, savevars, pns, nsteps=10**4, tmax=1000, return_timestep=False, expectedstart=0):
    '''
    Integrating with ias15! We control the time because the 
    stepsize is flexible. 
    '''
    times = np.linspace(0, tmax, nsteps)
    savearr = np.zeros((len(pns), len(savevars), nsteps))
    timestep = np.zeros((nsteps,))

    if sys.t != expectedstart:
        print('Returning zero arrays because not at expected start-time.')
        return times, savearr

    for i in tqdm(range(nsteps)):
        try: # to get the first step!
            particles = sys.particles
            for pnum, pn in enumerate(pns):
                savearr[pnum, :, i] = np.array([getattr(particles[pn], e) for e in savevars])
            sys.integrate(times[i])
            timestep[i] = sys.dt

        except rebound.Encounter as error:
            print(f'{error} at {np.round(sys.t, 2)} years!')
            return times[:i], formatsave(savearr[:, :, :i], savevars, pns)

    if return_timestep:
        return times, timestep, formatsave(savearr[:, :, :i+1], savevars, pns)
    return times, formatsave(savearr[:, :, :i+1], savevars, pns)
    

## Plotting for the orbits!
def orbitplot(sim, fig, axs, boundd):
    '''
    Visualise the initial system layout, a modification
    from Rebound because of our ~90 degree inclination.
    Controls the bounds by a2 and e2 as this is the outer planet.

    sim (REBOUND simulation)
    axs (list or array): should be length = 3
    boundd (dict): in the form of: {'x': (-0.1, 0.1), \
                                    'y': (-0.1, 0.1), \
                                    'z': (-0.1, 0.1)}
    '''
    rebound.OrbitPlot(sim, fig=fig, ax=axs[0], projection="xz", color=True, figsize=(3, 3), periastron=True)
    rebound.OrbitPlot(sim, fig=fig, ax=axs[1], projection="xy", color=True, figsize=(4, 4), periastron=True)
    rebound.OrbitPlot(sim, fig=fig, ax=axs[2], projection="yz", color=True, figsize=(4, 4), periastron=True)
    axislabels = [['x', 'z'], ['x', 'y'], ['y', 'z']]
    for i in range(0, len(axs)):
        axs[i].tick_params(axis="both", length=0)
        axs[i].set_xlabel(f'{axislabels[i][0]} (au)')
        axs[i].set_ylabel(f'{axislabels[i][1]} (au)')
        axs[i].set_title(f'{axislabels[i][0]}-{axislabels[i][1]} Plane')
        axs[i].set_xlim(*boundd[axislabels[i][0]])
        axs[i].set_ylim(*boundd[axislabels[i][1]]) 

    return axs

class SysPars():
    
    def __init__(self, sysname):
        self.sysname = sysname
        self.ilocs = (0, 1)
        self.qndverbose = False
        self.simelements = re.simelements
        self.configverbose = False
        self.avgverbose = False
        self.simplot = False
        self.boundd0 = None
        self.boundd1 = None
        self.integrator = 'whfast'
        self.dt = 0.05/365.25 # days to years
        self.saveorbs = re.saveorbs
        self.mkcheck = False
        self.order = (1, 2)
        self.save = False
        self.savedir = lambda file: f'/home/sliu/memordia/data/systems/{file}'

    def syscharacter(self, sysdf):
        '''
        gjdf = gen.validate(table[table.sysname == 'GJ 876'].copy(), \
                      index=True, convertdict=gjconvert)
        '''
        diagvals = np.zeros((4, 2))
        diagdf = sysdf.copy()
        diagvals[0] = mm.quickndirty(diagdf, verbose=self.qndverbose, ilocs=self.ilocs)
    
        ## Initialising the simulation
        sim, shash = re.configSim(diagdf, self.simelements, verbose=self.configverbose)
        sim, rotated, rotcom = re.rotatesimdf(sim, shash, diagdf, plot=self.simplot, \
                                         boundd0=self.boundd0, boundd1=self.boundd1)
        diagvals[1] = mm.quickndirty(rotated.copy(), ilocs=self.ilocs)
        sim.integrator = self.integrator
        sim.dt = self.dt # days to years
    
        time5000, intdict5000 = re.nbodyint_whfast(sim.copy(), self.saveorbs, list(shash.keys()),\
                                                   nsteps=10**5, tmax=5000)
        time20, intdict20 = re.nbodyint_whfast(sim.copy(), self.saveorbs, list(shash.keys()), tmax=20)
        diagvals[2] = mm.quickndirty(gen.timeAverage(intdict5000, df=diagdf.copy(), \
                                    verbose=self.avgverbose), ilocs=self.ilocs)
        diagvals[-1] = mm.quickndirty(gen.timeAverage(intdict5000, df=rotated.copy(), \
                                      verbose=self.avgverbose), ilocs=self.ilocs)
        gen.timeAverage(intdict20, verbose=self.avgverbose)
        longdf = mm.mkdiagdf(time5000, intdict5000, sysdf, check=self.mkcheck, order=self.order)
        shortdf = mm.mkdiagdf(time20, intdict20, sysdf, check=self.mkcheck, order=self.order)
        if self.save:
            longdf.to_csv(self.savedir(f'{self.sysname}(5000).csv'))
            shortdf.to_csv(self.savedir(f'{self.sysname}(20).csv'))
    
        return diagvals, (time5000, intdict5000), (time20, intdict20)


formatd = {'marker':'.', 's':1, 'bcolour':'tab:blue', \
           'ccolour':'tab:orange', 'thetacolour':'tab:green'}
def evolutionplot(time, kep9b, kep9c, axs=None, verbose=False, formatdict=None):
    '''
    '''
    if axs is None:
        fig, axs = gen.setupPlots((10, 12.5), row=5, logscale=0)
        showfig = True
    else: showfig = False
    
    thetab = gen.thetacalc(kep9c['l'], kep9b['l'], kep9b['pomega'])
    thetac = gen.thetacalc(kep9c['l'], kep9b['l'], kep9c['pomega'])
    astar = astarb, astarc = np.mean(kep9b['a']), np.mean(kep9c['a'])
    Pstar = Pstarb, Pstarc = np.mean(kep9b['P']), np.mean(kep9c['P'])
    if verbose:
        print(f'The time-averaged semi-major axis for b is {np.round(astarb, 3)}, and {np.round(astarc, 3)} for c.')
        print(f'The time-averaged period for b is {np.round(Pstarb, 3)} and {np.round(Pstarc, 3)} for c.')

    if formatdict is None: formatdict = copy.deepcopy(formatd)
    colours = (formatdict.pop('bcolour'), formatdict.pop('ccolour'), \
               formatdict.pop('thetacolour'))
    
    for i, kep9plan in enumerate([kep9b, kep9c]):
        axs[0, i].scatter(time, kep9plan['a'], color=colours[i], **formatdict)
        axs[0, i].axhline(y=astar[i], ls='dashed', color='black', lw=2)
        axs[1, i].scatter(time, kep9plan['e'], color=colours[i], **formatdict)
        axs[2, i].scatter(time, kep9plan['inc']*180/np.pi, color=colours[i], **formatdict)
        pomega = kep9plan['pomega']%(2*np.pi)
        axs[3, i].scatter(time, pomega*180/np.pi, color=colours[i], **formatdict)

        if i == 0:
            axs[0, i].set_ylabel('a (AU)')
            axs[1, i].set_ylabel('eccentricity')
            axs[2, i].set_ylabel('i (degrees)')
            axs[3, i].set_ylabel('pomega (degrees)')

    axs[4, 0].scatter(time, thetab*180/np.pi, color=colours[-1], **formatdict)
    axs[4, 0].set_ylabel(r'$\theta$')
    axs[4, 1].scatter(time, thetac*180/np.pi, color=colours[-1], **formatdict)

    if not showfig: return axs
    fig.tight_layout()
    fig.show()