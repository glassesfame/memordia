import sys
from pathlib import Path
funcdir = str((Path(__file__).resolve()).parent)

import cmath
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from collections import defaultdict

sys.path.insert(0, funcdir)
import general as gen

## CONSTANTS
G = (2*np.pi)**2
deltastar = (27/32)**(1/3)
two2one = pd.Series(index=['k', 'f1', 'f2'], data=[2, -1.19, 0.428])

# Calculating  for n and a
ncalc_period = lambda period: 2*np.pi/period
ncalc_a = lambda a, mass, mstar: np.sqrt(G*(mass + mstar)/(a)**3)
modkep3rd = lambda P, masses: (G/(2*np.pi)**2 * P**2 *np.sum(masses))**(1/3)
# The Hamiltonian and Psifac
Psifac = lambda Psi, psi: Psi * np.cos(psi)
H = lambda Psi, delta, psi: -(Psi - delta)**2 - np.sqrt(2*Psi)*np.cos(psi)

def delaunayelements(bodypars):
    '''
    Calculating the Delaunay elements as well as the combined masses and
    the relative masses! 
    '''
    M = bodypars.m + bodypars.mstar
    mu = (bodypars.m * bodypars.mstar)/M   
    Lamda = mu * np.sqrt(G*M*bodypars.a)
    LamdaStar = mu * np.sqrt(G*M*bodypars.astar)
    Gamma = Lamda * (1 - np.sqrt(1 - bodypars.e**2))
    gamma = -1 * bodypars.pomega

    delaun = pd.Series(data=[M, mu, Lamda, LamdaStar, Gamma, bodypars.l, gamma], \
            index=['M', 'mu', 'Lamda', 'LamdaStar', 'Gamma', 'lamda', 'gamma'])

    return delaun
    
def resonantelements(k, delaun1, delaun2):
    '''
    Transformation from the planetary orbital elements,
    i.e. semi-major axis (we need period as well), eccentricity, etc,
    into the resonant canonical variables (without longitude tho)

    K1, K2, sigma1, sigma2
    '''
    K1 = delaun1.Lamda + (k-1)*(delaun1.Gamma + delaun2.Gamma)
    K2 = delaun2.Lamda - k*(delaun1.Gamma + delaun2.Gamma)
    sigmatemp = k*delaun2.lamda - (k-1)*delaun1.lamda
    sigma1 = sigmatemp + delaun1.gamma
    sigma2 = sigmatemp + delaun2.gamma
    
    resvars = pd.Series(index=['K1', 'K2', 'Gamma1', 'Gamma2', 'sigma1', 'sigma2'], \
              data=[K1, K2, delaun1.Gamma, delaun2.Gamma, sigma1, sigma2])
    
    return resvars

def kepHfactors(k, n1, n2, Lamda1, Lamda2, K1, K2):
    '''
    The factors for H_K, where we approximate the small
    semi-major axis changes with Lamda1, Lamda2 and nstar.
    '''
    N1 = n1/Lamda1
    N2 = n2/Lamda2
    ns1 = 4 * (k*n2 - (k-1)*n1)
    ns2 = 3 * (k*K2*N2 - (k-1)*K1*N1)
    nu = 3/2 * (N1*(k-1)**2 + N2*k**2)

    return ns1-ns2, nu
    
def reducingtrans(A, B, resvars):
    '''
    Note that z1 and z2 are complex numbers, and we take
    the modulus by np.abs and then square it, i.e. z*z_conjugate.

    The angle from the positive real axis is the argument of z, 
    also known as the phase.
    '''

    z1 = cmath.rect(np.sqrt(2*resvars.Gamma1), resvars.sigma1)
    z2 = cmath.rect(np.sqrt(2*resvars.Gamma2), resvars.sigma2)
    abfac = A**2 + B**2
    temp1 = A*z1 + B*z2
    temp2 = B*z1 - A*z2    
    Phi1 = 1/2 * np.abs(temp1)**2/abfac
    Phi2 = 1/2 * np.abs(temp2)**2/abfac

    return pd.Series(index=['Phi1', 'Phi2', 'phi1', 'phi2'], \
                data=[Phi1, Phi2, cmath.phase(temp1), cmath.phase(temp2)])  

def n_and_a(body, period):
    '''
    body (Series): the planet parameters + stellar mass
    
    If period is true: using the period to derive a as well as
    mean notion (n). This accounts for stellar and planet mass
    but is not in Jacobi coordinates, unfortunately. If not,
    just deriving n and returning. 

    n is only used within the kepHfactors function, and it is
    defined as sqrt(G*M_j/(a*_j)^3), see below Eq 15 N&V 2016.
    '''

    if period:
        body.a = modkep3rd(body.P, np.array([body.mstar, body.m]))
        body.astar = modkep3rd(body.Pstar, np.array([body.mstar, body.m]))

    # because our astar is now defined by Pstar.     
    body.n = ncalc_a(body.astar, body.m, body.mstar)
    return body

def calcsysvals(body1, body2, resinfo, period=False, return_df=False):
    '''
    Computes the delta value and Phi*cosine(phi) values for a given
    planet pair (i.e. body1 and body2).

    period: use p to derive the semi-axis and n
            and pstar to derive astar etc. 
    '''
    body1 = n_and_a(body1, period)
    body2 = n_and_a(body2, period)
    delaun1 = delaunayelements(body1)
    delaun2 = delaunayelements(body2)

    resvars = resonantelements(resinfo.k, delaun1, delaun2)
    # print(resvars.K1 + resvars.K2)
    # print(delaun1.Lamda + delaun2.Lamda - resvars.Gamma1 - resvars.Gamma2)
    
    ns, nu = kepHfactors(resinfo.k, body1.n, body2.n, \
        delaun1.LamdaStar, delaun2.LamdaStar, resvars.K1, resvars.K2)

    A = resinfo.f1/np.sqrt(delaun1.LamdaStar)
    B = resinfo.f2/np.sqrt(delaun2.LamdaStar)
    Cfac = np.sqrt(A**2 + B**2)
    C = G*Cfac*body1.m*body2.m/body2.astar
    
    polarvars = reducingtrans(A, B, resvars)
    etafac = (nu/C)**(2/3)
    # print(polarvars.Phi2 + polarvars.Phi1)
    # print(resvars.Gamma1 + resvars.Gamma2)

    delta = etafac*(ns/(2*nu) - polarvars.Phi2)
    Psi = etafac*polarvars.Phi1
    
    return pd.Series(index=['delta', 'Psi', 'psi'], \
           data=[delta, Psi, polarvars.phi1])


def psi(body1, body2, resinfo, period=False):
    '''
    Similar to main but we can neglect C and the kepH
    factors, which simplify the situation, we just retun
    the resonant angle, psi.
    '''
    body1 = n_and_a(body1, period)
    body2 = n_and_a(body2, period)
    delaun1 = delaunayelements(body1)
    delaun2 = delaunayelements(body2)
    resvars = resonantelements(resinfo.k, delaun1, delaun2)
    A = resinfo.f1/np.sqrt(delaun1.LamdaStar)
    B = resinfo.f2/np.sqrt(delaun2.LamdaStar)
    polarvars = reducingtrans(A, B, resvars)

    return polarvars.phi1
    

existcol = lambda col, df: col in df.columns
def quickndirty(df, a=False, verbose=False, ilocs=(0,1)):
    '''
    'Cheating' to run calcsysvals() by ensuring that all the inputs
    are corrected for and printing out the conditions.
    '''
    if verbose:
        print(f'Do we have Pstar? {existcol('Pstar', df)} \n',
              f'Do we have astar? {existcol('astar', df)} \n',
              f'Do we have pomega? {not np.all(np.isnan(df['pomega']))} \n',
              f'Do we have lamda? {not np.all(np.isnan(df['l']))} \n')
    
    if not existcol('Pstar', df): df['Pstar'] = df.P
    if not existcol('astar', df): df['astar'] = df.a

    if np.all(np.isnan(df['pomega'])):
        df['pomega'] = df.Omega + df.omega
    if np.all(np.isnan(df['l'])):
        df['l'] = df.pomega + df.M

    inner = df.iloc[np.min(ilocs)]
    outer = df.iloc[np.max(ilocs)]
    
    rpP = calcsysvals(inner.copy(), outer.copy(), two2one, True)
    rpA = calcsysvals(inner.copy(), outer.copy(), two2one, False)
    deltaP, psifacP = rpP.delta, Psifac(rpP.Psi, rpP.psi)
    deltaA, psifacA = rpA.delta, Psifac(rpA.Psi, rpA.psi)
    
    if verbose:
        print(f'Period-calculated: delta is {deltaP} and Psi*cos(psi) is {psifacP}.')
        print(f'a-calculated: delta is {deltaA} and Psi*cos(psi) is {psifacA}.')

    if a: return deltaA, psifacA
    return deltaP, psifacP

def mkdiagdf(time, integratedict, pardf, check=False, order=(1,2)):
    '''
    Ensure the innermost planet is first by using pardf and 
    sorting by the period. Then, we can get the corresponding
    planet by indexing from the keys of integratedict. 
    
    time (arr or list): the integration timestamps.
    integratedict (dict): in the form of nested dictionaries.
        {'planet1': {orbitalelements}, 'planet2': {orbitalelements}}
    pardf (pd DataFrame): the DataFrame with planet masses and stellar mass
        (used to initialise rebound sim and for the delta value calculation.)
    '''
    
    pardf = pardf.sort_values(by='P', ascending=True)
    dflist = []
    for row in pardf.itertuples():
        if row.order in order: # e.g. GJ 876 has four planets.
            df = pd.DataFrame(integratedict[row.Index], index=time)
            df['astar'], df['Pstar'] = np.mean(df['a']), np.mean(df['P'])
            df['mstar'], df['m'] = row.mstar, row.m
            if check: gen.checkangles(df)
            dflist.append(df)
    if len(dflist) > 2: return print('Too many planets! Incorrect order?')
    # The lambda function loops through each row and row.name gives the time. 
    diagdf = dflist[0].apply(lambda row: calcsysvals(row, dflist[1].loc[row.name], two2one.copy(), False), axis=1)

    return diagdf

## Generating the diagram: mmr!/resonant diagram/diagram (D2).ipynb
def getequilibrium(delta):
    roots = np.roots([1, -2*delta, delta**2, -1/8])
    if delta < deltastar: return -1*(roots[np.isreal(roots)]).real
    stable, eqtrix, stable2 = roots.real
    return -1*stable, eqtrix, stable2

def equiPsi(deltas):
    '''
    Define the array: we have three roots given the cubic.
    We solve for the roots based on a given delta value.
    However, the roots are only real given delta > delta*,
    so we separate out the two domains.
    '''
    PsiSol = np.zeros((len(deltas), 3), dtype=np.complex128)
    for i, delta in enumerate(deltas):
        PsiSol[i] = np.roots([1, -2*delta, delta**2, -1/8])

    # The two regimes of solutions. 
    underPsiSol = PsiSol[deltas <= deltastar]
    underPsiSol = underPsiSol[np.isreal(underPsiSol)].real
    return underPsiSol, PsiSol[deltas > deltastar].real

def resonantbounds(delta, initialguess, zerothres, overP=None):
    '''
    1. Calculate the value of the Hamiltonian on the separatrix
       i.e. Htrix =  -(Psi-delta)^2 - (2*Psi)^(1/2)*np.cos(psi)

    2. Divide into three regimes. 
       - when one value is above zero (psi = 0) and the other is below.
       - when the upper bound is closer to zero.
       - when both bounds are in the regime of psi = pi.
       This requires three forms of initial guesses!

    delta (array): the range of delta to solve over, ought to be > deltastar
    initial guess (list): 2, 1, 2 because solver should not go too near zero.
    overP (array): previously solved solution if given.
    '''

    if overP is None: underP, overP = equiPsi(delta)
    trixH = H(overP[:, 1], delta[delta > deltastar], 0)
    PsiBounds = np.zeros((len(delta), 2))

    # Identifying the different regime bounds.
    disdeltas = np.abs(np.sqrt(-1*trixH) - delta[delta > deltastar])
    disidx = np.where(disdeltas < zerothres)[0]
    possolve = lambda Psi2: -(Psi2**2-d)**2 - np.sqrt(2)*Psi2 - trixH[i]
    negsolve = lambda Psi2: -(Psi2**2-d)**2 + np.sqrt(2)*Psi2 - trixH[i]
    regimes = np.array([0, disidx[0], disidx[-1]+1])
    rcount = 0

    for i, d in enumerate(delta):
        # iterating over the various regimes
        if i in regimes:
            guess = np.sqrt(initialguess[rcount])
            rcount += 1   
        else: guess = PsiBounds[i-1]

        if i < regimes[1]: # upper close to separatrix
            uproot = fsolve(possolve, guess[0:1])
            downroot = fsolve(negsolve, guess[-1:])
            PsiBounds[i] = np.array([*uproot, *downroot])

        elif i in disidx: # discontinuity at zero
            root = fsolve(negsolve, guess[-1])
            PsiBounds[i] = np.array([0, *root])

        else: # both bounds are psi = pi.
            roots = fsolve(negsolve, guess)
            PsiBounds[i] = roots

    # Fixing the bounds by the np.cos factor and square
    PsiBounds = np.square(PsiBounds)
    PsiBounds[regimes[1]:, 0] *= -1
    PsiBounds[:, 1] *= -1
        
    return PsiBounds

def getdiagrambounds(deltas, initialguess, zerothres):
    '''
    '''
    # Calculating the equilibrium points
    underP, overP = equiPsi(deltas)  
    truncdelta = deltas[deltas > deltastar]
    stable1psifac = Psifac(np.append(underP, overP[:, 0]), np.pi)
    bounddict = {'stable1': (deltas, stable1psifac), \
          'stable2': (truncdelta, Psifac(overP[:, 2], 0)), \
          'unstable': (truncdelta, Psifac(overP[:, 1], 0))} 
    
    # Calculating the bounds of libration as given by the separatrix
    bounds = resonantbounds(truncdelta, initialguess, zerothres, overP=overP)
    bounddict['reslower'] = (truncdelta, bounds[:, 1])
    bounddict['resupper'] = (truncdelta, bounds[:, 0])

    return bounddict

class DiagramParameters():

    def __init__(self, pairvals, sysnames):

        self.numpt = 10**3
        self.defaultcolours = ['#E40303', '#FF8C00', '#FFED00', '#008026', \
                               '#004DFF', '#750787', '#FFAFC8', '#74D7EE', '#613915']
        self.colours = self.getcolours(len(sysnames))
        self.pairvals = pairvals
        if len(sysnames) > 0: self.sysnames = sysnames
        else: self.sysnames = np.tile([''], len(pairvals))
        self.legendict = {'loc': 'lower left', 'fontsize': 18, 'ncols': 2, 'show': False}

    def formatlegend(self, ax):
        '''
        '''
        showlegend = (self.legendict).pop('show', True)
        if showlegend: ax.legend(**self.legendict)

        return ax

    def getcolours(self, length):
        '''
        '''
        if length > len(self.defaultcolours):
            tilelen = np.ceil(length/len(self.defaultcolours)).astype(int)
            self.defaultcolours = np.tile(self.defaultcolours, tilelen)
            
        return self.defaultcolours[:length]

    def reinitialise(self, pairvals, sysnames):
        self.pairvals = pairvals
        self.sysnames = sysnames
        self.colours = self.getcolours(len(sysnames))

class StructureDiagram(DiagramParameters):
    
    def __init__(self, pairvals=[], sysnames=[]):
        
        super().__init__(pairvals, sysnames)
        # Calculation-based diagram parameters.
        self.deltamin = -2
        self.deltamax = 7.5
        self.zero_threshold = 0.01
        self.initial_guess = [[0.2, 2.8], [3.2], [1e-3, 4]]
        # The information needed for plotting. 
        self.datadict = None
        
        # Line and bounds formatting. 
        self.equidict = {'color':'black', 'lw': 2, 'zorder':10}
        self.stable1 = dict({'label':'Stable #1'}, **self.equidict)
        self.stable2 = dict({'ls':'dashdot', 'label':'Stable #2'}, **self.equidict)
        self.unstable = dict({'ls':'dashed', 'label':'Unstable'}, **self.equidict)
        self.resbounds = (self.equidict).copy()
        self.resfill = {'alpha':0.35, 'color':'slategrey', 'label':'Resonance', 'zorder':0}
        
        # If annotating with the scatter, we have:
        self.sysmarkers = lambda colour, label: {'marker': '*', 's': 350, 'zorder': 8, \
                                    'edgecolor': 'black', 'color': colour, 'label': label}
        self.offset = None
        self.annotatedict = dict(ha='center', textcoords='offset points')
        self.line = lambda colour, label: {'lw': 0.8, 'alpha': 0.65, 'marker': '1', \
            'markersize': 1, 'markerfacecolor': 'black', 'color': colour, 'label': label}
        self.set_labels = True

    def getdeltas(self):
        self.deltas = np.linspace(self.deltamin, self.deltamax, self.numpt)
        return self.deltas

    def diagram(self, ax):
        '''
        Getting the equilibrium lines to define the different dynamical
        regimes as well as the libration boundaries. 
        '''
        deltas = self.getdeltas()
        if self.datadict is None:
            self.datadict = getdiagrambounds(deltas, self.initial_guess, self.zero_threshold)
            
        ax.plot(*self.datadict['stable1'], **self.stable1)
        ax.plot(*self.datadict['stable2'], **self.stable2)
        ax.plot(*self.datadict['unstable'], **self.unstable)
        ax.plot(*self.datadict['reslower'], **self.resbounds)
        ax.plot(*self.datadict['resupper'], **self.resbounds)
        ax.fill_between(*self.datadict['resupper'], self.datadict['reslower'][1], **self.resfill)
    
        return ax

    def annotatesys(self, ax):
        '''
        How we add the labels to our scatter points.
        '''
        if self.offset is not None: offset = self.offset
        else: offset = np.tile(np.array([10, -15]), ((len(deltas), 1)))
    
        for i, xy in enumerate(self.pairvals):
            ax.annotate(self.sysnames[i], xy=xy, xytext=offset[i], **self.annotatedict)
    
        return ax

    def scattersys(self, ax):
        '''
        '''
        scatters = []
        for i, colour in enumerate(self.colours):
            sdict = self.sysmarkers(colour, self.sysnames[i])
            scatters.append(ax.scatter(*self.pairvals[i], **sdict))
        return ax, scatters
    
    def scatterdiagram(self, ax):
        '''
        '''
        ax = self.diagram(ax)
        ax, scatters = self.scattersys(ax)
        ax = self.formatlegend(ax)
        ax.set_xlim(self.deltamin, self.deltamax)
        if self.set_labels:
            ax.set_xlabel(r'$\delta$')
            ax.set_ylabel(r'$\Psi$ cos($\psi$)')
    
        return ax, scatters


## For the contour dynamical portraits.
def deltaslicescatter(df, slicenum=350):
    '''
    '''
    deltabins = np.linspace(np.min(df.delta), np.max(df.delta), slicenum+1)
    binwidth = deltabins[1]-deltabins[0]
    deltamean = np.mean(df.delta)
    meanbounds = deltamean - 1/2*binwidth, deltamean + 1/2*binwidth

    scatterdata = []
    lowb = [deltabins[0], meanbounds[0], deltabins[-2]]
    upperb = [deltabins[1], meanbounds[-1], deltabins[-1]]
    for i, (l, u) in enumerate(zip(lowb, upperb)):
        m = np.logical_and(df.delta >= l, df.delta < u)
        x, y = Psifac(df.Psi, df.psi)[m], (df.Psi*np.sin(df.psi))[m]
        scatterdata.append([x, y])
        
    return np.transpose([lowb, upperb]), scatterdata

class ContourParameters():
    
    def __init__(self, sysname):

        self.sysname = sysname
        self.numpt = 10**3
        self.slicenum = 25
        self.axbounds = 15
        self.set_labels = True
        self.efunc = None
        self.elevels = None

        self.colour = '#fdd110'
        self.colourlist = ['#D60270', '#9B4F96', '#0038A8']
        self.equimarker = dict(color='tab:red', marker='X', edgecolor='black', s=150, zorder=10)
        self.alldatamarker = dict(color='silver', s=1, alpha=0.1)
        self.contour = dict(colors='black', linestyles='solid')
        self.trixdict = dict({'linewidths': 5}, **self.contour)
        self.trixlabel = dict(fmt=lambda x: gen.formatSig(x, 3), fontsize=10)
        self.subdatamarker = dict(s=25, alpha=0.5)

    def getelevels(self, hvals=[-40, 5]):
        '''
        '''
        
        if self.efunc is not None: self.elevels = np.sort(self.efunc(hvals))
        else: # we don't have a function for our energy levels
            if self.elevels is None: 
                conlower = np.linspace(np.min(hvals), -10, 12)
                conupper = np.linspace(-6, np.max(hvals), 5)
                self.elevels = np.sort(np.array([*conlower, *conupper]))
            else: self.elevels = np.sort(self.elevels)
        return self.elevels

    def contourfromdelta(self, ax, delta):
        '''
        '''
        gridpt = np.linspace(-1*self.axbounds, self.axbounds, self.numpt)
        X, Y = np.meshgrid(gridpt, gridpt)
        
        psi = np.atan2(Y, X)
        Psi = np.sqrt(X**2 + Y**2)
        Hcalced = H(Psi, delta, psi)
        roots = getequilibrium(delta)
        self.getelevels(Hcalced)
        
        ax.contour(X, Y, Hcalced, levels=self.elevels, **self.contour)
        if delta > deltastar:
            htrix = H(roots[1], delta, 0)
            showtrix = (self.trixlabel).pop('show', True)
            separatrix = ax.contour(X, Y, Hcalced, levels=[htrix], **self.trixdict)
            if showtrix: ax.clabel(separatrix, separatrix.levels, **self.trixlabel)            
        ax.scatter(roots, np.zeros((len(roots),)), **self.equimarker)
    
        return ax
    
    def contourportrait(self, ax, df):
        '''
        '''
        delta = np.mean(df.delta)
        ax = self.contourfromdelta(ax, delta)
        alldata = Psifac(df.Psi, df.psi), df.Psi*np.sin(df.psi)
        ax.scatter(*alldata, **self.alldatamarker)
        ax.set_title(f'{self.sysname} with {r'$\delta\sim$'}{np.round(delta, 2)}')
        if self.set_labels:
            ax.set_xlabel(r'$\Psi \cos(\psi)$')
            ax.set_ylabel(r'$\Psi \sin(\psi)$')

        return ax
    
    def contourportrait3(self, axs, df):
        '''
        df = pd.read_csv(sysdir('HD-82943(5000).csv'))
        '''
        dfscatter = deltaslicescatter(df, slicenum=self.slicenum)
        gridpt = np.linspace(-1*self.axbounds, self.axbounds, self.numpt)
        X, Y = np.meshgrid(gridpt, gridpt)
        psi = np.atan2(Y, X)
        Psi = np.sqrt(X**2 + Y**2)
        showtrix = (self.trixlabel).pop('show', True)
            
        for i, ax in enumerate(axs):    
            deltabounds, scatterdata = dfscatter
            delta = np.mean(deltabounds[i])
            Hcalced = H(Psi, delta, psi)
            roots = getequilibrium(delta)
            self.getelevels(Hcalced)

            ax.contour(X, Y, Hcalced, levels=self.elevels, **self.contour)
            if delta > deltastar:
                htrix = H(roots[1], delta, 0)
                separatrix = ax.contour(X, Y, Hcalced, levels=[htrix], **self.trixdict)
                if showtrix: ax.clabel(separatrix, separatrix.levels, **self.trixlabel)
            ax.scatter(roots, np.zeros((len(roots),)), **self.equimarker)
            alldata = Psifac(df.Psi, df.psi), df.Psi*np.sin(df.psi)
            ax.scatter(*alldata, **self.alldatamarker)
            ax.scatter(*scatterdata[i], color=self.colourlist[i], **self.subdatamarker)
            ax.set_title(r'$\delta$: ' + f'{np.round(deltabounds[i][0], 2)} - {np.round(deltabounds[i][-1], 2)}')

        for ax in axs: ax.set_xlabel(r'$\Psi \cos(\psi)$')
        axs[0].set_ylabel(r'$\Psi \sin(\psi)$')
    
        return axs