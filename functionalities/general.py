from pathlib import Path
currdir = str((Path(__file__).resolve()).parent)
basedir = lambda path: f'/home/sliu/{path}'

import matplotlib
import numpy as np
import pandas as pd
import astropy.constants as a
import matplotlib.pyplot as plt

## CONSTANTS
G = (2*np.pi)**2
RSRJ = (a.R_sun/a.R_jup).value
RERJ = (a.R_earth/a.R_jup).value
MEMJ = (a.M_earth/a.M_jup).value
MSMJ = (a.M_sun/a.M_jup).value
MSME = (a.M_sun/a.M_earth).value
## mass and radius thresholds from (2)massrad.ipynb
MTHRES = 0.12
RTHRES = 0.68

## Formating functionality
formatSig = np.vectorize(lambda val, sf: f"{val:.{sf}g}")
formatAngle = lambda angle: wrapzero(angle) * 180/np.pi

## Lists or column selection for NASA exoplanet archive dataframe
DISCM = ['Radial Velocity', 'Transit', 'Transit Timing Variations']
QUICKCOLS = ['pl_name', 'hostname', 'discoverymethod', 'sy_pnum', \
             'pl_orbper', 'pl_rade', 'pl_radj', 'pl_ratror', \
             'pl_masse', 'pl_massj', 'pl_msinie', 'pl_msinij']
MISHOSTS = np.array([['55 Cnc', '55 Cnc B'], ['HD 133131 A', 'HD 133131 B'], \
                    ['HD 20781', 'HD 20782'], ['TOI-2267 A', 'TOI-2267 B'], \
                    ['XO-2 S', 'XO-2 N']])
## Lists or column selection for rebound functionality.
anglecols = ['pomega', 'l', 'Omega', 'omega', 'inc', 'M']
# Columns which should be numerical depending on df format.
numcols = np.append(np.array(['mstar', 'order', 'P', 'a', 'm', 'e']), anglecols)
# The orbital elements in different configurations
orbels = ['P', 'a', 'e', 'inc', 'Omega', 'omega', 'pomega', 'f', 'M', 'l']
# if coplanar, Omega and omega are not well-defined: pos should be either 'M' or 'l'
coplorbels = lambda pos: ['P', 'e', 'inc', 'pomega', pos]
# if mutually inclined, we take Omega and omega instead of pomega
planarorbels = lambda pos: ['P', 'e', 'inc', 'Omega', 'omega', pos]
# To get the dictionaries in which we can use the validate function.
angled = dict(zip(anglecols, np.repeat(np.pi/180, len(anglecols))))
convertd = {'P': 1/365.25, 'm': 1/MSMJ}|angled

def conversion(col, convertd):
    '''
    To convert the original system dataframe from:
    - mass: jupiter mass to solar mass
    - period: days to years
    - omega, lamda, mean_anomaly, i: degrees to radians
    However, the conversion can be modified by a different convdict.

    df.apply(conversion, axis=0) 
    where axis=0 applies this to columns and not rows. 
    '''
    if col.name in convertd.keys():
        col *= convertd[col.name]
    return col

def validate(df, index=False, convertdict=convertd, numcols=numcols):
    '''
    Unnamed columns are droppped as rows if so necessary.
    Fix the index if specified not to be the default pandas.
    The columns which should be numeric are switched to numerical.
    Then, multiply by the conversion factor if columns are in 
    the convert dictionary.

    df (DataFrame): on which changes are applied.
    strindex (bool): to handle if the index should be changed.
    convertdict (dict): dictionary which maps conversion factors.
    numcols (list/array): columns which should have numerical values.
    '''
    while np.all(df.columns.str.contains('Unnamed')):
        df.columns = df.iloc[0].values
        df.drop(index=df.index[0], inplace=True)

    if index:
        try: df.set_index('Unnamed: 0', inplace=True)
        except: df.set_index(df.columns[0], inplace=True)

    df[numcols] = df[numcols].apply(pd.to_numeric)
    df = df.apply(lambda x: conversion(x, convertdict), axis=0)
    return df 

def timeAverage(intdict, df=None, verbose=True):
    '''
    Writing in the time-averaged values into df if not None, otherwise
    just calculating the time-averaged a and period.
    
    intdict (dict): {planetidx: {planetdict}}
    '''
    astar = np.zeros((len(intdict),))
    Pstar = np.zeros((len(intdict),))
    for i, (planetidx, planetarr) in enumerate(intdict.items()):

        astar[i] = np.mean(planetarr['a'])
        Pstar[i] = np.mean(planetarr['P'])
        if verbose:
            print(f'For {planetidx}: the time-averaged semi-major axis is', \
            f'{formatSig(astar[i], 5)}, the time-averaged period is {formatSig(Pstar[i], 5)}.')
    
    if df is not None:
        df['astar'] = astar
        df['Pstar'] = Pstar
        return df
        
    return astar, Pstar

# Simple calculations
wrapzero = lambda theta: (theta + np.pi)  % (2*np.pi) - np.pi
def thetacalc(outlam, inlam, pomega, abtzero=True):
    '''
    If the oscillation is about zero, then we want the bounds
    of [-np.pi, np.pi). Otherwise, we take the notation from 
    Murray & Dermott, e.g. Eq. 8.8 from p. 324.
    '''
    theta = 2*outlam - inlam - pomega
    if abtzero: return wrapzero(theta)
    return theta %(2*np.pi)

def getResAngles(innerd, outerd):
    '''
    Calculating the resonant angles and formating them to degrees.
    Requires the individual integration planetary orbital
    elements from reboundfuncs.py.
    '''
    dpomega = formatAngle(outerd['pomega'] - innerd['pomega'])
    intheta = thetacalc(outerd['l'], innerd['l'], innerd['pomega']) * 180/np.pi
    outheta = thetacalc(outerd['l'], innerd['l'], outerd['pomega']) * 180/np.pi
    return dpomega, intheta, outheta

modkep3rd = lambda P, masses: (G/(2*np.pi)**2 * P**2 *np.sum(masses))**(1/3)
def calca(sys, colname='a_J', Jacobi=True):
    '''
    Adding the calculated semi-major axis. Depending on whether 
    or not Jacobi is specified, the outer calculation includes
    the mass of the inner planet.
    '''

    a = np.zeros((len(sys),))
    for i, row in enumerate(sys.itertuples()):

        if Jacobi:
            masses = sys.m[sys.order <= row.order]
            masses = np.append([row.mstar], masses)
        else: masses = np.array([row.mstar, row.m])
        a[i] = modkep3rd(row.P, masses)

    if colname == '':
        return a

    sys[colname] = a
    return sys

def checkangles(df):
    '''
    Checks if pomega corresponds to its calculated values!
    We typically don't use the mean anomaly beyond setting up the
    simulations, so we do not check the lambda = mean anomaly + pomega
    relationship.
    '''
    summangles = np.round(df.omega + df.Omega, 5)%(2*np.pi)
    pomega = np.round(df.pomega, 5)%(2*np.pi)
    equalm = summangles == pomega
    diff = np.abs(summangles - pomega)[~equalm]
    print(f'The fraction of equality is {np.round(np.sum(equalm)/len(df), 5)},' \
          f'the sum of absolute differences is {np.round(np.sum(diff), 5)},' \
          f'count of differences > 0.1 are {np.sum(diff > 0.1)}.')

## PLOTTING FUNCTIONALITIES
def plotSpecs(ax, logscale):
    '''
    Specifications for the tick parameters and for the logscale.
    logscale: can be an integer or a list.
    '''
    ax.tick_params(direction='in', which='major', bottom=True, top=True, \
                   left=True, right=True, length=10, width=2)
    ax.tick_params(direction='in', which='minor', bottom=True, top=True,  \
                   left=True, right=True, length=4, width=2)

    if logscale==1:
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif logscale==2:
        ax.set_xscale('log')
    elif logscale==3:
        ax.set_yscale('log')
        
    return ax
    
def setupPlot(sizeTuple, logscale=1, fontsize=16, scalar=True):
    '''
    To format single plots uniformly. The sizeTuple is a tuple.
    logscale: should be an integer if trying to get axis in logscale.
    scalar: allows us to format the x and y axis as scalars.
    '''
    matplotlib.rcParams.update({'font.size': fontsize}) #adjust font
    matplotlib.rcParams['axes.linewidth'] = 1.5
    
    fig = plt.figure(figsize=sizeTuple) #adjust size of figure
    ax = plt.axes()
    ax = plotSpecs(ax, logscale)

    if scalar:
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    return fig,ax

def setupPlots(sizeTuple, row=1, col=2, logscale=1, fontsize=16, scalar=True, **kwargs):
    '''
    Uses the same style as setupPlot but allows for multiple plots.
    logscale can be an integer or it can be an iterable, i.e. list.
    scalar: allows us to format the x and y axis as scalars.
    '''
    
    matplotlib.rcParams.update({'font.size': fontsize}) # adjust font
    matplotlib.rcParams['axes.linewidth'] = 1.5

    # If all the plots should be on the same scale, then the user can input an integer.
    if isinstance(logscale, int):
        logscale = np.repeat(logscale, row*col)
    # adjust size of figure
    fig, axs = plt.subplots(nrows=row, ncols=col, figsize=sizeTuple, **kwargs) 
    for i, ax in enumerate(axs.flatten()):
        ax = plotSpecs(ax, logscale[i])
        if scalar:
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    return fig, axs