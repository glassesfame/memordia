import sys
from pathlib import Path
funcdir = str((Path(__file__).resolve()).parent)

import matplotlib
import numpy as np
import pandas as pd
import astropy.constants as a
import matplotlib.pyplot as plt

sys.path.insert(0, funcdir)
import general as gen

## CONSTANTS
FIRSTORDER = np.transpose([np.arange(2, 7), np.arange(1, 6)])
SECONDORDER = np.transpose([np.arange(3, 10, 2), np.arange(1, 8, 2)])
MMRCOLS = np.array(['hostname', 'pnumflag', 'nanflag', 'pairname', \
                   'firstMMR', 'firstdelta', 'secondMMR', 'seconddelta'])
## To categorise the mass and radius columns.
RADCOLS = np.array(['st_rad', 'pl_ratror', 'pl_rade', 'pl_radj'])
MASSCOLS = np.array(['pl_orbincl', 'pl_masse', 'pl_massj', 'pl_msinie', 'pl_msinij', 'pl_cmasse', 'pl_cmassj'])
CATECOLS = ['pl_name', 'hostname', 'sy_pnum', 'discoverymethod', 'pl_orbper']
MASSCATECOLS = ['mjoveflag', 'mass', 'siniflag']
RADCATECOLS = ['rjoveflag', 'radius']

def fixcolformat(arr):
    '''
    Example row: # COLUMN pl_name:        Planet Name
    Therefore, our delimiter is '#' and then we treat the elements
    accordingly to extract the column name.
    '''
    rmedcol = arr[-1].strip(' COLUMN ')
    return rmedcol.split(':')[0]

def getExoCols(file):
    '''
    Load in the txt with the column info. This looks like just the 
    copy and pasted info from the first couple rows of the exoplanet
    archive csv file. 
    '''
    txt = np.loadtxt(file, comments='+++', delimiter='#', dtype=object)
    return np.array([fixcolformat(row) for row in txt], dtype=object)

def handlebinary(df, verbose=True, return_mismatch=False, \
                return_stcomp=False):
    '''
    Used with default_flag == 1! 
    '''
    hosts, hostidx, hostcounts = np.unique(df.hostname, return_index=True, return_counts=True)
    pnum, pcount = np.unique(df.sy_pnum, return_counts=True)
    checkpnum = df.iloc[hostidx].sy_pnum
    mismatchdf = df[np.isin(df.hostname, hosts[hostcounts != checkpnum])] 
    mishost, miscount = np.unique(mismatchdf.hostname, return_counts=True)
    rmhost = mishost[miscount == 1]

    binarydf = df[df.sy_snum > 1]
    stcomphosts = np.setdiff1d(binarydf.hostname, mismatchdf.hostname)
    stcompdf = df[np.isin(df.hostname, stcomphosts)]

    if verbose:
        cbhosts = np.unique(df.hostname[df.cb_flag == 1])
        print(f'At a glance: {len(df)} planets in {len(hosts)} systems. BUT:', 
              '\n' + f'1. {np.sum(df.cb_flag == 1)} circumbinary planets orbiting {len(cbhosts)} pairs.',
              '\n' + f'2. {len(stcomphosts)} hosts with stellar companion(s) ({len(stcompdf)} planets about such hosts)',
              '\n' + f'3. {len(mismatchdf)} planets in systems with >= 2 planet-hosting stars.' , \
              f'Of such {len(mishost)} stars, those hosting one planet are removed: {rmhost}.')

    if return_mismatch: return mismatchdf
    if return_stcomp: return stcompdf

    return df[~np.isin(df.hostname, rmhost)]

def getDefaultCat(df, cols, savefile=None, treatbinary=True, verbose=True):
    '''
    Filtering for only systems which have planet number > 1, then we take
    the default parameters. Finally, we remove binary systems which do not
    fit this criteria. 
    
    df (DataFrame): the downloaded exoplanet archive csv with all columns/rows
    cols (str or list): if str, then a path + file name, which will
          allow us to read the columns from a txt file, or just a list
    savefile (str): also the path + save filename

    '''
    if isinstance(cols, str): cols = getExoCols(cols)
    cat = df.loc[df.sy_pnum > 1, cols]
    cat = (cat[cat.default_flag == 1]).drop('default_flag', axis=1)
    if treatbinary: cat = handlebinary(cat, verbose=verbose)
    if savefile is not None: cat.to_csv(f'{savefile}.csv', index=False)
    return cat

def getSysNum(df):
    '''
    '''
    pnum, pcount = np.unique(df.sy_pnum, return_counts=True)
    return (pcount/pnum).astype(int)

def catstats(df, allcat=None, counthres=50, verbose=True):
    '''
    Check if the default_flag is present. If it is, filter for default and drop.
    Gets an overview of the catalogue: number of planets, host stars and systems,
    as well as binary+ information and discoverymethod information.
    '''

    if 'default_flag' in df.columns:
         df = (df[df.default_flag == 1]).drop('default_flag', axis=1)
    
    hosts, hostcounts = np.unique(df.hostname, return_counts=True)
    pnum, pcount = np.unique(df.sy_pnum, return_counts=True)
    numsys = getSysNum(df)
    try: pl2count = numsys[pnum == 2][0]
    except: pl2count = 0
    
    discmthd, discount = np.unique(df.discoverymethod, return_counts=True)
    if isinstance(counthres, int): m = discount >= counthres
    else: m = np.isin(discmthd, counthres)
    discmthd, discount = discmthd[m], discount[m]
    discstr = [f'{discmthd[i]} ({count})' for i, count in enumerate(discount)]
    
    mstardf = df[df.sy_snum > 1]
    mstar_numsys = np.sum(getSysNum(mstardf))
    mstarname = np.unique(mstardf.hostname)

    if verbose: 
        print(f'There are {len(df)} planets orbiting {len(hosts)} stars in {np.sum(numsys)} systems. Of such:',
              '\n' + '\t' + f'{pl2count} 2-planet systems and ', \
              f'{np.sum(numsys[pnum > 2])} systems with > 2 planets.',
              '\n' + '\t' + f'Discovery methods: {'; '.join(discstr)}.', 
              '\n' + '\t' + f'{np.sum(df.cb_flag == 1)} circumbinary planets and ', \
              f'{len(mstarname)} companioned stars in {mstar_numsys} systems with {len(mstardf)} planets.')
    
    index = ['# Planets', '# Host Stars', '# Systems', '# 2-Planet Systems', \
             '# > 2-Planet Systems', *discmthd, '# Circumbinary Planets', \
             '# Companioned Stars', '> 1 Star Systems', '# Planets in > 1 Star Systems']
    data = [len(df), len(hosts), np.sum(numsys), pl2count, \
             np.sum(numsys[pnum > 2]), *discount, np.sum(df.cb_flag == 1), \
            len(mstarname), mstar_numsys, len(mstardf)]
    
    return pd.Series(data=data, index=index)

def mixeddiscovery(hostname):
    '''
    discdf = cat.groupby('hostname')[cat.columns].apply(mixeddiscovery)
    '''
    discmthd, discount = np.unique(hostname.discoverymethod, return_counts=True)
    disdf = pd.DataFrame([discount], columns=discmthd)
    if len(discmthd) > 1: mixed = True 
    else: mixed = False
    disdf['Mixed'] = mixed
    return disdf

def mixeddiscoveryStats(discs, df):
    '''
    '''
    discdf = df.groupby('hostname')[df.columns].apply(mixeddiscovery)
    if isinstance(discs, str): discs = [discs]
    for disc in discs:
        m = np.logical_and(~np.isnan(discdf[disc]), discdf.Mixed)
        mixedcounts = np.sum(~np.isnan(discdf[m]), axis=0)
        rmmask = np.logical_and(mixedcounts != 0, \
                                ~np.isin(discdf.columns, [disc, 'Mixed']))
        writestr = [f'{mixedcounts[rmmask][i]} {discmthd}' for i, \
                    discmthd in enumerate(discdf.columns[rmmask])]
        print(f'For hoststars/systems with {disc} detections, there are', \
              f'{len(discdf[m])} mixed detections with {', '.join(writestr)} mixes.')
              
## MMR Calculation
## For the MMR calculation!
fpairs = lambda arr: np.transpose([arr[:-1], arr[1:]])
deltaEq = lambda Pout, Pin, p, q: (Pout/Pin)/(p/q) - 1

def pairMMR(pout, pin, order):
    '''
    Finding the pair values for the MMR, this looks through all the possible ratios.
    '''
    testdelta = deltaEq(pout, pin, order[:, 0], order[:, 1])
    minIdx = np.argmin(np.abs(testdelta))
    mmrPair = ':'.join(np.array(order[minIdx], dtype=str))
    return testdelta[minIdx], mmrPair

def sysMMRdf(sys, infodict={'pairname': 'pl_name'}):
    '''
    Separate our cases into those without pairs, one pair and multiple pairs.
    Considering the exceptions:
    - There is only one planet in the system and we can't advance further.
    - There are missing periods in this system.

    Ideally, we'd be able to generalise 


    '''
    # Handling columns and creating MMR dataframe for the system. 
    coltemp = np.append(MMRCOLS, list(infodict.keys()))
    cols, reorder = np.unique(coltemp, return_index=True)
    df = pd.DataFrame(columns=coltemp[reorder])    
    ## System-level information
    pairL = len(sys) - 1 # possible pairs
    if pairL == 0: # Only one planet!! Should not happen!
        df.loc[0, ['hostname', 'pnumflag']] = [sys.name, 1]
        return df
    df['hostname'] = np.repeat(sys.name, pairL)
    df['pnumflag'] = 0
    # Double check that there are the correct number of planets vs entries!
    if len(sys.index) != np.array(sys.sy_pnum)[0]:
        df['pnumflag'] = 1

    ## Pair information.
    # Sorting the periods so that the longest period is first.
    sys = sys.sort_values('pl_orbper', ascending=False)
    for mmrcol, syscol in infodict.items():
        pairarr = fpairs(sys[syscol]).astype(str)
        df[mmrcol] = np.array(['-'.join(pair) for pair in pairarr])

    ## Handling the periods!
    periodp = np.array(fpairs(sys.pl_orbper), dtype=float)        
    noNan = np.all(~np.isnan(periodp), axis=1)
    df['nanflag'] = (~noNan).astype(int)
    if np.all(~noNan): return df
    periodp = periodp[noNan]
    
    ## Looping through the periods to determine their MMR 
    ppairL = np.sum(noNan) # the number of pairs fufilling our criteria
    mmr1, mmr2 = np.zeros(ppairL, dtype=object), np.zeros(ppairL, dtype=object)
    deltas1, deltas2 = np.zeros((ppairL,)), np.zeros((ppairL,))
    for i, pp in enumerate(periodp):
        deltas1[i], mmr1[i] = pairMMR(*pp, FIRSTORDER)
        deltas2[i], mmr2[i] = pairMMR(*pp, SECONDORDER)
    df.loc[noNan, MMRCOLS[-4:]] = np.transpose([mmr1, deltas1, mmr2, deltas2])

    return df

## MMR statistics
def rmDeltaNan(df, mask=False):
    '''
    If mask, find a mask which removes the null values for the
    deltas, i.e. the planet pairs which did not go through the deltaEq.
    E.g. mmrdf = allmmrdf[rmDeltaNan(allmmrdf)]
    '''
    nullm = np.all(df[MMRCOLS[-4:]].isnull(), axis=1)
    if mask: return ~nullm
    return df[~nullm]

def whichdelta(df, returndf=True):
    '''
    We want to find the minimum delta between first and second order.
    Writes an extra column into the dataframe to tell you whether
    the first or second order is the best approximation. 
    '''
    delta = np.array([df.firstdelta, df.seconddelta])
    minidex = np.argmin(np.abs(delta), axis=0)
    minidex += 1

    if not returndf:
        return minidex

    df = df.copy()
    df['which'] = minidex
    return df

def rmOverlap(m, overlap, checkm):
    '''
    overlap: where the condition for both first-order and second-order
             MMR is satisfied.
    checkm: depending on whether this is checking for first-order or second-
            order, this will check if the minimum delta value is for the first-
            order MMR or the second-order MMR.
    '''
    notm = np.logical_and(overlap, checkm)
    m[notm] = False
    return m

def mmrmasks(df, overlap=False):
    '''
    Obtains the mask of the dataframe where there is first order or
    second order mmr. If overlap is not allowed, then we should treat
    the values which are both in first and second order by comparing
    the minimum delta values!
    '''

    m1 = np.logical_and(np.array(df.firstdelta, dtype=float) >= -0.015, np.array(df.firstdelta, dtype=float) < 0.03)
    m2 = np.logical_and(np.array(df.seconddelta, dtype=float) >= -0.015, np.array(df.seconddelta, dtype=float) < 0.015)
    if overlap:
        return m1, m2

    overlap = np.logical_and(m1, m2)  
    try: # in case the df does not have the 'which' column.
        deltacheck = np.array(df.which)
    except AttributeError: 
        deltacheck = whichdelta(df, returndf=False)
    m1 = rmOverlap(m1, overlap, deltacheck == 2)
    m2 = rmOverlap(m2, overlap, deltacheck == 1)
    
    return m1, m2

def multiMMRdf(df):
    '''
    Returns the dataframe of those pairs which are in systems with > 1
    pairs in first-order or second-order MMR, but note this will include
    pairs that are not in MMR!
    '''
    m1, m2 = mmrmasks(df)
    mmrpresent = np.logical_or(m1, m2)
    sysN, sysC = np.unique(df[mmrpresent].hostname, return_counts=True)
    multim = np.isin(df.hostname, sysN[sysC > 1])
    return df[multim]

def separateOrders(df):
    '''
    We separate the first order from the second order as to 
    properly replicate Dai et.al. This is necessary because there
    are those in first order resonance which have minimum deltas as 2.
    '''
    m1, m2 = mmrmasks(df)

    try: np.array(df.which) # to check for required column
    except AttributeError: 
        df = whichdelta(df)
        
    order1 = np.logical_or(m1, df.which == 1)
    order2 = np.logical_or(m2, df.which == 2)

    return order1, order2

def getSysDiff(df, mismatchosts=gen.MISHOSTS, starname=False):
    '''
    The discrepancy between planet-hosting stars and systems
    is due to systems with > 1 gravitationally bound stars for which
    multiple host planets. For our multi-planet systems, these consist
    of only binary star systems, and should only be 55 Cnc and 55 Cnc B.
    '''
    mismatchmask = np.all(np.isin(mismatchosts, df.hostname), axis=1)
    sysstardiff = np.sum(mismatchmask)
    probstars = ['-'.join(stararr) for stararr in mismatchosts[mismatchmask]]
    if starname: return sysstardiff, probstars
    return sysstardiff

def mmrStats(alldf, verbose=True, mismatchosts=gen.MISHOSTS):
    '''
    
    '''
    df = rmDeltaNan(alldf)
    df = whichdelta(df)
    starN, sysC = np.unique(df.hostname, return_counts=True)
    sysstardiff, probstarn = getSysDiff(df, starname=True)
    if sysstardiff != 0:
        sysnum = len(starN) - sysstardiff
        print(f'Because both {', '.join(probstarn)} are within the DataFrame, the number of systems', \
              'is not interchangeable with the number of planet-hosting stars.')
    else: sysnum = len(starN)
    
    mmrm = [mmrmasks(df, True), mmrmasks(df, False)]
    ## Getting the statistics for those pairs which are in resonance. 
    mmrdf = df[np.logical_or(*mmrm[-1])] # using overlap = False.
    mstarnum = len(np.unique(mmrdf.hostname))
    multidf = multiMMRdf(df)
    multistarnum = len(np.unique(multidf.hostname))

    if verbose:
        print(f'{len(df)} possible pairs catalogued {len(starN)} planet-hosting stars', \
              f'within {sysnum} systems. If overlapping, {np.sum(mmrm[0][0])} first-order', \
              f'versus {np.sum(mmrm[1][0])} first-order; {np.sum(mmrm[0][1])} second-order', \
              f'to {np.sum(mmrm[1][1])} respectively. Such resonant pairs are hosted by', \
              f'{mstarnum} stars within {mstarnum-getSysDiff(mmrdf, starname=False)} systems.', \
              f'For multi-resonant systems: {multistarnum} stars host >= 2 MMR pairs in', \
              f'{multistarnum-getSysDiff(multidf, starname=False)} systems with {len(multidf)}', \
              f'total pairs. The # stars with >= 2 pairs is {len(starN[sysC > 1])}.')

    index = ['Pairs', 'Host Stars', 'Systems', '1st Order', '2nd Order', 'Overlap 1st Order', \
             'Overlap 2nd Order', 'MMR Host Stars', 'MMR Systems', 'Multi-Resonant Stars', \
             'Multi-Resonant Systems', 'Multi-Resonant Pairs', '>= 2 Pair Host Stars']
    data = [len(df), len(starN), sysnum, np.sum(mmrm[1][0]), np.sum(mmrm[1][1]), \
            np.sum(mmrm[0][0]), np.sum(mmrm[0][1]), mstarnum, mstarnum-getSysDiff(mmrdf, starname=False), \
            multistarnum, multistarnum-getSysDiff(multidf, starname=False), len(multidf), \
            len(starN[sysC > 1])]

    nandf = alldf[np.any(alldf.isnull(), axis=1)]
    index.append('Nan Values'), data.append(len(nandf))
    ## To handle the nan-cases if they exist:~
    if nandf.empty:
        if verbose: print(f'No nan values in the original dataframe!')
        return pd.Series(data=data, index=index).astype(int), None    
    othercols = df.columns[~np.isin(df.columns, np.append(np.array(['which']), MMRCOLS))]
    if not othercols.empty: # Checking the cases for if there are other than the default columns.
        othernullm = np.any(df[othercols].isnull(), axis=1)
        print(f'There are the other columns {othercols}, for which {np.sum(othernullm)} rows have nan value in them.')
    
    # Checking the missing pairs.
    mmrnan = alldf[~rmDeltaNan(alldf, mask=True)]
    index = index + ['Nan Delta', 'Mismatch Planet Count', 'Nan Periods']
    data = data + [len(mmrnan), np.sum(alldf.pnumflag > 0), np.sum(alldf.nanflag > 0)]
    if verbose:
        print(f'However, there are {len(mmrnan)} pairs for which deltas were not calculated for, this may be due to \
            \n 1. The catalogue planet count and catalogue number of entries misalignment: {np.sum(alldf.pnumflag > 0)} \
            \n 2. Missing periods from the entries: {np.sum(alldf.nanflag > 0)} pairs with one or more missing periods \
            \n 3. Using a mask to filter out certain planets if they do not match a given criteria.')
    
    return pd.Series(data=data, index=index).astype(int), nandf

## Replicating the plots of Dai et al. 2024
ABSTHRES = 0.05
ORDERPARS = {'colour':['tab:blue', 'tab:green'], 'limit':[0.03, 0.015]}
ALLHIST = dict(alpha=0.35, edgecolor='white', label='All Pairs')
MULTIHIST = dict(alpha=0.85, edgecolor='white', label='Multi Resonant Pairs')
DAI1ST = dict(color='tab:blue', marker='o', ms=8, capsize=3, \
              label='1st Order', annotatept=10)
DAI2nd = dict(color='tab:red', marker='*', ms=12, capsize=3, \
              label='2nd Order', annotatept=6)

def mmrBar(data, m, ax, annotate, **kwargs):
    '''
    Plotting the fractional line for either 1st or 2nd order,
    with annotations as 
    '''
    labels, Nres = np.unique(data[m], return_counts=True)
    ratios = [int(ratio.split(':')[0])/int(ratio.split(':')[1]) for ratio in labels]
    Ntot = len(data)
    errors = np.sqrt(Nres*(Ntot - Nres)/(Ntot**3))
    if annotate: annotatept = kwargs.pop('annotatept', 10)
    ax.errorbar(ratios, (Nres/len(data))*100, yerr=errors*100, **kwargs)
    
    for i, (ratio, label) in enumerate(zip(ratios, labels)):
        ax.axvline(x=ratio, color='slategrey', ls=':', lw=2, alpha=0.75)
        if annotate:
            if label == '5:4': place = (ratio, annotatept-2)
            else: place = (ratio, annotatept)
            ax.annotate(label, place, color=kwargs['color'], fontsize=12, \
                        horizontalalignment='center')
    return ax, (ratios, labels)

def mmrbarplot(df, ax, annotate=False, o1dict=DAI1ST, o2dict=DAI2nd):
    '''
    '''
    m1, m2 = mmrmasks(df)
    ax, first = mmrBar(df.firstMMR, m1, ax, annotate, **o1dict)
    ax, second = mmrBar(df.secondMMR, m2, ax, annotate, **o2dict)
    return ax, first, second

def mmrbarSingleWrapper(df, ax, yformat=True, xformat=True):
    '''
    '''
    ax, first, second = mmrbarplot(df, ax, annotate=True, o1dict=DAI1ST, o2dict=DAI2nd)
    ax.set_xticks(np.append(first[0], second[0]), labels=np.append(first[1], \
                            second[1]), rotation='vertical', fontsize=13)
    ax.set_yticks(np.arange(0, 14, 2))
    ax.set_ylabel('Fraction (%)')
    ax.set_xlabel('Period Ratio')
    ax.legend()
    
    return ax

def mmrHist(data, ax, bins, **kwargs):
    '''
    Plotting the histogram.
    bins (array or int): if int, then generate bins.
    '''
    if isinstance(bins, int): 
        bins = np.linspace(-ABSTHRES, ABSTHRES, bins+1)
    delta = np.array(data, dtype=float)
    mmrH, mmrBins = np.histogram(delta, bins=bins)
    ax.hist(mmrBins[:-1], bins, weights=mmrH, **kwargs)
    
    return ax, mmrBins

def mmrHistPlot(delta, mult, ax, binNum, order):
    '''
    The plot of the histogram, complete with axis labels.
    '''
    ax.axvline(x=-0.015, color='tab:red', ls=':', lw=3, label='Threshold')
    ax, bins = mmrHist(delta, ax, binNum, color=ORDERPARS['colour'][order], **ALLHIST)
    ax, _ = mmrHist(mult, ax, bins, color=ORDERPARS['colour'][order], **MULTIHIST)
    ax.axvline(x=ORDERPARS['limit'][order], color='tab:red', ls=':', lw=3)
    ax.set_ylabel('Number of Pairs')
    ax.set_xlabel(r'$\Delta$')
    ax.legend(fontsize=13)

    return ax

def mmrHistFig(axs, df, binNum=40, which=True):
    '''
    The figure of first and second order MMR!
    '''
    
    multires = multiMMRdf(df)
    if which:
        o1, o2 = separateOrders(df)
        om1, om2 = separateOrders(multires)
        axs[0] = mmrHistPlot(df.firstdelta[o1], multires.firstdelta[om1], axs[0], binNum, 0)
        axs[1] = mmrHistPlot(df.seconddelta[o2], multires.seconddelta[om2], axs[1], binNum, 1)
    else:
        axs[0] = mmrHistPlot(df.firstdelta, multires.firstdelta, axs[0], binNum, 0)
        axs[1] = mmrHistPlot(df.seconddelta, multires.seconddelta, axs[1], binNum, 1)
        
    axs[0].set_title('1st Order MMR')
    axs[1].set_title('2nd Order MMR')
    
    return axs

## To categorise by mass or by radius. 
def columnsJE(df, m, col, const):
    '''
    If the value in units of Jupiter is not available,
    then we multiply the value in Earth units with the ratio.
    '''
    jcol = col + 'j'
    ecol = col + 'e'
    par = np.zeros(len(df))
    par[m[jcol]] = df.loc[m[jcol], jcol]
    earthm = np.logical_and(~m[jcol], m[ecol])
    par[earthm] = df.loc[earthm, ecol] * const

    return par

def getrad(df):
    '''
    Check for where there are no radius values. Then, write in
    those already in units of Jupiter's radius. Convert those
    remaining with units in Earth's radius into Jupiter's radius. All those
    that remain must be in the form of Rp/R*. However, we must consider
    if the stellar radius is present.

    df: the DataFrame we will get the radii for.
    '''
    notnanrad = np.any(df[RADCOLS[2:]].notnull(), axis=1)
    notnanrprs = np.all(df[RADCOLS[:2]].notnull(), axis=1)
    notnanm = np.logical_or(notnanrad, notnanrprs)
    raddf = df.loc[notnanm, RADCOLS]
    radm = ~raddf.isnull()
    
    rads = columnsJE(raddf, radm, 'pl_rad', gen.RERJ)
    zerom = rads == 0
    rpsr = raddf.pl_ratror[zerom] * raddf.st_rad[zerom]
    rads[zerom] = rpsr * gen.RSRJ

    return np.array(df[notnanm].pl_name), rads

def getmass(df):
    '''
    Check for where there are no mass values. Then, write in
    those already in units of Jupiter's mass. Convert those
    remaining with units in Earth's mass into Jupiter's mass.
    Repeat the procedure for the mass*sini/sini values; there are two
    cases for m*sini, where i exists and where i does not exist.

    df: the DataFrame we will get the mass for.
    '''
    
    notnanm = np.any(df[MASSCOLS[1:]], axis=1)
    massdf = df.loc[notnanm, MASSCOLS]
    massm = ~massdf.isnull()

    # We prioritise those with proper masses.
    masses = columnsJE(massdf, massm, 'pl_mass', gen.MEMJ)
    # Then, we handle those with mass*sin(i)/sin(i)
    tempmass = columnsJE(massdf, massm, 'pl_cmass', gen.MEMJ)
    zerom = masses == 0
    masses[zerom] = tempmass[zerom]
    # Finally, we can handle mass*sin(i)
    masssini = columnsJE(massdf, massm, 'pl_msini', gen.MEMJ)
    ## those which have sini values:
    zerom = np.logical_and(masses == 0, ~np.isnan(massdf.pl_orbincl))
    sini = np.sin(massdf.pl_orbincl[zerom] * 180/np.pi)
    masses[zerom] = masssini[zerom]/np.abs(sini)
    ## those which do not have sinivalues
    zerom = masses == 0
    masses[zerom] = masssini[zerom]

    return np.array(df[notnanm].pl_name), masses, zerom

def getParDF(data, cutoff, parcols, cat, syscols=CATECOLS):
    '''
    data should contain: pln, par and if it is mass, then the siniflag.
    '''
    pln = data[0]
    df = pd.DataFrame(columns=np.append(syscols, parcols))

    # Organising the system-related statistics.
    df = cat.loc[np.isin(cat.pl_name, pln), syscols]
    _, dfdex, pldex = np.intersect1d(df.pl_name, pln, return_indices=True)
    df[syscols] = np.array(df)[dfdex]
    # the parameter-based statistics.
    flag = np.array(data[1, pldex] >= cutoff).astype(int)
    df[parcols] = np.transpose(np.array([flag, *data[1:, pldex]]))
    
    return df

def getCateDF(df, mthres=gen.MTHRES, rthres=gen.RTHRES, \
              verbose=True, returnmissing=False):
    '''
    getCateDF(rvcat, returnmissing=True, verbose=True)
    '''
    rpln, radj = getrad(df)
    mpln, massj, siniflag = getmass(df)

    ## Getting the dataframe which categories the original dataframe (df)
    raddf = getParDF(np.array([rpln, radj]), rthres, RADCATECOLS, df)
    massdf = getParDF(np.array([mpln, massj, siniflag]), mthres, MASSCATECOLS, df)
    ## Adding the two dataframes together!
    catedf = raddf.merge(massdf, on=CATECOLS, how='outer')
    if verbose:
        print(f'There are {len(raddf)} available radii and {len(massdf)} masses with an overlap of', \
              f'{np.intersect1d(rpln, mpln).shape[0]} entries; merged dataframe has {len(catedf)} planets.')

    if not returnmissing: return catedf

    # Adding back in the missing entries.
    missednames = np.setdiff1d(df.pl_name, np.append(rpln, mpln))
    if verbose: print(f'{len(missednames)} planets without mass/radius: {', '.join(missednames)}!')
    missdf = df.loc[np.isin(df.pl_name, missednames), CATECOLS]
    return catedf.merge(missdf, on=CATECOLS, how='outer')

def planetparstats(cat, returndf=True, **catepars):
    '''
    how many nan values do we have?
    how many mass values do we have? how many sin(i) values do we have? have many radius values do we have?
    how many pass the mass test? how many pass the radius test? how many pass both?

    distribution plot of mass, radius and orbital period if we have greater than 100 values.

    also we want to add a jove flag column
    '''
    df = getCateDF(cat, **catepars)
    bflag = np.full((len(df),), 'N/A', dtype=object)
    bflag[np.logical_and(df.mjoveflag == 0, df.rjoveflag == 0)] = 'nonJove'
    bflag[np.logical_and(df.mjoveflag == 1, df.rjoveflag == 1)] = 'Jovian'
    bflag[np.logical_and(df.mjoveflag == 1, df.rjoveflag == 0)] = 'Superdense'
    bflag[np.logical_and(df.mjoveflag == 0, df.rjoveflag == 1)] = 'Superpuff'
    bothnum = np.sum(bflag != 'N/A')
    df['planettype'] = bflag
    joveflag = np.zeros((len(df),))
    joveflag[np.logical_or(df.mjoveflag == 1, df.rjoveflag == 1)] = 1
    df['joveflag'] = joveflag.astype(int)
    
    mdf = df.notnull()
    masscount, radcount = np.sum(mdf.mass), np.sum(mdf.radius)     
    jovetypes = [f'{np.sum(df.planettype == t)}/{bothnum}' for t \
                 in ['nonJove', 'Superpuff', 'Superdense', 'Jovian']]
    
    index = ['Periods', 'Masses', 'Sin(i) Flag', 'Radius', 'Jovian by Mass', 'Jovian by Radius', \
             'Non-Jovian by Both', 'Superpuff', 'Superdense', 'Jovian by Both']
    data = [f'{np.sum(mdf.pl_orbper)}/{len(df)}', f'{masscount}/{len(df)}', \
            f'{np.sum(df.siniflag)}/{masscount}', f'{radcount}/{len(df)}', \
            np.sum(df.mjoveflag), np.sum(df.rjoveflag), *jovetypes]
    
    if not returndf: return pd.Series(data=data, index=index)
    return pd.Series(data=data, index=index), df