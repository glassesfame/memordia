## MAKING ANIMATED PLOTS

class AnimationParameters(DiagramParameters):
    
    def __init__(self):
        # Inherit from the Diagram Parameters class
        super().__init__() 

        self.path = ''
        self.filename = 'diagram.mp4'
        self.skipframes = 25
        self.startframe = 'all'
        self.interval = 50 # in units of milliseconds
        self.timethreshold = 100 # in units of years
        self.tstampx = 0.1
        self.tstampy = 0.9

        self.psifacmin = None
        self.psifacmax = None
        self.psifaclims = (self.psifacmin, self.psifacmax)

    def plotlimits(self, deltarr, psifacarr, ax=None):
        '''
        '''
        self.deltamin = min(np.min(deltarr)-self.buffer, self.deltamin)
        self.deltamax = max(np.max(deltarr)+self.buffer, self.deltamax)
        self.psifacmin = np.min(psifacarr)-self.buffer
        self.psifacmax = np.max(psifacarr)+self.buffer
        
        if ax is not None:
            pfaxmin, pfaxmax = ax.get_ylim()
            self.psifacmin = min(self.psifacmin, pfaxmin)
            self.psifacmax = max(self.psifacmax, pfaxmax)

    def gettstampp(self, ax):
        '''
        '''
        xtup = ax.get_xlim()
        ytup = ax.get_ylim()
        
        return xtup[0] + self.tstampx*(xtup[-1]-xtup[0]), \
        ytup[0] + self.tstampy*(ytup[-1]-ytup[0])

def preparegif(dflist):
    '''
    '''
    plt.close('all') # checking everything else is closed!
    checklen = [len(df) for df in dflist]
    if len(set(checklen)) != 1:
        print('Mismatched DataFrame lengths!! No!!')
        return None, [], []

    deltarr = np.zeros((len(dflist), checklen[0]))
    psifacarr = np.zeros((len(dflist), checklen[0]))
    times = np.zeros((len(dflist), checklen[0]))
    
    for i, df in enumerate(dflist):
        deltarr[i], psifacarr[i] = np.array(df.delta), np.array(df.psifac)
        times[i] = np.array(df.index)
    times = np.unique(times)
    if times.ndim > 1:
        print('Times between DataFrames do not match!')
        return None, deltarr, psifacarr

    return times, deltarr, psifacarr

def longevolutiongif(dflist, pairvallist, gpars, sysname):
    '''
    The time arrays must match!!
    dflist (list): a list of dataframes
    '''

    times, deltarr, psifacarr = preparegif(dflist)
    if times is None: return print('Exiting GIF functionality.')

    fig, ax, _ = scatterdiagram(pairvallist, gpars) # No labels for original scatter
    gpars.plotlimits(deltarr, psifacarr, ax)
    tstampp = gpars.gettstampp(ax)
    timestamp = ax.text(*tstampp, f'Time: {np.round(times[0], 3)} Yrs', zorder=5)
    traject = []
    
    for i, sysn in enumerate(sysname): # setting up line parameters
        linedict = gpars.line(gpars.colours[i], sysn)
        traject.append(ax.plot(deltarr[i, 0], psifacarr[i, 0], **linedict)[0])

    def init():
        for i, traj in enumerate(traject):
            traj.set_data(deltarr[i], psifacarr[i])

    def update(frame):
        
        modframe = frame * gpars.skipframes
        timestamp.set_text(np.round(times[modframe], 3))

        trelative = np.abs(times - gpars.timethreshold)
        restep = np.where(trelative == np.min(trelative))[0][0]
        for i, traj in enumerate(traject):
            if times[modframe] < gpars.timethreshold:
                traj.set_data(deltarr[i, :modframe], psifacarr[i, :modframe])
            else:
                start = modframe - restep
                traj.set_data(deltarr[i, start:modframe], psifacarr[i, start:modframe])

        return traj

    fnum = len(times)//gpars.skipframes
    anim = animation.FuncAnimation(fig=fig, func=update, frames=fnum, \
                                   init_func=init, interval=gpars.interval)
    anim.save(gpars.filename)

def shortevolutiongif(dflist, pairvallist, gpars, sysname):
    '''
    '''
    
    times, deltarr, psifacarr = preparegif(dflist)
    if times is None: return print('Exiting GIF functionality.')

    fig, ax, scatters = scatterdiagram(pairvallist, gpars, sysname) # No labels for original scatter
    gpars.plotlimits(deltarr, psifacarr, ax)
    tstampp = gpars.gettstampp(ax)
    timestamp = ax.text(*tstampp, f'Time: {np.round(times[0], 3)} Yrs', zorder=5)
    # if gpars.skipframes > 20: gpars.skipframes = 15

    def update(frame):
        modframe = frame * gpars.skipframes
        timestamp.set_text(np.round(times[modframe], 3))
        for i, scat in enumerate(scatters):
            scat.set_offsets((deltarr[i, modframe], psifacarr[i, modframe]))
            
        return scatters

    fnum = len(times)//gpars.skipframes
    anim = animation.FuncAnimation(fig=fig, func=update, frames=fnum, interval=gpars.interval)
    anim.save(gpars.filename)