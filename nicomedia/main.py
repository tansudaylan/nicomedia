import os, sys, json
import numpy as np
import pandas as pd
from tqdm import tqdm

import astropy
import astroquery
import scipy

import tdpy
from tdpy import summgene
import chalcedon


def retr_radieins_inft( \
                       # velocity dispersion [km/s]
                       dispvelo, \

                      ):
    '''
    Calculate the Einstein radius for a source position at infinity
    '''
    """
            :param deflector_dict: deflector properties
            :param v_sigma: velocity dispersion in km/s
            :return: Einstein radius in arc-seconds
            """
    if v_sigma is None:
        if deflector_dict is None:
            raise ValueError("Either deflector_dict or v_sigma must be provided")
        else:
            v_sigma = deflector_dict['vel_disp']

    theta_E_infinity = 4 * np.pi * (dispvelo / 3e5)**2 * (180. / np.pi * 3600.)
    return theta_E_infinity


def quer_mast(request):
    '''
    Query the MAST catalog
    '''
    
    from urllib.parse import quote as urlencode
    import http.client as httplib 

    server='mast.stsci.edu'

    # Grab Python Version
    version = '.'.join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {'Content-type': 'application/x-www-form-urlencoded',
               'Accept': 'text/plain',
               'User-agent':'python-requests/'+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request('POST', '/api/v0/invoke', 'request='+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head, content


def xmat_tici(listtici):
    '''
    Crossmatch with the TESS Input Catalog (TIC)'''
    if len(listtici) == 0:
        raise Exception('')
    
    # make sure the input is a python list of strings
    if isinstance(listtici[0], str):
        if isinstance(listtici, np.ndarray):
            listtici = list(listtici)
    else:
        if isinstance(listtici, list):
            listtici = np.array(listtici)
        if isinstance(listtici, np.ndarray):
            listtici = listtici.astype(str)
        listtici = list(listtici)

    request = {'service':'Mast.Catalogs.Filtered.Tic', 'format':'json', 'params':{'columns':'rad, mass', \
                                                                        'filters':[{'paramName':'ID', 'values':listtici}]}}
    headers, outString = quer_mast(request)
    dictquertemp = json.loads(outString)['data']
    dictquer = dict()
    dictquer['rad'] = [dictquertemp['rad'], 'R_{\oplus}']
    dictquer['mass'] = [dictquertemp['mass'], 'M_{\oplus}']
    
    return dictquer


def retr_toiitici(tici, typeverb=1, dicttoii=None):
    
    if dicttoii is None:
        dicttoii = retr_dicttoii()
    
    toii = None
    indx = np.where(dicttoii['TICID'] == tici)[0]
    if indx.size > 0:
        toii = int(str(dicttoii['TOIID'][indx[0]]).split('.')[0])
        if typeverb > 0:
            print('Matched the input TIC ID with TOI-%d.' % toii)
    
    return toii


def retr_dictpopltic8( \
                      typepopl, \
                      numbsyst=None, \
                      # type of verbosity
                      ## -1: absolutely no text
                      ##  0: no text output except critical warnings
                      ##  1: minimal description of the execution
                      ##  2: detailed description of the execution
                      typeverb=1, \
                      # Boolean flag to turn on diagnostic mode
                      booldiag=True, \
                     ):
    '''
    Get a dictionary of the sources in the TIC8 with the fields in the TIC8.
    
    Keyword arguments   
        typepopl: type of the population
            'ticiprmshcon': TIC targets with contamination larger than
            'ticim060': TIC targets brighter than TESS magnitude 6.0
            'ticim100': TIC targets brighter than TESS magnitude 10.0
            'ticim140': TIC targets brighter than TESS magnitude 14.0
            'targtess_prms_ffimm060': TESS targets observed during PM on FFIs brighter than mag 6.0
            'targtess_prms_2min': 2-minute TESS targets obtained by merging the SPOC 2-min bulk downloads

    Returns a dictionary with keys:
        rasc: RA
        decl: declination
        tmag: TESS magnitude
        radistar: radius of the star
        massstar: mass of the star
    '''
    
    if typeverb > 0:
        print('Retrieving a dictionary of TIC8 for population %s...' % typepopl)
    
    if typepopl.startswith('CTL'):
        strgtypepoplsplt = typepopl.split('_')
        
        if booldiag:
            if len(strgtypepoplsplt) < 2:
                print('')
                print('')
                print('')
                print('typepopl')
                print(typepopl)
                print('strgtypepoplsplt')
                print(strgtypepoplsplt)
                raise Exception('len(strgtypepoplsplt) < 2')
        
        strgtimetess = strgtypepoplsplt[1]
        if strgtimetess == 'yr01':
            listtsec = np.arange(1, 14) # [1-13]
        elif strgtimetess == 'S1':
            listtsec = np.arange(1, 2) # [1]
        elif strgtimetess == 'yr02':
            listtsec = np.arange(13, 27) # [13-26]
        elif strgtimetess == 'yr03':
            listtsec = np.arange(27, 40) # [27-39]
        elif strgtimetess == 'yr04':
            listtsec = np.arange(40, 56) # [40-55]
        elif strgtimetess == 'yr05':
            listtsec = np.arange(56, 70) # [56-69]
        elif strgtimetess == 'sc01':
            listtsec = np.arange(1, 2)
        elif strgtimetess == 'prms':
            listtsec = np.arange(1, 27)
        elif strgtimetess == 'e1ms':
            listtsec = np.arange(27, 56)
        elif strgtimetess == 'e2ms':
            listtsec = np.arange(56, 70)
        else:
            print('typepopl')
            print(typepopl)
            raise Exception('')
        numbtsec = len(listtsec)
        indxtsec = np.arange(numbtsec)

    pathlistticidata = os.environ['EPHESOS_DATA_PATH'] + '/data/listticidata/'
    os.system('mkdir -p %s' % pathlistticidata)

    path = pathlistticidata + 'listticidata_%s.csv' % typepopl
    if not os.path.exists(path):
        
        # dictionary of strings that will be keys of the output dictionary
        dictstrg = dict()
        dictstrg['ID'] = 'TICID'
        dictstrg['ra'] = 'rascstar'
        dictstrg['dec'] = 'declstar'
        dictstrg['Tmag'] = 'tmag'
        dictstrg['rad'] = 'radistar'
        dictstrg['mass'] = 'massstar'
        dictstrg['Teff'] = 'tmptstar'
        dictstrg['logg'] = 'loggstar'
        dictstrg['MH'] = 'metastar'
        liststrg = list(dictstrg.keys())
        
        print('typepopl')
        print(typepopl)
        if typepopl.startswith('CTL'):
            
            if strgtypepoplsplt[2] == '20sc':
                strgurll = '_20s_'
                labltemp = '20-second'
            elif strgtypepoplsplt[2] == '2min':
                strgurll = '_'
                labltemp = '2-minute'
            else:
                print('')
                print('')
                print('')
                print('typepopl')
                print(typepopl)
                raise Exception('')

            dictquer = dict()
            listtici = []
            for o in indxtsec:
                if typepopl.endswith('bulk'):
                    pathtess = os.environ['TESS_DATA_PATH'] + '/data/lcur/sector-%02d' % listtsec[o]
                    listnamefile = fnmatch.filter(os.listdir(pathtess), '*.fits')
                    listticitsec = []
                    for namefile in listnamefile:
                        listticitsec.append(str(int(namefile.split('-')[2])))
                    listticitsec = np.array(listticitsec)
                else:
                    urlo = 'https://tess.mit.edu/wp-content/uploads/all_targets%sS%03d_v1.csv' % (strgurll, listtsec[o])
                    print('urlo')
                    print(urlo)
                    c = pd.read_csv(urlo, header=5)
                    listticitsec = c['TICID'].values
                    listticitsec = listticitsec.astype(str)
                numbtargtsec = listticitsec.size
                
                if typeverb > 0:
                    print('%d observed %s targets in Sector %d...' % (numbtargtsec, labltemp, listtsec[o]))
                
                if numbtargtsec > 0:
                    dictquertemp = xmat_tici(listticitsec)
                
                if o == 0:
                    dictquerinte = dict()
                    for name in dictstrg.keys():
                        dictquerinte[dictstrg[name]] = [[] for o in indxtsec]
                
                for name in dictstrg.keys():
                    for k in range(len(dictquertemp)):
                        dictquerinte[dictstrg[name]][o].append(dictquertemp[k][name])

            print('Concatenating arrays from different sectors...')
            for name in dictstrg.keys():
                dictquer[dictstrg[name]] = np.concatenate(dictquerinte[dictstrg[name]])
            
            u, indxuniq, cnts = np.unique(dictquer['TICID'], return_index=True, return_counts=True)
            for name in dictstrg.keys():
                dictquer[dictstrg[name]] = dictquer[dictstrg[name]][indxuniq]
            dictquer['numbtsec'] = cnts

        elif typepopl.startswith('TIC'):
            if typepopl.endswith('hcon'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass, contratio', \
                                                             'filters':[{'paramName':'contratio', 'values':[{"min":10., "max":1e3}]}]}}
            elif typepopl.endswith('m060'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass', \
                                                             'filters':[{'paramName':'Tmag', 'values':[{"min":-100., "max":6.0}]}]}}
            elif typepopl.endswith('m100'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass', \
                                                             'filters':[{'paramName':'Tmag', 'values':[{"min":-100., "max":10.0}]}]}}
            elif typepopl.endswith('m140'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass', \
                                                             'filters':[{'paramName':'Tmag', 'values':[{"min":-100., "max":14.0}]}]}}
            else:
                raise Exception('')
    
            # temp
            ## this can be alternatively done as
            #catalog_data = Catalogs.query_criteria(catalog='Tic', Vmag=[0., 5], objtype='STAR')
            #print(catalog_data.keys())
            #x_value = catalog_data['ID']
            #y_value = catalog_data['HIP']

            headers, outString = quer_mast(request)
            listdictquer = json.loads(outString)['data']
            if typeverb > 0:
                print('%d matches...' % len(listdictquer))
            dictquer = dict()
            for name in listdictquer[0].keys():
                if name == 'ID':
                    namedict = 'TICID'
                if name == 'Tmag':
                    namedict = 'tmag'
                if name == 'ra':
                    namedict = 'rascstar'
                if name == 'dec':
                    namedict = 'declstar'
                if name == 'rad':
                    namedict = 'radistar'
                if name == 'mass':
                    namedict = 'massstar'
                dictquer[namedict] = np.empty(len(listdictquer))
                for k in range(len(listdictquer)):
                    dictquer[namedict][k] = listdictquer[k][name]
        else:
            raise Exception('Unrecognized population name: %s' % typepopl)
        
        numbtarg = dictquer['radistar'].size
            
        if typeverb > 0:
            print('%d targets...' % numbtarg)
            print('Writing to %s...' % path)
        #columns = ['TICID', 'radi', 'mass']
        dictquertemp = dict()
        listnamefeat = list(dictquertemp.keys())
        listhead = []
        for namefeat in listnamefeat:
            strghead = namefeat
            if dictquer[namefeat][1] != '':
                strghead += '[%s]' % dictquer[namefeat][1]
            listhead.append(strghead)
            dictquertemp[namefeat] = dictquer[namefeat][0]
        pd.DataFrame.from_dict(dictquertemp).to_csv(path, header=listhead, index=False)#, columns=columns)
    else:
        if typeverb > 0:
            print('Reading from %s...' % path)
        dictquer = pd.read_csv(path, nrows=numbsyst).to_dict(orient='list')
        
        for name in dictquer.keys():
            dictquer[name] = np.array(dictquer[name])

    #if gdat.typedata == 'simuinje':
    #    indx = np.where((~np.isfinite(gdat.dictfeat['true']['ssys']['massstar'])) | (~np.isfinite(gdat.dictfeat['true']['ssys']['radistar'])))[0]
    #    gdat.dictfeat['true']['ssys']['radistar'][indx] = 1.
    #    gdat.dictfeat['true']['ssys']['massstar'][indx] = 1.
    #    gdat.dictfeat['true']['totl']['tmag'] = dicttic8['tmag']
        
    # check if dictquer is properly defined, whose leaves should be a list of two items (of values and labels, respectively)
    if booldiag:
        for namefeat in dictquer:
            if len(dictquer[namefeat]) != 2 or len(dictquer[namefeat][1]) > 0 and not isinstance(dictquer[namefeat][1][1], str):
                print('namefeat')
                print(namefeat)
                print('dictquer[namefeat]')
                print(dictquer[namefeat])
                print('')
                print('')
                print('')
                raise Exception('dictquer is not properly defined.')

    return dictquer


def retr_objtlinefade(x, y, colr='black', initalph=1., alphfinl=0.):
    
    colr = get_color(colr)
    cdict = {'red':   ((0.,colr[0],colr[0]),(1.,colr[0],colr[0])),
             'green': ((0.,colr[1],colr[1]),(1.,colr[1],colr[1])),
             'blue':  ((0.,colr[2],colr[2]),(1.,colr[2],colr[2])),
             'alpha': ((0.,initalph, initalph), (1., alphfinl, alphfinl))}
    
    Npts = len(x)
    if len(y) != Npts:
        raise AttributeError("x and y must have same dimension.")
   
    segments = np.zeros((Npts-1,2,2))
    segments[0][0] = [x[0], y[0]]
    for i in range(1,Npts-1):
        pt = [x[i], y[i]]
        segments[i-1][1] = pt
        segments[i][0] = pt 
    segments[-1][1] = [x[-1], y[-1]]

    individual_cm = mpl.colors.LinearSegmentedColormap('indv1', cdict)
    lc = mpl.collections.LineCollection(segments, cmap=individual_cm)
    lc.set_array(np.linspace(0.,1.,len(segments)))
    
    return lc


def retr_liststrgcomp(numbcomp):
    
    liststrgcomp = np.array(['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'])[:numbcomp]

    return liststrgcomp


def retr_listcolrcomp(numbcomp):
    
    listcolrcomp = np.array(['magenta', 'orange', 'red', 'green', 'purple', 'cyan'])[:numbcomp]

    return listcolrcomp


def retr_dictfluxband(tmptstar, liststrgband, gdatfluxband=None, pathvisutarg=None, strgtarg=None, typefileplot='png', \
              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1, \
              ):
            
    print('Calculating the band fluxes of the target...')
    
    dictfluxband = dict()
    
    boolconsgdat = gdatfluxband is None
    
    if boolconsgdat:
        # construct global object
        gdatfluxband = tdpy.gdatstrt()

        # define spectral grid (this may need to be taken outside the if statements for other purposes)
        minmwlen = 0.1
        maxmwlen = 10.
        numbwlen = 1000
        gdatfluxband.binswlen, gdatfluxband.midpwlen, _, _, _ = tdpy.retr_axis(minmwlen, maxmwlen, numbpntsgrid=numbwlen, scalpara='logt')
    
    specbbod = tdpy.retr_specbbod(tmptstar, gdatfluxband.midpwlen)
    
    gdatfluxband.numbband = len(liststrgband)
    gdatfluxband.indxband = np.arange(gdatfluxband.numbband)
    
    if pathvisutarg is not None:
        path = pathvisutarg + 'fluxband_%s' % strgtarg
        for pl in gdatfluxband.indxband:
            path += '_' + liststrgband[pl]
        path += '.%s' % typefileplot
        if not os.path.exists(path):
            figr, axis = plt.subplots(figsize=(8, 5))
            axistwin = axis.twinx()
            axis.plot(gdatfluxband.midpwlen, gdatfluxband.fluxbandsyst, color=colrdraw, ls='-', ms=1, rasterized=True)
    
    gdatfluxband.functran = [[] for pl in gdatfluxband.indxband]
    for pl in gdatfluxband.indxband:
        
        if boolconsgdat:
        
            if liststrgband[pl].startswith('LSST'):
                import sncosmo
                strgband = liststrgband[pl][4]
                functraninit = sncosmo.get_bandpass(f'lsst%s' % strgband)
                print('')
                print('')
                print('')
                print('')
                print('')
                print('gdatfluxband.midpwlen')
                summgene(gdatfluxband.midpwlen)
                print('functraninit.wave')
                summgene(functraninit.wave)
                print('functraninit.trans')
                summgene(functraninit.trans)
                print('specbbod')
                summgene(specbbod)
                gdatfluxband.functran[pl] = np.interp(gdatfluxband.midpwlen, 1e-4 * functraninit.wave, functraninit.trans)
                print('gdatfluxband.functran[pl]')
                summgene(gdatfluxband.functran[pl])
            else:
                gdatfluxband.functran[pl] = np.zeros_like(gdatfluxband.midpwlen)
                
                if liststrgband[pl] == 'ULTRASAT':
                    indxwlen = np.where((gdatfluxband.midpwlen < 0.29) & (gdatfluxband.midpwlen > 0.23))[0]
                    gdatfluxband.functran[pl][indxwlen] = 1.
                elif liststrgband[pl] == 'TESS-GEO-UV':
                    indxwlen = np.where((gdatfluxband.midpwlen < 0.29) & (gdatfluxband.midpwlen > 0.23))[0]
                    gdatfluxband.functran[pl][indxwlen] = 1.
                elif liststrgband[pl] == 'TESS-GEO-VIS':
                    indxwlen = np.where((gdatfluxband.midpwlen < 0.7) & (gdatfluxband.midpwlen > 0.4))[0]
                    gdatfluxband.functran[pl][indxwlen] = 1.
                elif liststrgband[pl] == 'TESS':
                    indxwlen = np.where((gdatfluxband.midpwlen < 1.) & (gdatfluxband.midpwlen > 0.6))[0]
                    gdatfluxband.functran[pl][indxwlen] = 1.
                
                #elif liststrgband[pl].startswith('LSST'):
                #    print('strgband')
                #    print(strgband)
                #    print(liststrgband[pl])
                #    if strgband == 'u':
                #        indxwlen = np.where((gdatfluxband.midpwlen < 0.4) & (gdatfluxband.midpwlen > 0.3))[0]
                #    if strgband == 'b':
                #        indxwlen = np.where((gdatfluxband.midpwlen < 0.4) & (gdatfluxband.midpwlen > 0.3))[0]
                #    if strgband == 'r':
                #        indxwlen = np.where((gdatfluxband.midpwlen < 1.) & (gdatfluxband.midpwlen > 0.6))[0]
                #    gdatfluxband.functran[pl][indxwlen][:] = 1.
                
                elif liststrgband[pl] == 'Bolometric':
                    gdatfluxband.functran[pl][indxwlen][:] = 1.
                else:
                    print('')
                    print('')
                    print('')
                    print('liststrgband[pl]')
                    print(liststrgband[pl])
                    raise Exception('Undefined liststrgband[pl]')
        
            if pathvisutarg is not None and not os.path.exists(path) and liststrgband[pl] != 'Bolometric':
                axistwin.plot(gdatfluxband.midpwlen, gdatfluxband[liststrgband[pl]], ls='-', ms=1, rasterized=True, label=liststrgband[pl])
    
        print('specbbod')
        summgene(specbbod)
        print('gdatfluxband.functran[pl]')
        summgene(gdatfluxband.functran[pl])
        dictfluxband[liststrgband[pl]] = np.trapz(specbbod * gdatfluxband.functran[pl], x=gdatfluxband.midpwlen)
    
    if boolconsgdat:
        if pathvisutarg is not None:
            axis.set_xscale('log')
            axis.set_xlabel('Wavelength [$\mu$m]')
            axis.set_ylabel('Spectrum')
            axistwin.legend()
            axistwin.set_ylabel('Transfer function')
            plt.subplots_adjust(hspace=0.)
            if gdat.typeverb > 0:
                print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
    
    return dictfluxband, gdatfluxband


def plot_orbt( \
              # size of the figure
              sizefigr=(8, 8), \
              listcolrcomp=None, \
              liststrgcomp=None, \
              boolsingside=True, \
              ## file type of the plot
              typefileplot='png', \

              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1, \
             ):

    if listcolrcomp is None:
        listcolrcomp = retr_listcolrcomp(numbcomp)

    if liststrgcomp is None:
        liststrgcomp = retr_liststrgcomp(numbcomp)
    
    ## scale factor for the star
    factstar = 5.
    
    ## scale factor for the planets
    factplan = 20.
    
    # maximum y-axis value
    maxmyaxi = 0.05

    if boolinclmerc:
        # Mercury
        smaxmerc = 0.387 # [AU]
        radicompmerc = 0.3829 # [R_E]
    
    time = np.arange(0., 30., 2. / 60. / 24.)
    numbtime = time.size
    indxtime = np.arange(numbtime)
   
    if boolanim:
        numbiter = min(500, numbtime)
    else:
        numbiter = 1
    indxiter = np.arange(numbiter)
    
    xposmaxm = smax
    yposmaxm = factpers * xposmaxm
    numbtimequad = 10
    
    # get transit model based on TESS ephemerides
    rratcomp = radicomp / radistar
    
    rflxtranmodl = eval_modl(time, 'PlanetarySystem', pericomp=peri, epocmtracomp=epoc, rsmacomp=rsmacomp, cosicomp=cosi, rratcomp=rratcomp)['rflx'] - 1.
    
    lcur = rflxtranmodl + np.random.randn(numbtime) * 1e-6
    ylimrflx = [np.amin(lcur), np.amax(lcur)]
    
    phas = np.random.rand(numbcomp)[None, :] * 2. * np.pi + 2. * np.pi * time[:, None] / peri[None, :]
    yposelli = yposmaxm[None, :] * np.sin(phas)
    xposelli = xposmaxm[None, :] * np.cos(phas)
    
    # time indices for iterations
    indxtimeiter = np.linspace(0., numbtime - numbtime / numbiter, numbiter).astype(int)
    
    if typevisu.startswith('cart'):
        colrstar = 'k'
        colrface = 'w'
        plt.style.use('default')
    else:
        colrface = 'k'
        colrstar = 'w'
        plt.style.use('dark_background')
    
    if boolanim:
        gdat.cmndmakeanim = 'convert -delay 5'
        listpathtemp = []
    for k in indxiter:
        
        if typeplotlcurposi == 'lowr':
            numbrows = 2
        else:
            numbrows = 1
        figr, axis = plt.subplots(figsize=sizefigr)

        ### lower half of the star
        w1 = mpl.patches.Wedge((0, 0), radistarscal, 180, 360, fc=colrstar, zorder=1, edgecolor=colrstar)
        axis.add_artist(w1)
        
        for jj, j in enumerate(indxcomp):
            xposellishft = np.roll(xposelli[:, j], -indxtimeiter[k])[-numbtimequad:][::-1]
            yposellishft = np.roll(yposelli[:, j], -indxtimeiter[k])[-numbtimequad:][::-1]
        
            # add cartoon-like disks for planets
            if typeplotplan.startswith('colr'):
                colrplan = listcolrcomp[j]
                radi = radicomp[j] / dictfact['rsre'] / dictfact['aurs'] * factplan
                w1 = mpl.patches.Circle((xposelli[indxtimeiter[k], j], yposelli[indxtimeiter[k], j], 0), radius=radi, color=colrplan, zorder=3)
                axis.add_artist(w1)
            
            # add trailing tails to planets
            if typeplotplan.startswith('colrtail'):
                objt = retr_objtlinefade(xposellishft, yposellishft, colr=listcolrcomp[j], initalph=1., alphfinl=0.)
                axis.add_collection(objt)
            
            # add labels to planets
            if typeplotplan == 'colrtaillabl':
                axis.text(.6 + 0.03 * jj, 0.1, liststrgcomp[j], color=listcolrcomp[j], transform=axis.transAxes)
        
        ## upper half of the star
        w1 = mpl.patches.Wedge((0, 0), radistarscal, 0, 180, fc=colrstar, zorder=4, edgecolor=colrstar)
        axis.add_artist(w1)
        
        if boolinclmerc:
            ## add Mercury
            axis.text(.387, 0.01, 'Mercury', color='gray', ha='right')
            radi = radicompmerc / dictfact['rsre'] / dictfact['aurs'] * factplan
            w1 = mpl.patches.Circle((smaxmerc, 0), radius=radi, color='gray')
            axis.add_artist(w1)
        
        # temperature axis
        #axistwin = axis.twiny()
        ##axistwin.set_xlim(axis.get_xlim())
        #xpostemp = axistwin.get_xticks()
        ##axistwin.set_xticks(xpostemp[1:])
        #axistwin.set_xticklabels(['%f' % tmpt for tmpt in listtmpt])
        
        # temperature contours
        #for tmpt in [500., 700,]:
        #    smaj = tmpt
        #    axis.axvline(smaj, ls='--')
        
        axis.get_yaxis().set_visible(False)
        axis.set_aspect('equal')
        
        if boolinclmerc:
            maxmxaxi = max(1.2 * np.amax(smaxcomp), 0.4)
        else:
            maxmxaxi = 1.2 * np.amax(smaxcomp)
        
        if boolsingside:
            minmxaxi = 0.
        else:
            minmxaxi = -maxmxaxi

        axis.set_xlim([minmxaxi, maxmxaxi])
        axis.set_ylim([-maxmyaxi, maxmyaxi])
        axis.set_xlabel('Distance from the star [AU]')
        
        #plt.subplots_adjust()
        #axis.legend()
        
        strgvisu = ''
        
        print('Writing to %s...' % pathtemp)
        plt.savefig(pathtemp)
        plt.close()
        

def retr_dictpoplrvel():
    
    if typeverb > 0:
        print('Reading Sauls Gaia high RV catalog...')
    path = os.environ['TROIA_DATA_PATH'] + '/data/Gaia_high_RV_errors.txt'
    for line in open(path):
        listnamesaul = line[:-1].split('\t')
        break
    if typeverb > 0:
        print('Reading from %s...' % path)
    data = np.loadtxt(path, skiprows=1)
    dictcatl = dict()
    dictcatl['rascstar'] = data[:, 0]
    dictcatl['declstar'] = data[:, 1]
    dictcatl['stdvrvel'] = data[:, -4]
    
    return dictcatl


def retr_dicthostplan(namepopl, \
                      # type of verbosity
                      ## -1: absolutely no text
                      ##  0: no text output except critical warnings
                      ##  1: minimal description of the execution
                      ##  2: detailed description of the execution
                      typeverb=1, \
                      ):
    
    pathephe = os.environ['EPHESOS_DATA_PATH'] + '/'
    path = pathephe + 'data/dicthost%s.csv' % namepopl
    if os.path.exists(path):
        if typeverb > 0:
            print('Reading from %s...' % path)
        dicthost = pd.read_csv(path).to_dict(orient='list')
        for name in dicthost.keys():
            dicthost[name] = np.array(dicthost[name])
        
    else:
        dicthost = dict()
        if namepopl == 'TOIID':
            dictplan = retr_dicttoii()
        else:
            dictplan = retr_dictexar()
        listnamestar = np.unique(dictplan['namestar'])
        dicthost['namestar'] = listnamestar
        numbstar = listnamestar.size
        indxstar = np.arange(numbstar)
        
        listnamefeatstar = ['numbplanstar', 'numbplantranstar', 'radistar', 'massstar']
        listnamefeatcomp = ['epocmtracomp', 'pericomp', 'duratrantotl', 'radicomp', 'masscomp']
        for namefeat in listnamefeatstar:
            dicthost[namefeat] = np.empty(numbstar)
        for namefeat in listnamefeatcomp:
            dicthost[namefeat] = [[] for k in indxstar]
        for k in indxstar:
            indx = np.where(dictplan['namestar'] == listnamestar[k])[0]
            for namefeat in listnamefeatstar:
                dicthost[namefeat][k] = dictplan[namefeat][indx[0]]
            for namefeat in listnamefeatcomp:
                dicthost[namefeat][k] = dictplan[namefeat][indx]
                
        print('Writing to %s...' % path)
        pd.DataFrame.from_dict(dicthost).to_csv(path, index=False)

    return dicthost


def retr_dicttoii(toiitarg=None, boolreplexar=False, \
                  # type of verbosity
                  ## -1: absolutely no text
                  ##  0: no text output except critical warnings
                  ##  1: minimal description of the execution
                  ##  2: detailed description of the execution
                  typeverb=1, strgelem='comp'):
    
    dictfact = tdpy.retr_factconv()
    
    pathephe = os.environ['EPHESOS_DATA_PATH'] + '/'
    pathexof = pathephe + 'data/exofop_toilists.csv'
    if typeverb > 0:
        print('Reading from %s...' % pathexof)
    objtexof = pd.read_csv(pathexof, skiprows=0)
    
    strgradielem = 'radi' + strgelem
    strgstdvradi = 'stdv' + strgradielem
    strgmasselem = 'mass' + strgelem
    strgstdvmass = 'stdv' + strgmasselem
    
    strgperielem = 'peri' + strgelem
    
    dicttoii = {}
    tdpy.setp_dict(dicttoii, 'TOIID', objtexof['TOI'].values)
    numbcomp = dicttoii['TOIID'][0].size
    indxcomp = np.arange(numbcomp)
    toiitargexof = np.empty(numbcomp, dtype=object)
    for k in indxcomp:
        toiitargexof[k] = int(dicttoii['TOIID'][0][k])
        
    if toiitarg is None:
        indxcomp = np.arange(numbcomp)
    else:
        indxcomp = np.where(toiitargexof == toiitarg)[0]
    
    dicttoii['TOIID'][0] = dicttoii['TOIID'][0][indxcomp]
    
    numbcomp = indxcomp.size
    
    if indxcomp.size == 0:
        if typeverb > 0:
            print('The host name, %s, was not found in the ExoFOP TOI Catalog.' % toiitarg)
        return None
    else:
        tdpy.setp_dict(dicttoii, 'namestar', np.empty(numbcomp, dtype=object))
        tdpy.setp_dict(dicttoii, 'nametoii', np.empty(numbcomp, dtype=object))
        for kk, k in enumerate(indxcomp):
            dicttoii['nametoii'][0][kk] = 'TOI-' + str(dicttoii['TOIID'][0][kk])
            dicttoii['namestar'][0][kk] = 'TOI-' + str(dicttoii['TOIID'][0][kk])[:-3]
        
        tdpy.setp_dict(dicttoii, 'depttrancomp', objtexof['Depth (ppm)'].values[indxcomp] * 1e-3, 'ppt')
        tdpy.setp_dict(dicttoii, 'rratcomp', np.sqrt(dicttoii['depttrancomp'][0] * 1e-3))
        tdpy.setp_dict(dicttoii, strgradielem, objtexof['Planet Radius (R_Earth)'][indxcomp].values, 'R$_{\oplus}$')
        tdpy.setp_dict(dicttoii, strgstdvradi, objtexof['Planet Radius (R_Earth) err'][indxcomp].values, 'R$_{\oplus}$')
        
        rascstarstrg = objtexof['RA'][indxcomp].values
        declstarstrg = objtexof['Dec'][indxcomp].values
        tdpy.setp_dict(dicttoii, 'rascstar', np.empty(numbcomp))
        tdpy.setp_dict(dicttoii, 'declstar', np.empty(numbcomp))
        for k in range(dicttoii[strgradielem][0].size):
            objt = astropy.coordinates.SkyCoord('%s %s' % (rascstarstrg[k], declstarstrg[k]), unit=(astropy.units.hourangle, astropy.units.deg))
            dicttoii['rascstar'][0][k] = objt.ra.degree
            dicttoii['declstar'][0][k] = objt.dec.degree

        # a string holding the comments
        tdpy.setp_dict(dicttoii, 'strgcomm', objtexof['Comments'][indxcomp].values)
        
        # transit duration
        tdpy.setp_dict(dicttoii, 'duratrantotl', objtexof['Duration (hours)'].values[indxcomp], 'hours')
        
        # coordinates
        objticrs = astropy.coordinates.SkyCoord(ra=dicttoii['rascstar'][0], dec=dicttoii['declstar'][0], frame='icrs', unit='deg')
        
        ## galactic longitude
        tdpy.setp_dict(dicttoii, 'lgalstar', np.array([objticrs.galactic.l])[0, :], 'degrees')
        
        ## galactic latitude
        tdpy.setp_dict(dicttoii, 'bgalstar', np.array([objticrs.galactic.b])[0, :], 'degrees')
        
        ## ecliptic longitude
        tdpy.setp_dict(dicttoii, 'loecstar', np.array([objticrs.barycentricmeanecliptic.lon.degree])[0, :], 'degrees')
        
        ## ecliptic latitude
        tdpy.setp_dict(dicttoii, 'laecstar', np.array([objticrs.barycentricmeanecliptic.lat.degree])[0, :], 'degrees')

        # SNR
        tdpy.setp_dict(dicttoii, 's2nr', objtexof['Planet SNR'][indxcomp].values)
        
        # amount of follow-up data
        tdpy.setp_dict(dicttoii, 'numbobsvtime', objtexof['Time Series Observations'][indxcomp].values)
        tdpy.setp_dict(dicttoii, 'numbobsvspec', objtexof['Spectroscopy Observations'][indxcomp].values)
        tdpy.setp_dict(dicttoii, 'numbobsvimag', objtexof['Imaging Observations'][indxcomp].values)
        
        # alert year
        tdpy.setp_dict(dicttoii, 'yearaler', objtexof['Date TOI Alerted (UTC)'][indxcomp].values)
        for k in range(len(dicttoii['yearaler'][0])):
            dicttoii['yearaler'][0][k] = astropy.time.Time(dicttoii['yearaler'][0][k] + ' 00:00:00.000').decimalyear
        dicttoii['yearaler'][0] = dicttoii['yearaler'][0].astype(float)

        tdpy.setp_dict(dicttoii, 'tsmmacwg', objtexof['TSM'][indxcomp].values)
        tdpy.setp_dict(dicttoii, 'esmmacwg', objtexof['ESM'][indxcomp].values)
    
        tdpy.setp_dict(dicttoii, 'facidisc', np.empty(numbcomp, dtype=object))
        dicttoii['facidisc'][0][:] = 'Transiting Exoplanet Survey Satellite (TESS)'
        
        tdpy.setp_dict(dicttoii, strgperielem, objtexof['Period (days)'][indxcomp].values)
        dicttoii[strgperielem][0][np.where(dicttoii[strgperielem][0] == 0)] = np.nan

        tdpy.setp_dict(dicttoii, 'epocmtra' + strgelem, objtexof['Epoch (BJD)'][indxcomp].values)

        tdpy.setp_dict(dicttoii, 'tmagsyst', objtexof['TESS Mag'][indxcomp].values)
        tdpy.setp_dict(dicttoii, 'stdvtmagsyst', objtexof['TESS Mag err'][indxcomp].values)

        # transit duty cycle
        tdpy.setp_dict(dicttoii, 'dcyc', dicttoii['duratrantotl'][0] / dicttoii[strgperielem][0] / 24.)
        
        liststrgfeatstartici = ['massstar', 'vmagsyst', 'jmagsyst', 'hmagsyst', 'kmagsyst', 'distsyst', 'metastar', 'radistar', 'tmptstar', 'loggstar']
        liststrgfeatstarticiinhe = ['mass', 'Vmag', 'Jmag', 'Hmag', 'Kmag', 'd', 'MH', 'rad', 'Teff', 'logg']
        
        numbstrgfeatstartici = len(liststrgfeatstartici)
        indxstrgfeatstartici = np.arange(numbstrgfeatstartici)

        for strgfeat in liststrgfeatstartici:
            tdpy.setp_dict(dicttoii, strgfeat, np.zeros(numbcomp))
            tdpy.setp_dict(dicttoii, 'stdv' + strgfeat, np.zeros(numbcomp))
        
        ## crossmatch with TIC
        print('Retrieving TIC columns of TOI hosts...')
        tdpy.setp_dict(dicttoii, 'TICID', objtexof['TIC ID'][indxcomp].values)
        listticiuniq = np.unique(dicttoii['TICID'][0].astype(str))
        request = {'service':'Mast.Catalogs.Filtered.Tic', 'format':'json', 'params':{'columns':"*", \
                                                              'filters':[{'paramName':'ID', 'values':list(listticiuniq)}]}}
        headers, outString = quer_mast(request)
        listdictquer = json.loads(outString)['data']
        for k in range(len(listdictquer)):
            indxtemp = np.where(dicttoii['TICID'][0] == listdictquer[k]['ID'])[0]
            if indxtemp.size == 0:
                raise Exception('')
            for n in indxstrgfeatstartici:
                dicttoii[liststrgfeatstartici[n]][0][indxtemp] = listdictquer[k][liststrgfeatstarticiinhe[n]]
                dicttoii['stdv' + liststrgfeatstartici[n]][0][indxtemp] = listdictquer[k]['e_' + liststrgfeatstarticiinhe[n]]
        
        tdpy.setp_dict(dicttoii, 'typedisptess', objtexof['TESS Disposition'][indxcomp].values)
        tdpy.setp_dict(dicttoii, 'boolfpos', objtexof['TFOPWG Disposition'][indxcomp].values == 'FP')
        
        # augment
        tdpy.setp_dict(dicttoii, 'numb%sstar' % strgelem, np.zeros(numbcomp))
        boolfrst = np.zeros(numbcomp, dtype=bool)
        for kk, k in enumerate(indxcomp):
            indxcompthis = np.where(dicttoii['namestar'][0][kk] == dicttoii['namestar'][0])[0]
            if kk == indxcompthis[0]:
                boolfrst[kk] = True
            dicttoii['numb%sstar' % strgelem][0][kk] = indxcompthis.size
        
        tdpy.setp_dict(dicttoii, 'numb%stranstar' % strgelem, dicttoii['numb%sstar' % strgelem][0])
        tdpy.setp_dict(dicttoii, 'lumistar', dicttoii['radistar'][0]**2 * (dicttoii['tmptstar'][0] / 5778.)**4)
        tdpy.setp_dict(dicttoii, 'stdvlumistar', dicttoii['lumistar'][0] * np.sqrt((2 * dicttoii['stdvradistar'][0] / dicttoii['radistar'][0])**2 + \
                                                                                            (4 * dicttoii['stdvtmptstar'][0] / dicttoii['tmptstar'][0])**2))
        
        # predicted mass from radii
        path = pathephe + 'data/exofop_toi_mass_saved.csv'
        if not os.path.exists(path):
            dicttemp = dict()
            dicttemp[strgmasselem] = np.ones_like(dicttoii[strgradielem][0]) + np.nan
            dicttemp[strgstdvmass] = np.ones_like(dicttoii[strgradielem][0]) + np.nan
            
            numbsamppopl = 10
            indx = np.where(np.isfinite(dicttoii[strgradielem][0]))[0]
            for n in tqdm(range(indx.size)):
                k = indx[n]
                meanvarb = dicttoii[strgradielem][0][k]
                stdvvarb = dicttoii['stdvradi' + strgelem][0][k]
                
                # if radius uncertainty is not available, assume that it is small, so the mass uncertainty will be dominated by population uncertainty
                if not np.isfinite(stdvvarb):
                    stdvvarb = 1e-3 * dicttoii[strgradielem][0][k]
                else:
                    stdvvarb = dicttoii['stdvradi' + strgelem][0][k]
                
                # sample from a truncated Gaussian
                listradicomp = tdpy.samp_gaustrun(1000, dicttoii[strgradielem][0][k], stdvvarb, 0., np.inf)
                
                # estimate the mass from samples
                listmassplan = retr_massfromradi(listradicomp)
                
                dicttemp[strgmasselem][k] = np.mean(listmassplan)
                dicttemp[strgstdvmass][k] = np.std(listmassplan)
                
            if typeverb > 0:
                print('Writing to %s...' % path)
            pd.DataFrame.from_dict(dicttemp).to_csv(path, index=False)
        else:
            if typeverb > 0:
                print('Reading from %s...' % path)
            dicttemp = pd.read_csv(path).to_dict(orient='list')
            
            for name in dicttemp:
                dicttemp[name] = np.array(dicttemp[name])
                if toiitarg is not None:
                    dicttemp[name] = dicttemp[name][indxcomp]
        
        tdpy.setp_dict(dicttoii, strgmasselem, dicttemp[strgmasselem])
        tdpy.setp_dict(dicttoii, strgstdvmass, dicttemp[strgstdvmass])
        
        perielem = dicttoii[strgperielem][0]
        masselem = dicttoii[strgmasselem][0]

        rvelsemapred = retr_rvelsema(perielem, dicttoii['massstar'][0], masselem, 90., 0.)
        tdpy.setp_dict(dicttoii, 'rvelsemapred', rvelsemapred, 'degrees')
        
        masstotl = dicttoii['massstar'][0] + dicttoii[strgmasselem][0] / dictfact['msme']
        tdpy.setp_dict(dicttoii, 'masstotl', masstotl)
        
        tdpy.setp_dict(dicttoii, 'smax' + strgelem, retr_smaxkepl(dicttoii[strgperielem][0], dicttoii['masstotl'][0]))
        
        rsmaelem = (dicttoii[strgradielem][0] / dictfact['rsre'] + dicttoii['radistar'][0]) / (dictfact['aurs'] * dicttoii['smax'+strgelem][0])
        tdpy.setp_dict(dicttoii, 'rsma' + strgelem, rsmaelem)
        
        tdpy.setp_dict(dicttoii, 'irra', dicttoii['lumistar'][0] / dicttoii['smax'+strgelem][0]**2)
        
        
        # temp check if factor of 2 is right
        tmptcomp = dicttoii['tmptstar'][0] * np.sqrt(dicttoii['radistar'][0] / dicttoii['smax'+strgelem][0] / 2. / dictfact['aurs'])
        tdpy.setp_dict(dicttoii, 'tmpt' + strgelem, tmptcomp)
        
        # temp check if factor of 2 is right
        stdvtmptcomp = np.sqrt((dicttoii['stdvtmptstar'][0] / dicttoii['tmptstar'][0])**2 + \
                                                        0.5 * (dicttoii['stdvradistar'][0] / dicttoii['radistar'][0])**2) / np.sqrt(2.)
        tdpy.setp_dict(dicttoii, 'stdvtmpt' + strgelem, stdvtmptcomp)
        
        tdpy.setp_dict(dicttoii, 'dens' + strgelem, 5.51 * dicttoii[strgmasselem][0] / dicttoii[strgradielem][0]**3, 'g/cm^3')
        
        tdpy.setp_dict(dicttoii, 'booltran', np.ones(numbcomp, dtype=bool))
        
        tdpy.setp_dict(dicttoii, 'vesc', retr_vesc(dicttoii[strgmasselem][0], dicttoii[strgradielem][0]))
        
        print('temp: vsiistar and projoblq are NaNs')
        tdpy.setp_dict(dicttoii, 'vsiistar', np.full(numbcomp, np.nan))
        tdpy.setp_dict(dicttoii, 'projoblq', np.full(numbcomp, np.nan))
        
        # replace confirmed planet features
        if boolreplexar:
            dictexar = retr_dictexar()
            listdisptess = objtexof['TESS Disposition'][indxcomp].values.astype(str)
            listdisptfop = objtexof['TFOPWG Disposition'][indxcomp].values.astype(str)
            indxexofcpla = np.where((listdisptfop == 'CP') & (listdisptess == 'PC'))[0]
            listticicpla = dicttoii['TICID'][0][indxexofcpla]
            numbticicpla = len(listticicpla)
            indxticicpla = np.arange(numbticicpla)
            for k in indxticicpla:
                indxexartici = np.where((dictexar['TICID'] == int(listticicpla[k])) & \
                                                    (dictexar['facidisc'] == 'Transiting Exoplanet Survey Satellite (TESS)'))[0]
                indxexoftici = np.where(dicttoii['TICID'][0] == int(listticicpla[k]))[0]
                for strg in dictexar.keys():
                    if indxexartici.size > 0:
                        dicttoii[strg][0] = np.delete(dicttoii[strg][0], indxexoftici)
                    dicttoii[strg][0] = np.concatenate((dicttoii[strg][0], dictexar[strg][indxexartici]))

        # derived quantities
        ## photometric noise in the TESS passband
        tdpy.setp_dict(dicttoii, 'noistess', retr_noisphot(dicttoii['tmagsyst'][0], 'TESS'))
        
        ## atmospheric characterization
        # calculate TSM and ESM
        calc_tsmmesmm(dicttoii, strgelem=strgelem)
    
        # turn zero TSM ACWG or ESM ACWG into NaN
        indx = np.where(dicttoii['tsmmacwg'][0] == 0)[0]
        dicttoii['tsmmacwg'][0][indx] = np.nan
        
        indx = np.where(dicttoii['esmmacwg'][0] == 0)[0]
        dicttoii['esmmacwg'][0][indx] = np.nan
        
        # surface gravity of the companion
        tdpy.setp_dict(dicttoii, 'logg' + strgelem, dicttoii[strgmasselem][0] / dicttoii[strgradielem][0]**2)

    return dicttoii


def calc_tsmmesmm(dictpopl, strgelem='comp', boolsamp=False):
    
    if boolsamp:
        numbsamp = 1000
    else:
        numbsamp = 1

    strgradielem = 'radi' + strgelem
    strgmasselem = 'mass' + strgelem
    
    numbcomp = dictpopl[strgmasselem][0].size
    listtsmm = np.empty((numbsamp, numbcomp)) + np.nan
    listesmm = np.empty((numbsamp, numbcomp)) + np.nan
    
    for n in range(numbcomp):
        
        if not np.isfinite(dictpopl['tmpt%s' % strgelem][0][n]):
            continue
        
        if not np.isfinite(dictpopl[strgradielem][0][n]):
            continue
        
        if boolsamp:
            if not np.isfinite(dictpopl['stdvradi' + strgelem][0][n]):
                stdv = dictpopl[strgradielem][0][n]
            else:
                stdv = dictpopl['stdvradi' + strgelem][0][n]
            listradicomp = tdpy.samp_gaustrun(numbsamp, dictpopl[strgradielem][0][n], stdv, 0., np.inf)
            
            listmassplan = tdpy.samp_gaustrun(numbsamp, dictpopl[strgmasselem][0][n], dictpopl['stdvmass' + strgelem][0][n], 0., np.inf)

            if not np.isfinite(dictpopl['stdvtmpt%s' % strgelem][0][n]):
                stdv = dictpopl['tmpt%s' % strgelem][0][n]
            else:
                stdv = dictpopl['stdvtmpt%s' % strgelem][0][n]
            listtmptplan = tdpy.samp_gaustrun(numbsamp, dictpopl['tmpt%s' % strgelem][0][n], stdv, 0., np.inf)
            
            if not np.isfinite(dictpopl['stdvradistar'][0][n]):
                stdv = dictpopl['radistar'][0][n]
            else:
                stdv = dictpopl['stdvradistar'][0][n]
            listradistar = tdpy.samp_gaustrun(numbsamp, dictpopl['radistar'][0][n], stdv, 0., np.inf)
            
            listkmagsyst = tdpy.icdf_gaus(np.random.rand(numbsamp), dictpopl['kmagsyst'][0][n], dictpopl['stdvkmagsyst'][0][n])
            listjmagsyst = tdpy.icdf_gaus(np.random.rand(numbsamp), dictpopl['jmagsyst'][0][n], dictpopl['stdvjmagsyst'][0][n])
            listtmptstar = tdpy.samp_gaustrun(numbsamp, dictpopl['tmptstar'][0][n], dictpopl['stdvtmptstar'][0][n], 0., np.inf)
        
        else:
            listradicomp = dictpopl[strgradielem][0][None, n]
            listtmptplan = dictpopl['tmpt%s' % strgelem][0][None, n]
            listmassplan = dictpopl[strgmasselem][0][None, n]
            listradistar = dictpopl['radistar'][0][None, n]
            listkmagsyst = dictpopl['kmagsyst'][0][None, n]
            listjmagsyst = dictpopl['jmagsyst'][0][None, n]
            listtmptstar = dictpopl['tmptstar'][0][None, n]
        
        # TSM
        listtsmm[:, n] = retr_tsmm(listradicomp, listtmptplan, listmassplan, listradistar, listjmagsyst)

        # ESM
        listesmm[:, n] = retr_esmm(listtmptplan, listtmptstar, listradicomp, listradistar, listkmagsyst)
        
        #if (listesmm[:, n] < 1e-10).any():
        #    print('listradicomp')
        #    summgene(listradicomp)
        #    print('listtmptplan')
        #    summgene(listtmptplan)
        #    print('listmassplan')
        #    summgene(listmassplan)
        #    print('listradistar')
        #    summgene(listradistar)
        #    print('listkmagsyst')
        #    summgene(listkmagsyst)
        #    print('listjmagsyst')
        #    summgene(listjmagsyst)
        #    print('listtmptstar')
        #    summgene(listtmptstar)
        #    print('listesmm[:, n]')
        #    summgene(listesmm[:, n])
        #    raise Exception('')
    tdpy.setp_dict(dictpopl, 'tsmm', np.nanmedian(listtsmm, 0))
    tdpy.setp_dict(dictpopl, 'stdvtsmm', np.nanstd(listtsmm, 0))
    tdpy.setp_dict(dictpopl, 'esmm', np.nanmedian(listesmm, 0))
    tdpy.setp_dict(dictpopl, 'stdvesmm', np.nanstd(listesmm, 0))
    
    #print('listesmm')
    #summgene(listesmm)
    #print('dictpopl[tsmm]')
    #summgene(dictpopl['tsmm'])
    #print('dictpopl[esmm]')
    #summgene(dictpopl['esmm'])
    #print('dictpopl[stdvtsmm]')
    #summgene(dictpopl['stdvtsmm'])
    #print('dictpopl[stdvesmm]')
    #summgene(dictpopl['stdvesmm'])
    #raise Exception('')


def retr_reso(listperi, maxmordr=10):
    
    if np.where(listperi == 0)[0].size > 0:
        raise Exception('')

    numbsamp = listperi.shape[0]
    numbcomp = listperi.shape[1]
    indxcomp = np.arange(numbcomp)
    listratiperi = np.zeros((numbsamp, numbcomp, numbcomp))
    intgreso = np.zeros((numbcomp, numbcomp, 2))
    for j in indxcomp:
        for jj in indxcomp:
            if j >= jj:
                continue
                
            rati = listperi[:, j] / listperi[:, jj]
            #print('listperi')
            #print(listperi)
            #print('rati')
            #print(rati)
            if rati < 1:
                listratiperi[:, j, jj] = 1. / rati
            else:
                listratiperi[:, j, jj] = rati

            minmdiff = 1e100
            for a in range(1, maxmordr):
                for aa in range(1, maxmordr):
                    diff = abs(float(a) / aa - listratiperi[:, j, jj])
                    if np.mean(diff) < minmdiff:
                        minmdiff = np.mean(diff)
                        minmreso = a, aa
            intgreso[j, jj, :] = minmreso
            #print('minmdiff') 
            #print(minmdiff)
            #print('minmreso')
            #print(minmreso)
            #print
    
    return intgreso, listratiperi


def retr_dilu(tmpttarg, tmptcomp, strgwlentype='tess'):
    
    if strgwlentype != 'tess':
        raise Exception('')
    else:
        binswlen = np.linspace(0.6, 1.)
    meanwlen = (binswlen[1:] + binswlen[:-1]) / 2.
    diffwlen = (binswlen[1:] - binswlen[:-1]) / 2.
    
    fluxtarg = tdpy.retr_specbbod(tmpttarg, meanwlen)
    fluxtarg = np.sum(diffwlen * fluxtarg)
    
    fluxcomp = tdpy.retr_specbbod(tmptcomp, meanwlen)
    fluxcomp = np.sum(diffwlen * fluxcomp)
    
    dilu = 1. - fluxtarg / (fluxtarg + fluxcomp)
    
    return dilu


def retr_toiifstr():
    '''
    Return the TOI IDs that have been alerted by the FaintStar project
    '''
    
    dicttoii = retr_dicttoii(toiitarg=None, boolreplexar=False, \
                             # type of verbosity
                             ## -1: absolutely no text
                             ##  0: no text output except critical warnings
                             ##  1: minimal description of the execution
                             ##  2: detailed description of the execution
                             typeverb=1, \
                             )
    listtoiifstr = []
    for k in range(len(dicttoii['strgcomm'])):
        if isinstance(dicttoii['strgcomm'][k], str) and 'found in faint-star QLP search' in dicttoii['strgcomm'][k]:
            listtoiifstr.append(dicttoii['nametoii'][k][4:])
    listtoiifstr = np.array(listtoiifstr)

    return listtoiifstr
    

def retr_radifrommass( \
                      # list of planet masses in units of Earth mass
                      listmassplan, \
                      # type of radius-mass model
                      strgtype='mine', \
                      ):
    '''
    Estimate planetary radii from samples of masses.
    '''
    
    if len(listmassplan) == 0:
        raise Exception('')

    if strgtype == 'mine':
        # interpolate masses
        listradi = np.empty_like(listmassplan)
        
        indx = np.where(listmassplan < 2.)[0]
        listradi[indx] = listmassplan[indx]**0.28
        
        indx = np.where((listmassplan > 2.) & (listmassplan < 130.))[0]
        listradi[indx] = 5. * (listmassplan[indx] / 20.)**(-0.59)
        
        indx = np.where((listmassplan > 130.) & (listmassplan < 2.66e4))[0]
        listradi[indx] = 10. * (listmassplan[indx] / 1e5)**(-0.04)
        
        indx = np.where(listmassplan > 2.66e4)[0]
        listradi[indx] = 20. * (listmassplan[indx] / 5e4)**0.88
    
    return listradi


def retr_massfromradi( \
                      # list of planet radius in units of Earth radius
                      listradicomp, \
                      # type of radius-mass model
                      strgtype='mine', \
                      
                      # type of verbosity
                      ## -1: absolutely no text
                      ##  0: no text output except critical warnings
                      ##  1: minimal description of the execution
                      ##  2: detailed description of the execution
                      typeverb=1, \
                      ):
    '''
    Estimate planetary mass from samples of radii.
    '''
    
    if len(listradicomp) == 0:
        raise Exception('')


    if strgtype == 'mine':
        # get interpolation data
        path = os.environ['EPHESOS_DATA_PATH'] + '/data/massfromradi.csv'
        if os.path.exists(path):
            if typeverb > 0:
                print('Reading from %s...' % path)
            arry = np.loadtxt(path)
        else:
            # features of confirmed exoplanets
            dictpopl = dict()
            dictpopl['totl'] = retr_dictexar()
            ## planets with good measured radii and masses
            #indx = []
            #for n  in range(dictpopl['totl'][strgstrgrefrmasselem].size):
            #    if not ('Calculated Value' in dictpopl['totl'][strgstrgrefrmasselem][n] or \
            #            'Calculated Value' in dictpopl['totl']['strgrefrradicomp'][n]):
            #        indx.append(n)
            #indxmeas = np.array(indx)
            #indxgood = np.where(dictpopl['totl']['stdvmasscomp'] / dictpopl['totl']['stdvmasscomp'] > 5.)[0]
            #indx = np.setdiff1d(indxmeas, indxgood)
            #retr_subp(dictpopl, dictnumbsamp, dictindxsamp, 'totl', 'gmas', indxgood)
            
            minmradi = np.nanmin(dictpopl['totl']['radicomp'])
            maxmradi = np.nanmax(dictpopl['totl']['radicomp'])
            binsradi = np.linspace(minmradi, 24., 15)
            meanradi = (binsradi[1:] + binsradi[:-1]) / 2.
            arry = np.empty((meanradi.size, 5))
            arry[:, 0] = meanradi
            for l in range(meanradi.size):
                indx = np.where((dictpopl['totl']['radicomp'] > binsradi[l]) & (dictpopl['totl']['radicomp'] < binsradi[l+1]) & \
                                                                                            (dictpopl['totl']['masscomp'] / dictpopl['totl']['stdvmasscomp'] > 5.))[0]
                arry[l, 1] = np.nanmedian(dictpopl['totl']['masscomp'][indx])
                arry[l, 2] = np.nanstd(dictpopl['totl']['masscomp'][indx])
                arry[l, 3] = np.nanmedian(dictpopl['totl']['densplan'][indx])
                arry[l, 4] = np.nanstd(dictpopl['totl']['densplan'][indx])
            
            print('Writing to %s...' % path)
            np.savetxt(path, arry, fmt='%8.4g')

        # interpolate masses
        listmass = np.interp(listradicomp, arry[:, 0], arry[:, 1])
        liststdvmass = np.interp(listradicomp, arry[:, 0], arry[:, 2])
    
    if strgtype == 'Wolfgang2016':
        # (Wolgang+2016 Table 1)
        listmass = (2.7 * (listradicomp * 11.2)**1.3 + np.random.randn(listradicomp.size) * 1.9) / 317.907
        listmass = np.maximum(listmass, np.zeros_like(listmass))
    
    return listmass


def retr_tmptplandayynigh(tmptirra, epsi):
    '''
    Estimate the dayside and nightside temperatures [K] of a planet given its irradiation temperature in K and recirculation efficiency.
    '''
    
    tmptdayy = tmptirra * (2. / 3. - 5. / 12. * epsi)**.25
    tmptnigh = tmptirra * (epsi / 4.)**.25
    
    return tmptdayy, tmptnigh


def retr_esmm(tmptplanequi, tmptstar, radicomp, radistar, kmag):
    
    tmptplanirra = tmptplanequi
    tmptplandayy, tmptplannigh = retr_tmptplandayynigh(tmptplanirra, 0.1)
    esmm = 1.1e3 * tdpy.util.retr_specbbod(tmptplandayy, 7.5) / tdpy.util.retr_specbbod(tmptstar, 7.5) * (radicomp / radistar)*2 * 10**(-kmag / 5.)

    return esmm


def retr_tsmm(radicomp, tmptplan, massplan, radistar, jmag):
    
    tsmm = 1.53 / 1.2 * radicomp**3 * tmptplan / massplan / radistar**2 * 10**(-jmag / 5.)
    
    return tsmm


def retr_scalheig(tmptplan, massplan, radicomp):
    
    # tied to Jupier's scale height for H/He at 110 K   
    scalheig = 27. * (tmptplan / 160.) / (massplan / radicomp**2) / 71398. # [R_J]

    return scalheig


def retr_pntszero(strginst):

    if strginst == 'TESS':
        pntszero = 20.4

    elif strginst == 'LSSTuband':
        pntszero = 20.4
    elif strginst == 'LSSTgband':
        pntszero = 20.4
    elif strginst == 'LSSTrband':
        pntszero = 20.4
    elif strginst == 'LSSTiband':
        pntszero = 20.4
    elif strginst == 'LSSTzband':
        pntszero = 20.4
    elif strginst == 'LSSTyband':
        pntszero = 20.4
    
    return pntszero


def retr_magtfromflux(flux, strginst):
    
    pntszero = retr_pntszero(strginst)

    magt = -2.5 * np.log10(flux) + pntszero
    
    #mlikmagttemp = 10**((mlikmagttemp - 20.424) / (-2.5))
    #stdvmagttemp = mlikmagttemp * stdvmagttemp / 1.09
    #gdat.stdvmagtrefr = 1.09 * gdat.stdvrefrrflx[o] / gdat.refrrflx[o]
    
    return magt


def retr_fluxfrommagt(dmag, strginst, stdvmagt=None):
    
    pntszero = retr_pntszero(strginst)
    
    flux = 10**(-(magt - pntszero) / 2.5)

    if stdvmagt is not None:
        stdvrflx = np.log(10.) / 2.5 * rflx * stdvdmag
        return flux, stdvflux
    else:
        return flux


def retr_rflxfromdmag(dmag, stdvdmag=None):
    
    rflx = 10**(-dmag / 2.5)

    if stdvdmag is not None:
        stdvrflx = np.log(10.) / 2.5 * rflx * stdvdmag
        return rflx, stdvrflx
    else:
        return rflx


def retr_dictexar( \
                  strgexar=None, \
                  
                  # type of verbosity
                  ## -1: absolutely no text
                  ##  0: no text output except critical warnings
                  ##  1: minimal description of the execution
                  ##  2: detailed description of the execution
                  typeverb=1, \
                  
                  strgelem='comp', \
                 ):
    
    strgradielem = 'radi' + strgelem
    strgstdvradi = 'stdv' + strgradielem
    strgmasselem = 'mass' + strgelem
    strgstdvmass = 'stdv' + strgmasselem
    
    strgstrgrefrradielem = 'strgrefrradi' + strgelem
    strgstrgrefrmasselem = 'strgrefrmass' + strgelem
    
    # variables for which both the quantity and its uncertainty will be retrieved
    liststrgstdv = ['radistar', 'massstar', 'tmptstar', 'loggstar', strgradielem, strgmasselem, 'tmpt'+strgelem, 'tagestar', \
                    'vmagsyst', 'jmagsyst', 'hmagsyst', 'kmagsyst', 'tmagsyst', 'metastar', 'distsyst', 'lumistar', \
                   ]
        
    strgnumbelemstar = 'numb%sstar' % strgelem
    strgnumbelemtranstar = 'numb%stranstar' % strgelem
    # get NASA Exoplanet Archive data
    path = os.environ['EPHESOS_DATA_PATH'] + '/data/PSCompPars_2023.10.12_16.58.05.csv'
    if typeverb > 0:
        print('Reading from %s...' % path)
    objtexar = pd.read_csv(path, skiprows=318)
    if strgexar is None:
        indx = np.arange(objtexar['hostname'].size)
        #indx = np.where(objtexar['default_flag'].values == 1)[0]
    else:
        indx = np.where(objtexar['hostname'] == strgexar)[0]
        #indx = np.where((objtexar['hostname'] == strgexar) & (objtexar['default_flag'].values == 1))[0]
    
    dictfact = tdpy.retr_factconv()
    
    if indx.size == 0:
        print('The target name, %s, was not found in the NASA Exoplanet Archive composite table.' % strgexar)
        return None
    else:
        dictexar = {}

        listnamefeat = ['namestar', 'nameplan', 'TICID', 'rascstar', 'declstar', 'TOIID', 'methdisc', 'eccecomp', 'facidisc', 'yeardisc', \
                        'irra', 'pericomp', 'smaxcomp', 'epocmtracomp', 'cosicomp', 'duratrantotl', 'booltran', 'strgprovmass', strgstrgrefrradielem, strgstrgrefrmasselem, \
                        'vesc', 'masstotl', 'densplan', 'vsiistar', 'projoblq', 'boolcibp', strgnumbelemstar, strgnumbelemtranstar, 'dcyc', 'lgalstar', 'bgalstar', \
                        'loecstar', 'laecstar', 'rratcomp', 'rsmacomp', 'esmm', 'tsmm', 'loggcomp', 'depttrancomp', \
                        'esmm', 'tsmm', 'stdvesmm', 'stdvtsmm', \
                        ]
        for name in liststrgstdv:
            listnamefeat += [name]
            listnamefeat += ['stdv' + name]
        
        for name in listnamefeat:
            dictexar[name] = [[], '']
        
        dictexar['namestar'][0] = objtexar['hostname'][indx].values
        dictexar['nameplan'][0] = objtexar['pl_name'][indx].values
        
        numbplanexar = len(dictexar['nameplan'][0])
        indxplanexar = np.arange(numbplanexar)

        listticitemp = objtexar['tic_id'][indx].values
        dictexar['TICID'][0] = np.empty(numbplanexar, dtype=int)
        for k in indxplanexar:
            if isinstance(listticitemp[k], str):
                dictexar['TICID'][0][k] = listticitemp[k][4:]
            else:
                dictexar['TICID'][0][k] = 0
        
        dictexar['rascstar'][0] = objtexar['ra'][indx].values
        dictexar['declstar'][0] = objtexar['dec'][indx].values
        
        # err1 have positive values or zero
        # err2 have negative values or zero
        
        dictexar['TOIID'][0] = np.empty(numbplanexar, dtype=object)
        
        # discovery method
        dictexar['methdisc'][0] = objtexar['discoverymethod'][indx].values
        
        # discovery facility
        dictexar['facidisc'][0] = objtexar['disc_facility'][indx].values
        
        # discovery year
        dictexar['yeardisc'][0] = objtexar['disc_year'][indx].values
        
        dictexar['irra'][0] = objtexar['pl_insol'][indx].values
        dictexar['irra'][0][np.where(dictexar['irra'][0] <= 0.)] = np.nan
        
        dictexar['pericomp'][0] = objtexar['pl_orbper'][indx].values # [days]
        
        dictexar['smaxcomp'][0] = objtexar['pl_orbsmax'][indx].values # [AU]
        
        # eccentricity
        dictexar['eccecomp'][0] = objtexar['pl_orbeccen'][indx].values
        
        dictexar['epocmtracomp'][0] = objtexar['pl_tranmid'][indx].values # [BJD]
        dictexar['cosicomp'][0] = np.cos(objtexar['pl_orbincl'][indx].values / 180. * np.pi)
        dictexar['duratrantotl'][0] = objtexar['pl_trandur'][indx].values # [hour]
        dictexar['depttrancomp'][0] = 10. * objtexar['pl_trandep'][indx].values # ppt
        
        dictexar['booltran'][0] = objtexar['tran_flag'][indx].values.astype(bool)
        
        # mass provenance
        dictexar['strgprovmass'][0] = objtexar['pl_bmassprov'][indx].values

        # radius reference
        dictexar[strgstrgrefrradielem][0] = objtexar['pl_rade_reflink'][indx].values
        for a in range(dictexar[strgstrgrefrradielem][0].size):
            if isinstance(dictexar[strgstrgrefrradielem][0][a], float) and not np.isfinite(dictexar[strgstrgrefrradielem][0][a]):
                dictexar[strgstrgrefrradielem][0][a] = ''
        
        # mass reference
        dictexar[strgstrgrefrmasselem][0] = objtexar['pl_bmasse_reflink'][indx].values
        for a in range(dictexar[strgstrgrefrmasselem][0].size):
            if isinstance(dictexar[strgstrgrefrmasselem][0][a], float) and not np.isfinite(dictexar[strgstrgrefrmasselem][0][a]):
                dictexar[strgstrgrefrmasselem][0][a] = ''
        
        for strg in liststrgstdv:
            strgvarbexar = None
            if strg.endswith('syst'):
                strgvarbexar = 'sy_'
                if strg[:-4].endswith('mag'):
                    strgvarbexar += '%smag' % strg[0]
                if strg[:-4] == 'dist':
                    strgvarbexar += 'dist'
            if strg.endswith('star'):
                strgvarbexar = 'st_'
                if strg[:-4] == 'logg':
                    strgvarbexar += 'logg'
                if strg[:-4] == 'tage':
                    strgvarbexar += 'age'
                if strg[:-4] == 'meta':
                    strgvarbexar += 'met'
                if strg[:-4] == 'radi':
                    strgvarbexar += 'rad'
                if strg[:-4] == 'mass':
                    strgvarbexar += 'mass'
                if strg[:-4] == 'tmpt':
                    strgvarbexar += 'teff'
                if strg[:-4] == 'lumi':
                    strgvarbexar += 'lum'
            if strg.endswith('plan') or strg.endswith(strgelem):
                strgvarbexar = 'pl_'
                if strg[:-4].endswith('mag'):
                    strgvarbexar += '%smag' % strg[0]
                if strg[:-4] == 'tmpt':
                    strgvarbexar += 'eqt'
                if strg[:-4] == 'radi':
                    strgvarbexar += 'rade'
                if strg[:-4] == 'mass':
                    strgvarbexar += 'bmasse'
            if strgvarbexar is None:
                print('strg')
                print(strg)
                raise Exception('')
            
            stdv = (objtexar['%serr1' % strgvarbexar][indx].values - objtexar['%serr2' % strgvarbexar][indx].values) / 2.
            
            dictexar[strg][0] = objtexar[strgvarbexar][indx].values
            dictexar['stdv%s' % strg][0] = stdv
            if strg == strgradielem:
                dictexar[strg][1] = '$R_\oplus$'
                dictexar['stdv%s' % strg][1] = '$R_\oplus$'
            elif strg == strgmasselem:
                dictexar[strg][1] = '$M_\oplus$'
                dictexar['stdv%s' % strg][1] = '$M_\oplus$'
       
        #dictexar['fxuvpred'] = 
        dictexar['vesc'][0] = retr_vesc(dictexar[strgmasselem][0], dictexar[strgradielem][0])
        dictexar['masstotl'][0] = dictexar['massstar'][0] + dictexar[strgmasselem][0] / dictfact['msme']
        
        dictexar['densplan'][0] = objtexar['pl_dens'][indx].values # [g/cm3]
        dictexar['vsiistar'][0] = objtexar['st_vsin'][indx].values # [km/s]
        dictexar['projoblq'][0] = objtexar['pl_projobliq'][indx].values # [deg]
        
        # Boolean flag indicating if the planet is part of a circumbinary planetary system
        dictexar['boolcibp'][0] = objtexar['cb_flag'][indx].values == 1
        
        dictexar[strgnumbelemstar][0] = np.empty(numbplanexar)
        dictexar[strgnumbelemtranstar][0] = np.empty(numbplanexar, dtype=int)
        boolfrst = np.zeros(numbplanexar, dtype=bool)
        #dictexar['booltrantotl'][0] = np.empty(numbplanexar, dtype=bool)
        for k, namestar in enumerate(dictexar['namestar'][0]):
            indxexarstar = np.where(namestar == dictexar['namestar'][0])[0]
            if k == indxexarstar[0]:
                boolfrst[k] = True
            dictexar['numb%sstar' % strgelem][0][k] = indxexarstar.size
            indxexarstartran = np.where((namestar == dictexar['namestar'][0]) & dictexar['booltran'][0])[0]
            dictexar['numb%stranstar' % strgelem][0][k] = indxexarstartran.size
            #dictexar['booltrantotl'][0][k] = dictexar['booltran'][0][indxexarstar].all()
        
        # transit duty cycle
        dictexar['dcyc'][0] = dictexar['duratrantotl'][0] / dictexar['pericomp'][0] / 24.
        
        # coordinates
        objticrs = astropy.coordinates.SkyCoord(ra=dictexar['rascstar'][0], \
                                               dec=dictexar['declstar'][0], frame='icrs', unit='deg')
        
        ## galactic longitude
        dictexar['lgalstar'][0] = np.array([objticrs.galactic.l])[0, :]
        
        ## galactic latitude
        dictexar['bgalstar'][0] = np.array([objticrs.galactic.b])[0, :]
        
        ## ecliptic longitude
        dictexar['loecstar'][0] = np.array([objticrs.barycentricmeanecliptic.lon.degree])[0, :]
        
        ## ecliptic latitude
        dictexar['laecstar'][0] = np.array([objticrs.barycentricmeanecliptic.lat.degree])[0, :]

        # radius ratio
        dictexar['rratcomp'][0] = dictexar[strgradielem][0] / dictexar['radistar'][0] / dictfact['rsre']
        
        # sum of the companion and stellar radii divided by the semi-major axis
        dictexar['rsmacomp'][0] = (dictexar[strgradielem][0] / dictfact['rsre'] + dictexar['radistar'][0]) / (dictexar['smaxcomp'][0] * dictfact['aurs'])
        
        # calculate TSM and ESM
        calc_tsmmesmm(dictexar, strgelem=strgelem)
        
        indxnonntran = np.where(~dictexar['booltran'][0])[0]
        dictexar['esmm'][0][indxnonntran] = np.nan
        dictexar['tsmm'][0][indxnonntran] = np.nan
        
        # surface gravity of the companion
        dictexar['loggcomp'][0] = np.log10(9.8 * dictexar[strgmasselem][0] / dictexar[strgradielem][0]**2)

    return dictexar


# physics

def retr_vesc(massplan, radicomp):
    
    vesc = 11.2 * np.sqrt(massplan / radicomp) # km/s

    return vesc


def retr_rs2a(rsmacomp, rrat):
    
    rs2a = rsmacomp / (1. + rrat)
    
    return rs2a


def retr_rsmacomp(peri, dura, cosi):
    
    rsmacomp = np.sqrt(np.sin(dura * np.pi / peri / 24.)**2 + cosi**2)
    
    return rsmacomp


def retr_duratranfull(
                      # orbital period [days]
                      pericomp, \
                      # sum of the radii of th star and the companion divided by the semi-major axis
                      rsmacomp, \
                      # cosine of the orbital inclination
                      cosicomp, \
                      # radius ratio of the companion and the star
                      rratcomp, \
                     ):
    '''
    Return the full transit duration in hours.
    '''    
    
    # radius of the star minus the radius of the companion
    rdiacomp = rsmacomp * (1. - 2. / (1. + rratcomp))

    fact = rdiacomp**2 - cosicomp**2
    
    duratranfull = np.full_like(pericomp, np.nan)
    indxtran = np.where(fact > 0)[0]
    
    if indxtran.size > 0:
        # sine of inclination
        sinicomp = np.sqrt(1. - cosicomp[indxtran]**2)
    
        duratranfull[indxtran] = 24. * pericomp[indxtran] / np.pi * np.arcsin(np.sqrt(fact[indxtran]) / sinicomp) # [hours]

    return duratranfull 


def retr_duratrantotl( \
                      # orbital period [days]
                      pericomp, \
                      # sum of the radii of th star and the companion divided by the semi-major axis
                      rsmacomp, \
                      # cosine of the orbital inclination
                      cosicomp, \
                      # Boolean flag to turn on diagnostic mode
                      booldiag=True, \
                     ):
    '''
    Return the total transit duration in hours.
    '''    
    
    if booldiag:
        if len(pericomp) != len(rsmacomp) or len(cosicomp) != len(rsmacomp):
            print('')
            print('pericomp')
            summgene(pericomp)
            print('rsmacomp')
            summgene(rsmacomp)
            print('cosicomp')
            summgene(cosicomp)
            raise Exception('')

    fact = rsmacomp**2 - cosicomp**2
    
    duratrantotl = np.full_like(pericomp, np.nan)
    indx = np.where(fact >= 0.)[0]
        
    if indx.size > 0:
        # sine of inclination
        sinicomp = np.sqrt(1. - cosicomp[indx]**2)
    
        duratrantotl[indx] = 24. * pericomp[indx] / np.pi * np.arcsin(np.sqrt(fact[indx]) / sinicomp) # [hours]
    
    return duratrantotl


#def retr_fracrtsa(fracrprs, fracsars):
#    
#    fracrtsa = (fracrprs + 1.) / fracsars
#    
#    return fracrtsa
#
#
#def retr_fracsars(fracrprs, fracrtsa):
#    
#    fracsars = (fracrprs + 1.) / fracrtsa
#    
#    return fracsars


def retr_rflxmodlrise(time, timerise, coeflinerise, coefquadrise):
    
    timeoffs = time - timerise
    indxpost = np.where(timeoffs > 0)[0]
    dflxrise = np.zeros_like(time)
    dflxrise[indxpost] = coeflinerise * timeoffs[indxpost] + coefquadrise * timeoffs[indxpost]**2
    rflx = 1. + dflxrise
    
    return rflx, dflxrise


def retr_rvel( \
              # times in days at which to evaluate the radial velocity
              time, \
              # epoch of midtransit
              epocmtracomp, \
              # orbital period in days
              pericomp, \
              # mass of the secondary in Solar mass [M_S]
              masscomp, 
              # mass of the primary in Solar mass [M_S]
              massstar, \
              # orbital inclination in degrees
              inclcomp, \
              # orbital eccentricity
              eccecomp, \
              # argument of periastron in degrees
              arpacomp, \
              ):
    '''
    Calculate the time-series of radial velocity (RV) of a two-body system.
    '''
    
    # phase
    phas = (time - epocmtracomp) / pericomp
    phas = phas % 1.
    
    # radial velocity (RV) semi-amplitude
    rvelsema = retr_rvelsema(pericomp, massstar, masscomp, inclcomp, eccecomp)
    
    # radial velocity time-series
    rvel = rvelsema * (np.cos(np.pi * arpacomp / 180. + 2. * np.pi * phas) + eccecomp * np.cos(np.pi * arpacomp / 180.))

    return rvel


def retr_coeflmdkkipp(u1, u2):
    
    q1 = (u1 + u2)**2
    q2 = u1 / 2. / (u1 + u2)
    
    return q1, q2


def retr_coeflmdkfromkipp(q1, q2):
    
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1. - 2. * q2)

    return u1, u2


def retr_rvelsema( \
                  # orbital period in days
                  pericomp, \
                  # mass of the primary in Solar mass [M_S]
                  massstar, \
                  # mass of the secondary in Solar mass [M_S]
                  masscomp, \
                  # orbital inclination in degrees
                  inclcomp, \
                  # orbital eccentricity
                  eccecomp, \
                 ):
    '''
    Calculate the semi-amplitude of radial velocity (RV) of a two-body system.
    '''
    
    dictfact = tdpy.retr_factconv()
    
    rvelsema = 203. * pericomp**(-1. / 3.) * masscomp * np.sin(inclcomp / 180. * np.pi) / \
                                                    (masscomp + massstar * dictfact['msme'])**(2. / 3.) / np.sqrt(1. - eccecomp**2) # [m/s]

    return rvelsema


def retr_brgtlmdk(cosg, coeflmdk, brgtraww=None, typelmdk='quad'):
    
    if brgtraww is None:
        brgtraww = 1.
    
    if typelmdk == 'linr':
        factlmdk = 1. - coeflmdk[0] * (1. - cosg)
    
    if typelmdk == 'quad' or typelmdk == 'quadkipp':
        factlmdk = 1. - coeflmdk[0] * (1. - cosg) - coeflmdk[1] * (1. - cosg)**2
    
    if typelmdk == 'nlin':
        factlmdk = 1. - coeflmdk[0] * (1. - cosg) - coeflmdk[1] * (1. - cosg)**2
    
    if typelmdk == 'none':
        factlmdk = np.ones_like(cosg)
    
    brgtlmdk = brgtraww * factlmdk
    
    return brgtlmdk


def retr_logg(radi, mass):
    
    logg = mass / radi**2

    return logg


def retr_noisphot(magtinpt, strginst, typeoutp='intplite'):
    '''
    TESS photometric precision (over what time scale?)
    ''' 
    
    if np.isscalar(magtinpt):
        #magtinpt = np.array(magtinpt)
        magtinpt = np.full(1, magtinpt)
    if isinstance(magtinpt, float):
        magtinpt = np.array(magtinpt)
    
    nois = np.zeros_like(magtinpt) + np.inf
    
    if strginst.startswith('LSST'):
        if strginst.endswith('band'):
            strgband = strginst[-5]
        else:
            strgband = strginst[-1]
        
        indx = np.where((magtinpt < 20.) & (magtinpt > 15.))
        nois[indx] = 6. # [ppt]
        
        indx = np.where((magtinpt >= 20.) & (magtinpt < 24.))
        nois[indx] = 6. * 10**((magtinpt[indx] - 20.) / 3.) # [ppt]
    
    elif strginst == 'TESS':
        # interpolate literature values
        if typeoutp == 'intplite':
            nois = np.array([40., 40., 40., 90., 200., 700., 3e3, 2e4]) * 1e-3 # [ppt]
            magt = np.array([ 2.,  4.,  6.,  8.,  10.,  12., 14., 16.])
            objtspln = scipy.interpolate.interp1d(magt, nois, fill_value='extrapolate')
            nois = objtspln(magtinpt)
        elif typeoutp == 'calcspoc':
            pass
        else:
            raise Exception('')
    elif strginst == 'TESS-GEO-UV':
        nois = 0.5 * 1e3 * 0.2 * 10**(-22. + magtinpt) # [ppt over one hour]
    elif strginst == 'TESS-GEO-VIS':
        nois = 0.5 * 1e3 * 0.2 * 10**(-25. + magtinpt) # [ppt over one hour]
    elif strginst == 'ULTRASAT':
        nois = 0.5 * 1e3 * 0.2 * 10**(-22.4 + magtinpt) # [ppt over one hour]
    else:
        print('')
        print('')
        print('')
        print('strginst')
        print(strginst)
        raise Exception('')

    return nois


def retr_subp(dictpopl, dictnumbsamp, dictindxsamp, namepoplinit, namepoplfinl, indx):
    
    if isinstance(indx, list):
        raise Exception('')

    if len(indx) == 0:
        indx = np.array([], dtype=int)

    if indx.size == 0:
        print('Warning! indx is zero.')

    dictpopl[namepoplfinl] = dict()
    for name in dictpopl[namepoplinit].keys():
        dictpopl[namepoplfinl][name] = [[], []]
        
        # copy the subset of the array
        
        if indx.size > 0:
            dictpopl[namepoplfinl][name][0] = dictpopl[namepoplinit][name][0][indx]
        else:
            dictpopl[namepoplfinl][name][0] = np.array([])

        # copy the unit
        dictpopl[namepoplfinl][name][1] =  dictpopl[namepoplinit][name][1]
    
    dictindxsamp[namepoplinit][namepoplfinl] = indx
    dictnumbsamp[namepoplfinl] = indx.size
    dictindxsamp[namepoplfinl] = dict()


def retr_dictpoplstarcomp( \
                          # type of target systems
                          typesyst, \
                          
                          # type of the population of target systems
                          typepoplsyst, \
                          
                          # number of systems
                          numbsyst=None, \
                          
                          # epochs of mid-transits
                          epocmtracomp=None, \
                          
                          # offset for mid-transit epochs
                          timeepoc=None, \
                          
                          # type of sampling of orbital period and semi-major axes
                          ## 'smax': semi-major axes are sampled first and then orbital periods are calculated
                          ## 'peri': orbital periods are sampled first and then semi-major axies are calculated
                          typesamporbtcomp='smax', \

                          # minimum number of components per star
                          minmnumbcompstar=1, \
                          
                          # maximum number of components per star
                          maxmnumbcompstar=1, \
                          
                          # minimum ratio of semi-major axis to radius of the host star
                          minmsmaxradistar=3., \
                          
                          # maximum ratio of semi-major axis to radius of the host star
                          maxmsmaxradistar=1e4, \
                          
                          # minimum mass of the companions
                          minmmasscomp=None, \
                          
                          # minimum orbital period, only taken into account when typesamporbtcomp == 'peri'
                          minmpericomp=0.5, \
                          #minmpericomp=0.1, \
                          
                          # maximum orbital period, only taken into account when typesamporbtcomp == 'peri'
                          maxmpericomp=0.5, \
                          #maxmpericomp=1000., \
                          
                          # Boolean flag to force all companions to be transiting
                          booltrancomp=True, \

                          # maximum cosine of inclination if booltrancomp is False
                          maxmcosicomp=None, \
                          
                          # Boolean flag to include exomoons
                          boolinclmoon=False, \
                          
                          # type of stellar population
                          ## 'sunl': 1 Solar mass with the Sun's density (i.e., 1 Solar radius)
                          ## 'drawkrou': draw from Kroupa IMF with densities consistent with mass
                          ## 'wdwf': 1 Solar mass and 0.01 Solar radius
                          typestar='sunl', \
                          
                          # Boolean flag to diagnose
                          booldiag=True, \
                          
                          # list of bands
                          liststrgband=None, \
                          
                         ):
    '''
    Sample a synthetic population of the features of companions (e.g., exoplanets )and the companions to companions (e.g., exomoons) 
    hosted by a specified or random population of stellar systems.
    '''
    
    print('typesyst')
    print(typesyst)
    print('typepoplsyst')
    print(typepoplsyst)
    
    # Boolean flag indicating if the system is a star with a companion
    boolhavecomp = typesyst.startswith('PlanetarySystem') or typesyst == 'CompactObjectStellarCompanion' or typesyst == 'StellarBinary'
       
    # Boolean flag indicating if the system is a star with a companion with moons
    boolhavemoon = typesyst.startswith('PlanetarySystemWithMoons')
       
    # Boolean flag indicating if the system is a planetary system
    boolsystpsys = typesyst.startswith('PlanetarySystem')
        
    # Boolean flag indicating if the system is a compact object with stellar companion
    boolsystcosc = typesyst == 'CompactObjectStellarCompanion'

    # Boolean flag indicating if the star is flaring
    boolflar = typesyst == 'StarFlaring'
    
    if booldiag:
        if not (boolflar or boolhavecomp):
            print('')
            print('')
            print('')
            print('typesyst')
            print(typesyst)
            raise Exception('typesyst is undefined.')

    # dictionary keys of the populations
    namepoplstartotl = 'star_%s_All' % typepoplsyst
    namepoplstaroccu = 'star_%s_Occurrent' % typepoplsyst
    
    dictnico = dict()
    dictpopl = dict()
    dictpopl['star'] = dict()
    dictstarnumbsamp = dict()
    dictstarindxsamp = dict()
    dictstarnumbsamp[namepoplstartotl] = dict()
    dictstarindxsamp[namepoplstartotl] = dict()
    
    dictfact = tdpy.retr_factconv()
    
    # number of systems
    if numbsyst is None:
        if typepoplsyst == 'SyntheticPopulation':
            numbsyst = 10000
    
    indxsyst = np.arange(numbsyst)

    # get the features of the star population
    if typepoplsyst.startswith('CTL') or typepoplsyst.startswith('TIC'):
        dictstar = retr_dictpopltic8(typepoplsyst, numbsyst=numbsyst)
        
        print('Removing stars that do not have radii or masses...')
        indx = np.where(np.isfinite(dictstar['radistar']) & \
                        np.isfinite(dictstar['massstar']))[0]
        for name in dictstar.keys():
            dictstar[name] = dictstar[name][indx]

        if (dictstar['rascstar'] > 1e4).any():
            raise Exception('')

        if (dictstar['radistar'] == 0.).any():
            raise Exception('')

        dictstar['densstar'] = 1.41 * dictstar['massstar'] / dictstar['radistar']**3
        dictstar['idenstar'] = dictstar['TICID']
    

    elif typepoplsyst == 'SyntheticPopulation':
        
        dictstar = dict()
        
        dictstar['distsyst'] = [tdpy.icdf_powr(np.random.rand(numbsyst), 100., 7000., -2.), 'pc']
        
        if typestar == 'sunl':
            dictstar['radistar'] = [np.ones(numbsyst), '$R_{\odot}$']
            dictstar['massstar'] = [np.ones(numbsyst), '$M_{\odot}$']
            dictstar['densstar'] = [1.4 * np.ones(numbsyst), 'g cm$^{-3}$']
        elif typestar == 'drawkrou':
            dictstar['massstar'] = [tdpy.icdf_powr(np.random.rand(numbsyst), 0.1, 10., 2.), '$M_{\odot}$']
            dictstar['densstar'] = [1.4 * (1. / dictstar['massstar'][0])**(0.7), 'g cm$^{-3}$']
            dictstar['radistar'] = [(1.4 * dictstar['massstar'][0] / \
                                                                                        dictstar['densstar'][0])**(1. / 3.), '$R_{\odot}$']
            raise Exception('To be implemented')
        elif typestar == 'wdwf':
            dictstar['radistar'] = [0.01 * np.ones(numbsyst), '$R_{\odot}$']
            dictstar['massstar'] = [np.ones(numbsyst), '$M_{\odot}$']
            dictstar['densstar'] = [1.4e6 * np.ones(numbsyst), 'g cm$^{-3}$']
        else:
            raise Exception('')

        dictstar['coeflmdklinr'] = [0.4 * np.ones(numbsyst), '']
        dictstar['coeflmdkquad'] = [0.25 * np.ones(numbsyst), '']

        dictstar['tmptstar'] = [6000 * dictstar['massstar'][0], 'K']
        
        dictstar['lumistar'] = [4. * np.pi * dictstar['tmptstar'][0]**4 * dictstar['radistar'][0]**2, '$L_{\odot}$']
        
        dictstar['fluxbolostar'] = [1361. * dictstar['lumistar'][0] / dictstar['distsyst'][0]**2 / 4. / np.pi, 'W/m^2']

        if liststrgband is None:
            liststrgband = []
            if typepoplsyst == 'lsstwfds':
                liststrgband += ['r']
        
        gdatfluxband = None
        for k in indxsyst: 
            print('liststrgband')
            print(liststrgband)
            dictfluxband, gdatfluxband = retr_dictfluxband(dictstar['tmptstar'][0][k], liststrgband, gdatfluxband=gdatfluxband)
            for strgband in liststrgband:
                dictstar['magtsyst%s' % strgband] = [retr_magtfromflux(dictfluxband[strgband], strgband), 'mag']

    else:
        print('')
        print('')
        print('')
        print('typepoplsyst')
        print(typepoplsyst)
        raise Exception('Undefined typepoplsyst.')
    
    dictstarnumbsamp[namepoplstartotl] = numbsyst
    dictpopl['star'][namepoplstartotl] = dictstar
    # total mass
    dictstar['masssyst'] = [[], []]
    dictstar['masssyst'][0] = np.copy(dictstar['massstar'][0])
    dictstar['masssyst'][1] = dictstar['massstar'][1]
    
    numbstar = numbsyst

    dictindx = dict()
    dictnumbsamp = dict()
    dictindxsamp = dict()

    if boolhavecomp:
        # minimum companion mass
        if minmmasscomp is None:
            if boolsystcosc:
                minmmasscomp = 5. # [Solar mass]
            elif typesyst == 'PlanetarySystem' or typesyst == 'PlanetarySystemEmittingCompanion' or typesyst == 'PlanetarySystemWithMoons':
                if typesyst == 'PlanetarySystem' or typesyst == 'PlanetarySystemWithMoons':
                    # ~ Mars mass
                    minmmasscomp = 0.1 # [Earth mass]
                if typesyst == 'PlanetarySystemEmittingCompanion':
                    minmmasscomp = 30. # [Earth mass]
            elif typesyst == 'StellarBinary':
                minmmasscomp = 0.5 # [Earth mass]
    
        # maximum companion mass
        if boolsystcosc:
            maxmmasscomp = 200. # [Solar mass]
        elif boolsystpsys:
            # Deuterium burning mass
            maxmmasscomp = 4400. # [Earth mass]
        elif typesyst == 'StellarBinary':
            maxmmasscomp = np.inf # [Earth mass]
        else:
            print('')
            print('')
            print('')
            print('typesyst')
            print(typesyst)
            raise Exception('Could not define maxmmasscomp')
    
    if boolhavecomp or boolflar:
        
        strgbody = 'star'

        if boolhavecomp:
            strglimb = 'comp'
        else:
            strglimb = 'flar'
        
        strgnumblimbbody = 'numb%s%s' % (strglimb, strgbody)
        strgnumblimbbodymean = 'numb%s%smean' % (strglimb, strgbody)
        
        namepopllimbtotl = '%s%s_%s_All' % (strglimb, strgbody, typepoplsyst)
        if strglimb == 'comp':
            namepoplcomptran = '%s%s_%s_Transiting' % (strglimb, strgbody, typepoplsyst)
        
        dictpopl[strglimb] = dict()
        dictnumbsamp[strglimb] = dict()
        dictindxsamp[strglimb] = dict()
        dictnumbsamp[strglimb][namepopllimbtotl] = dict()
        dictindxsamp[strglimb][namepopllimbtotl] = dict()
        dictpopl[strglimb][namepopllimbtotl] = dict()
        
        if boolhavemoon:
            namepoplmoontotl = 'mooncompstar_%s_All'
            dictpopl['moon'] = dict()
            dictmoonnumbsamp = dict()
            dictmoonindxsamp = dict()
            dictmoonnumbsamp[namepoplmoontotl] = dict()
            dictmoonindxsamp[namepoplmoontotl] = dict()
            dictpopl['moon'][namepoplmoontotl] = dict()
    
        if typesyst.startswith('PlanetarySystem') or typesyst == 'StarFlaring':
            
            if typesyst.startswith('PlanetarySystem'):
                # mean number of companions per star
                dictpopl[strgbody][namepoplstartotl][strgnumblimbbodymean] = [0.5 * dictpopl[strgbody][namepoplstartotl]['massstar'][0]**(-1.), '']
            if typesyst == 'StarFlaring':
                # mean number of flares per star
                dictpopl[strgbody][namepoplstartotl][strgnumblimbbodymean] = [0.5 * dictpopl[strgbody][namepoplstartotl]['massstar'][0]**(-1.), '']
            
            # mean number per star
            dictpopl[strgbody][namepoplstartotl][strgnumblimbbodymean] = [0.5 * dictpopl[strgbody][namepoplstartotl]['massstar'][0]**(-1.), '']
            
            # number per star
            dictpopl[strgbody][namepoplstartotl][strgnumblimbbody] = [np.random.poisson(dictpopl[strgbody][namepoplstartotl][strgnumblimbbodymean][0]), '']
            
            if booldiag:
                if maxmnumbcompstar is not None and minmnumbcompstar is not None:
                    if maxmnumbcompstar < minmnumbcompstar:
                        print('')
                        print('')
                        print('')
                        raise Exception('maxmnumbcompstar < minmnumbcompstar')

            if minmnumbcompstar is not None:
                dictpopl[strgbody][namepoplstartotl][strgnumblimbbody] = [np.maximum(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0], minmnumbcompstar), '']

            if maxmnumbcompstar is not None:
                dictpopl[strgbody][namepoplstartotl][strgnumblimbbody] = [np.minimum(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0], maxmnumbcompstar), '']

        elif typesyst == 'CompactObjectStellarCompanion' or typesyst == 'StellarBinary':
            # number of companions per star
            dictpopl[strgbody][namepoplstartotl][strgnumblimbbody] = np.ones(dictpopl[strgbody][namepoplstartotl]['radistar'].size).astype(int)
            
        else:
            print('')
            print('')
            print('')
            print('typesyst')
            print(typesyst)
            raise Exception('typesyst is undefined.')
    
        if booldiag:
            if np.isscalar(dictpopl[strgbody][namepoplstartotl]['distsyst'][0]):
                print('')
                print('')
                print('')
                print('dictpopl[strgbody][namepoplstartotl][distsyst][0]')
                summgene(dictpopl[strgbody][namepoplstartotl]['distsyst'][0])
                raise Exception('')

        # Boolean flag of occurence
        dictpopl[strgbody][namepoplstartotl]['booloccu'] = [dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0] > 0, '']
    
        # subpopulation where companions or flares occur
        indx = np.where(dictpopl[strgbody][namepoplstartotl]['booloccu'][0])[0]
        retr_subp(dictpopl[strgbody], dictstarnumbsamp, dictstarindxsamp, namepoplstartotl, namepoplstaroccu, indx)
    
        # indices of companions or flares for each star
        dictindx[strglimb] = dict()
        dictindx[strglimb][strgbody] = [[] for k in indxsyst]
        cntr = 0
        print('indxsyst')
        print(indxsyst)
        for k in indxsyst:
            dictindx[strglimb][strgbody][k] = np.arange(cntr, cntr + dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]).astype(int)
            cntr += dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]
        dictnumbsamp[strglimb][namepopllimbtotl] = cntr
    
        # prepare to load star features into component features
        for name in list(dictpopl[strgbody][namepoplstartotl].keys()):
            dictpopl[strglimb][namepopllimbtotl][name] = [np.empty(dictnumbsamp[strglimb][namepopllimbtotl]), '']
    
        dictnumbsamp[strglimb][namepopllimbtotl] = dictpopl[strglimb][namepopllimbtotl]['radistar'][0].size
        

        listnamecatr = ['masssyst', 'radistar']
        if strglimb == 'comp':
            listnamecatr += ['pericomp', 'cosicomp', 'smaxcomp', 'eccecomp', 'arpacomp', 'loancomp', 'masscomp', 'epocmtracomp', 'rsmacomp']
            if typesyst == 'PlanetarySystemWithMoons':
                listnamecatr += ['masscompmoon']
            if typesyst == 'PlanetarySystemWithNonKeplerianObjects':
                listnamecatr += ['factnonkcomp']
            if not boolsystcosc:
                listnamecatr += ['radicomp', 'denscomp']
        elif strglimb == 'flar':
            listnamecatr += ['amplflar', 'timeflar', 'enerflar', 'tsclflar']
        else:
            raise Exception('')

        for name in listnamecatr:
            dictpopl[strglimb][namepopllimbtotl][name] = [np.empty(dictnumbsamp[strglimb][namepopllimbtotl]), '']

        if booldiag:
            if np.isscalar(dictpopl[strglimb][namepopllimbtotl]['distsyst'][0]):
                print('')
                print('')
                print('')
                print('dictpopl[strglimb][namepopllimbtotl][distsyst][0]')
                summgene(dictpopl[strglimb][namepopllimbtotl]['distsyst'][0])
                raise Exception('')

            cntr = 0
            for k in indxsyst:
                cntr += len(dictindx[strglimb][strgbody][k])
            if cntr != dictnumbsamp[strglimb][namepopllimbtotl]:
                print('')
                print('')
                print('')
                print('cntr')
                print(cntr)
                print('dictnumbsamp[strglimb][namepopllimbtotl]')
                print(dictnumbsamp[strglimb][namepopllimbtotl])
                raise Exception('cntr != dictnumbsamp[strglimb][namepopllimbtotl]')
    
        print('Sampling features...')
    
        for k in tqdm(range(numbstar)):
            
            if dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k] == 0:
                continue

            # load star features into component features
            for namefeat in dictpopl[strgbody][namepoplstartotl].keys():
                dictpopl[strglimb][namepopllimbtotl][namefeat][0][dictindx[strglimb][strgbody][k]] = dictpopl[strgbody][namepoplstartotl][namefeat][0][k]
            
            if strglimb == 'comp':

                numb = dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]

                # eccentricities
                dictpopl[strglimb][namepopllimbtotl]['eccecomp'][0][dictindx[strglimb][strgbody][k]] = np.random.rand(numb)
                
                # arguments of periapsis
                dictpopl[strglimb][namepopllimbtotl]['arpacomp'][0][dictindx[strglimb][strgbody][k]] = 2. * np.pi * np.random.rand(numb)
                
                # longtides of ascending node
                dictpopl[strglimb][namepopllimbtotl]['loancomp'][0][dictindx[strglimb][strgbody][k]] = 2. * np.pi * np.random.rand(numb)
                
                # companion mass
                dictpopl[strglimb][namepopllimbtotl]['masscomp'][0][dictindx[strglimb][strgbody][k]] = tdpy.util.icdf_powr(np.random.rand(numb), minmmasscomp, maxmmasscomp, 2.)
                
                if boolsystpsys or typesyst == 'StellarBinary':
                    # companion radius
                    dictpopl[strglimb][namepopllimbtotl]['radicomp'][0][dictindx[strglimb][strgbody][k]] = \
                                                        retr_radifrommass(dictpopl[strglimb][namepopllimbtotl]['masscomp'][0][dictindx[strglimb][strgbody][k]])
        
                    # companion density
                    dictpopl[strglimb][namepopllimbtotl]['denscomp'][0][dictindx[strglimb][strgbody][k]] = \
                                        5.51 * dictpopl[strglimb][namepopllimbtotl]['masscomp'][0][dictindx[strglimb][strgbody][k]] / \
                                                                                           dictpopl[strglimb][namepopllimbtotl]['radicomp'][0][dictindx[strglimb][strgbody][k]]**3
                
                # total mass
                if boolsystcosc or typesyst == 'StellarBinary':
                    dictpopl[strgbody][namepoplstartotl]['masssyst'][k] += np.sum(dictpopl[strglimb][namepopllimbtotl]['masscomp'][0][dictindx[strglimb][strgbody][k]])
                
                if typesamporbtcomp == 'peri':
                
                    ratiperi = tdpy.util.icdf_powr(np.random.rand(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][k] - 1), 1.2, 1.3, 5.)
                    
                    listpericomp = []
                    for mm in range(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][k]):
                        if mm == 0:
                            peri = minmpericomp
                        else:
                            peri = ratiperi[mm-1] * listpericomp[mm-1]
                        listpericomp.append(peri)
                    dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]] = np.array(listpericomp)

                    if booldiag:
                        ratiperi = dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]][1:] / \
                                                    dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]][:-1]
                        indx = np.where(ratiperi < 1.2)[0]
                        if indx.size > 0:
                            print('indx')
                            summgene(indx)
                            print('dictpopl[comp][namepopllimbtotl][pericomp][dictindx[strglimb][strgbody][k]]')
                            print(dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]])
                            print('dictpopl[star][namepoplstartotl][numbcompstar][k]')
                            print(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][k])
                            raise Exception('')
                    
                    if typesyst == 'PlanetarySystemWithNonKeplerianObjects':
                        raise Exception('')
                    else:
                        factnonk = 1.

                    dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k]] = \
                                            retr_smaxkepl(dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]], \
                                                                                                                dictpopl[strgbody][namepoplstartotl]['masssyst'][k], factnonk=factnonk)
                
                else:
                    # semi-major axes
                    #if np.isfinite(dictpopl[strgbody][namepoplstartotl]['densstar'][k]):
                    #    densstar = dictpopl[strgbody][namepoplstartotl]['densstar'][k]
                    #else:
                    #    densstar = 1.
                    #dictpopl[strglimb][namepopllimbtotl]['radiroch'][k] = retr_radiroch(radistar, densstar, denscomp)
                    #minmsmax = 2. * dictpopl[strglimb][namepopllimbtotl]['radiroch'][k]

                    dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k]] = dictpopl[strgbody][namepoplstartotl]['radistar'][0][k] * \
                                                                                 tdpy.util.icdf_powr(np.random.rand(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]), \
                                                                                                    minmsmaxradistar, maxmsmaxradistar, 2.) / dictfact['aurs']
                    
                    if typesyst == 'PlanetarySystemWithNonKeplerianObjects':
                        factnonk = tdpy.util.icdf_powr(np.random.rand(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][k]), 0.1, 1., -2.)
                        dictpopl[strglimb][namepopllimbtotl]['factnonkcomp'][0][dictindx[strglimb][strgbody][k]] = factnonk
                    else:
                        factnonk = 1.
                    
                    dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]] = \
                                            retr_perikepl(dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k]], \
                                                                                                  dictpopl[strgbody][namepoplstartotl]['masssyst'][0][k], factnonk=factnonk)
                    
        
                
                if booldiag:
                    if not np.isfinite(dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]]).all():
                        print('')
                        print('')
                        print('')
                        print('dictpopl[comp][namepopllimbtotl][masscomp][dictindx[comp][star][k]]')
                        print(dictpopl[strglimb][namepopllimbtotl]['masscomp'][0][dictindx[strglimb][strgbody][k]])
                        print('dictpopl[comp][namepopllimbtotl][masssyst][dictindx[comp][star][k]]')
                        print(dictpopl[strglimb][namepopllimbtotl]['masssyst'][dictindx[strglimb][strgbody][k]])
                        print('dictpopl[comp][namepopllimbtotl][smaxcomp][dictindx[comp][star][k]]')
                        print(dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k]])
                        print('dictpopl[comp][namepopllimbtotl][pericomp][dictindx[comp][star][k]]')
                        print(dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]])
                        raise Exception('')

                # conjunction epochs
                if epocmtracomp is not None:
                    dictpopl[strglimb][namepopllimbtotl]['epocmtracomp'][0][dictindx[strglimb][strgbody][k]] = \
                                                            np.full(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][k], epocmtracomp)
                else:
                    dictpopl[strglimb][namepopllimbtotl]['epocmtracomp'][0][dictindx[strglimb][strgbody][k]] = \
                                        1e8 * np.random.rand(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k])
                if timeepoc is not None:
                    dictpopl[strglimb][namepopllimbtotl]['epocmtracomp'][0][dictindx[strglimb][strgbody][k]] = dictpopl[strglimb][namepopllimbtotl]['epocmtracomp'][k] + \
                                                   dictpopl[strglimb][namepopllimbtotl]['pericomp'][k] * \
                                                   np.round((dictpopl[strglimb][namepopllimbtotl]['epocmtracomp'][k] - timeepoc) / dictpopl[strglimb][namepopllimbtotl]['pericomp'][k])
    
        if strglimb == 'comp':

            if typesyst == 'PlanetarySystemWithMoons':
                # initialize the total mass of the companion + moons system as the mass of the companion
                dictpopl[strglimb][namepopllimbtotl]['masscompmoon'] = np.copy(dictpopl[strglimb][namepopllimbtotl]['masscomp'])
                        
            rsum = dictpopl[strglimb][namepopllimbtotl]['radistar'][0]
            if not boolsystcosc:
                rsum += dictpopl[strglimb][namepopllimbtotl]['radicomp'][0] / dictfact['rsre']    
            dictpopl[strglimb][namepopllimbtotl]['rsmacomp'][0] = rsum / dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0] / dictfact['aurs']
            
            if booltrancomp is True and maxmcosicomp is not None:
                raise Exception('maxmcosicomp cannot be specified if booltrancomp is True.')

            for k in tqdm(range(numbstar)):
                
                if dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k] == 0:
                    continue

                if booltrancomp:
                    maxmcosicomptemp = dictpopl[strglimb][namepopllimbtotl]['rsmacomp'][0][k]
                elif maxmcosicomp is not None:
                    maxmcosicomptemp = maxmcosicomp
                else:
                    maxmcosicomptemp = 1.
                
                # cosine of orbital inclinations
                dictpopl[strglimb][namepopllimbtotl]['cosicomp'][0][dictindx[strglimb][strgbody][k]] = maxmcosicomptemp * np.random.rand(numb)
                
        if strglimb == 'comp':

            # orbital inclinations of the companions
            dictpopl[strglimb][namepopllimbtotl]['inclcomp'] = [180. / np.pi * np.arccos(dictpopl[strglimb][namepopllimbtotl]['cosicomp'][0]), 'degree']
            
            dictpopl[strglimb][namepopllimbtotl]['inclcomp'][0] = 90. + (dictpopl[strglimb][namepopllimbtotl]['inclcomp'][0] - 90.) * \
                                                                            (2 * np.random.randint(2, size=dictpopl[strglimb][namepopllimbtotl]['cosicomp'][0].size) - 1.)

            if boolsystpsys:
                
                if booldiag:
                    
                    if not np.isfinite(dictpopl[strglimb][namepopllimbtotl]['radistar'][0]).all():
                        print('')
                        print('')
                        print('')
                        raise Exception('not np.isfinite(dictpopl[comp][namepopllimbtotl][radistar]).all()')
                    
                    if not np.isfinite(dictpopl[strglimb][namepopllimbtotl]['radicomp'][0]).all():
                        print('')
                        print('')
                        print('')
                        raise Exception('not np.isfinite(dictpopl[comp][namepopllimbtotl][radicomp]).all()')

                # radius ratio
                dictpopl[strglimb][namepopllimbtotl]['rratcomp'] = [dictpopl[strglimb][namepopllimbtotl]['radicomp'][0] / \
                                                                        dictpopl[strglimb][namepopllimbtotl]['radistar'][0] / dictfact['rsre'], '']
                
                if booldiag:
                    if not np.isfinite(dictpopl[strglimb][namepopllimbtotl]['rratcomp'][0]).all():
                        print('')
                        print('')
                        print('')
                        raise Exception('not np.isfinite(dictpopl[comp][namepopllimbtotl][rratcomp]).all()')
                    
            # Boolean flag indicating whether a companion is transiting
            dictpopl[strglimb][namepopllimbtotl]['booltran'] = [dictpopl[strglimb][namepopllimbtotl]['rsmacomp'][0] > dictpopl[strglimb][namepopllimbtotl]['cosicomp'][0], '']

            # subpopulation where object transits
            indx = np.where(dictpopl[strglimb][namepopllimbtotl]['booltran'][0])[0]
            retr_subp(dictpopl[strglimb], dictnumbsamp[strglimb], dictindxsamp[strglimb], namepopllimbtotl, namepoplcomptran, indx)

            # transit duration
            dictpopl[strglimb][namepoplcomptran]['duratrantotl'] = [retr_duratrantotl(dictpopl[strglimb][namepoplcomptran]['pericomp'][0], \
                                                                                     dictpopl[strglimb][namepoplcomptran]['rsmacomp'][0], \
                                                                                     dictpopl[strglimb][namepoplcomptran]['cosicomp'][0]), '']
            dictpopl[strglimb][namepoplcomptran]['dcyc'] = [dictpopl[strglimb][namepoplcomptran]['duratrantotl'][0] / dictpopl[strglimb][namepoplcomptran]['pericomp'][0] / 24., '']
            
            if boolsystcosc:
                # amplitude of self-lensing
                dictpopl[strglimb][namepoplcomptran]['amplslen'] = chalcedon.retr_amplslen(dictpopl[strglimb][namepoplcomptran]['pericomp'][0], \
                                                                                           dictpopl[strglimb][namepoplcomptran]['radistar'][0], \
                                                                                           dictpopl[strglimb][namepoplcomptran]['masscomp'][0], \
                                                                                           dictpopl[strglimb][namepoplcomptran]['massstar'][0])
            
            if typesyst == 'PlanetarySystem':
                # transit depth
                dictpopl[strglimb][namepoplcomptran]['depttrancomp'] = [1e3 * dictpopl[strglimb][namepoplcomptran]['rratcomp'][0]**2, 'ppt']
            
            # define parent population's features that are valid only for transiting systems
            listtemp = ['duratrantotl', 'dcyc']
            if typesyst == 'PlanetarySystem':
                listtemp += ['depttrancomp']
            if boolsystcosc:
                listtemp += ['amplslen']
            for strg in listtemp:
                dictpopl[strglimb][namepopllimbtotl][strg] = [np.full_like(dictpopl[strglimb][namepopllimbtotl]['pericomp'][0], np.nan), '']
                dictpopl[strglimb][namepopllimbtotl][strg][0][indx] = dictpopl[strglimb][namepoplcomptran][strg][0]

            indxmooncompstar = [[[] for j in dictindx[strglimb][strgbody][k]] for k in indxsyst]
            if typesyst == 'PlanetarySystemWithMoons':
                # Hill radius of the companion
                dictpopl[strglimb][namepopllimbtotl]['radihill'] = retr_radihill(dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0], \
                                                                                 dictpopl[strglimb][namepopllimbtotl]['masscomp'][0] / dictfact['msme'], \
                                                                                 dictpopl[strglimb][namepopllimbtotl]['massstar'][0])
            
                # maximum semi-major axis of the moons 
                dictpopl[strglimb][namepopllimbtotl]['maxmsmaxmoon'] = 0.2 * dictpopl[strglimb][namepopllimbtotl]['radihill']
                    
                # mean number of moons per companion
                dictpopl[strglimb][namepopllimbtotl]['numbmooncompmean'] = 1000. * dictpopl[strglimb][namepopllimbtotl]['masscomp']**(-1.)
                
                # number of moons per companion
                #dictpopl[strglimb][namepopllimbtotl]['numbmooncomp'] = np.random.poisson(dictpopl[strglimb][namepopllimbtotl]['numbmooncompmean'])
                dictpopl[strglimb][namepopllimbtotl]['numbmooncomp'] = np.ones_like(dictpopl[strglimb][namepopllimbtotl]['numbmooncompmean'])
                
                cntr = 0
                for k in range(len(dictpopl[strgbody][namepoplstartotl]['radistar'])):
                    for j in range(len(dictindx[strglimb][strgbody][k])):
                        indxmooncompstar[k][j] = np.arange(cntr, cntr + dictpopl[strglimb][namepopllimbtotl]['numbmooncomp'][0][dictindx[strglimb][strgbody][k][j]]).astype(int)
                        cntr += int(dictpopl[strglimb][namepopllimbtotl]['numbmooncomp'][j])
                dictmoonnumbsamp[namepoplmoontotl] = cntr
            
                numbmoontotl = int(np.sum(dictpopl[strglimb][namepopllimbtotl]['numbmooncomp']))
                
                # prepare to load component features into moon features
                for name in list(dictpopl[strglimb][namepopllimbtotl].keys()):
                    dictpopl['moon'][namepoplmoontotl][name] = np.empty(dictmoonnumbsamp[namepoplmoontotl])
            
                for k in tqdm(range(numbstar)):
                    
                    if dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][k] == 0:
                        continue
                    
                    numbcomp = dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][k]
                    
                    # number of exomoons to the companion
                    numbmoon = dictpopl[strglimb][namepopllimbtotl]['numbmooncomp'][0][dictindx[strglimb][strgbody][k]].astype(int)
                    for name in ['radi', 'mass', 'dens', 'peri', 'epocmtra', 'smax', 'minmsmax']:
                        dictpopl['moon'][namepoplmoontotl][name+'moon'] = np.empty(numbmoontotl)
                    
                    indxmoon = [[] for j in dictindx[strglimb][strgbody][k]]
                    for j in range(dictindx[strglimb][strgbody][k].size):
                        
                        #print('')
                        #print('')
                        #print('')
                        #print('')
                        #print('j')
                        #print(j)
                        #print('dictindx[strglimb][strgbody][k]')
                        #print(dictindx[strglimb][strgbody][k])
                        #print('numbmoon')
                        #print(numbmoon)
                        #print('indxmoon')
                        #print(indxmoon)
                        #print('indxmooncompstar[k][j]')
                        #print(indxmooncompstar[k][j])
                        #print('')
                        
                        if numbmoon[j] == 0:
                            continue

                        indxmoon[j] = np.arange(numbmoon[j])
                        # properties of the moons
                        ## mass [M_E]
                        dictpopl['moon'][namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]] = dictpopl[strglimb][namepopllimbtotl]['masscomp'][0][dictindx[strglimb][strgbody][k][j]] * \
                                              tdpy.icdf_powr(np.random.rand(int(dictpopl[strglimb][namepopllimbtotl]['numbmooncomp'][0][dictindx[strglimb][strgbody][k][j]])), 0.005, 0.1, 2.)
                        
                        ## radii [R_E]
                        dictpopl['moon'][namepoplmoontotl]['radimoon'][indxmooncompstar[k][j]] = \
                                                    retr_radifrommass(dictpopl['moon'][namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]])
                        
                        ## densities [g/cm^3]
                        dictpopl['moon'][namepoplmoontotl]['densmoon'][indxmooncompstar[k][j]] = 5.51 * dictpopl['moon'][namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]] / \
                                                                                                      dictpopl['moon'][namepoplmoontotl]['radimoon'][indxmooncompstar[k][j]]**3
                        
                        # minimum semi-major axes for the moons
                        dictpopl['moon'][namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j]] = (1. / dictfact['rsre'] / dictfact['aurs']) * \
                                                                             retr_radiroch(dictpopl[strglimb][namepopllimbtotl]['radicomp'][0][dictindx[strglimb][strgbody][k][j]], \
                                                                                           dictpopl[strglimb][namepopllimbtotl]['denscomp'][0][dictindx[strglimb][strgbody][k][j]], \
                                                                                                                  dictpopl['moon'][namepoplmoontotl]['densmoon'][indxmooncompstar[k][j]])
                        
                        if booldiag:
                            if (dictpopl['moon'][namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j]] >= \
                                                                    dictpopl[strglimb][namepopllimbtotl]['maxmsmaxmoon'][dictindx[strglimb][strgbody][k][j]]).any():
                                print('')
                                print('')
                                print('')
                                print('typesamporbtcomp')
                                print(typesamporbtcomp)
                                print('dictpopl[comp][namepopllimbtotl][massstar][k]')
                                print(dictpopl[strglimb][namepopllimbtotl]['massstar'][k])
                                print('dictpopl[comp][namepopllimbtotl][masssyst][k]')
                                print(dictpopl[strglimb][namepopllimbtotl]['masssyst'][k])
                                print('dictpopl[comp][namepopllimbtotl][pericomp][dictindx[strglimb][strgbody][k][j]]')
                                print(dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k][j]])
                                print('dictpopl[comp][namepopllimbtotl][smaxcomp][dictindx[strglimb][strgbody][k][j]]')
                                print(dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k][j]])
                                print('dictpopl[comp][namepopllimbtotl][radihill][dictindx[strglimb][strgbody][k][j]]')
                                print(dictpopl[strglimb][namepopllimbtotl]['radihill'][dictindx[strglimb][strgbody][k][j]])
                                print('dictpopl[comp][namepopllimbtotl][maxmsmaxmoon][dictindx[strglimb][strgbody][k][j]]')
                                print(dictpopl[strglimb][namepopllimbtotl]['maxmsmaxmoon'][dictindx[strglimb][strgbody][k][j]])
                                print('dictpopl[moon][namepoplmoontotl][minmsmaxmoon][indxmooncompstar[k][j]]')
                                print(dictpopl['moon'][namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j]])
                                print('dictpopl[moon][namepoplmoontotl][radimoon][indxmooncompstar[k][j]]')
                                print(dictpopl['moon'][namepoplmoontotl]['radimoon'][indxmooncompstar[k][j]])
                                print('dictpopl[moon][namepoplmoontotl][massmoon][indxmooncompstar[k][j]]')
                                print(dictpopl['moon'][namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]])
                                print('dictpopl[moon][namepoplmoontotl][densmoon][indxmooncompstar[k][j]]')
                                print(dictpopl['moon'][namepoplmoontotl]['densmoon'][indxmooncompstar[k][j]])
                                print('dictpopl[comp][namepopllimbtotl][radicomp][dictindx[strglimb][strgbody][k][j]]')
                                print(dictpopl[strglimb][namepopllimbtotl]['radicomp'][0][dictindx[strglimb][strgbody][k][j]])
                                print('dictpopl[comp][namepopllimbtotl][masscomp][dictindx[strglimb][strgbody][k][j]]')
                                print(dictpopl[strglimb][namepopllimbtotl]['masscomp'][0][dictindx[strglimb][strgbody][k][j]])
                                print('dictpopl[comp][namepopllimbtotl][denscomp][dictindx[strglimb][strgbody][k][j]]')
                                print(dictpopl[strglimb][namepopllimbtotl]['denscomp'][0][dictindx[strglimb][strgbody][k][j]])
                                raise Exception('Minimum semi-major axes for the moons are greater than the maximum semi-major axes.')
                        
                        # semi-major axes of the moons
                        for jj in indxmoon[j]:
                            dictpopl['moon'][namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j][jj]] = tdpy.icdf_powr(np.random.rand(), \
                                                                                                dictpopl['moon'][namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j][jj]], \
                                                                                                dictpopl[strglimb][namepopllimbtotl]['maxmsmaxmoon'][dictindx[strglimb][strgbody][k][j]], 2.)
                      
                        # add the moon masses to the total mass of the companion + moons system
                        dictpopl[strglimb][namepopllimbtotl]['masscompmoon'][dictindx[strglimb][strgbody][k][j]] += \
                                                        np.sum(dictpopl['moon'][namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]])
                        
                        # orbital period of the moons
                        dictpopl['moon'][namepoplmoontotl]['perimoon'][indxmooncompstar[k][j]] = retr_perikepl(dictpopl['moon'][namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j]], \
                                                                              dictpopl[strglimb][namepopllimbtotl]['masscompmoon'][dictindx[strglimb][strgbody][k][j]] / dictfact['msme'])
                        
                        # load component features into moon features
                        ## temp the code crashed here once
                        for name in dictpopl[strglimb][namepopllimbtotl].keys():
                            dictpopl['moon'][namepoplmoontotl][name][indxmooncompstar[k][j]] = dictpopl[strglimb][namepopllimbtotl][name][dictindx[strglimb][strgbody][k][j]]
                        
                        if booldiag:
                            if (dictpopl['moon'][namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j]] > \
                                                0.7 * dictpopl[strglimb][namepopllimbtotl]['radihill'][dictindx[strglimb][strgbody][k][j]]).any():
                            
                            
                                print('')
                                print('')
                                print('')
                                print('numbmoon[j]')
                                print(numbmoon[j])
                                print('dictpopl[comp][namepopllimbtotl][radihill][dictindx[strglimb][strgbody][k][j]]')
                                print(dictpopl[strglimb][namepopllimbtotl]['radihill'][dictindx[strglimb][strgbody][k][j]])
                                
                                print('dictpopl[moon][namepoplmoontotl][smaxmoon][indxmooncompstar[k][j]]')
                                print(dictpopl['moon'][namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j]])
                                
                                print('dictpopl[comp][namepopllimbtotl][smaxcomp][dictindx[strglimb][strgbody][k][j]]')
                                print(dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k][j]])
                                
                                print('dictpopl[moon][namepoplmoontotl][minmsmaxmoon][indxmooncompstar[k][j]]')
                                print(dictpopl['moon'][namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j]])
                                print('dictpopl[comp][namepopllimbtotl][maxmsmaxmoon][dictindx[strglimb][strgbody][k][j]]')
                                print(dictpopl[strglimb][namepopllimbtotl]['maxmsmaxmoon'][dictindx[strglimb][strgbody][k][j]])
                                
                                raise Exception('Semi-major axis of a moon is larger than 0.7 times the Hill radius of the companion.')
            
                    # planet-moon conjunction times
                    dictpopl['moon'][namepoplmoontotl]['epocmtramoon'] = 1e8 * np.random.rand(numbmoontotl)
        
        dictnico['listnamefeatbody'] = np.array(list(dictpopl[strgbody][namepoplstartotl].keys()))
        dictnico['listnamefeatlimb'] = np.array(list(dictpopl[strglimb][namepopllimbtotl].keys()))
        dictnico['listnamefeatlimbonly'] = np.setdiff1d(dictnico['listnamefeatlimb'], dictnico['listnamefeatbody'])
    
    # check if dictpopl is properly defined, whose leaves should be a list of two items (of values and labels, respectively)
    if booldiag:
        for namepopl in dictpopl:
            for namespop in dictpopl[namepopl]:
                for namefeat in dictpopl[namepopl][namespop]:
                    if len(dictpopl[namepopl][namespop][namefeat]) != 2 or \
                            len(dictpopl[namepopl][namespop][namefeat][1]) > 0 and not isinstance(dictpopl[namepopl][namespop][namefeat][1][0], str):
                        print('')
                        print('')
                        print('')
                        print('namepopl')
                        print(namepopl)
                        print('namespop')
                        print(namespop)
                        print('namefeat')
                        print(namefeat)
                        print('dictpopl[namepopl][namespop][namefeat]')
                        print(dictpopl[namepopl][namespop][namefeat])
                        raise Exception('dictpopl is not properly defined.')

    dictnico['dictpopl'] = dictpopl
    dictnico['dictindx'] = dictindx
    dictnico['dictnumbsamp'] = dictnumbsamp
    dictnico['dictindxsamp'] = dictindxsamp
    
    return dictnico
       

def retr_listtablobsv(strgmast):
    
    if strgmast is None:
        raise Exception('strgmast should not be None.')

    listtablobsv = astroquery.mast.Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries', objectname=strgmast)

    if len(listtablobsv) == 0:
        print('No SPOC data is found...')
    
    return listtablobsv


def setp_coorstrgmast(rasctarg=None, decltarg=None, strgmast=None):
    
    if strgmast is not None and (rasctarg is not None or decltarg is not None) or strgmast is None and (rasctarg is None or decltarg is None):
        raise Exception('')

    # determine RA and DEC if not already provided
    if rasctarg is None:
        rasctarg, decltarg = retr_rascdeclfromstrgmast(strgmast)
    
    # determine strgmast if not already provided
    if strgmast is None:
        strgmast = '%g %g' % (rasctarg, decltarg)

    return strgmast, rasctarg, decltarg


def retr_rascdeclfromstrgmast(strgmast):

    print('Querying the TIC using the key %s, in order to get the RA and DEC of the closest TIC source...' % strgmast)
    listdictcatl = astroquery.mast.Catalogs.query_object(strgmast, catalog='TIC')
    #listdictcatl = astroquery.mast.Catalogs.query_object(strgmast, catalog='TIC', radius='40s')
    rasctarg = listdictcatl[0]['ra']
    decltarg = listdictcatl[0]['dec']
    print('TIC, RA, and DEC of the closest match are %d, %.5g, and %.5g' % (int(listdictcatl[0]['ID']), rasctarg, decltarg))
    
    return rasctarg, decltarg

    
def retr_lcurmodl_flarsing(meantime, timeflar, amplflar, scalrise, scalfall):
    
    numbtime = meantime.size
    if numbtime == 0:
        raise Exception('')
    indxtime = np.arange(numbtime)
    indxtimerise = np.where(meantime < timeflar)[0]
    indxtimefall = np.setdiff1d(indxtime, indxtimerise)
    lcur = np.empty_like(meantime)
    lcur[indxtimerise] = np.exp((meantime[indxtimerise] - timeflar) / scalrise)
    lcur[indxtimefall] = np.exp(-(meantime[indxtimefall] - timeflar) / scalfall)
    lcur *= amplflar / np.amax(lcur) 
    
    return lcur


def retr_imfa(cosi, rs2a, ecce, sinw):
    
    imfa = cosi / rs2a * (1. - ecce)**2 / (1. + ecce * sinw)

    return imfa


def retr_deptbeam(peri, massstar, masscomp):
    '''
    Calculate the beaming amplitude.
    '''
    
    deptbeam = 2.8 * peri**(-1. / 3.) * (massstar + masscomp)**(-2. / 3.) * masscomp # [ppt]
    
    return deptbeam


def retr_deptelli(peri, densstar, massstar, masscomp):
    '''
    Calculate the ellipsoidal variation amplitude.
    '''
    
    deptelli = 18.9 * peri**(-2.) / densstar * (1. / (1. + massstar / masscomp)) # [ppt]
    
    return deptelli


def retr_masscomp(amplslen, peri):
    
    print('temp: this mass calculation is an approximation.')
    masscomp = 1e-3 * amplslen / 7.15e-5 / gdat.radistar**(-2.) / peri**(2. / 3.) / (gdat.massstar)**(1. / 3.)
    
    return masscomp


def retr_smaxkepl(peri, masstotl, \
                  # a non-Keplerian factor, controlling the orbital period at a given semi-major axis
                  factnonk=1., \
                 ):
    '''
    Get the semi-major axis of a Keplerian orbit (in AU) from the orbital period (in days) and total mass (in Solar masses).

    Arguments
        peri: orbital period [days]
        masstotl: total mass of the system [Solar Masses]
    Returns
        smax: the semi-major axis of a Keplerian orbit [AU]
    '''
    
    smax = (7.496e-6 * masstotl * (peri / factnonk)**2)**(1. / 3.) # [AU]
    
    return smax


def retr_perikepl(smax, masstotl, \
                  # a non-Keplerian factor, controlling the orbital period at a given semi-major axis
                  factnonk=1., \
                 ):
    '''
    Get the period of a Keplerian orbit (in days) from the semi-major axis (in AU) and total mass (in Solar masses).

    Arguments
        smax: the semi-major axis of a Keplerian orbit [AU]
        masstotl: total mass of the system [Solar Masses]
    Returns
        peri: orbital period [days]
    '''
    
    peri = np.sqrt(smax**3 / 7.496e-6 / masstotl) * factnonk
    
    return peri


def retr_radiroch(radistar, densstar, denscomp):
    '''
    Return the Roche limit.

    Arguments
        radistar: radius of the primary star
        densstar: density of the primary star
        denscomp: density of the companion
    '''    
    radiroch = radistar * (2. * densstar / denscomp)**(1. / 3.)
    
    return radiroch


def retr_radihill(smax, masscomp, massstar):
    '''
    Return the Hill radius of a companion.

    Arguments
        peri: orbital period
        rsmacomp: the sum of radii of the two bodies divided by the semi-major axis
        cosi: cosine of the inclination
    '''    
    radihill = smax * (masscomp / 3. / massstar)**(1. / 3.) # [AU]
    
    return radihill


def retr_alphelli(u, g):
    
    alphelli = 0.15 * (15 + u) * (1 + g) / (3 - u)
    
    return alphelli


def plot_anim():

    pathbase = os.environ['PEXO_DATA_PATH'] + '/imag/'
    radistar = 0.9
    
    booldark = True
    
    boolsingside = False
    boolanim = True
    listtypevisu = ['real', 'cart']
    path = pathbase + 'orbt'
    
    for a in range(2):
    
        radicomp = [1.6, 2.1, 2.7, 3.1]
        rsmacomp = [0.0895, 0.0647, 0.0375, 0.03043]
        epoc = [2458572.1128, 2458572.3949, 2458571.3368, 2458586.5677]
        peri = [3.8, 6.2, 14.2, 19.6]
        cosi = [0., 0., 0., 0.]
        
        if a == 1:
            radicomp += [2.0]
            rsmacomp += [0.88 / (215. * 0.1758)]
            epoc += [2458793.2786]
            peri += [29.54115]
            cosi += [0.]
        
        for typevisu in listtypevisu:
            
            if a == 0:
                continue
    
            pexo.main.plot_orbt( \
                                path, \
                                radicomp, \
                                rsmacomp, \
                                epoc, \
                                peri, \
                                cosi, \
                                typevisu, \
                                radistar=radistar, \
                                boolsingside=boolsingside, \
                                boolanim=boolanim, \
                               )
        



