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
    dictquer = json.loads(outString)['data']
    
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
        pd.DataFrame.from_dict(dictquer).to_csv(path, index=False)#, columns=columns)
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
    
    dicttoii = {}
    dicttoii['TOIID'] = objtexof['TOI'].values
    numbcomp = dicttoii['TOIID'].size
    indxcomp = np.arange(numbcomp)
    toiitargexof = np.empty(numbcomp, dtype=object)
    for k in indxcomp:
        toiitargexof[k] = int(dicttoii['TOIID'][k])
        
    if toiitarg is None:
        indxcomp = np.arange(numbcomp)
    else:
        indxcomp = np.where(toiitargexof == toiitarg)[0]
    
    dicttoii['TOIID'] = dicttoii['TOIID'][indxcomp]
    
    numbcomp = indxcomp.size
    
    if indxcomp.size == 0:
        if typeverb > 0:
            print('The host name, %s, was not found in the ExoFOP TOI Catalog.' % toiitarg)
        return None
    else:
        dicttoii['namestar'] = np.empty(numbcomp, dtype=object)
        dicttoii['nametoii'] = np.empty(numbcomp, dtype=object)
        for kk, k in enumerate(indxcomp):
            dicttoii['nametoii'][kk] = 'TOI-' + str(dicttoii['TOIID'][kk])
            dicttoii['namestar'][kk] = 'TOI-' + str(dicttoii['TOIID'][kk])[:-3]
        
        dicttoii['depttrancomp'] = objtexof['Depth (ppm)'].values[indxcomp] * 1e-3 # [ppt]
        dicttoii['rratcomp'] = np.sqrt(dicttoii['depttrancomp'] * 1e-3)
        dicttoii[strgradielem] = objtexof['Planet Radius (R_Earth)'][indxcomp].values
        dicttoii['stdvradi' + strgelem] = objtexof['Planet Radius (R_Earth) err'][indxcomp].values
        
        rascstarstrg = objtexof['RA'][indxcomp].values
        declstarstrg = objtexof['Dec'][indxcomp].values
        dicttoii['rascstar'] = np.empty_like(dicttoii[strgradielem])
        dicttoii['declstar'] = np.empty_like(dicttoii[strgradielem])
        for k in range(dicttoii[strgradielem].size):
            objt = astropy.coordinates.SkyCoord('%s %s' % (rascstarstrg[k], declstarstrg[k]), unit=(astropy.units.hourangle, astropy.units.deg))
            dicttoii['rascstar'][k] = objt.ra.degree
            dicttoii['declstar'][k] = objt.dec.degree

        # a string holding the comments
        dicttoii['strgcomm'] = np.empty(numbcomp, dtype=object)
        dicttoii['strgcomm'][:] = objtexof['Comments'][indxcomp].values
        
        #objticrs = astropy.coordinates.SkyCoord(ra=dicttoii['rascstar']*astropy.units.degree, \
        #                                       dec=dicttoii['declstar']*astropy.units.degree, frame='icrs')
        
        objticrs = astropy.coordinates.SkyCoord(ra=dicttoii['rascstar'], \
                                               dec=dicttoii['declstar'], frame='icrs', unit='deg')
        
        # transit duration
        dicttoii['duratrantotl'] = objtexof['Duration (hours)'].values[indxcomp] # [hours]
        
        # galactic longitude
        dicttoii['lgalstar'] = np.array([objticrs.galactic.l])[0, :]
        
        # galactic latitude
        dicttoii['bgalstar'] = np.array([objticrs.galactic.b])[0, :]
        
        # ecliptic longitude
        dicttoii['loecstar'] = np.array([objticrs.barycentricmeanecliptic.lon.degree])[0, :]
        
        # ecliptic latitude
        dicttoii['laecstar'] = np.array([objticrs.barycentricmeanecliptic.lat.degree])[0, :]

        # SNR
        dicttoii['s2nr'] = objtexof['Planet SNR'][indxcomp].values
        
        dicttoii['numbobsvtime'] = objtexof['Time Series Observations'][indxcomp].values
        dicttoii['numbobsvspec'] = objtexof['Spectroscopy Observations'][indxcomp].values
        dicttoii['numbobsvimag'] = objtexof['Imaging Observations'][indxcomp].values
        # alert year
        dicttoii['yearaler'] = objtexof['Date TOI Alerted (UTC)'][indxcomp].values
        for k in range(len(dicttoii['yearaler'])):
            dicttoii['yearaler'][k] = astropy.time.Time(dicttoii['yearaler'][k] + ' 00:00:00.000').decimalyear
        dicttoii['yearaler'] = dicttoii['yearaler'].astype(float)

        dicttoii['tsmmacwg'] = objtexof['TSM'][indxcomp].values
        dicttoii['esmmacwg'] = objtexof['ESM'][indxcomp].values
    
        dicttoii['facidisc'] = np.empty(numbcomp, dtype=object)
        dicttoii['facidisc'][:] = 'Transiting Exoplanet Survey Satellite (TESS)'
        
        dicttoii['peri'+strgelem] = objtexof['Period (days)'][indxcomp].values
        dicttoii['peri'+strgelem][np.where(dicttoii['peri'+strgelem] == 0)] = np.nan

        dicttoii['epocmtra'+strgelem] = objtexof['Epoch (BJD)'][indxcomp].values

        dicttoii['tmagsyst'] = objtexof['TESS Mag'][indxcomp].values
        dicttoii['stdvtmagsyst'] = objtexof['TESS Mag err'][indxcomp].values

        # transit duty cycle
        dicttoii['dcyc'] = dicttoii['duratrantotl'] / dicttoii['peri'+strgelem] / 24.
        
        boolfrst = np.zeros(numbcomp)
        dicttoii['numb%sstar' % strgelem] = np.zeros(numbcomp)
        
        liststrgfeatstartici = ['massstar', 'vmagsyst', 'jmagsyst', 'hmagsyst', 'kmagsyst', 'distsyst', 'metastar', 'radistar', 'tmptstar', 'loggstar']
        liststrgfeatstarticiinhe = ['mass', 'Vmag', 'Jmag', 'Hmag', 'Kmag', 'd', 'MH', 'rad', 'Teff', 'logg']
        
        numbstrgfeatstartici = len(liststrgfeatstartici)
        indxstrgfeatstartici = np.arange(numbstrgfeatstartici)

        for strgfeat in liststrgfeatstartici:
            dicttoii[strgfeat] = np.zeros(numbcomp)
            dicttoii['stdv' + strgfeat] = np.zeros(numbcomp)
        
        ## crossmatch with TIC
        print('Retrieving TIC columns of TOI hosts...')
        dicttoii['TICID'] = objtexof['TIC ID'][indxcomp].values
        listticiuniq = np.unique(dicttoii['TICID'].astype(str))
        request = {'service':'Mast.Catalogs.Filtered.Tic', 'format':'json', 'params':{'columns':"*", \
                                                              'filters':[{'paramName':'ID', 'values':list(listticiuniq)}]}}
        headers, outString = quer_mast(request)
        listdictquer = json.loads(outString)['data']
        for k in range(len(listdictquer)):
            indxtemp = np.where(dicttoii['TICID'] == listdictquer[k]['ID'])[0]
            if indxtemp.size == 0:
                raise Exception('')
            for n in indxstrgfeatstartici:
                dicttoii[liststrgfeatstartici[n]][indxtemp] = listdictquer[k][liststrgfeatstarticiinhe[n]]
                dicttoii['stdv' + liststrgfeatstartici[n]][indxtemp] = listdictquer[k]['e_' + liststrgfeatstarticiinhe[n]]
        
        dicttoii['typedisptess'] = objtexof['TESS Disposition'][indxcomp].values
        dicttoii['boolfpos'] = objtexof['TFOPWG Disposition'][indxcomp].values == 'FP'
        
        # augment
        dicttoii['numb%sstar' % strgelem] = np.empty(numbcomp)
        boolfrst = np.zeros(numbcomp, dtype=bool)
        for kk, k in enumerate(indxcomp):
            indxcompthis = np.where(dicttoii['namestar'][kk] == dicttoii['namestar'])[0]
            if kk == indxcompthis[0]:
                boolfrst[kk] = True
            dicttoii['numb%sstar' % strgelem][kk] = indxcompthis.size
        
        dicttoii['numb%stranstar' % strgelem] = dicttoii['numb%sstar' % strgelem]
        dicttoii['lumistar'] = dicttoii['radistar']**2 * (dicttoii['tmptstar'] / 5778.)**4
        dicttoii['stdvlumistar'] = dicttoii['lumistar'] * np.sqrt((2 * dicttoii['stdvradistar'] / dicttoii['radistar'])**2 + \
                                                                        (4 * dicttoii['stdvtmptstar'] / dicttoii['tmptstar'])**2)
        
        # predicted mass from radii
        path = pathephe + 'data/exofop_toi_mass_saved.csv'
        if not os.path.exists(path):
            dicttemp = dict()
            dicttemp[strgmasselem] = np.ones_like(dicttoii[strgradielem]) + np.nan
            dicttemp['stdvmass' + strgelem] = np.ones_like(dicttoii[strgradielem]) + np.nan
            
            numbsamppopl = 10
            indx = np.where(np.isfinite(dicttoii[strgradielem]))[0]
            for n in tqdm(range(indx.size)):
                k = indx[n]
                meanvarb = dicttoii[strgradielem][k]
                stdvvarb = dicttoii['stdvradi' + strgelem][k]
                
                # if radius uncertainty is not available, assume that it is small, so the mass uncertainty will be dominated by population uncertainty
                if not np.isfinite(stdvvarb):
                    stdvvarb = 1e-3 * dicttoii[strgradielem][k]
                else:
                    stdvvarb = dicttoii['stdvradi' + strgelem][k]
                
                # sample from a truncated Gaussian
                listradicomp = tdpy.samp_gaustrun(1000, dicttoii[strgradielem][k], stdvvarb, 0., np.inf)
                
                # estimate the mass from samples
                listmassplan = retr_massfromradi(listradicomp)
                
                dicttemp[strgmasselem][k] = np.mean(listmassplan)
                dicttemp['stdvmass' + strgelem][k] = np.std(listmassplan)
                
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

        dicttoii[strgmasselem] = dicttemp['mass' + strgelem]
        
        perielem = dicttoii['peri'+strgelem]
        masselem = dicttoii['mass'+strgelem]

        dicttoii['rvelsemapred'] = retr_rvelsema(perielem, dicttoii['massstar'], masselem, 90., 0.)
        
        dicttoii['stdvmass' + strgelem] = dicttemp['stdvmass' + strgelem]
        
        dicttoii['masstotl'] = dicttoii['massstar'] + dicttoii[strgmasselem] / dictfact['msme']
        dicttoii['smax'+strgelem] = retr_smaxkepl(dicttoii['peri'+strgelem], dicttoii['masstotl'])
        
        dicttoii['rsma'+strgelem] = (dicttoii[strgradielem] / dictfact['rsre'] + dicttoii['radistar']) / (dictfact['aurs'] * dicttoii['smax'+strgelem])
        
        dicttoii['irra'] = dicttoii['lumistar'] / dicttoii['smax'+strgelem]**2
        
        dicttoii['tmpt%s' % strgelem] = dicttoii['tmptstar'] * np.sqrt(dicttoii['radistar'] / dicttoii['smax'+strgelem] / 2. / dictfact['aurs'])
        # temp check if factor of 2 is right
        dicttoii['stdvtmpt%s' % strgelem] = np.sqrt((dicttoii['stdvtmptstar'] / dicttoii['tmptstar'])**2 + \
                                                        0.5 * (dicttoii['stdvradistar'] / dicttoii['radistar'])**2) / np.sqrt(2.)
        
        dicttoii['dens%s' % strgelem] = 5.51 * dicttoii[strgmasselem] / dicttoii[strgradielem]**3 # [g/cm^3]
        dicttoii['booltran'] = np.ones_like(dicttoii['TOIID'], dtype=bool)
    
        dicttoii['vesc'] = retr_vesc(dicttoii[strgmasselem], dicttoii[strgradielem])
        print('temp: vsiistar and projoblq are NaNs')
        dicttoii['vsiistar'] = np.ones(numbcomp) + np.nan
        dicttoii['projoblq'] = np.ones(numbcomp) + np.nan
        
        # replace confirmed planet features
        if boolreplexar:
            dictexar = retr_dictexar()
            listdisptess = objtexof['TESS Disposition'][indxcomp].values.astype(str)
            listdisptfop = objtexof['TFOPWG Disposition'][indxcomp].values.astype(str)
            indxexofcpla = np.where((listdisptfop == 'CP') & (listdisptess == 'PC'))[0]
            listticicpla = dicttoii['TICID'][indxexofcpla]
            numbticicpla = len(listticicpla)
            indxticicpla = np.arange(numbticicpla)
            for k in indxticicpla:
                indxexartici = np.where((dictexar['TICID'] == int(listticicpla[k])) & \
                                                    (dictexar['facidisc'] == 'Transiting Exoplanet Survey Satellite (TESS)'))[0]
                indxexoftici = np.where(dicttoii['TICID'] == int(listticicpla[k]))[0]
                for strg in dictexar.keys():
                    if indxexartici.size > 0:
                        dicttoii[strg] = np.delete(dicttoii[strg], indxexoftici)
                    dicttoii[strg] = np.concatenate((dicttoii[strg], dictexar[strg][indxexartici]))

        # calculate TSM and ESM
        calc_tsmmesmm(dicttoii, strgelem=strgelem)
    
        # turn zero TSM ACWG or ESM ACWG into NaN
        indx = np.where(dicttoii['tsmmacwg'] == 0)[0]
        dicttoii['tsmmacwg'][indx] = np.nan
        
        indx = np.where(dicttoii['esmmacwg'] == 0)[0]
        dicttoii['esmmacwg'][indx] = np.nan

    return dicttoii


def calc_tsmmesmm(dictpopl, strgelem='comp', boolsamp=False):
    
    if boolsamp:
        numbsamp = 1000
    else:
        numbsamp = 1

    strgradielem = 'radi' + strgelem
    strgmasselem = 'mass' + strgelem
    
    numbcomp = dictpopl[strgmasselem].size
    listtsmm = np.empty((numbsamp, numbcomp)) + np.nan
    listesmm = np.empty((numbsamp, numbcomp)) + np.nan
    
    for n in range(numbcomp):
        
        if not np.isfinite(dictpopl['tmpt%s' % strgelem][n]):
            continue
        
        if not np.isfinite(dictpopl[strgradielem][n]):
            continue
        
        if boolsamp:
            if not np.isfinite(dictpopl['stdvradi' + strgelem][n]):
                stdv = dictpopl[strgradielem][n]
            else:
                stdv = dictpopl['stdvradi' + strgelem][n]
            listradicomp = tdpy.samp_gaustrun(numbsamp, dictpopl[strgradielem][n], stdv, 0., np.inf)
            
            listmassplan = tdpy.samp_gaustrun(numbsamp, dictpopl[strgmasselem][n], dictpopl['stdvmass' + strgelem][n], 0., np.inf)

            if not np.isfinite(dictpopl['stdvtmpt%s' % strgelem][n]):
                stdv = dictpopl['tmpt%s' % strgelem][n]
            else:
                stdv = dictpopl['stdvtmpt%s' % strgelem][n]
            listtmptplan = tdpy.samp_gaustrun(numbsamp, dictpopl['tmpt%s' % strgelem][n], stdv, 0., np.inf)
            
            if not np.isfinite(dictpopl['stdvradistar'][n]):
                stdv = dictpopl['radistar'][n]
            else:
                stdv = dictpopl['stdvradistar'][n]
            listradistar = tdpy.samp_gaustrun(numbsamp, dictpopl['radistar'][n], stdv, 0., np.inf)
            
            listkmagsyst = tdpy.icdf_gaus(np.random.rand(numbsamp), dictpopl['kmagsyst'][n], dictpopl['stdvkmagsyst'][n])
            listjmagsyst = tdpy.icdf_gaus(np.random.rand(numbsamp), dictpopl['jmagsyst'][n], dictpopl['stdvjmagsyst'][n])
            listtmptstar = tdpy.samp_gaustrun(numbsamp, dictpopl['tmptstar'][n], dictpopl['stdvtmptstar'][n], 0., np.inf)
        
        else:
            listradicomp = dictpopl[strgradielem][None, n]
            listtmptplan = dictpopl['tmpt%s' % strgelem][None, n]
            listmassplan = dictpopl[strgmasselem][None, n]
            listradistar = dictpopl['radistar'][None, n]
            listkmagsyst = dictpopl['kmagsyst'][None, n]
            listjmagsyst = dictpopl['jmagsyst'][None, n]
            listtmptstar = dictpopl['tmptstar'][None, n]
        
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
    dictpopl['tsmm'] = np.nanmedian(listtsmm, 0)
    dictpopl['stdvtsmm'] = np.nanstd(listtsmm, 0)
    dictpopl['esmm'] = np.nanmedian(listesmm, 0)
    dictpopl['stdvesmm'] = np.nanstd(listesmm, 0)
    
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

                             strgelem='plan')
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


def retr_rflxfromdmag(dmag, stdvdmag=None):
    
    rflx = 10**(-dmag / 2.5)

    if stdvdmag is not None:
        stdvrflx = np.log(10.) / 2.5 * rflx * stdvdmag
    
    return rflx, stdvrflx


def retr_dictexar( \
                  strgexar=None, \
                  
                  # type of verbosity
                  ## -1: absolutely no text
                  ##  0: no text output except critical warnings
                  ##  1: minimal description of the execution
                  ##  2: detailed description of the execution
                  typeverb=1, \
                  
                  strgelem='plan', \
                 ):
    
    strgradielem = 'radi' + strgelem
    strgstdvradi = 'stdv' + strgradielem
    strgmasselem = 'mass' + strgelem
    strgstdvmass = 'stdv' + strgmasselem
    
    strgstrgrefrradielem = 'strgrefrradi' + strgelem
    strgstrgrefrmasselem = 'strgrefrmass' + strgelem

    # get NASA Exoplanet Archive data
    path = os.environ['EPHESOS_DATA_PATH'] + '/data/PSCompPars_2023.01.07_17.02.16.csv'
    if typeverb > 0:
        print('Reading from %s...' % path)
    objtexar = pd.read_csv(path, skiprows=316)
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
        dictexar['namestar'] = objtexar['hostname'][indx].values
        dictexar['nameplan'] = objtexar['pl_name'][indx].values
        
        numbplanexar = len(dictexar['nameplan'])
        indxplanexar = np.arange(numbplanexar)

        listticitemp = objtexar['tic_id'][indx].values
        dictexar['TICID'] = np.empty(numbplanexar, dtype=int)
        for k in indxplanexar:
            if isinstance(listticitemp[k], str):
                dictexar['TICID'][k] = listticitemp[k][4:]
            else:
                dictexar['TICID'][k] = 0
        
        dictexar['rascstar'] = objtexar['ra'][indx].values
        dictexar['declstar'] = objtexar['dec'][indx].values
        
        # err1 have positive values or zero
        # err2 have negative values or zero
        
        dictexar['TOIID'] = np.empty(numbplanexar, dtype=object)
        
        # discovery method
        dictexar['methdisc'] = objtexar['discoverymethod'][indx].values
        
        # eccentricity
        dictexar['ecce'] = objtexar['pl_orbeccen'][indx].values
        
        # discovery facility
        dictexar['facidisc'] = objtexar['disc_facility'][indx].values
        
        # discovery year
        dictexar['yeardisc'] = objtexar['disc_year'][indx].values
        
        dictexar['irra'] = objtexar['pl_insol'][indx].values
        dictexar['pericomp'] = objtexar['pl_orbper'][indx].values # [days]
        dictexar['smaxcomp'] = objtexar['pl_orbsmax'][indx].values # [AU]
        dictexar['epocmtracomp'] = objtexar['pl_tranmid'][indx].values # [BJD]
        dictexar['cosicomp'] = np.cos(objtexar['pl_orbincl'][indx].values / 180. * np.pi)
        dictexar['duratrantotl'] = objtexar['pl_trandur'][indx].values # [hour]
        dictexar['depttrancomp'] = 10. * objtexar['pl_trandep'][indx].values # ppt
        
        # to be deleted
        #dictexar['boolfpos'] = np.zeros(numbplanexar, dtype=bool)
        
        dictexar['booltran'] = objtexar['tran_flag'][indx].values
        
        # mass provenance
        dictexar['strgprovmass'] = objtexar['pl_bmassprov'][indx].values
        
        dictexar['booltran'] = dictexar['booltran'].astype(bool)

        # radius reference
        dictexar[strgstrgrefrradielem] = objtexar['pl_rade_reflink'][indx].values
        for a in range(dictexar[strgstrgrefrradielem].size):
            if isinstance(dictexar[strgstrgrefrradielem][a], float) and not np.isfinite(dictexar[strgstrgrefrradielem][a]):
                dictexar[strgstrgrefrradielem][a] = ''
        
        # mass reference
        dictexar[strgstrgrefrmasselem] = objtexar['pl_bmasse_reflink'][indx].values
        for a in range(dictexar[strgstrgrefrmasselem].size):
            if isinstance(dictexar[strgstrgrefrmasselem][a], float) and not np.isfinite(dictexar[strgstrgrefrmasselem][a]):
                dictexar[strgstrgrefrmasselem][a] = ''

        for strg in ['radistar', 'massstar', 'tmptstar', 'loggstar', strgradielem, strgmasselem, 'tmpt'+strgelem, 'tagestar', \
                     'vmagsyst', 'jmagsyst', 'hmagsyst', 'kmagsyst', 'tmagsyst', 'metastar', 'distsyst', 'lumistar']:
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
            dictexar[strg] = objtexar[strgvarbexar][indx].values
            dictexar['stdv%s' % strg] = (objtexar['%serr1' % strgvarbexar][indx].values - objtexar['%serr2' % strgvarbexar][indx].values) / 2.
       
        dictexar['vesc'] = retr_vesc(dictexar[strgmasselem], dictexar[strgradielem])
        dictexar['masstotl'] = dictexar['massstar'] + dictexar[strgmasselem] / dictfact['msme']
        
        dictexar['densplan'] = objtexar['pl_dens'][indx].values # [g/cm3]
        dictexar['vsiistar'] = objtexar['st_vsin'][indx].values # [km/s]
        dictexar['projoblq'] = objtexar['pl_projobliq'][indx].values # [deg]
        
        # Boolean flag indicating if the planet is part of a circumbinary planetary system
        dictexar['boolcibp'] = objtexar['cb_flag'][indx].values == 1
        
        dictexar['numbplanstar'] = np.empty(numbplanexar)
        dictexar['numbplantranstar'] = np.empty(numbplanexar, dtype=int)
        boolfrst = np.zeros(numbplanexar, dtype=bool)
        #dictexar['booltrantotl'] = np.empty(numbplanexar, dtype=bool)
        for k, namestar in enumerate(dictexar['namestar']):
            indxexarstar = np.where(namestar == dictexar['namestar'])[0]
            if k == indxexarstar[0]:
                boolfrst[k] = True
            dictexar['numbplanstar'][k] = indxexarstar.size
            indxexarstartran = np.where((namestar == dictexar['namestar']) & dictexar['booltran'])[0]
            dictexar['numbplantranstar'][k] = indxexarstartran.size
            #dictexar['booltrantotl'][k] = dictexar['booltran'][indxexarstar].all()
        
        objticrs = astropy.coordinates.SkyCoord(ra=dictexar['rascstar'], \
                                               dec=dictexar['declstar'], frame='icrs', unit='deg')
        
        # transit duty cycle
        dictexar['dcyc'] = dictexar['duratrantotl'] / dictexar['pericomp'] / 24.
        
        # galactic longitude
        dictexar['lgalstar'] = np.array([objticrs.galactic.l])[0, :]
        
        # galactic latitude
        dictexar['bgalstar'] = np.array([objticrs.galactic.b])[0, :]
        
        # ecliptic longitude
        dictexar['loecstar'] = np.array([objticrs.barycentricmeanecliptic.lon.degree])[0, :]
        
        # ecliptic latitude
        dictexar['laecstar'] = np.array([objticrs.barycentricmeanecliptic.lat.degree])[0, :]

        # radius ratio
        dictexar['rratcomp'] = dictexar[strgradielem] / dictexar['radistar'] / dictfact['rsre']
        
        # sum of the companion and stellar radii divided by the semi-major axis
        dictexar['rsmacomp'] = (dictexar[strgradielem] / dictfact['rsre'] + dictexar['radistar']) / (dictexar['smaxcomp'] * dictfact['aurs'])

        # calculate TSM and ESM
        calc_tsmmesmm(dictexar, strgelem=strgelem)
        
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


def retr_noislsst(magtinpt):
    
    if np.isscalar(magtinpt):
        #magtinpt = np.array(magtinpt)
        magtinpt = np.full(1, magtinpt)
    if isinstance(magtinpt, float):
        magtinpt = np.array(magtinpt)
    
    nois = np.zeros_like(magtinpt) + np.inf
    
    indx = np.where((magtinpt < 20.) & (magtinpt > 15.))
    nois[indx] = 6. # [ppt]
    
    indx = np.where((magtinpt >= 20.) & (magtinpt < 24.))
    nois[indx] = 6. * 10**((magtinpt[indx] - 20.) / 3.) # [ppt]
    
    return nois


def retr_noistess(magtinpt, typeoutp='intplite', typeinst='TESS'):
    '''
    TESS photometric precision (over what time scale?)
    ''' 
    
    # interpolate literature values
    if typeoutp == 'intplite':
        nois = np.array([40., 40., 40., 90., 200., 700., 3e3, 2e4]) * 1e-3 # [ppt]
        magt = np.array([ 2.,  4.,  6.,  8.,  10.,  12., 14., 16.])
        objtspln = scipy.interpolate.interp1d(magt, nois, fill_value='extrapolate')
        nois = objtspln(magtinpt)
    if typeoutp == 'calcspoc':
        pass
    
    if typeinst == 'TESS' or typeinst == 'TGEO-IR':
        pass
    elif typeinst in ['TGEO-UV', 'TGEO-VIS']:
        nois *= 0.2
    else:
        print('typeinst')
        print(typeinst)
        raise Exception('')

    return nois


def retr_tmag(gdat, cntp):
    
    tmag = -2.5 * np.log10(cntp / 1.5e4 / gdat.listcade) + 10
    #tmag = -2.5 * np.log10(mlikfluxtemp) + 20.424
    
    #mlikmagttemp = 10**((mlikmagttemp - 20.424) / (-2.5))
    #stdvmagttemp = mlikmagttemp * stdvmagttemp / 1.09
    #gdat.stdvmagtrefr = 1.09 * gdat.stdvrefrrflx[o] / gdat.refrrflx[o]
    
    return tmag


def retr_subp(dictpopl, dictnumbsamp, dictindxsamp, namepoplinit, namepoplfinl, indx):
    
    if isinstance(indx, list):
        raise Exception('')

    if len(indx) == 0:
        indx = np.array([], dtype=int)

    if indx.size == 0:
        print('Warning! indx is zero.')

    dictpopl[namepoplfinl] = dict()
    for name in dictpopl[namepoplinit].keys():
        
        if indx.size > 0:
            dictpopl[namepoplfinl][name] = dictpopl[namepoplinit][name][indx]
        else:
            dictpopl[namepoplfinl][name] = np.array([])

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
                          minmnumbcompstar=None, \
                          
                          # maximum number of components per star
                          maxmnumbcompstar=None, \
                          
                          # minimum ratio of semi-major axis to radius of the host star
                          minmsmaxradistar=3., \
                          
                          # maximum ratio of semi-major axis to radius of the host star
                          maxmsmaxradistar=1e4, \
                          
                          # minimum mass of the companions
                          minmmasscomp=None, \
                          
                          # minimum orbital period, only taken into account when typesamporbtcomp == 'peri'
                          minmpericomp=0.1, \
                          
                          # maximum orbital period, only taken into account when typesamporbtcomp == 'peri'
                          maxmpericomp=1000., \
                          
                          # maximum cosine of inclination
                          maxmcosicomp=1., \
                          
                          # Boolean flag to include exomoons
                          boolinclmoon=False, \
                          
                          # Boolean flag to make the generative model produce Suns
                          booltoyysunn=False, \
                          
                          # Boolean flag to diagnose
                          booldiag=True, \
                          
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
    namepoplstartotl = 'star' + typepoplsyst + 'totl'
    namepoplstaroccu = 'star' + typepoplsyst + 'occu'
    namepoplcomptotl = 'compstar' + typepoplsyst + 'totl'
    namepoplcomptran = 'compstar' + typepoplsyst + 'tran'
    
    namepoplmoontotl = 'mooncompstar' + typepoplsyst + 'totl'
    
    dictpoplstar = dict()
    dictstarnumbsamp = dict()
    dictstarindxsamp = dict()
    dictstarnumbsamp[namepoplstartotl] = dict()
    dictstarindxsamp[namepoplstartotl] = dict()
    
    if boolhavecomp:
        dictpoplcomp = dict()
        dictcompnumbsamp = dict()
        dictcompindxsamp = dict()
        dictcompnumbsamp[namepoplcomptotl] = dict()
        dictcompindxsamp[namepoplcomptotl] = dict()
        dictpoplcomp[namepoplcomptotl] = dict()
        
        if boolhavemoon:
            dictpoplmoon = dict()
            dictmoonnumbsamp = dict()
            dictmoonindxsamp = dict()
            dictmoonnumbsamp[namepoplmoontotl] = dict()
            dictmoonindxsamp[namepoplmoontotl] = dict()
            dictpoplmoon[namepoplmoontotl] = dict()
    
    dictfact = tdpy.retr_factconv()
    
    # get the features of the star population
    if typepoplsyst.startswith('CTL') or typepoplsyst.startswith('TIC'):
        dictpoplstar[namepoplstartotl] = retr_dictpopltic8(typepoplsyst, numbsyst=numbsyst)
        
        print('Removing stars that do not have radii or masses...')
        indx = np.where(np.isfinite(dictpoplstar[namepoplstartotl]['radistar']) & \
                        np.isfinite(dictpoplstar[namepoplstartotl]['massstar']))[0]
        for name in dictpoplstar[namepoplstartotl].keys():
            dictpoplstar[namepoplstartotl][name] = dictpoplstar[namepoplstartotl][name][indx]

        if (dictpoplstar[namepoplstartotl]['rascstar'] > 1e4).any():
            raise Exception('')

        if (dictpoplstar[namepoplstartotl]['radistar'] == 0.).any():
            raise Exception('')

        dictpoplstar[namepoplstartotl]['densstar'] = 1.41 * dictpoplstar[namepoplstartotl]['massstar'] / dictpoplstar[namepoplstartotl]['radistar']**3
        dictpoplstar[namepoplstartotl]['idenstar'] = dictpoplstar[namepoplstartotl]['TICID']
    

    elif typepoplsyst == 'Synthetic':
        
        if numbsyst is None:
            numbsyst = 10000
        
        dictpoplstar[namepoplstartotl] = dict()
        
        dictpoplstar[namepoplstartotl]['distsyst'] = tdpy.icdf_powr(np.random.rand(numbsyst), 100., 7000., -2.)
        
        if booltoyysunn:
            dictpoplstar[namepoplstartotl]['radistar'] = np.ones(numbsyst)
            dictpoplstar[namepoplstartotl]['massstar'] = np.ones(numbsyst)
            dictpoplstar[namepoplstartotl]['densstar'] = 1.4 * np.ones(numbsyst)
        else:
            dictpoplstar[namepoplstartotl]['massstar'] = tdpy.icdf_powr(np.random.rand(numbsyst), 0.1, 10., 2.)
            dictpoplstar[namepoplstartotl]['densstar'] = 1.4 * (1. / dictpoplstar[namepoplstartotl]['massstar'])**(0.7)
            dictpoplstar[namepoplstartotl]['radistar'] = (1.4 * dictpoplstar[namepoplstartotl]['massstar'] / dictpoplstar[namepoplstartotl]['densstar'])**(1. / 3.)
        
        dictpoplstar[namepoplstartotl]['coeflmdklinr'] = 0.4 * np.ones_like(dictpoplstar[namepoplstartotl]['densstar'])
        dictpoplstar[namepoplstartotl]['coeflmdkquad'] = 0.25 * np.ones_like(dictpoplstar[namepoplstartotl]['densstar'])

        dictpoplstar[namepoplstartotl]['lumistar'] = dictpoplstar[namepoplstartotl]['massstar']**4
        
        dictpoplstar[namepoplstartotl]['tmag'] = 1. * (-2.5) * np.log10(dictpoplstar[namepoplstartotl]['lumistar'] / dictpoplstar[namepoplstartotl]['distsyst']**2)
        
        if typepoplsyst == 'lsstwfds':
            dictpoplstar[namepoplstartotl]['rmag'] = -2.5 * np.log10(dictpoplstar[namepoplstartotl]['lumistar'] / dictpoplstar[namepoplstartotl]['distsyst']**2)
            
            indx = np.where((dictpoplstar[namepoplstartotl]['rmag'] < 24.) & (dictpoplstar[namepoplstartotl]['rmag'] > 15.))[0]
            for name in ['distsyst', 'rmag', 'massstar', 'densstar', 'radistar', 'lumistar']:
                dictpoplstar[namepoplstartotl][name] = dictpoplstar[namepoplstartotl][name][indx]

    else:
        print('')
        print('')
        print('')
        print('typepoplsyst')
        print(typepoplsyst)
        raise Exception('Undefined typepoplsyst.')
    
    # number of stars
    numbstar = dictpoplstar[namepoplstartotl]['radistar'].size
    
    dictstarnumbsamp[namepoplstartotl] = numbstar

    numbsyst = len(dictpoplstar[namepoplstartotl]['radistar'])
    indxsyst = np.arange(numbsyst)

    # total mass
    dictpoplstar[namepoplstartotl]['masssyst'] = np.copy(dictpoplstar[namepoplstartotl]['massstar'])
    
    if boolhavecomp:
        if typesyst.startswith('PlanetarySystem'):
            
            # mean number of companions per star
            dictpoplstar[namepoplstartotl]['numbcompstarmean'] = 0.5 * dictpoplstar[namepoplstartotl]['massstar']**(-1.)
            
            # number of companions per star
            dictpoplstar[namepoplstartotl]['numbcompstar'] = np.random.poisson(dictpoplstar[namepoplstartotl]['numbcompstarmean'])
            
            if booldiag:
                if maxmnumbcompstar is not None and minmnumbcompstar is not None:
                    if maxmnumbcompstar < minmnumbcompstar:
                        print('')
                        print('')
                        print('')
                        raise Exception('maxmnumbcompstar < minmnumbcompstar')

            if minmnumbcompstar is not None:
                dictpoplstar[namepoplstartotl]['numbcompstar'] = np.maximum(dictpoplstar[namepoplstartotl]['numbcompstar'], minmnumbcompstar)

            if maxmnumbcompstar is not None:
                dictpoplstar[namepoplstartotl]['numbcompstar'] = np.minimum(dictpoplstar[namepoplstartotl]['numbcompstar'], maxmnumbcompstar)

        elif typesyst == 'CompactObjectStellarCompanion' or typesyst == 'StellarBinary':
            # number of companions per star
            dictpoplstar[namepoplstartotl]['numbcompstar'] = np.ones(dictpoplstar[namepoplstartotl]['radistar'].size).astype(int)
            
        else:
            print('')
            print('')
            print('')
            print('typesyst')
            print(typesyst)
            raise Exception('typesyst is undefined.')
    
    if boolflar:
        
        # mean number of flares per star
        dictpoplstar[namepoplstartotl]['numbflarstarmean'] = 0.5 * dictpoplstar[namepoplstartotl]['massstar']**(-1.)
        
        # number of flares per star
        dictpoplstar[namepoplstartotl]['numbflarstar'] = np.random.poisson(dictpoplstar[namepoplstartotl]['numbflarstarmean'])
        
    if boolhavecomp:
        # Boolean flag of occurence
        dictpoplstar[namepoplstartotl]['booloccu'] = dictpoplstar[namepoplstartotl]['numbcompstar'] > 0
    
        # subpopulation where companions occur
        indx = np.where(dictpoplstar[namepoplstartotl]['booloccu'])[0]
        retr_subp(dictpoplstar, dictstarnumbsamp, dictstarindxsamp, namepoplstartotl, namepoplstaroccu, indx)
    
        if minmmasscomp is None:
            if boolsystcosc:
                minmmasscomp = 5. # [Solar mass]
            elif typesyst == 'PlanetarySystem' or typesyst == 'PlanetarySystemWithPhaseCurve' or typesyst == 'PlanetarySystemWithMoons':
                if typesyst == 'PlanetarySystem' or typesyst == 'PlanetarySystemWithMoons':
                    # ~ Mars mass
                    minmmasscomp = 0.1 # [Earth mass]
                if typesyst == 'PlanetarySystemWithPhaseCurve':
                    minmmasscomp = 30. # [Earth mass]
            elif typesyst == 'StellarBinary':
                minmmasscomp = 0.5 # [Earth mass]
    
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

        print('Sampling companion features...')
    
        # indices of companions for each star
        indxcompstar = [[] for k in indxsyst]
        cntr = 0
        for k in range(len(dictpoplstar[namepoplstartotl]['radistar'])):
            indxcompstar[k] = np.arange(cntr, cntr + dictpoplstar[namepoplstartotl]['numbcompstar'][k]).astype(int)
            cntr += dictpoplstar[namepoplstartotl]['numbcompstar'][k]
        dictcompnumbsamp[namepoplcomptotl] = cntr
    
        # prepare to load star features into component features
        for name in list(dictpoplstar[namepoplstartotl].keys()):
            dictpoplcomp[namepoplcomptotl][name] = np.empty(dictcompnumbsamp[namepoplcomptotl])
    
        listnamecomp = ['masssyst', 'radistar']
        listnamecomp += ['pericomp', 'cosicomp', 'smaxcomp', 'eccecomp', 'arpacomp', 'loancomp', 'masscomp', 'epocmtracomp']
        if typesyst == 'PlanetarySystemWithMoons':
            listnamecomp += ['masscompmoon']
        if typesyst == 'PlanetarySystemWithNonKeplerianObjects':
            listnamecomp += ['factnonkcomp']
        if not boolsystcosc:
            listnamecomp += ['radicomp', 'denscomp']
        for name in listnamecomp:
            dictpoplcomp[namepoplcomptotl][name] = np.empty(dictcompnumbsamp[namepoplcomptotl])

        if booldiag:
            cntr = 0
            for k in range(len(indxcompstar)):
                cntr += indxcompstar[k].size
            if cntr != dictcompnumbsamp[namepoplcomptotl]:
                raise Exception('')
    
        for k in tqdm(range(numbstar)):
            
            if dictpoplstar[namepoplstartotl]['numbcompstar'][k] == 0:
                continue

            # load star features into component features
            for name in dictpoplstar[namepoplstartotl].keys():
                dictpoplcomp[namepoplcomptotl][name][indxcompstar[k]] = dictpoplstar[namepoplstartotl][name][k]
            
            # eccentricities
            dictpoplcomp[namepoplcomptotl]['eccecomp'][indxcompstar[k]] = np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k])
            
            # arguments of periapsis
            dictpoplcomp[namepoplcomptotl]['arpacomp'][indxcompstar[k]] = 2. * np.pi * np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k])
            
            # cosine of orbital inclinations
            dictpoplcomp[namepoplcomptotl]['cosicomp'][indxcompstar[k]] = np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k]) * maxmcosicomp
            
            # longtides of ascending node
            dictpoplcomp[namepoplcomptotl]['loancomp'][indxcompstar[k]] = 2. * np.pi * np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k])
            
            # companion mass
            dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k]] = tdpy.util.icdf_powr(np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k]), \
                                                                                                                                          minmmasscomp, maxmmasscomp, 2.)
            
            if boolsystpsys or typesyst == 'StellarBinary':
                # companion radius
                dictpoplcomp[namepoplcomptotl]['radicomp'][indxcompstar[k]] = retr_radifrommass(dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k]])
        
                # companion density
                dictpoplcomp[namepoplcomptotl]['denscomp'][indxcompstar[k]] = 5.51 * dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k]] / \
                                                                                                         dictpoplcomp[namepoplcomptotl]['radicomp'][indxcompstar[k]]**3
            
            # total mass
            if boolsystcosc or typesyst == 'StellarBinary':
                dictpoplstar[namepoplstartotl]['masssyst'][k] += np.sum(dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k]])
            
            if typesamporbtcomp == 'peri':
            
                ratiperi = tdpy.util.icdf_powr(np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k] - 1), 1.2, 1.3, 5.)
                
                listpericomp = []
                for mm in range(dictpoplstar[namepoplstartotl]['numbcompstar'][k]):
                    if mm == 0:
                        peri = minmpericomp
                    else:
                        peri = ratiperi[mm-1] * listpericomp[mm-1]
                    listpericomp.append(peri)
                dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]] = np.array(listpericomp)

                if booldiag:
                    ratiperi = dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]][1:] / dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]][:-1]
                    indx = np.where(ratiperi < 1.2)[0]
                    if indx.size > 0:
                        print('indx')
                        summgene(indx)
                        print('dictpoplcomp[namepoplcomptotl][pericomp][indxcompstar[k]]')
                        print(dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]])
                        print('dictpoplstar[namepoplstartotl][numbcompstar][k]')
                        print(dictpoplstar[namepoplstartotl]['numbcompstar'][k])
                        raise Exception('')
                
                if typesyst == 'PlanetarySystemWithNonKeplerianObjects':
                    raise Exception('')
                else:
                    factnonk = 1.

                dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k]] = retr_smaxkepl(dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]], \
                                                                                                                            dictpoplstar[namepoplstartotl]['masssyst'][k], factnonk=factnonk)
            
            else:
                # semi-major axes
                #if np.isfinite(dictpoplstar[namepoplstartotl]['densstar'][k]):
                #    densstar = dictpoplstar[namepoplstartotl]['densstar'][k]
                #else:
                #    densstar = 1.
                #dictpoplcomp[namepoplcomptotl]['radiroch'][k] = retr_radiroch(radistar, densstar, denscomp)
                #minmsmax = 2. * dictpoplcomp[namepoplcomptotl]['radiroch'][k]
                dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k]] = dictpoplstar[namepoplstartotl]['radistar'][k] * \
                                                                             tdpy.util.icdf_powr(np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k]), \
                                                                                                minmsmaxradistar, maxmsmaxradistar, 2.) / dictfact['aurs']
                
                if typesyst == 'PlanetarySystemWithNonKeplerianObjects':
                    factnonk = tdpy.util.icdf_powr(np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k]), 0.1, 1., -2.)
                    dictpoplcomp[namepoplcomptotl]['factnonkcomp'][indxcompstar[k]] = factnonk
                else:
                    factnonk = 1.
                
                dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]] = retr_perikepl(dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k]], \
                                                                                                                            dictpoplstar[namepoplstartotl]['masssyst'][k], factnonk=factnonk)
                
        
            
            if booldiag:
                if not np.isfinite(dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]]).all():
                    
                    print('dictpoplcomp[namepoplcomptotl][masscomp][indxcompstar[k]]')
                    print(dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k]])
                    print('dictpoplcomp[namepoplcomptotl][masssyst][indxcompstar[k]]')
                    print(dictpoplcomp[namepoplcomptotl]['masssyst'][indxcompstar[k]])
                    print('dictpoplcomp[namepoplcomptotl][smaxcomp][indxcompstar[k]]')
                    print(dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k]])
                    print('dictpoplcomp[namepoplcomptotl][pericomp][indxcompstar[k]]')
                    print(dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k]])
                    raise Exception('')

            # conjunction epochs
            if epocmtracomp is not None:
                dictpoplcomp[namepoplcomptotl]['epocmtracomp'][indxcompstar[k]] = np.full(dictpoplstar[namepoplstartotl]['numbcompstar'][k], epocmtracomp)
            else:
                dictpoplcomp[namepoplcomptotl]['epocmtracomp'][indxcompstar[k]] = 1e8 * np.random.rand(dictpoplstar[namepoplstartotl]['numbcompstar'][k])
            if timeepoc is not None:
                dictpoplcomp[namepoplcomptotl]['epocmtracomp'][indxcompstar[k]] = dictpoplcomp[namepoplcomptotl]['epocmtracomp'][k] + \
                                               dictpoplcomp[namepoplcomptotl]['pericomp'][k] * \
                                               np.round((dictpoplcomp[namepoplcomptotl]['epocmtracomp'][k] - timeepoc) / dictpoplcomp[namepoplcomptotl]['pericomp'][k])
    
        if typesyst == 'PlanetarySystemWithMoons':
            # initialize the total mass of the companion + moons system as the mass of the companion
            dictpoplcomp[namepoplcomptotl]['masscompmoon'] = np.copy(dictpoplcomp[namepoplcomptotl]['masscomp'])
                    
        rsum = dictpoplcomp[namepoplcomptotl]['radistar']
        if not boolsystcosc:
            rsum += dictpoplcomp[namepoplcomptotl]['radicomp'] / dictfact['rsre']    
        dictpoplcomp[namepoplcomptotl]['rsmacomp'] = rsum / dictpoplcomp[namepoplcomptotl]['smaxcomp'] / dictfact['aurs']
        
        # orbital inclinations of the companions
        dictpoplcomp[namepoplcomptotl]['inclcomp'] = 180. / np.pi * np.arccos(dictpoplcomp[namepoplcomptotl]['cosicomp'])
        
        dictpoplcomp[namepoplcomptotl]['inclcomp'] = 90. + (dictpoplcomp[namepoplcomptotl]['inclcomp'] - 90.) * \
                                                                        (2 * np.random.randint(2, size=dictpoplcomp[namepoplcomptotl]['cosicomp'].size) - 1.)

        if boolsystpsys:
            
            if booldiag:
                
                if not np.isfinite(dictpoplcomp[namepoplcomptotl]['radistar']).all():
                    print('')
                    print('')
                    print('')
                    raise Exception('not np.isfinite(dictpoplcomp[namepoplcomptotl][radistar]).all()')
                
                if not np.isfinite(dictpoplcomp[namepoplcomptotl]['radicomp']).all():
                    print('')
                    print('')
                    print('')
                    raise Exception('not np.isfinite(dictpoplcomp[namepoplcomptotl][radicomp]).all()')

            # radius ratio
            dictpoplcomp[namepoplcomptotl]['rratcomp'] = dictpoplcomp[namepoplcomptotl]['radicomp'] / dictpoplcomp[namepoplcomptotl]['radistar'] / dictfact['rsre']
            
            if booldiag:
                if not np.isfinite(dictpoplcomp[namepoplcomptotl]['rratcomp']).all():
                    print('')
                    print('')
                    print('')
                    raise Exception('not np.isfinite(dictpoplcomp[namepoplcomptotl][rratcomp]).all()')
                
        # Boolean flag indicating whether a companion is transiting
        dictpoplcomp[namepoplcomptotl]['booltran'] = dictpoplcomp[namepoplcomptotl]['rsmacomp'] > dictpoplcomp[namepoplcomptotl]['cosicomp']

        # subpopulation where object transits
        indx = np.where(dictpoplcomp[namepoplcomptotl]['booltran'])[0]
        retr_subp(dictpoplcomp, dictcompnumbsamp, dictcompindxsamp, namepoplcomptotl, namepoplcomptran, indx)

        # transit duration
        dictpoplcomp[namepoplcomptran]['duratrantotl'] = retr_duratrantotl(dictpoplcomp[namepoplcomptran]['pericomp'], \
                                                                       dictpoplcomp[namepoplcomptran]['rsmacomp'], \
                                                                       dictpoplcomp[namepoplcomptran]['cosicomp'])
        dictpoplcomp[namepoplcomptran]['dcyc'] = dictpoplcomp[namepoplcomptran]['duratrantotl'] / dictpoplcomp[namepoplcomptran]['pericomp'] / 24.
        
        if boolsystcosc:
            # amplitude of self-lensing
            dictpoplcomp[namepoplcomptran]['amplslen'] = chalcedon.retr_amplslen(dictpoplcomp[namepoplcomptran]['pericomp'], dictpoplcomp[namepoplcomptran]['radistar'], \
                                                                                dictpoplcomp[namepoplcomptran]['masscomp'], dictpoplcomp[namepoplcomptran]['massstar'])
        
        if typesyst == 'PlanetarySystem':
            # transit depth
            dictpoplcomp[namepoplcomptran]['depttrancomp'] = 1e3 * dictpoplcomp[namepoplcomptran]['rratcomp']**2 # [ppt]
        
        # define parent population's features that are valid only for transiting systems
        listtemp = ['duratrantotl', 'dcyc']
        if typesyst == 'PlanetarySystem':
            listtemp += ['depttrancomp']
        if boolsystcosc:
            listtemp += ['amplslen']
        for strg in listtemp:
            dictpoplcomp[namepoplcomptotl][strg] = np.full_like(dictpoplcomp[namepoplcomptotl]['pericomp'], np.nan)
            dictpoplcomp[namepoplcomptotl][strg][indx] = dictpoplcomp[namepoplcomptran][strg]

        dictcompnumbsamp[namepoplcomptotl] = dictpoplcomp[namepoplcomptotl]['radistar'].size
        
        indxmooncompstar = [[[] for j in indxcompstar[k]] for k in indxsyst]
        if typesyst == 'PlanetarySystemWithMoons':
            # Hill radius of the companion
            dictpoplcomp[namepoplcomptotl]['radihill'] = retr_radihill(dictpoplcomp[namepoplcomptotl]['smaxcomp'], \
                                                                        dictpoplcomp[namepoplcomptotl]['masscomp'] / dictfact['msme'], \
                                                                        dictpoplcomp[namepoplcomptotl]['massstar'])
        
            # maximum semi-major axis of the moons 
            dictpoplcomp[namepoplcomptotl]['maxmsmaxmoon'] = 0.2 * dictpoplcomp[namepoplcomptotl]['radihill']
                
            # mean number of moons per companion
            dictpoplcomp[namepoplcomptotl]['numbmooncompmean'] = 1000. * dictpoplcomp[namepoplcomptotl]['masscomp']**(-1.)
            
            # number of moons per companion
            #dictpoplcomp[namepoplcomptotl]['numbmooncomp'] = np.random.poisson(dictpoplcomp[namepoplcomptotl]['numbmooncompmean'])
            print('hede')
            print('temp')
            print('hede')
            dictpoplcomp[namepoplcomptotl]['numbmooncomp'] = np.ones_like(dictpoplcomp[namepoplcomptotl]['numbmooncompmean'])
            
            cntr = 0
            for k in range(len(dictpoplstar[namepoplstartotl]['radistar'])):
                for j in range(len(indxcompstar[k])):
                    indxmooncompstar[k][j] = np.arange(cntr, cntr + dictpoplcomp[namepoplcomptotl]['numbmooncomp'][indxcompstar[k][j]]).astype(int)
                    cntr += int(dictpoplcomp[namepoplcomptotl]['numbmooncomp'][j])
            dictmoonnumbsamp[namepoplmoontotl] = cntr
        
            numbmoontotl = int(np.sum(dictpoplcomp[namepoplcomptotl]['numbmooncomp']))
            
            # prepare to load component features into moon features
            for name in list(dictpoplcomp[namepoplcomptotl].keys()):
                dictpoplmoon[namepoplmoontotl][name] = np.empty(dictmoonnumbsamp[namepoplmoontotl])
        
            for k in tqdm(range(numbstar)):
                
                if dictpoplstar[namepoplstartotl]['numbcompstar'][k] == 0:
                    continue
                
                numbcomp = dictpoplstar[namepoplstartotl]['numbcompstar'][k]
                
                # number of exomoons to the companion
                numbmoon = dictpoplcomp[namepoplcomptotl]['numbmooncomp'][indxcompstar[k]].astype(int)
                for name in ['radi', 'mass', 'dens', 'peri', 'epocmtra', 'smax', 'minmsmax']:
                    dictpoplmoon[namepoplmoontotl][name+'moon'] = np.empty(numbmoontotl)
                
                indxmoon = [[] for j in indxcompstar[k]]
                for j in range(indxcompstar[k].size):
                    
                    #print('')
                    #print('')
                    #print('')
                    #print('')
                    #print('j')
                    #print(j)
                    #print('indxcompstar[k]')
                    #print(indxcompstar[k])
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
                    dictpoplmoon[namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]] = dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k][j]] * \
                                                   tdpy.icdf_powr(np.random.rand(int(dictpoplcomp[namepoplcomptotl]['numbmooncomp'][indxcompstar[k][j]])), 0.005, 0.1, 2.)
                    
                    ## radii [R_E]
                    dictpoplmoon[namepoplmoontotl]['radimoon'][indxmooncompstar[k][j]] = \
                                                retr_radifrommass(dictpoplmoon[namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]])
                    
                    ## densities [g/cm^3]
                    dictpoplmoon[namepoplmoontotl]['densmoon'][indxmooncompstar[k][j]] = 5.51 * dictpoplmoon[namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]] / \
                                                                                                  dictpoplmoon[namepoplmoontotl]['radimoon'][indxmooncompstar[k][j]]**3
                    
                    # minimum semi-major axes for the moons
                    dictpoplmoon[namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j]] = (1. / dictfact['rsre'] / dictfact['aurs']) * \
                                                                                                    retr_radiroch(dictpoplcomp[namepoplcomptotl]['radicomp'][indxcompstar[k][j]], \
                                                                                                                dictpoplcomp[namepoplcomptotl]['denscomp'][indxcompstar[k][j]], \
                                                                                                                dictpoplmoon[namepoplmoontotl]['densmoon'][indxmooncompstar[k][j]])
                    
                    if booldiag:
                        if (dictpoplmoon[namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j]] >= \
                                                                dictpoplcomp[namepoplcomptotl]['maxmsmaxmoon'][indxcompstar[k][j]]).any():
                            print('')
                            print('')
                            print('')
                            print('typesamporbtcomp')
                            print(typesamporbtcomp)
                            print('dictpoplcomp[namepoplcomptotl][massstar][k]')
                            print(dictpoplcomp[namepoplcomptotl]['massstar'][k])
                            print('dictpoplcomp[namepoplcomptotl][masssyst][k]')
                            print(dictpoplcomp[namepoplcomptotl]['masssyst'][k])
                            print('dictpoplcomp[namepoplcomptotl][pericomp][indxcompstar[k][j]]')
                            print(dictpoplcomp[namepoplcomptotl]['pericomp'][indxcompstar[k][j]])
                            print('dictpoplcomp[namepoplcomptotl][smaxcomp][indxcompstar[k][j]]')
                            print(dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k][j]])
                            print('dictpoplcomp[namepoplcomptotl][radihill][indxcompstar[k][j]]')
                            print(dictpoplcomp[namepoplcomptotl]['radihill'][indxcompstar[k][j]])
                            print('dictpoplcomp[namepoplcomptotl][maxmsmaxmoon][indxcompstar[k][j]]')
                            print(dictpoplcomp[namepoplcomptotl]['maxmsmaxmoon'][indxcompstar[k][j]])
                            print('dictpoplmoon[namepoplmoontotl][minmsmaxmoon][indxmooncompstar[k][j]]')
                            print(dictpoplmoon[namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j]])
                            print('dictpoplmoon[namepoplmoontotl][radimoon][indxmooncompstar[k][j]]')
                            print(dictpoplmoon[namepoplmoontotl]['radimoon'][indxmooncompstar[k][j]])
                            print('dictpoplmoon[namepoplmoontotl][massmoon][indxmooncompstar[k][j]]')
                            print(dictpoplmoon[namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]])
                            print('dictpoplmoon[namepoplmoontotl][densmoon][indxmooncompstar[k][j]]')
                            print(dictpoplmoon[namepoplmoontotl]['densmoon'][indxmooncompstar[k][j]])
                            print('dictpoplcomp[namepoplcomptotl][radicomp][indxcompstar[k][j]]')
                            print(dictpoplcomp[namepoplcomptotl]['radicomp'][indxcompstar[k][j]])
                            print('dictpoplcomp[namepoplcomptotl][masscomp][indxcompstar[k][j]]')
                            print(dictpoplcomp[namepoplcomptotl]['masscomp'][indxcompstar[k][j]])
                            print('dictpoplcomp[namepoplcomptotl][denscomp][indxcompstar[k][j]]')
                            print(dictpoplcomp[namepoplcomptotl]['denscomp'][indxcompstar[k][j]])
                            raise Exception('Minimum semi-major axes for the moons are greater than the maximum semi-major axes.')
                    
                    # semi-major axes of the moons
                    for jj in indxmoon[j]:
                        dictpoplmoon[namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j][jj]] = tdpy.icdf_powr(np.random.rand(), \
                                                                                            dictpoplmoon[namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j][jj]], \
                                                                                            dictpoplcomp[namepoplcomptotl]['maxmsmaxmoon'][indxcompstar[k][j]], 2.)
                  
                    # add the moon masses to the total mass of the companion + moons system
                    dictpoplcomp[namepoplcomptotl]['masscompmoon'][indxcompstar[k][j]] += np.sum(dictpoplmoon[namepoplmoontotl]['massmoon'][indxmooncompstar[k][j]])
                    
                    # orbital period of the moons
                    dictpoplmoon[namepoplmoontotl]['perimoon'][indxmooncompstar[k][j]] = retr_perikepl(dictpoplmoon[namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j]], \
                                                                                                    dictpoplcomp[namepoplcomptotl]['masscompmoon'][indxcompstar[k][j]] / dictfact['msme'])
                    
                    # load component features into moon features
                    ## temp the code crashed here once
                    for name in dictpoplcomp[namepoplcomptotl].keys():
                        dictpoplmoon[namepoplmoontotl][name][indxmooncompstar[k][j]] = dictpoplcomp[namepoplcomptotl][name][indxcompstar[k][j]]
                    
                    if booldiag:
                        if (dictpoplmoon[namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j]] > 0.7 * dictpoplcomp[namepoplcomptotl]['radihill'][indxcompstar[k][j]]).any():
                        
                        
                            print('')
                            print('')
                            print('')
                            print('numbmoon[j]')
                            print(numbmoon[j])
                            print('dictpoplcomp[namepoplcomptotl][radihill][indxcompstar[k][j]]')
                            print(dictpoplcomp[namepoplcomptotl]['radihill'][indxcompstar[k][j]])
                            
                            print('dictpoplmoon[namepoplmoontotl][smaxmoon][indxmooncompstar[k][j]]')
                            print(dictpoplmoon[namepoplmoontotl]['smaxmoon'][indxmooncompstar[k][j]])
                            
                            print('dictpoplcomp[namepoplcomptotl][smaxcomp][indxcompstar[k][j]]')
                            print(dictpoplcomp[namepoplcomptotl]['smaxcomp'][indxcompstar[k][j]])
                            
                            print('dictpoplmoon[namepoplmoontotl][minmsmaxmoon][indxmooncompstar[k][j]]')
                            print(dictpoplmoon[namepoplmoontotl]['minmsmaxmoon'][indxmooncompstar[k][j]])
                            print('dictpoplcomp[namepoplcomptotl][maxmsmaxmoon][indxcompstar[k][j]]')
                            print(dictpoplcomp[namepoplcomptotl]['maxmsmaxmoon'][indxcompstar[k][j]])
                            
                            #print('dictpoplcomp[namepoplcomptotl][radicomp][indxcompstar[k][j]]')
                            #print(dictpoplcomp[namepoplcomptotl]['radicomp'][indxcompstar[k][j]])
                            #print('dictpoplcomp[namepoplcomptotl][denscomp][indxcompstar[k][j]]')
                            #print(dictpoplcomp[namepoplcomptotl]['denscomp'][indxcompstar[k][j]])
                            #print('dictpoplmoon[namepoplmoontotl][densmoon][indxmooncompstar[k][j]]')
                            #print(dictpoplmoon[namepoplmoontotl]['densmoon'][indxmooncompstar[k][j]])
                            
                            raise Exception('Semi-major axis of a moon is larger than 0.7 times the Hill radius of the companion.')
        
                # planet-moon conjunction times
                dictpoplmoon[namepoplmoontotl]['epocmtramoon'] = 1e8 * np.random.rand(numbmoontotl)
    
    dictnico = dict()
    dictnico['dictpoplstar'] = dictpoplstar
    if boolhavecomp:
        dictnico['dictpoplcomp'] = dictpoplcomp
        dictnico['dictpoplmoon'] = dictpoplmoon
        dictnico['dictcompnumbsamp'] = dictcompnumbsamp
        dictnico['dictcompindxsamp'] = dictcompindxsamp
        dictnico['indxcompstar'] = indxcompstar
        dictnico['indxmooncompstar'] = indxmooncompstar
    if boolflar:
        dictnico['dictpoplflar'] = dictpoplflar
        dictnico['indxflarstar'] = indxflarstar
    
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
        



