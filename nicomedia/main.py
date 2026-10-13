from tdpy.verbosity import print
import os, sys, json
import numpy as np
import pandas as pd
from tdpy.verbosity import tqdm

import astropy
import astroquery
import scipy

import tdpy
from tdpy import retr_doubgaus, retr_doubking, retr_gausking, retr_singgaus, retr_singking, retr_xposypos
from tdpy import summgene
from tdpy.exoplanet import (
    keplerian_radial_velocity,
    kipping_to_quadratic_limb_darkening,
    quadratic_limb_darkening,
    quadratic_to_kipping_limb_darkening,
)
from tdpy.population import subset_population
from tdpy.spectral import evaluate_continuum_spectrum, evaluate_line_profile
import chalcedon
from tdpy.paths import RepositoryPaths
from nicomedia.paths import get_visuals_path


def _repository_data_path(repository, *parts):
    return os.path.join(str(RepositoryPaths(repository + '_PATH').get_data_path()), *parts)


retr_psfnwdth = tdpy.retr_psfnwdth
retr_gang = tdpy.retr_gang
retr_aang = tdpy.retr_aang


def retr_psfn(
    dictpara,
    indxenertemp,
    arryangl,
    typemodlpsfn,
    typenormangl='none',
    booldiag=True,
    *,
    psfp=None,
    indxpsfpinit=None,
    fermscalfact=None,
):
    """Evaluate a supported PSF profile from explicit model parameters."""
    angle = np.asarray(arryangl, dtype=float)
    if angle.ndim != 1:
        raise ValueError('arryangl must be one-dimensional.')
    if typenormangl not in ('none', 'ferm'):
        raise ValueError("typenormangl must be 'none' or 'ferm'.")

    if typenormangl == 'ferm':
        if fermscalfact is None:
            raise ValueError("fermscalfact is required for typenormangl='ferm'.")
        scale = np.asarray(fermscalfact, dtype=float)
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError('fermscalfact must contain finite positive values.')
        angle = 2.0 * np.arcsin(np.sqrt(2.0 - 2.0 * np.cos(angle)) / 2.0)
        angle_scale = angle[None, :, None] / scale[:, None, :]
    else:
        angle_scale = angle[None, :, None]

    def shape_parameters(number):
        if psfp is None or indxpsfpinit is None:
            raise ValueError('psfp and indxpsfpinit are required for King PSFs.')
        values = np.asarray(psfp[indxpsfpinit:indxpsfpinit + number], dtype=float)
        if values.ndim != 3 or values.shape[0] != number:
            raise ValueError('psfp must have shape (parameters, energy, event class).')
        return [values[index, :, None, :] for index in range(number)]

    def evaluate(profile_angle):
        if typemodlpsfn == 'singgaus':
            if 'sigc' not in dictpara:
                raise ValueError("dictpara['sigc'] is required for a Gaussian PSF.")
            sigma = np.asarray(dictpara['sigc'], dtype=float)
            if booldiag and (not np.isfinite(sigma).all() or np.any(sigma == 0.0)):
                raise ValueError('Gaussian PSF width must be finite and nonzero.')
            return retr_singgaus(profile_angle, sigma)
        if typemodlpsfn == 'singking':
            sigma_core, gamma_core = shape_parameters(2)
            return retr_singking(profile_angle, sigma_core, gamma_core)
        if typemodlpsfn == 'doubking':
            sigma_core, gamma_core, sigma_tail, gamma_tail, fraction = shape_parameters(5)
            return retr_doubking(
                profile_angle, fraction, sigma_core, gamma_core, sigma_tail, gamma_tail
            )
        raise ValueError('Unsupported PSF model %r.' % typemodlpsfn)

    profile = evaluate(angle_scale)
    if typenormangl == 'ferm':
        normalization = 2.0 * np.pi * np.trapz(
            profile * np.sin(angle[None, :, None]), angle, axis=1
        )[:, None, :]
        profile = profile / normalization
    return profile


# photometry related
retr_spec = tdpy.retr_spec


# databases
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


def xmat_tici(listtici, dictnamefeatmast):
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
    for namefeat in dictnamefeatmast:
        dictquer[dictnamefeatmast[namefeat]] = []
        for k in range(len(dictquertemp)):
            dictquer[dictnamefeatmast[namefeat]].append(dictquertemp[k][namefeat])
        dictquer[dictnamefeatmast[namefeat]] = [np.array([dictquer[dictnamefeatmast[namefeat]]]), '']
    dictquer['radistar'][1] = r'R_{\oplus}'
    dictquer['massstar'][1] = r'M_{\oplus}'
    
    return dictquer


def retr_toiitici(tici, typeverb=1, dicttoii=None):
    
    if dicttoii is None:
        dicttoii = retr_dicttoii()
    
    toii = None
    indx = np.where(dicttoii['TICID'][0] == tici)[0]
    if indx.size > 0:
        toii = int(str(dicttoii['TOIID'][0][indx[0]]).split('.')[0])
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
            'TIC_prmshcon': TIC targets with contamination larger than
            'TIC_m060': TIC targets brighter than TESS magnitude 6.0
            'TIC_m100': TIC targets brighter than TESS magnitude 10.0
            'TIC_m140': TIC targets brighter than TESS magnitude 14.0
            'targtess_prms_ffimm060': TESS targets observed during PM on FFIs brighter than mag 6.0
            'targtess_prms_2min': 2-minute TESS targets obtained by merging the SPOC 2-min bulk downloads

    Returns a dictionary with keys:
        rasc: RA
        decl: declination
        magtsystTESS: TESS magnitude
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

    pathlistticidata = _repository_data_path('EPHESOS', 'listticidata') + os.sep
    os.makedirs(pathlistticidata, exist_ok=True)

    path = pathlistticidata + 'listticidata_%s.csv' % typepopl
    if not os.path.exists(path):
        
        # dictionary of strings that will be keys of the output dictionary
        dictnamefeatmast = dict()
        dictnamefeatmast['ID'] = 'TICID'
        dictnamefeatmast['ra'] = 'rascstar'
        dictnamefeatmast['dec'] = 'declstar'
        dictnamefeatmast['Tmag'] = 'magtsystTESS'
        dictnamefeatmast['rad'] = 'radistar'
        dictnamefeatmast['mass'] = 'massstar'
        dictnamefeatmast['Teff'] = 'tmptstar'
        dictnamefeatmast['logg'] = 'loggstar'
        dictnamefeatmast['MH'] = 'metastar'
        liststrg = list(dictnamefeatmast.keys())
        
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
                    pathtess = _repository_data_path('TESS', 'lcur', 'sector-%02d' % listtsec[o])
                    listnamefile = fnmatch.filter(os.listdir(pathtess), '*.fits')
                    listticitsec = []
                    for namefile in listnamefile:
                        listticitsec.append(str(int(namefile.split('-')[2])))
                    listticitsec = np.array(listticitsec)
                else:
                    urlo = 'https://tess.mit.edu/wp-content/uploads/all_targets%sS%03d_v1.csv' % (strgurll, listtsec[o])
                    c = pd.read_csv(urlo, header=5)
                    listticitsec = c['TICID'].values
                    listticitsec = listticitsec.astype(str)
                numbtargtsec = listticitsec.size
                
                if typeverb > 0:
                    print('%d observed %s targets in Sector %d...' % (numbtargtsec, labltemp, listtsec[o]))
                
                if numbtargtsec > 0:
                    dictquertemp = xmat_tici(listticitsec, dictnamefeatmast)
                
                if o == 0:
                    dictquerinte = dict()
                    for name in dictnamefeatmast.keys():
                        dictquerinte[dictnamefeatmast[name]] = [[] for o in indxtsec]
                
                for name in dictnamefeatmast.keys():
                    print('dictquertemp')
                    print(dictquertemp)
                    for k in range(len(dictquertemp[dictnamefeatmast[name]][0])):
                        dictquerinte[dictnamefeatmast[name]][o].extend(dictquertemp[dictnamefeatmast[name]][k][0])
                    print('name')
                    print(name)
                    print('dictquerinte[dictnamefeatmast[name]]')
                    summgene(dictquerinte[dictnamefeatmast[name]])
                    print('')

            print('Concatenating arrays from different sectors...')
            for name in dictnamefeatmast.keys():
                dictquer[dictnamefeatmast[name]] = [np.concatenate(dictquerinte[dictnamefeatmast[name]]), '']
            
            u, indxuniq, cnts = np.unique(dictquer['TICID'][0], return_index=True, return_counts=True)
            for name in dictnamefeatmast.keys():
                dictquer[dictnamefeatmast[name]][0] = dictquer[dictnamefeatmast[name]][0][indxuniq]
            dictquer['numbtsec'] = [cnts, '']

        elif typepopl.startswith('TIC'):
            if typepopl.endswith('hcon'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass, contratio', \
                                                             'filters':[{'paramName':'contratio', 'values':[{"min":10., "max":1e3}]}]}}
            elif typepopl.endswith('m060'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass', \
                                                             'filters':[{'paramName':'Tmag', 'values':[{"min":-100., "max":6.0}]}]}}
            elif typepopl.endswith('m090'):
                request = {'service':'Mast.Catalogs.Filtered.Tic.Rows', 'format':'json', 'params':{ \
                'columns':'ID, ra, dec, Tmag, rad, mass', \
                                                             'filters':[{'paramName':'Tmag', 'values':[{"min":-100., "max":9.0}]}]}}
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
                    namefeat = 'TICID'
                    lablunit = ''
                if name == 'Tmag':
                    namefeat = 'magtsystTESS'
                    lablunit = 'mag'
                if name == 'ra':
                    namefeat = 'rascstar'
                    lablunit = 'degree'
                if name == 'dec':
                    namefeat = 'declstar'
                    lablunit = 'degree'
                if name == 'rad':
                    namefeat = 'radistar'
                    lablunit = r'$R_{\odot}$'
                if name == 'mass':
                    namefeat = 'massstar'
                    lablunit = r'$M_{\odot}$'
                dictquer[namefeat] = [np.empty(len(listdictquer)), lablunit]
                for k in range(len(listdictquer)):
                    dictquer[namefeat][0][k] = listdictquer[k][name]
        else:
            raise Exception('Unrecognized population name: %s' % typepopl)
        
        numbtarg = dictquer['radistar'][0].size
            
        if typeverb > 0:
            print('%d targets...' % numbtarg)
            print('Writing to %s...' % path)
        #columns = ['TICID', 'radi', 'mass']
        dictquertemp = dict()
        listnamefeat = list(dictquer.keys())
        listhead = []
        for namefeat in listnamefeat:
            strghead = namefeat
            if dictquer[namefeat][1] != '':
                strghead += ' [%s]' % dictquer[namefeat][1]
            listhead.append(strghead)
            dictquertemp[namefeat] = dictquer[namefeat][0]
        print('dictquertemp')
        print(dictquertemp)
        print('listhead')
        print(listhead)
        pd.DataFrame.from_dict(dictquertemp).to_csv(path, header=listhead, index=False)#, columns=columns)
        dictquer = dictquertemp
    else:
        if typeverb > 0:
            print('Reading from %s...' % path)
        dictquertemp = pd.read_csv(path, nrows=numbsyst).to_dict(orient='list')
        dictquer = dict()
        for name in dictquer.keys():
            print('name')
            print(name)
            dictquer[name] = [[], []]
            dictquer[name][0] = np.array(dictquertemp[name])
            nameunit = name.split('[')[1].split(']')[0]
            dictquer[name][1] = nameunit

    #if gdat.typedata == 'simuinje':
    #    indx = np.where((~np.isfinite(gdat.dictfeat['true']['ssys']['massstar'])) | (~np.isfinite(gdat.dictfeat['true']['ssys']['radistar'])))[0]
    #    gdat.dictfeat['true']['ssys']['radistar'][indx] = 1.
    #    gdat.dictfeat['true']['ssys']['massstar'][indx] = 1.
    #    gdat.dictfeat['true']['totl']['magtsystTESS'] = dicttic8['magtsystTESS']
        
    # check if dictquer is properly defined, whose leaves should be a list of two items (of values and labels, respectively)
    if booldiag:
        for namefeat in dictquer:
            if len(dictquer[namefeat]) != 2 or len(dictquer[namefeat][1]) > 0 and not isinstance(dictquer[namefeat][1][1], str):
                print('')
                print('')
                print('')
                print('namefeat')
                print(namefeat)
                
                print('dictquer[namefeat]')
                print(dictquer[namefeat])
                print('typepopl')
                print(typepopl)
                raise Exception('dictquer is not properly defined.')

    return dictquer


retr_liststrgcomp = tdpy.retr_liststrgcomp
retr_listcolrcomp = tdpy.retr_listcolrcomp


def retr_dictfluxband(tmptstar, liststrgband, gdatfluxband=None, pathvisutarg=None, strgtarg=None, typefileplot='png', \
              # type of verbosity
              ## -1: absolutely no text
              ##  0: no text output except critical warnings
              ##  1: minimal description of the execution
              ##  2: detailed description of the execution
              typeverb=1, \
              ):
            
    dictfluxband = dict()
    
    boolconsgdat = gdatfluxband is None
    
    if boolconsgdat:
        # construct global object
        gdatfluxband = tdpy.gdatstrt()
        
        # dictionary for the transfer functions
        gdatfluxband.dictfunctran = dict()

        # define spectral grid (this may need to be taken outside the if statements for other purposes)
        minmwlen = 0.1
        maxmwlen = 10.
        numbwlen = 1000
        gdatfluxband.binswlen, gdatfluxband.midpwlen, _, _, _ = tdpy.retr_axis(minmwlen, maxmwlen, numbpntsgrid=numbwlen, scalpara='logt')
    
        gdatfluxband.numbband = len(liststrgband)
        gdatfluxband.indxband = np.arange(gdatfluxband.numbband)
    
    specbbod = retr_fluxspecbbod(tmptstar, xdat=gdatfluxband.midpwlen, typexdat='wlen')
    
    if pathvisutarg is not None:
        path = pathvisutarg + 'fluxband_%s' % strgtarg
        for pl in gdatfluxband.indxband:
            path += '_' + liststrgband[pl]
        path += '.%s' % typefileplot
        if not os.path.exists(path):
            figr, axis = plt.subplots(figsize=(8, 5))
            axistwin = axis.twinx()
            axis.plot(gdatfluxband.midpwlen, gdatfluxband.fluxbandsyst, color=colrdraw, ls='-', ms=1, rasterized=True)
    
    for pl in gdatfluxband.indxband:
        
        if boolconsgdat:
        
            if liststrgband[pl].startswith('LSST'):
                import sncosmo
                strgband = liststrgband[pl][4]
                functraninit = sncosmo.get_bandpass(f'lsst%s' % strgband)
                gdatfluxband.dictfunctran[liststrgband[pl]] = np.interp(gdatfluxband.midpwlen, 1e-4 * functraninit.wave, functraninit.trans)
            elif liststrgband[pl] == 'Bolometric':
                gdatfluxband.dictfunctran[liststrgband[pl]] = np.ones_like(gdatfluxband.midpwlen)
            else:
                gdatfluxband.dictfunctran[liststrgband[pl]] = np.zeros_like(gdatfluxband.midpwlen)
                
                if liststrgband[pl] == 'ULTRASAT':
                    indxwlen = np.where((gdatfluxband.midpwlen < 0.29) & (gdatfluxband.midpwlen > 0.23))[0]
                    gdatfluxband.dictfunctran[liststrgband[pl]][indxwlen] = 1.
                elif liststrgband[pl] == 'TESS-GEO-UV':
                    indxwlen = np.where((gdatfluxband.midpwlen < 0.29) & (gdatfluxband.midpwlen > 0.23))[0]
                    gdatfluxband.dictfunctran[liststrgband[pl]][indxwlen] = 1.
                elif liststrgband[pl] == 'TESS-GEO-VIS':
                    indxwlen = np.where((gdatfluxband.midpwlen < 0.7) & (gdatfluxband.midpwlen > 0.4))[0]
                    gdatfluxband.dictfunctran[liststrgband[pl]][indxwlen] = 1.
                elif liststrgband[pl] == 'TESS':
                    indxwlen = np.where((gdatfluxband.midpwlen < 1.) & (gdatfluxband.midpwlen > 0.6))[0]
                    gdatfluxband.dictfunctran[liststrgband[pl]][indxwlen] = 1.
                
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
                #    gdatfluxband.dictfunctran[liststrgband[pl]][indxwlen][:] = 1.
                
                else:
                    print('')
                    print('')
                    print('')
                    print('liststrgband[pl]')
                    print(liststrgband[pl])
                    raise Exception('Undefined liststrgband[pl]')
        
            if pathvisutarg is not None and not os.path.exists(path) and liststrgband[pl] != 'Bolometric':
                axistwin.plot(gdatfluxband.midpwlen, gdatfluxband[liststrgband[pl]], ls='-', ms=1, rasterized=True, label=liststrgband[pl])
    
        dictfluxband[liststrgband[pl]] = np.trapz(specbbod * gdatfluxband.dictfunctran[liststrgband[pl]], x=gdatfluxband.midpwlen)
    
    if boolconsgdat:
        if pathvisutarg is not None:
            axis.set_xscale('log')
            axis.set_xlabel(r'Wavelength [$\mu$m]')
            axis.set_ylabel('Spectrum')
            axistwin.legend()
            axistwin.set_ylabel('Transfer function')
            plt.subplots_adjust(hspace=0.)
            if gdat.typeverb > 0:
                print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
    
    return dictfluxband, gdatfluxband


def retr_dictpoplrvel():
    
    if typeverb > 0:
        print('Reading Sauls Gaia high RV catalog...')
    path = _repository_data_path('TROIA', 'Gaia_high_RV_errors.txt')
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
    
    pathephe = str(RepositoryPaths('EPHESOS_PATH').get_repository_path()) + os.sep
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
                  
                  strgelem='comp', \
                  
                  # type of verbosity
                  ## -1: absolutely no text
                  ##  0: no text output except critical warnings
                  ##  1: minimal description of the execution
                  ##  2: detailed description of the execution
                  typeverb=1, \
                  
                  ):
    
    dictfact = tdpy.retr_factconv()
    
    pathephe = str(RepositoryPaths('EPHESOS_PATH').get_repository_path()) + os.sep
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
        tdpy.setp_dict(dicttoii, strgradielem, objtexof['Planet Radius (R_Earth)'][indxcomp].values, r'R$_{\oplus}$')
        tdpy.setp_dict(dicttoii, strgstdvradi, objtexof['Planet Radius (R_Earth) err'][indxcomp].values, r'R$_{\oplus}$')
        
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

        tdpy.setp_dict(dicttoii, 'magtsystTESS', objtexof['TESS Mag'][indxcomp].values)
        tdpy.setp_dict(dicttoii, 'stdvmagtsystTESS', objtexof['TESS Mag err'][indxcomp].values)

        # transit duty cycle
        tdpy.setp_dict(dicttoii, 'dcyc', dicttoii['duratrantotl'][0] / dicttoii[strgperielem][0] / 24.)
        
        liststrgfeatstartici = ['massstar', 'magtsystVbnd', 'magtsystJbnd', 'magtsystHbnd', 'magtsystKbnd', 'distsyst', 'metastar', 'radistar', 'tmptstar', 'loggstar']
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
            dictelemmass = dict()
            dictelemmass[strgmasselem] = np.ones_like(dicttoii[strgradielem][0]) + np.nan
            dictelemmass[strgstdvmass] = np.ones_like(dicttoii[strgradielem][0]) + np.nan
            
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
                
                dictelemmass[strgmasselem][k] = np.mean(listmassplan)
                dictelemmass[strgstdvmass][k] = np.std(listmassplan)
                
            if typeverb > 0:
                print('Writing to %s...' % path)
            pd.DataFrame.from_dict(dictelemmass).to_csv(path, index=False)
        else:
            if typeverb > 0:
                print('Reading from %s...' % path)
            dictelemmass = pd.read_csv(path).to_dict(orient='list')
            
            for name in dictelemmass:
                dictelemmass[name] = np.array(dictelemmass[name])
                if toiitarg is not None:
                    dictelemmass[name] = dictelemmass[name][indxcomp]
            
            if dicttoii['radistar'][0].size != dictelemmass[strgmasselem].size:
                raise Exception('dictelemmass size is anomalous.')

        tdpy.setp_dict(dicttoii, strgmasselem, dictelemmass[strgmasselem])
        tdpy.setp_dict(dicttoii, strgstdvmass, dictelemmass[strgstdvmass])
        
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
                indxexartici = np.where((dictexar['TICID'][0] == int(listticicpla[k])) & \
                                                    (dictexar['facidisc'] == 'Transiting Exoplanet Survey Satellite (TESS)'))[0]
                indxexoftici = np.where(dicttoii['TICID'][0] == int(listticicpla[k]))[0]
                for strg in dictexar.keys():
                    if indxexartici.size > 0:
                        dicttoii[strg][0] = np.delete(dicttoii[strg][0], indxexoftici)
                    dicttoii[strg][0] = np.concatenate((dicttoii[strg][0], dictexar[strg][indxexartici]))

        # derived quantities
        ## photometric noise in the TESS passband over one hour
        tdpy.setp_dict(dicttoii, 'noistess', retr_noisphot(dicttoii['magtsystTESS'][0], 'TESS'))
        
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
            
            listmagtsystKbnd = tdpy.icdf_gaus(np.random.rand(numbsamp), dictpopl['magtsystKbnd'][0][n], dictpopl['stdvmagtsystKbnd'][0][n])
            listmagtsystJbnd = tdpy.icdf_gaus(np.random.rand(numbsamp), dictpopl['magtsystJbnd'][0][n], dictpopl['stdvmagtsystJbnd'][0][n])
            listtmptstar = tdpy.samp_gaustrun(numbsamp, dictpopl['tmptstar'][0][n], dictpopl['stdvtmptstar'][0][n], 0., np.inf)
        
        else:
            listradicomp = dictpopl[strgradielem][0][None, n]
            listtmptplan = dictpopl['tmpt%s' % strgelem][0][None, n]
            listmassplan = dictpopl[strgmasselem][0][None, n]
            listradistar = dictpopl['radistar'][0][None, n]
            listmagtsystKbnd = dictpopl['magtsystKbnd'][0][None, n]
            listmagtsystJbnd = dictpopl['magtsystJbnd'][0][None, n]
            listtmptstar = dictpopl['tmptstar'][0][None, n]
        
        # TSM
        listtsmm[:, n] = retr_tsmm(listradicomp, listtmptplan, listmassplan, listradistar, listmagtsystJbnd)

        # ESM
        listesmm[:, n] = retr_esmm(listtmptplan, listtmptstar, listradicomp, listradistar, listmagtsystKbnd)
        
        #if (listesmm[:, n] < 1e-10).any():
        #    print('listradicomp')
        #    summgene(listradicomp)
        #    print('listtmptplan')
        #    summgene(listtmptplan)
        #    print('listmassplan')
        #    summgene(listmassplan)
        #    print('listradistar')
        #    summgene(listradistar)
        #    print('listmagtsystKbnd')
        #    summgene(listmagtsystKbnd)
        #    print('listmagtsystJbnd')
        #    summgene(listmagtsystJbnd)
        #    print('listtmptstar')
        #    summgene(listtmptstar)
        #    print('listesmm[:, n]')
        #    summgene(listesmm[:, n])
        #    raise Exception('')
    for name, samples in [('tsmm', listtsmm), ('esmm', listesmm)]:
        valid = np.isfinite(samples).any(axis=0)
        median = np.full(numbcomp, np.nan)
        stdv = np.full(numbcomp, np.nan)
        median[valid] = np.nanmedian(samples[:, valid], axis=0)
        stdv[valid] = np.nanstd(samples[:, valid], axis=0)
        tdpy.setp_dict(dictpopl, name, median)
        tdpy.setp_dict(dictpopl, 'stdv' + name, stdv)
    
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


retr_reso = tdpy.retr_reso
retr_dilu = tdpy.retr_dilu


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


retr_radifrommass = tdpy.retr_radifrommass
retr_massfromradi = tdpy.retr_massfromradi


retr_tmptplandayynigh = tdpy.retr_tmptplandayynigh
retr_esmm = tdpy.retr_esmm
retr_tsmm = tdpy.retr_tsmm
retr_scalheig = tdpy.retr_scalheig
retr_pntszero = tdpy.retr_pntszero
retr_magtfromflux = tdpy.retr_magtfromflux
retr_fluxfrommagt = tdpy.retr_fluxfrommagt
retr_rflxfromdmag = tdpy.retr_rflxfromdmag


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
                    'magtsystVbnd', 'magtsystJbnd', 'magtsystHbnd', 'magtsystKbnd', 'magtsystTESS', 'metastar', 'distsyst', 'lumistar', \
                   ]
        
    strgnumbelemstar = 'numb%sstar' % strgelem
    strgnumbelemtranstar = 'numb%stranstar' % strgelem
    # get NASA Exoplanet Archive data
    path = _repository_data_path('EPHESOS', 'PSCompPars_2023.10.12_16.58.05.csv')
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
        
        print('The target name, %s, was **not** found in the NASA Exoplanet Archive composite table.' % strgexar)
        
        return None
    else:
        print('The target name, %s, has been found in the NASA Exoplanet Archive composite table.' % strgexar)
        dictexar = {}

        listnamefeat = ['namestar', 'nameplan', 'TICID', 'rascstar', 'declstar', 'TOIID', 'methdisc', 'eccecomp', 'facidisc', 'yeardisc', \
                        'irra', 'pericomp', 'smaxcomp', 'epocmtracomp', 'cosicomp', 'duratrantotl', 'booltran', 'strgprovmass', strgstrgrefrradielem, strgstrgrefrmasselem, \
                        'vesc', 'masstotl', 'densplan', 'vsiistar', 'projoblq', 'boolcibp', strgnumbelemstar, strgnumbelemtranstar, 'dcyc', 'lgalstar', 'bgalstar', \
                        'loecstar', 'laecstar', 'rratcomp', 'rsmacomp', 'esmm', 'tsmm', 'loggcomp', 'depttrancomp', \
                        
                        # derived
                        'lumibbodbolohost', \
                        'fracxrayboloaddihost', \
                        'fluxbbod0224host', \
                        'lumibbod0224host', \
                        'lumipred0224host', \
                        'fluxpred0224host', \

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
            if strg.startswith('magtsyst'):
                strgvarbexar = 'sy_'
                if strg.endswith('TESS'):
                    strgextn = 't'
                elif strg.endswith('bnd'):
                    strgextn = strg[-4].lower()
                strgvarbexar += '%smag' % strgextn
            if strg.endswith('syst'):
                strgvarbexar = 'sy_'
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
                dictexar[strg][1] = r'$R_\oplus$'
                dictexar['stdv%s' % strg][1] = r'$R_\oplus$'
            elif strg == strgmasselem:
                dictexar[strg][1] = r'$M_\oplus$'
                dictexar['stdv%s' % strg][1] = r'$M_\oplus$'
            
            if strg == 'tmptstar':
                dictexar[strg][1] = 'K'
            if strg == 'radistar':
                dictexar[strg][1] = r'R$_{\odot}$'
        
        
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
        
        # derived properties
        # bolometric flux of a star based on radius and temperature
        dictexar['lumibbodbolohost'] = retr_fluxbolobbod(dictexar['tmptstar'], dictexar['radistar'])
        
        # X-ray flux of the host star
        dictxraystar = retr_dictxraystar(dictexar['tmptstar'], dictexar['radistar'], dictexar['distsyst'])
        dictexar['fracxrayboloaddihost'] = dictxraystar['fracxrayboloaddi']
        dictexar['fluxbbod0224host'] = dictxraystar['fluxbbod0224']
        dictexar['lumibbod0224host'] = dictxraystar['lumibbod0224']
        dictexar['lumipred0224host'] = dictxraystar['lumipred0224']
        dictexar['fluxpred0224host'] = dictxraystar['fluxpred0224']
        
        # surface gravity of the companion
        dictexar['loggcomp'][0] = np.log10(9.8 * dictexar[strgmasselem][0] / dictexar[strgradielem][0]**2)

        # calculate TSM and ESM
        calc_tsmmesmm(dictexar, strgelem=strgelem)

        indxnonntran = np.where(~dictexar['booltran'][0])[0]
        dictexar['esmm'][0][indxnonntran] = np.nan
        dictexar['tsmm'][0][indxnonntran] = np.nan
        
    return dictexar


def retr_dictxraystar(tmptstar, radistar, diststar):
    '''
    Return a dictionary with predicted X-ray-related properties of stars 
    '''
    
    dictfact = tdpy.retr_factconv()
    
    listnameoutp = ['fracxrayboloaddi', 'fluxbbod0224', 'lumibbod0224', 'lumipred0224', 'fluxpred0224', 'lumibbodbolotest']

    dictxraystar = tdpy.retr_dict(listnameoutp)
    
    dictxraystar['lumibbodbolo'] = retr_fluxbolobbod(tmptstar, radistar)
    
    # M3 star, 3500 K, (FX=8.4e-17erg/s/cm2 ; 0.2-2.4keV) at a distance of 2.5 kpc
    blimener = retr_blimener(typeband='AXIS0224')
    functran = retr_functran(typeinst='AXIS', xdat=blimener)
    blimenerbolo = retr_blimener(typeband='bolo')
        
    # ratio of X-ray flux to bolometric flux
    # Solar LX/Lbol is 6e-7
    dictxraystar['fracxrayboloaddi'][0] = np.full_like(tmptstar[0], 6e-7)
    indxstarlmas = np.where(tmptstar[0] < 4000.)[0]
    dictxraystar['fracxrayboloaddi'][0][indxstarlmas] = 1e-4
    dictxraystar['fracxrayboloaddi'][1] = ''

    numbstar = tmptstar[0].size
    indxstar = np.arange(numbstar)
    dictxraystar['fluxbbod0224'][0] = np.empty(numbstar)
    dictxraystar['lumibbodbolotest'][0] = np.empty(numbstar)
    for k in indxstar:
        fluxspecbbod = retr_fluxspecbbodener(tmptstar[0][k], blimener) # ergs/s/cm^2/eV

        # Sun: 1e27 erg/s, Capella: 1e31 erg/s
        dictxraystar['fluxbbod0224'][0][k] = np.trapezoid(fluxspecbbod * functran, x=blimener)
        
        fluxspecbbodtest = retr_fluxspecbbodener(tmptstar[0][k], blimenerbolo) # ergs/s/cm^2/eV
        dictxraystar['lumibbodbolotest'][0][k] = np.trapezoid(fluxspecbbodtest, x=blimenerbolo)
        if False and abs(fluxbbodbolotest - dictxraystar['lumibbodbolo'][0][k]) / dictxraystar['lumibbodbolo'][0][k] > 0.01:
            print('')
            print('')
            print('')
            print('tmptstar[0][k]')
            print(tmptstar[0][k])
            print('blimenerbolo')
            summgene(blimenerbolo)
            print('fluxspecbbodtest')
            summgene(fluxspecbbodtest)
            print('fluxbbodbolotest')
            print(fluxbbodbolotest)
            print('dictxraystar[lumibbodbolo][0][k]')
            print(dictxraystar['lumibbodbolo'][0][k])
            raise Exception('')

    dictxraystar['fluxbbod0224'][1] = 'ergs s$^{-1}$cm$^{-2}$'
    
    dictxraystar['lumibbod0224'][0] = 4. * np.pi * (6.957e10 * radistar[0])**2 * dictxraystar['fluxbbod0224'][0] # [erg/s]
    dictxraystar['lumibbod0224'][1] = 'ergs s$^{-1}$'
    
    dictxraystar['lumipred0224'][0] = dictxraystar['lumibbod0224'][0] + dictxraystar['fracxrayboloaddi'][0] * dictxraystar['lumibbodbolo'][0]
    dictxraystar['lumipred0224'][1] = 'ergs s$^{-1}$'
    
    if diststar is None:
        dictxraystar['fluxpred0224'][0] = np.full(numbstar, np.nan)
    else:
        dictxraystar['fluxpred0224'][0] = dictxraystar['lumipred0224'][0] / 4. / np.pi / (diststar[0] * dictfact['pccm'])**2
    dictxraystar['fluxpred0224'][1] = 'ergs s$^{-1}$cm$^{-2}$'

    return dictxraystar


retr_fluxbolobbod = tdpy.retr_fluxbolobbod
retr_blimener = tdpy.retr_blimener
retr_fluxspecbbod = tdpy.retr_fluxspecbbod
retr_fluxspecbbodener = tdpy.retr_fluxspecbbodener
retr_fluxspecbbodwlen = tdpy.retr_fluxspecbbodwlen
retr_functran = tdpy.retr_functran

# physics
retr_vesc = tdpy.retr_vesc
retr_rs2a = tdpy.retr_rs2a
retr_rsmacomp = tdpy.retr_rsmacomp
retr_duratranfull = tdpy.retr_duratranfull
retr_duratrantotl = tdpy.retr_duratrantotl
retr_rflxmodlrise = tdpy.retr_rflxmodlrise
retr_rvel = tdpy.retr_rvel
retr_coeflmdkkipp = quadratic_to_kipping_limb_darkening
retr_coeflmdkfromkipp = kipping_to_quadratic_limb_darkening
retr_rvelsema = tdpy.retr_rvelsema
retr_brgtlmdk = tdpy.retr_brgtlmdk
retr_logg = tdpy.retr_logg
retr_albg = tdpy.retr_albg
retr_noisphot = tdpy.retr_noisphot


def retr_subp(dictpopl, dictnumbsamp, dictindxsamp, namepoplinit, namepoplfinl, indx):
    
    if isinstance(indx, list):
        raise Exception('')

    if len(indx) == 0:
        print('Warning! indx is zero.')
    subset_population(
        dictpopl,
        namepoplinit,
        namepoplfinl,
        indx,
        dictnumbsamp,
        dictindxsamp,
    )
    dictindxsamp[namepoplfinl] = dict()


retr_dictsols = tdpy.retr_dictsols


def retr_dictpoplstarcomp( \
                          # type of target systems
                          typesyst, \
                          
                          # type of the population of target systems
                          typepoplsyst=None, \
                          
                          # number of systems
                          numbsyst=None, \
                          
                          # epochs of mid-transits
                          epocmtracomp=None, \
                          
                          # offset for mid-transit epochs
                          timeepoc=None, \
                          
                          # type of sampling of orbital period and semi-major axes
                          ## 'smax': semi-major axes are sampled first and then orbital periods are calculated
                          ## 'peristab': orbital periods are sampled first, respecting stability, and then semi-major axies are calculated
                          ## 'peripowr': orbital periods are sampled from a power law regardless of stability and then semi-major axies are calculated
                          typesamporbtcomp='peristab', \
                          
                          # stellar properties (if scalar, resample from the same star)
                          ## distances to the systems
                          distsyst=None, \
                          ## mass of the star(s)
                          massstar=None, \

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
                          
                          # maximum mass of the companions
                          maxmmasscomp=None, \
                          
                          # minimum orbital period, only taken into account when typesamporbtcomp == 'peripowr' or 'peristab'
                          minmpericomp=0.5, \
                          
                          # maximum orbital period, only taken into account when typesamporbtcomp == 'peripowr'
                          maxmpericomp=1000., \
                          
                          # Boolean flag to force all companions to be transiting
                          booltrancomp=False, \

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

                          # legacy alias for stellar-population sampling behavior
                          booltoyysunn=False, \
                          
                          # list of bands
                          liststrgband=None, \
                          
                         ):
    '''
    Sample a synthetic population of the features of companions (e.g., exoplanets )and the companions to companions (e.g., exomoons) 
    hosted by a specified or random population of stellar systems.
    '''
    
    print('Simulating a population of systems via retr_dictpoplstarcomp()...')
    print('typesyst')
    print(typesyst)
    
    if typepoplsyst is None:
        typepoplsyst = 'SyntheticPopulation'
    elif typepoplsyst == 'Synthetic':
        typepoplsyst = 'SyntheticPopulation'

    print('typepoplsyst')
    print(typepoplsyst)
    
    if booldiag:
        if maxmcosicomp is not None and (maxmcosicomp > 1 or maxmcosicomp < 0):
            raise Exception('maxmcosicomp is bad.')
    
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
        if distsyst is not None:
            numbsyst = len(distsyst)
        elif typepoplsyst == 'SyntheticPopulation':
            numbsyst = 10000
    
    indxsyst = np.arange(numbsyst)

    # get the features of the star population
    if typepoplsyst.startswith('CTL') or typepoplsyst.startswith('TIC'):
        dictstar = retr_dictpopltic8(typepoplsyst, numbsyst=numbsyst)
        
        print('Removing stars that do not have radii or masses...')
        indx = np.where(np.isfinite(dictstar['radistar'][0]) & \
                        np.isfinite(dictstar['massstar'][0]))[0]
        for name in dictstar.keys():
            dictstar[name][0] = dictstar[name][0][indx]

        if (dictstar['rascstar'][0] > 1e4).any():
            raise Exception('')

        if (dictstar['radistar'][0] == 0.).any():
            raise Exception('')
        
        densstar = 1.41 * dictstar['massstar'][0] / dictstar['radistar'][0]**3
        tdpy.setp_dict(dictstar, 'densstar', densstar)
        
        dictstar['idenstar'] = dictstar['TICID']
    

    elif typepoplsyst == 'SyntheticPopulation':
        
        dictstar = dict()
        
        if distsyst is None:
            dictstar['distsyst'] = [tdpy.icdf_powr(np.random.rand(numbsyst), 100., 7000., -2.), 'pc']
        elif isinstance(distsyst, np.ndarray):
            dictstar['distsyst'] = [distsyst, 'pc']
        elif isinstance(distsyst, float):
            dictstar['distsyst'] = [np.full(numbsyst, distsyst), 'pc']
        
        # stellar masses, radii, and densities
        ## all masses and radii should be in solar units and densities should be in g/cm3
        dictstar['radistar'] = [[], []]
        dictstar['massstar'] = [[], []]
        dictstar['densstar'] = [[], []]
        if typestar == 'sunl':
            dictstar['radistar'][0] = np.ones(numbsyst)
            dictstar['massstar'][0] = np.ones(numbsyst)
            dictstar['densstar'][0] = np.full(numbsyst, 1.4)
        elif massstar is not None or typestar == 'drawkrou':
            if massstar is not None:
                dictstar['massstar'][0] = np.full(numbsyst, massstar)
            if typestar == 'drawkrou':
                dictstar['massstar'][0] = tdpy.icdf_powr(np.random.rand(numbsyst), 0.1, 10., 2.)
            if radistar is not None:
                dictstar['radistar'][0] = np.full(numbsyst, radistar)
                dictstar['densstar'][0] = 1.4 * dictstar['massstar'][0] / dictstar['radistar'][0]**3
            else:
                dictstar['densstar'][0] = 1.4 * dictstar['massstar'][0]**(-0.7)
                dictstar['radistar'][0] = 1.4 * dictstar['massstar'][0] / dictstar['densstar'][0]**(1. / 3.)
        elif typestar == 'wdwf':
            dictstar['radistar'] = [0.01 * np.ones(numbsyst), r'$R_{\odot}$']
            dictstar['massstar'] = [np.ones(numbsyst), r'$M_{\odot}$']
            dictstar['densstar'] = [1.4e6 * np.ones(numbsyst), 'g cm$^{-3}$']
        else:
            raise Exception('')
        dictstar['radistar'][1] = r'$R_{\odot}$'
        dictstar['massstar'][1] = r'$M_{\odot}$'
        dictstar['densstar'][1] = 'g cm$^{-3}$'

        dictstar['coeflmdklinr'] = [0.4 * np.ones(numbsyst), '']
        dictstar['coeflmdkquad'] = [0.25 * np.ones(numbsyst), '']

        dictstar['tmptstar'] = [6000 * dictstar['massstar'][0], 'K']
        
        dictstar['lumistar'] = [4. * np.pi * dictstar['tmptstar'][0]**4 * dictstar['radistar'][0]**2, r'$L_{\odot}$']
        
        dictstar['fluxbolostar'] = [1361. * dictstar['lumistar'][0] / dictstar['distsyst'][0]**2 / 4. / np.pi, 'W/m^2']

        if liststrgband is None:
            liststrgband = []
            if typepoplsyst == 'lsstwfds':
                liststrgband += ['r']

        gdatfluxband = None
        for strgband in liststrgband:
            dictstar['fluxbandsyst%s' % strgband] = [np.empty(numbsyst), 'W/m^2']
        for k in indxsyst: 
            # flux at the surface
            dictfluxband, gdatfluxband = retr_dictfluxband(dictstar['tmptstar'][0][k], liststrgband, gdatfluxband=gdatfluxband)
            
            # flux at the observer
            for strgband in liststrgband:
                dictstar['fluxbandsyst%s' % strgband][0][k] = dictfluxband[strgband] * (dictstar['radistar'][0][k] / 4.435e7 / dictstar['distsyst'][0][k])**2
        
        print('fluxbandsyst')
        for strgband in liststrgband:
            dictstar['magtsyst%s' % strgband] = [retr_magtfromflux(dictstar['fluxbandsyst%s' % strgband][0], strgband), 'mag']
            
            print('strgband')
            print(strgband)
            print('fluxbandsyst')
            summgene(dictstar['fluxbandsyst%s' % strgband][0])
            print('magtsyst')
            summgene(dictstar['magtsyst%s' % strgband][0])
            print('')
    else:
        print('')
        print('')
        print('')
        print('typepoplsyst')
        print(typepoplsyst)
        raise Exception('Undefined typepoplsyst.')
    
    dictstarnumbsamp[namepoplstartotl] = numbsyst
    dictpopl['star'][namepoplstartotl] = dictstar

    def _legacy_alias_popl(popl):
        out = dict()
        for key, val in popl.items():
            if isinstance(val, list) and len(val) == 2:
                out[key] = val[0]
            elif isinstance(val, dict):
                out[key] = _legacy_alias_popl(val)
            else:
                out[key] = val
        return out

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
                    minmmasscomp = 3000. # [Earth mass]
            elif typesyst == 'StellarBinary':
                minmmasscomp = 0.5 # [Earth mass]
    
        # maximum companion mass
        if maxmmasscomp is None:
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
            dictpopl[strgbody][namepoplstartotl][strgnumblimbbodymean] = [5. * dictpopl[strgbody][namepoplstartotl]['massstar'][0]**(-1.), '']
            
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
            dictpopl[strgbody][namepoplstartotl][strgnumblimbbody] = [np.ones(dictpopl[strgbody][namepoplstartotl]['radistar'][0].size).astype(int), '']
            
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
        for k in indxsyst:
            dictindx[strglimb][strgbody][k] = np.arange(cntr, cntr + dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]).astype(int)
            cntr += dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]
        dictnumbsamp[strglimb][namepopllimbtotl] = cntr
    
        # prepare to load star features into component features
        for name in list(dictpopl[strgbody][namepoplstartotl].keys()):
            dictpopl[strglimb][namepopllimbtotl][name] = [np.empty(dictnumbsamp[strglimb][namepopllimbtotl]), '']
        
        print('Loading star features into component features...')
        dictnumbsamp[strglimb][namepopllimbtotl] = dictpopl[strglimb][namepopllimbtotl]['radistar'][0].size

        listnamecatr = ['masssyst', 'radistar']
        if strglimb == 'comp':
            listnamecatr += ['pericomp', 'cosicomp', 'smaxcomp', 'eccecomp', 'arpacomp', 'loancomp', 'masscomp', 'epocmtracomp', 'rsmacomp', 'rsumcomp', 'distanglcomp']
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
    
        print('Iterating over the stars...')
        for k in tqdm(range(numbstar)):
            
            if dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k] == 0:
                print('Skipping')
                print('Skipping')
                print('Skipping')
                print('Skipping')
                print('Skipping')
                print('Skipping')
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
                
                # total radius
                dictpopl[strglimb][namepopllimbtotl]['rsumcomp'][0][dictindx[strglimb][strgbody][k]] = dictpopl[strgbody][namepoplstartotl]['radistar'][0][k]
                if not boolsystcosc:
                    dictpopl[strglimb][namepopllimbtotl]['rsumcomp'][0][dictindx[strglimb][strgbody][k]] += \
                                            dictpopl[strglimb][namepopllimbtotl]['radicomp'][0][dictindx[strglimb][strgbody][k]] / dictfact['rsre']
                
                if booldiag:
                    if (dictpopl[strglimb][namepopllimbtotl]['rsumcomp'][0][dictindx[strglimb][strgbody][k]] < 1e-30).any():
                        print('')
                        print('')
                        print('')
                        raise Exception('')

                    if (dictpopl[strglimb][namepopllimbtotl]['rsumcomp'][0][dictindx[strglimb][strgbody][k]] <= \
                       dictpopl[strglimb][namepopllimbtotl]['radicomp'][0][dictindx[strglimb][strgbody][k]] / dictfact['rsre']).any() or \
                       (dictpopl[strglimb][namepopllimbtotl]['rsumcomp'][0][dictindx[strglimb][strgbody][k]] <= \
                       dictpopl[strglimb][namepopllimbtotl]['radistar'][0][k]).any():
                        raise Exception('')
                    
                # total mass
                if boolsystcosc or typesyst == 'StellarBinary':
                    dictpopl[strgbody][namepoplstartotl]['masssyst'][0][k] += np.sum(dictpopl[strglimb][namepopllimbtotl]['masscomp'][0][dictindx[strglimb][strgbody][k]])
                
                if typesamporbtcomp == 'peristab' or typesamporbtcomp == 'peripowr':
                    
                    if typesamporbtcomp == 'peristab':
                        ratiperifrst = tdpy.util.icdf_powr(np.random.rand(1), 2., 1000., 0.5)[0]
                        ratiperi = tdpy.util.icdf_powr(np.random.rand(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k] - 1), 1.2, 100., 2.)

                        listpericomp = []
                        for mm in range(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]):
                            if mm == 0:
                                peri = minmpericomp * ratiperifrst
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
                                print(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k])
                                raise Exception('')
                    if typesamporbtcomp == 'peripowr':
                        dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]] = \
                                        tdpy.util.icdf_powr(np.random.rand(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]), minmpericomp, maxmpericomp, 2.)
                    
                    # determine the corresponding semi-major axes
                    if typesyst == 'PlanetarySystemWithNonKeplerianObjects':
                        raise Exception('')
                    else:
                        factnonk = 1.
                    
                    dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k]] = \
                                            retr_smaxkepl(dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]], \
                                                                                                     dictpopl[strgbody][namepoplstartotl]['masssyst'][0][k], factnonk=factnonk)
                
                else:
                    # semi-major axes
                    #if np.isfinite(dictpopl[strgbody][namepoplstartotl]['densstar'][k]):
                    #    densstar = dictpopl[strgbody][namepoplstartotl]['densstar'][k]
                    #else:
                    #    densstar = 1.
                    #dictpopl[strglimb][namepopllimbtotl]['radiroch'][k] = retr_radiroch(radistar, densstar, denscomp)
                    #minmsmax = 2. * dictpopl[strglimb][namepopllimbtotl]['radiroch'][k]
                    
                    dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k]] = dictpopl[strglimb][namepopllimbtotl]['rsumcomp'][0][k] * \
                                                                                 tdpy.util.icdf_powr(np.random.rand(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]), \
                                                                                                    minmsmaxradistar, maxmsmaxradistar, 2.) / dictfact['aurs']
                    
                    if typesyst == 'PlanetarySystemWithNonKeplerianObjects':
                        factnonk = tdpy.util.icdf_powr(np.random.rand(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]), 0.1, 1., -2.)
                        dictpopl[strglimb][namepopllimbtotl]['factnonkcomp'][0][dictindx[strglimb][strgbody][k]] = factnonk
                    else:
                        factnonk = 1.
                    
                    dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]] = \
                                            retr_perikepl(dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k]], \
                                                                                                  dictpopl[strgbody][namepoplstartotl]['masssyst'][0][k], factnonk=factnonk)
                    
        
                dictpopl[strglimb][namepopllimbtotl]['distanglcomp'][0][dictindx[strglimb][strgbody][k]] = \
                                        dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k]] / dictstar['distsyst'][0][k]

                if booldiag:
                    if not np.isfinite(dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]]).all():
                        print('')
                        print('')
                        print('')
                        print('dictpopl[comp][namepopllimbtotl][masscomp][dictindx[comp][star][k]]')
                        print(dictpopl[strglimb][namepopllimbtotl]['masscomp'][0][dictindx[strglimb][strgbody][k]])
                        print('dictpopl[comp][namepopllimbtotl][masssyst][dictindx[comp][star][k]]')
                        print(dictpopl[strglimb][namepopllimbtotl]['masssyst'][0][dictindx[strglimb][strgbody][k]])
                        print('dictpopl[comp][namepopllimbtotl][smaxcomp][dictindx[comp][star][k]]')
                        print(dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0][dictindx[strglimb][strgbody][k]])
                        print('dictpopl[comp][namepopllimbtotl][pericomp][dictindx[comp][star][k]]')
                        print(dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]])
                        raise Exception('')

                # conjunction epochs
                if epocmtracomp is not None:
                    dictpopl[strglimb][namepopllimbtotl]['epocmtracomp'][0][dictindx[strglimb][strgbody][k]] = \
                                                            np.full(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k], epocmtracomp)
                else:
                    dictpopl[strglimb][namepopllimbtotl]['epocmtracomp'][0][dictindx[strglimb][strgbody][k]] = \
                                        1e8 * np.random.rand(dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k])
                if timeepoc is not None:
                    dictpopl[strglimb][namepopllimbtotl]['epocmtracomp'][0][dictindx[strglimb][strgbody][k]] = dictpopl[strglimb][namepopllimbtotl]['epocmtracomp'][k] + \
                                                   dictpopl[strglimb][namepopllimbtotl]['pericomp'][k] * \
                                                   np.round((dictpopl[strglimb][namepopllimbtotl]['epocmtracomp'][k] - timeepoc) / dictpopl[strglimb][namepopllimbtotl]['pericomp'][k])
    
        # derived stellar features --  dictpopl[strgbody][namepoplstartotl] should not be enhanced with derived features before this
        # Boolean flag indicating whether the system has an inner, Earth-sized rocky planet
        dictpopl[strgbody][namepoplstartotl]['boolelik'] = [np.empty(numbstar, dtype=bool), '']
        # Boolean flag indicating whether the system has an outer, Jupiter-sized giant planet
        dictpopl[strgbody][namepoplstartotl]['booljlik'] = [np.empty(numbstar, dtype=bool), '']
        
        print('Iterating over the stars...')
        for k in tqdm(range(numbstar)):
            
            dictpopl[strgbody][namepoplstartotl]['boolelik'][0][k] = np.where( \
                                                                              (dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]] > 0.5 * 365.25) & \
                                                                              (dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]] < 2. * 365.25) & \
                                                                              (dictpopl[strglimb][namepopllimbtotl]['radicomp'][0][dictindx[strglimb][strgbody][k]] > 0.5) & \
                                                                              (dictpopl[strglimb][namepopllimbtotl]['radicomp'][0][dictindx[strglimb][strgbody][k]] < 2.) \
                                                                             )[0].size > 0
            
            dictpopl[strgbody][namepoplstartotl]['booljlik'][0][k] = np.where( \
                                                                        (dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]] > 0.5 * 11.86 * 365.25) & \
                                                                        (dictpopl[strglimb][namepopllimbtotl]['pericomp'][0][dictindx[strglimb][strgbody][k]] < 2. * 11.86 * 365.25) & \
                                                                        (dictpopl[strglimb][namepopllimbtotl]['radicomp'][0][dictindx[strglimb][strgbody][k]] > 0.6 * 11.2) & \
                                                                        (dictpopl[strglimb][namepopllimbtotl]['radicomp'][0][dictindx[strglimb][strgbody][k]] < 1.5 * 11.2) \
                                                                             )[0].size > 0
            
        # Boolean flag indicating whether the system has a an inner, Earth-sized rocky planet AND outer, Jupiter-sized giant planet
        dictpopl[strgbody][namepoplstartotl]['boolelikjlik'] = [dictpopl[strgbody][namepoplstartotl]['boolelik'][0] & dictpopl[strgbody][namepoplstartotl]['booljlik'][0], '']

        if strglimb == 'comp':
            if typesyst == 'PlanetarySystemWithMoons':
                # initialize the total mass of the companion + moons system as the mass of the companion
                dictpopl[strglimb][namepopllimbtotl]['masscompmoon'] = np.copy(dictpopl[strglimb][namepopllimbtotl]['masscomp'])
                        
            dictpopl[strglimb][namepopllimbtotl]['rsmacomp'][0] = dictpopl[strglimb][namepopllimbtotl]['rsumcomp'][0] \
                                                                            / (dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0] * dictfact['aurs'])
            
            if booldiag:
                if (dictpopl[strglimb][namepopllimbtotl]['rsmacomp'][0] >= 1.).any():
                    print('')
                    print('')
                    print('')
                    print('dictpopl[strglimb][namepopllimbtotl][rsmacomp][0]')
                    summgene(dictpopl[strglimb][namepopllimbtotl]['rsmacomp'][0])
                    print('dictpopl[strglimb][namepopllimbtotl][rsumcomp][0]')
                    summgene(dictpopl[strglimb][namepopllimbtotl]['rsumcomp'][0])
                    print(dictpopl[strglimb][namepopllimbtotl]['rsumcomp'][0])
                    print('dictpopl[strglimb][namepopllimbtotl][radistar][0]')
                    summgene(dictpopl[strglimb][namepopllimbtotl]['radistar'][0])
                    print('dictpopl[strglimb][namepopllimbtotl][radicomp][0] / dictfact[rsre]')
                    summgene(dictpopl[strglimb][namepopllimbtotl]['radicomp'][0] / dictfact['rsre'])
                    print('dictpopl[strglimb][namepopllimbtotl][smaxcomp][0] * dictfact[aurs]')
                    summgene(dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0] * dictfact['aurs'])
                    print(dictpopl[strglimb][namepopllimbtotl]['smaxcomp'][0] * dictfact['aurs'])
                    raise Exception('rsum is more than the semi-major axis!')

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
                
                numb = dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]

                # cosine of orbital inclinations
                dictpopl[strglimb][namepopllimbtotl]['cosicomp'][0][dictindx[strglimb][strgbody][k]] = maxmcosicomptemp * np.random.rand(numb)
                
            if booldiag:
                if (dictpopl[strglimb][namepopllimbtotl]['rsmacomp'][0] == 0).any():
                    print('dictpopl[strglimb][namepopllimbtotl][rsmacomp][0]')
                    print(dictpopl[strglimb][namepopllimbtotl]['rsmacomp'][0])
                    print('dictpopl[strglimb][namepopllimbtotl][rsumcomp][0]')
                    print(dictpopl[strglimb][namepopllimbtotl]['rsumcomp'][0])
                    print('')
                    print('')
                    print('')
                    raise Exception('')

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

            # Boolean flag indicating whether any companion is transiting
            booltran = np.empty(numbstar)
            for k in tqdm(range(numbstar)):
                booltran[k] = dictpopl[strglimb][namepopllimbtotl]['booltran'][0][dictindx[strglimb][strgbody][k]].any()
            dictpopl[strgbody][namepoplstartotl]['booltran'] = [booltran, '']

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
                    
                    if dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k] == 0:
                        continue
                    
                    numbcomp = dictpopl[strgbody][namepoplstartotl][strgnumblimbbody][0][k]
                    
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
                                print(dictpopl[strglimb][namepopllimbtotl]['massstar'][0][k])
                                print('dictpopl[comp][namepopllimbtotl][masssyst][k]')
                                print(dictpopl[strglimb][namepopllimbtotl]['masssyst'][0][k])
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
    
    # check if dictpopl is properly defined while allowing the modern flattened array format.
    # Legacy entries are intentionally stored as [value, label], but newer code often stores
    # the raw NumPy arrays directly and only uses the legacy alias names for compatibility.
    if booldiag:
        for namepopl in dictpopl:
            for namespop in dictpopl[namepopl]:
                if not isinstance(dictpopl[namepopl][namespop], dict):
                    continue
                for namefeat, value in dictpopl[namepopl][namespop].items():
                    if isinstance(value, np.ndarray):
                        continue
                    if not isinstance(value, (list, tuple)):
                        continue
                    if len(value) != 2:
                        continue
                    label = value[1]
                    if isinstance(label, str):
                        continue
                    if isinstance(label, (list, tuple)) and len(label) == 0:
                        continue
                    if isinstance(label, (list, tuple)) and len(label) > 0 and not isinstance(label[0], str):
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
                        print(value)
                        raise Exception('dictpopl is not properly defined.')

    def _flatten_legacy_alias(popl):
        out = dict()
        for key, val in popl.items():
            if isinstance(val, list) and len(val) == 2:
                out[key] = val[0]
            elif isinstance(val, dict):
                out[key] = _flatten_legacy_alias(val)
            else:
                out[key] = val
        return out

    # Build legacy key aliases from the canonical population names actually present in the dict.
    # The alias objects are intentionally flattened to the historical raw-array contract used by the tests.
    for namepopl, dictnamepopl in list(dictpopl.items()):
        if namepopl not in ['star', 'comp']:
            continue
        for canon_name in list(dictnamepopl.keys()):
            if namepopl == 'star' and canon_name.startswith('star_') and canon_name.endswith('_All'):
                legacy_name = 'star' + canon_name[len('star_'):-len('_All')].replace('SyntheticPopulation', 'Synthetic').replace('Population', '') + 'totl'
                if legacy_name != canon_name:
                    dictnamepopl[legacy_name] = _flatten_legacy_alias(dictnamepopl[canon_name])
            elif namepopl == 'star' and canon_name.startswith('star_') and canon_name.endswith('_Occurrent'):
                legacy_name = 'star' + canon_name[len('star_'):-len('_Occurrent')].replace('SyntheticPopulation', 'Synthetic').replace('Population', '') + 'occu'
                if legacy_name != canon_name:
                    dictnamepopl[legacy_name] = _flatten_legacy_alias(dictnamepopl[canon_name])
            elif namepopl == 'comp' and canon_name.startswith('compstar_') and canon_name.endswith('_All'):
                legacy_name = 'compstar' + canon_name[len('compstar_'):-len('_All')].replace('SyntheticPopulation', 'Synthetic').replace('Population', '') + 'totl'
                if legacy_name != canon_name:
                    dictnamepopl[legacy_name] = _flatten_legacy_alias(dictnamepopl[canon_name])
            elif namepopl == 'comp' and canon_name.startswith('compstar_') and canon_name.endswith('_Transiting'):
                legacy_name = 'compstar' + canon_name[len('compstar_'):-len('_Transiting')].replace('SyntheticPopulation', 'Synthetic').replace('Population', '') + 'tran'
                if legacy_name != canon_name:
                    dictnamepopl[legacy_name] = _flatten_legacy_alias(dictnamepopl[canon_name])

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

    
retr_lcurmodl_flarsing = tdpy.retr_lcurmodl_flarsing
retr_imfa = tdpy.retr_imfa
retr_deptbeam = tdpy.retr_deptbeam
retr_deptelli = tdpy.retr_deptelli
retr_smaxkepl = tdpy.retr_smaxkepl
retr_perikepl = tdpy.retr_perikepl
retr_radiroch = tdpy.retr_radiroch
retr_radihill = tdpy.retr_radihill
retr_alphelli = tdpy.retr_alphelli
        



