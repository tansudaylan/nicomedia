import nicomedia

'''
Calibrate the zero point for TESS
'''

dicttic8 = nicomedia.retr_dictpopltic8(typepopl='TIC_m090')

numbstar = dicttic8['radistar'].size
indxstar = np.arange(numbstar)

dictfluxband['magtsystTESSEstimate'] = np.empty(numbstar)
for k in indxstar:
    dictfluxband, gdatfluxband = retr_dictfluxband(dicttic8['tmptstar'][k], 'TESS', gdatfluxband=gdatfluxband)
    dictfluxband['magtsystTESSEstimate'][k] = retr_magtfromflux(dictfluxband['TESS'], 'TESS')

path = os.environ['NICOMEDIA_DATA_PATH'] + '/visuals/'
figr, axis = plt.subplots()
axis.scatter(dictstar['magtsystTESS'], dictfluxband['magtsystTESSEstimate'])
axis.set_xlabel('TIC8 TESS magnitude')
axis.set_ylabel('Estimated TESS magnitude')
print('Writing to %s...' % path)
plt.savefig(path)
plt.close()
    
