import rioxarray as rioxr
import os

filepath = 'data\\raw\\earth_engine_geowiki'
filepath2 = 'data\\raw\\earth_engine_geowiki_2'
base = 'data\\raw\\earth_engine_geowiki\\0-geowiki-landcover-2017_2016-02-07_2017-02-01.tif'

da = rioxr.open_rasterio(base)
band_names = da.attrs['long_name']
keys_toremove = ['VV','VH','total','temperature','elevation','slope']
indices_toremove = []
for idx, name in enumerate(band_names):
    if(any(map(name.__contains__, keys_toremove))):
        indices_toremove.append(idx+1)
indices_toremove2 = [x-1 for x in indices_toremove]
names = list(da.attrs['long_name'])
for index in sorted(indices_toremove2, reverse=True):
    del names[index]
descr = tuple(names)

for root, dirs, files in os.walk(filepath):
    for file in files:
        with rioxr.open_rasterio(os.path.join(filepath, file)) as da:

            da = da.drop_sel(band = indices_toremove)

            da.attrs['long_name'] = descr

            da.rio.to_raster(os.path.join(filepath2, file))