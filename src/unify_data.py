"""Unifies the ALFALFA and xGASS data sets."""

import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

HOME = Path('/home/jupyter')
PATH = HOME/'HI-diversity'

def combine_df():
    
    xg  = pd.read_csv(HOME/'alfalfa-convnets'/'data'/'xGASS_representative_sample.csv')
    a40 = pd.read_csv(HOME/'alfalfa-convnets'/'data'/'a40-SDSS_gas-frac.csv')
    
    # rename columns to get in right format
    xg['logfgas'] = xg.lgMHI - xg.lgMstar
    
    xg['id']  = xg. GASS .apply(lambda s: f'GASS_{s}')
    a40['id'] = a40.AGCNr.apply(lambda s: f'AGC_{s}')
    
    # overlap
    a40 = a40[~a40.AGCNr.isin(xg.AGCnr)]
    
    df = pd.concat([a40[['id', 'logfgas']], xg[['id', 'logfgas']]], join='inner')
    
    df.to_csv(PATH/'data'/'combined.csv')
    
    print(df.shape)
    

def combine_image_folders(legacy=False):
    
    if legacy:
        a40_imf = HOME/'alfalfa-convnets'/'images-legacy'
        xg_imf  = HOME/'alfalfa-convnets'/'images-xGASS-legacy'
        imf = PATH/'images-legacy'
    else:
        a40_imf = HOME/'alfalfa-convnets'/'images-OC'
        xg_imf  = HOME/'alfalfa-convnets'/'images-xGASS'
        imf = PATH/'images'
        
    xg  = pd.read_csv(HOME/'alfalfa-convnets'/'data'/'xGASS_representative_sample.csv')
    a40 = pd.read_csv(HOME/'alfalfa-convnets'/'data'/'a40-SDSS_gas-frac.csv')
    
    # run xGASS first
    for xg_id in tqdm(xg.GASS):
        shutil.copy(xg_imf/f'{xg_id}.jpg', imf/f'GASS_{xg_id}.jpg')

    for a40_id in tqdm(a40.AGCNr):
        shutil.copy(a40_imf/f'{a40_id}.jpg', imf/f'AGC_{a40_id}.jpg')

if __name__ == '__main__':
#     combine_df()
     combine_image_folders()
#    combine_image_folders(legacy=True)
