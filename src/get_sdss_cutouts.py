"""
John F. Wu (2020)

Fetch JPG images from SDSS cutout service based on 
catalog of spectra.
"""

from optparse import OptionParser
import pandas as pd
import skimage.io

import os
from pathlib import Path
import time
import sys
import urllib

from astropy.coordinates import SkyCoord
import astropy.units as u

PATH = Path(__file__).parent.parent.absolute()


class Printer:
    """Print things to stdout on one line dynamically"""

    def __init__(self, data):
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def process_coordinates(df):
    """Gets ra, dec from df in correct  SkyCoord.
    """

    ra = df['ra'].apply(lambda x: x.strip("'"))
    dec = df['dec'].apply(lambda x: x.strip("'"))

    return SkyCoord(ra, dec, unit=(u.hourangle, u.deg))

def process_identifiers(df):
    """Loads df and returns plate+mjd+fiberid spectral ids.
    """
    
    plate   = df.plate  .apply(lambda x: f'{x:>04d}') 
    mjd     = df.mjd    .apply(lambda x: f'{x:>06d}') 
    fiberid = df.fiberid.apply(lambda x: f'{x:>04d}') 

    return ['-'.join([p,m,f]) for p,m,f in zip(plate, mjd, fiberid)]

def cmdline():
    """Controls the command line argument handling for this little program.
    """

    # read in the cmd line arguments
    USAGE = "usage:\t %prog [options]\n"
    parser = OptionParser(usage=USAGE)

    # add options
    parser.add_option(
        "--output",
        dest="output",
        default=f"{PATH}/images-detectability",
        help="Path to save image data",
    )
    parser.add_option(
        "--width", dest="width", default=224, help="Default width of images"
    )
    parser.add_option(
        "--height", dest="height", default=224, help="Default height of images"
    )
    parser.add_option(
        "--cat",
        dest="cat",
        default=f"{PATH}/results/detectability_sample.csv",
        help="Catalog to get image names from.",
    )

    (options, args) = parser.parse_args()

    return options, args


def main():

    opt, arg = cmdline()
    width = opt.width
    height = opt.height

    # remove trailing slash in output path if it's there.
    opt.output = opt.output.rstrip("\/")

    # load data & get useful columns

    df = pd.read_csv(opt.cat)
    zipped = zip(df.id, df.ra, df.dec)

    # total number of images/spectra
    n_gals = df.shape[0]

    for i, (id_, ra, dec) in enumerate(zipped):

        url = (
            "http://skyserver.sdss.org/dr16/SkyserverWS/ImgCutout/getjpeg"
            "?ra={}"
            "&dec={}"
            "&width={}"
            "&height={}".format(ra, dec, width, height)
        )
        if not os.path.isfile(f"{opt.output}/{id_}.jpg"):
            try:
                img = skimage.io.imread(url)
                skimage.io.imsave(f"{opt.output}/{id_}.jpg", img)
                time.sleep(0.01)
            except urllib.error.HTTPError:
                pass
        current = i / n_gals * 100
        status = "{:.3f}% of {} completed.".format(current, n_gals)
        Printer(status)

    print("")


if __name__ == "__main__":
    main()
