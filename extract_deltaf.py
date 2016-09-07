import numpy as np
import sys
import pandas as pd
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt
import seaborn as sns
from pylab import rcParams
import os.path

plt.clf()
plt.close('all')
sns.set_context('paper', font_scale=2, rc={"lines.linewidth": 2})
sns.set_style('whitegrid')
rcParams['figure.figsize'] = 8, 6

def guess_tipatom_idx(df):
    n_db, n_cols = df.shape
    temp = df.T[3:360].T
    # get index of x coordinate of second atom in tip. Add 1 to have y coordinate == colvar
    tipatom_idx = np.argmin(temp.std()) + 1
    return tipatom_idx


def get_deltaf_olddb(dbname, tipatom_idx=None):

    data = pd.read_csv(dbname, header=None)
    basename = dbname.strip('.csv').strip('db-')
    
    if tipatom_idx is None:
        tipatom_idx = guess_tipatom_idx(data)
    
    n_db, n_cols = data.shape
    y = data[n_cols - 1].values
    x = data[tipatom_idx].values
    
    # sort values in order of increasing colvar. Should not be necessary, but sometimes the constraint is not perfect
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    # integrate constraint force
    iy = cumtrapz(y, x)

    df = pd.DataFrame(data=np.vstack([x[:-1], y[:-1], iy]).T, columns=['colvar', 'fcnstr', 'deltaf'])
    df.to_csv('deltaf-' + basename + '.csv')

    plt.plot(x[1:], iy - iy.max())
    sns.despine()
    plt.xlabel(r"Reaction coordinate [$\AA{}$]")
    plt.ylabel(r"$\Delta F$ [eV]")
    plt.xlim((x.min(), x.max()))
    plt.tight_layout()

    plt.savefig(basename + '.png')
    return df

def get_deltaf(dbname, sort=False):
     x, y = np.loadtxt(dbname, delimiter=',', skiprows=2).T

     if sort:
         sort_idx = np.argsort(x)
         x = x[sort_idx]
         y = y[sort_idx]
     iy = cumtrapz(y, x)

     plt.plot(x[1:], iy)
     plt.xlabel(r"Reaction coordinate [$\AA{}$]")
     plt.ylabel(r"$\Delta F$ [eV]")
     plt.xlim((x.min(), x.max()))
     sns.despine()
     plt.tight_layout()
     
     plt.savefig(os.path.splitext(dbname)[0] + '.pdf')
     
if __name__ == "__main__":
    
    dbname = sys.argv[1].strip()
    get_deltaf(dbname)
