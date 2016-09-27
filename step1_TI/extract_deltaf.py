import numpy as np
import sys
import pandas as pd
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt
import seaborn as sns
from pylab import rcParams
import os.path

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

def get_deltaf(dbname, sort=False, plot=True, color='b'):
    x, y = np.loadtxt(dbname, delimiter=',', skiprows=2).T

    if sort:
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
    iy = cumtrapz(y, x)
    iy -= iy[x[:-1] < 3.6].min()

    if plot:
        plt.plot(x[1:], iy, color=color)
        plt.xlabel("Reaction coordinate [A]")
        plt.ylabel("Free energy [eV]")
        plt.xlim((x.min(), x.max()))
        # sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.splitext(dbname)[0] + '.pdf')
        
    df = pd.DataFrame(data=np.vstack([x[:-1], y[:-1], iy]).T, columns=['colvar', 'fcnstr', 'deltaf'])
    return df


def all_together():
    dbs = ['c0.35/db-cracka-300K-000.csv',
           'c0.45/db-crack-300K-000.csv',
           'c0.60/db-crack-300K-000.csv',
           'c1.00/db-crack-300K-000.csv',
           'c2.00/db-crack-300K-000.csv',
           'c4.00/db-crack-300K-001.csv']
    gdata = pd.read_csv('potengs-vs_k-eps0.1-8tall.csv')
    k_springs = [float((s.split('/')[0]).strip('c')) for s in dbs]
    
    Gs = [gdata[gdata['k_spring'] == k]['G/Gc'].item() for k in k_springs]

    colors = sns.color_palette(palette='Set2', n_colors=len(Gs))
    
    alldata = []
    for (dbname, G, k, color) in zip(dbs, Gs, k_springs, colors):
        frame = get_deltaf(dbname, plot=False)
        frame['G'] = G # set G label
        frame['k'] = k # set k label
        # init_valley_cv = np.argmin(frame[frame['colvar'] < 3.5]['deltaf'])
        init_valley_f = np.min(frame[frame['colvar'] < 3.5]['deltaf'])
        
        frame['deltaf'] -= init_valley_f # align 0 to minimum of initial conf
        
        alldata.append(frame)

        plt.plot(frame['colvar'], frame['deltaf'], '-', label='%.02f' % G, color=color)

    alldata = pd.concat(alldata)
    # sns.factorplot(x='strain', y='PotEng', data=data, hue='k_spring', legend_out=True)
    # sns.lmplot(x="colvar", y="deltaf", hue="G", data=alldata, fit_reg=False, legend_out=True, size=6, aspect=1.3)
    # sns.tsplot(data=alldata, time="colvar",
    #            condition="G", value="deltaf")
    plt.xlim(3.2, 6)
    plt.ylim(-2, 2)
    plt.xlabel("Reaction coordinate [A]")
    plt.ylabel("Free energy [eV]")
    plt.legend(title=r'$G/G_c$', loc=2)
    plt.tight_layout()
    return alldata


def peak_pos():
    sns.set_context('paper', font_scale=2, rc={"lines.linewidth": 2})
    sns.set_style('whitegrid')
    # rcParams['figure.figsize'] = 8, 6
    dat = pd.read_csv('cryst_barrierdata.csv')
    sns.lmplot(x='G/Gc', y='ypeak', data=dat, hue='peakn', legend=False, palette='Greys', aspect=1.25, size=6, scatter_kws={"s": 100})
    plt.xlim(1, 2.2)
    plt.ylim(-2,2)
    plt.yticks([-2,-1,0,1,2])
    plt.ylabel('Peak height [eV]')
    plt.legend(loc=1, title="Peak n.")
    plt.tight_layout()
    sns.despine(top=False, right=False)
    plt.savefig('peak_heights.pdf')


    sns.lmplot(x='G/Gc', y='xpeak', data=dat, hue='peakn', legend=False, palette='Greys', aspect=1.25, size=6, scatter_kws={"s": 100})
    plt.xlim(1, 2.2)
    plt.ylim(3.5, 5)
    plt.yticks([3.5, 4, 4.5, 5])
    plt.ylabel('Peak position [A]')
    plt.legend(loc=1, title='Peak n.')
    sns.despine(top=False, right=False)
    plt.tight_layout()
    plt.savefig('peak_cvs.pdf')


def get_cosangle(conf, csvfile):
    from quippy import Atoms
    at = Atoms(conf)
    with open(csvfile, 'r') as fff:
        fff.readline()
        line = fff.readline()
    tipatoms = line.split(',')[2:]
    tipatoms = [int(i) for i in tipatoms]

    p1, p2 = at.positions[tipatoms]
    dvec = p2[:2] - p1[:2]
    d = np.linalg.norm(dvec)
    cosangle = dvec[1] / d
    print(tipatoms)
    return cosangle

if __name__ == "__main__":
    
    plt.close('all')
    plt.clf()
    dbname = sys.argv[1].strip()
    get_deltaf(dbname)
