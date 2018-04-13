import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=["C", "p", "p+q", "r", "E", "val"], dtype='object')

with open("computed_values_new.out", "r") as fn:
    for i, line in enumerate(fn):
        [C, E, p, pq, r, val] = line.split(" ")
        r = r[:-1]
        df.loc[i] = [C, p, pq, r, E, val]
df = df.apply(pd.to_numeric)
df = df.drop_duplicates(["C", "p", "p+q", "r", "E"], keep='last')
df = df.sort_values(["p", "p+q", "r"])

df['mult2'] = (df["p+q"] == 2*df["p"])
df['mult87'] = (df["p+q"] == 8.0/7*df["p"])
df['add4'] = (df["p+q"] == 4 + df["p"])

def maxr(ldf = df):
    return ldf[ldf['r'] == ldf['r'].max()]

def twicer(ldf = df):
    return ldf[ldf['r'] == 2*ldf['p+q']]

# remove deficient values
def remdef(ldf = df):
    ldf = ldf[(ldf['p'] != 7) | (ldf['p+q'] != 8)]
    return ldf[(ldf['val'] != -1.0) & ~((ldf['C'] == 1) & (ldf['p'] == 7) & (ldf['p+q'] == 8)) & (ldf['val'] < 1e10)]


def select(C, E, p=False, pq=False, ldf = df):
    if p:
        return ldf[(ldf["C"] == C) & (ldf["E"] == E) & (ldf["p"] == p) & (ldf["p+q"] == pq)]
    return ldf[(ldf["C"] == C) & (ldf["E"] == E)]

def mult(m, ldf = df):
    return ldf[ldf["p+q"] == m*ldf["p"]]

def add(q, ldf = df):
    return ldf[ldf["p+q"] == (ldf["p"] + q)]

rdf = remdef(df)

def rquotient(ldf):
    lldf = ldf[ldf['p+q'] <= 32]
    r4pq = lldf[lldf['r'] == 8*lldf['p+q']]['val']
    r2pq = lldf[lldf['r'] == 2*lldf['p+q']]['val']
    return np.divide(r4pq, r2pq)

def plotselected(C, E, df, ax=False):
    rdf = select(C, E, False, False, df)
    ldf = maxr(rdf)
    ldf2 = twicer(rdf)
    if not ax:
        fig, ax = plt.subplots()
    handles = {}
    for tpp, color in [('add4', 'blue'), ('mult2', 'orange'), ('mult87', 'red')]:
        handles[tpp] = ldf[ldf[tpp]].plot(x='p+q', y='val', ax=ax, label=tpp, color=color)
        ldf2[ldf2[tpp]].plot(x='p+q', y='val', ax=ax, color=color, style='--')
        ax.plot(ldf[ldf[tpp]]['p+q'], [1 for i in ldf[ldf[tpp]]['p+q']], 'k:')
    lines, labels = ax.get_legend_handles_labels()
    leglines = [lines[0], lines[2], lines[4]]
    leglabels = [labels[0], labels[2], labels[4]]
    ax.legend(leglines, leglabels, loc='best')

def plotgridC1(df):
    fig, axarr = plt.subplots(4, 4, sharex=True)
    c1es = (((-1, 1), (-1, 1)),((-1, 1), (    1,)),((-1, 1), (-1,   )),((-1, 1), (     )),
                ((-1,   ), (-1, 1)),((-1,   ), (    1,)),((-1,   ), (-1,   )),((-1,   ), (     )),
                ((    1,), (-1, 1)),((    1,), (    1,)),((    1,), (-1,   )),((    1,), (     )),
                ((     ), (-1, 1)),((     ), (    1,)),((     ), (-1,   )))
    for c1 in range(4):
        for c2 in range(4):
            if 4*c1 + c2 > 14:
                pass
            plotselected(1, 4*c1 + c2, df, ax=axarr[c1][c2])
            axarr[c1][c2].set_title("Dir BC=%s" % str(c1es[4*c1 + c2]))

def plotgridC2(df):
    fig, axarr = plt.subplots(4,3, sharex=True)
    c2es = (((((   ), (-1,)),((-1, 1), (1,)))),
            ((((   ), (-1,)),((-1, 1), (  )))),
            ((((   ), (-1,)),((-1,  ), (1,)))),
            ((((   ), (-1,)),((-1,  ), (  )))),
            ((((   ), (-1,)),((   1,), (1,)))),
            ((((   ), (-1,)),((   1,), (  )))),
            ((((   ), (-1,)),((     ), (1,)))),
            ((((-1,), (-1,)),((   1,), (1,)))),
            ((((-1,), (-1,)),((   1,), (  )))),
            ((((-1,), (-1,)),((     ), (1,)))),
            ((((-1,1), (-1,)),((     ), (1,)))))
    for c1 in range(4):
        for c2 in range(3):
            if 3*c1 + c2 > 10:
                pass
            plotselected(2, 3*c1 + c2, df, ax=axarr[c1][c2])
            axarr[c1][c2].set_title("Dir BC=%s; Nmn BC=%s" % (str(c2es[3*c1 + c2][0]), str(c2es[3*c1 + c2][1])))

def poep_tabel(ldf = rdf):
    C1df = {i: select(1, i, False, False, ldf) for i in [1, 2, 3, 4, 5]}
    C2df = {i: select(2, i, False, False, ldf) for i in [1, 2, 3, 4]}
    C3df = {0: select(3, 1, False, False, ldf)}
    dfs = [C1df[1], C1df[2], C1df[3], C1df[4], C1df[5], C2df[1], C2df[2], C2df[3], C2df[4], C3df[0]]
    qs = [8, 16, 32, 64, 128]
    ps = [lambda q: q-4, lambda q: q/8*7, lambda q: q/2]
    rs = [lambda q: 2*q, lambda q: 4*q, lambda q: 8*q]
    colnames = ["C1E1", "C1E2", "C1E3", "C1E4", "C1E5", "C2F1", "C2F2", "C2F3", "C2F4", "C3"]
    #nice_df = pd.DataFrame(columns=["p", "p+q", "r", "C1E1", "C1E2", "C1E3", "C1E4", "C1E5", "C2F1", "C2F2", "C2F3", "C2F4", "C3"], dtype="float64")
    dicts = []
    for p in ps:
        for q in qs:
            ldict = {"p": p(q), "q": q}
            for cdf,name in zip(dfs, colnames):
                for (rtype,r) in enumerate(rs):
                    vv = cdf[(cdf['p'] == p(q)) & (cdf['p+q'] == q) & (cdf['r'] == r(q))]['val']
                    assert(len(vv.values) <= 1)
                    if len(vv.values) == 1:
                        ldict[name + ("r%d" % rtype)] = ("%.4f" % vv.values[0])
            dicts.append(ldict)
    nice_df = pd.DataFrame(dicts)
    #nice_df.set_index(['p', 'q', 'r'], inplace=True)

    return nice_df
