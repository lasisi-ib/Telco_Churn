import pandas as pd

def indexer(var, data):
    rows,cols = data.isin([var]).to_numpy().nonzero()
    for r, c in zip(rows, cols):
        r=r
        c=c

    return r,c


def  stackedbar(var, comp_var, axs, data):
    tomp = pd.DataFrame(pd.crosstab(data[var], data[comp_var]))
    tomp1 = pd.DataFrame(pd.crosstab(data[var], data[comp_var]))
    for i in range(len(tomp)):
        tomp1.iloc[i]= round((tomp.iloc[i]/ tomp.iloc[i].sum()) * 100, 0)
    
    tt = tomp.plot(kind ='bar', ax=axs, stacked= True)
    for d in tt.patches:
        r,c = indexer(d.get_height(), tomp)
        axs.annotate(f'{tomp1[tomp.columns[c]][r]}%',
                     ((d.get_x() + d.get_width() / 2) * 0.96, (d.get_y() + d.get_height() / 2 )* 0.99),
                    color = 'black')
   