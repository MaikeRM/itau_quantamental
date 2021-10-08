import math
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_ret_ln(df):
    #Cria matriz para receber valores calculados
    df_ln = pd.DataFrame(columns = df.columns, index = df.index)

    #Calcula o ln para todos os valores
    for i in range(len(df)):
        for columns in df.columns:
            df_ln[columns][i] = math.log(df[columns][i])

    # Retorno diário OLS
    ret = df_ln - df_ln.shift(-1)

    return ret.dropna(), df_ln


def reg_m(x,y):
    X = sm.add_constant(x)
    modelo = sm.OLS(y, X)
    modelo_v1 = modelo.fit()
    return modelo_v1


def coint_model(residuos):
    try:
        adfTest = adfuller(residuos, autolag='AIC')
        return {
            'ADF': adfTest,
        }
    except:
        raise


def get_scatter_plot(series_x, series_y, ols):
    x = np.arange(series_x.values.min(), series_x.values.max())
    plt.clf()
    plt.cla()
    plt.scatter(series_x, series_y)
    plt.plot(x, ols.params.const + ols.params.x1 * x, color='red')


def half_life(ts):
    lagged = ts.shift(1).fillna(method="bfill")
    delta = ts-lagged
    X = sm.add_constant(lagged.values)
    ar_res = sm.OLS(delta, X).fit()
    half_life = -1*np.log(2)/ar_res.params['x1']
    return half_life, ar_res, delta, lagged