import pandas as pd
import numpy as np
import coin
import yfinance as yf
from tqdm import tqdm


def download_dados(tickers, interval, period):

    df = yf.download(tickers, interval=interval, period=period)['Adj Close']
    #retorno, df_ln = coin.calc_ret_ln(df)

    return coin.calc_ret_ln(df)

def teste_coint(df, step, coin_min, coin_max, coin_step):
    i=0
    testes = pd.DataFrame(columns=['Ativo 1', 'Ativo 2', 'Inicio' ,'Periodo' ,'ADF Test'], index = range(1000000))

    for column_1 in tqdm(df.columns):
        for column_2 in df.columns:
            if column_1 != column_2:
                print('\r',column_1, column_2, end='', flush=True)

                for inicio in range(0, len(base), step):
                    for periodo in np.arange(coin_min,(coin_max+coin_step), coin_step):
                        y = np.array(base[column_1].iloc[inicio:inicio + periodo], dtype = float)
                        x = np.array(base[column_2].iloc[inicio:inicio + periodo], dtype = float)

                        coef = coin.reg_m(x,y).params

                        residuos = pd.DataFrame(columns=['residuos'])
                        residuos['residuos'] = base[column_1] - (base[column_2]*coef[1] + coef[0])
                        
                        test = coin.coint_model(residuos)['ADF'][0]
                        
                        testes['Ativo 1'].iloc[i] = column_1
                        testes['Ativo 2'].iloc[i]  = column_2
                        testes['Inicio'].iloc[i]  = inicio
                        testes['Periodo'].iloc[i]  = periodo
                        testes['ADF Test'].iloc[i]  = test
                        print('\r', testes, end='', flush=True)
                        i+=1
    testes.to_csv('testes_coin_preco.csv')

if __name__ == '__main__':

    tickers = "ABEV3.SA AZUL4.SA B3SA3.SA BBAS3.SA BBDC3.SA BBDC4.SA BBSE3.SA BPAC11.SA BRAP4.SA BRDT3.SA BRFS3.SA BRKM5.SA BRML3.SA CCRO3.SA CIEL3.SA CMIG4.SA COGN3.SA CRFB3.SA CSAN3.SA CSNA3.SA CVCB3.SA CYRE3.SA ECOR3.SA EGIE3.SA ELET3.SA ELET6.SA EMBR3.SA ENBR3.SA EQTL3.SA FLRY3.SA GGBR4.SA GNDI3.SA GOAU4.SA GOLL4.SA HAPV3.SA HGTX3.SA HYPE3.SA IGTA3.SA IRBR3.SA ITSA4.SA ITUB4.SA JBSS3.SA KLBN11.SA LAME4.SA LREN3.SA MGLU3.SA MRFG3.SA MRVE3.SA MULT3.SA NTCO3.SA PETR3.SA PETR4.SA QUAL3.SA RADL3.SA RAIL3.SA RENT3.SA SANB11.SA SBSP3.SA SULA11.SA SUZB3.SA TAEE11.SA TOTS3.SA UGPA3.SA USIM5.SA VALE3.SA VVAR3.SA WEGE3.SA YDUQ3.SA"
    interval='1d'
    period='3y'
    retorno, df_ln = download_dados(tickers, interval, period)
    base = df_ln

    step = 50
    coin_min = 60
    coin_max = 200
    coin_step = 40
    teste_coint(base, step, coin_min, coin_max, coin_step)
