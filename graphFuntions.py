import matplotlib.pyplot as plt
import mplfinance as mpf
from myStocksFunctions import stock_information
import numpy as np

wick_threshold = 0.0001
def support(df1, l, n1, n2): #n1 n2 before and after candle l
    if ( df1.Low[l-n1:l].min() < df1.Low[l] or
        df1.Low[l+1:l+n2+1].min() < df1.Low[l] ):
        return 0

    candle_body = abs(df1.Open[l]-df1.Close[l])
    Lower_wick = min(df1.Open[l], df1.Close[l])-df1.Low[l]
    if (Lower_wick > candle_body) and (Lower_wick > wick_threshold): 
        return 1
    
    return 0

def resistance(df1, l, n1, n2): #n1 n2 before and after candle l
    if ( df1.High[l-n1:l].max() > df1.High[l] or
       df1.High[l+1:l+n2+1].max() > df1.High[l] ):
        return 0
    
    candle_body = abs(df1.Open[l]-df1.Close[l])
    upper_wick = df1.High[l]-max(df1.Open[l], df1.Close[l])
    if (upper_wick > candle_body) and (upper_wick > wick_threshold) :
        return 1

    return 0


def sup_res_point(l, n1, n2, backCandles, df):
    ss = []
    rr = []
    for subrow in range(l-backCandles, l-n2):
        if support(df, subrow, n1, n2):
            ss.append(df.Low[subrow])
        if resistance(df, subrow, n1, n2):
            rr.append(df.High[subrow])
    
    ss.sort() #keep Lowest support when popping a level
    for i in range(1,len(ss)):
        if(i>=len(ss)):
            break
        if abs(ss[i]-ss[i-1])<=0.0001: # merging Close distance levels
            ss.pop(i)

    rr.sort(reverse=True) # keep Highest resistance when popping one
    for i in range(1,len(rr)):
        if(i>=len(rr)):
            break
        if abs(rr[i]-rr[i-1])<=0.0001: # merging Close distance levels
            rr.pop(i)

    return rr,ss



def res_sup_line_graph(data):

    plt.style.use("dark_background")
    # Geçmiş veriyi alma
    

    # Market renklerini belirleme
    colors = mpf.make_marketcolors(up="#00ff00", down="#ff0000", wick="inherit", edge="inherit", volume="in")
    mpf_style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=colors)
    
    rr, ss = sup_res_point(len(data)-1, 8, 6, 150, data)

    # Grafik oluşturma
    fig, axlist = mpf.plot(data, type="candle", style=mpf_style, volume=True, returnfig=True)

     # Destek seviyelerini grafikte gösterme
    for s in ss:
        axlist[0].axhline(y=s, color='blue', linestyle='--', linewidth=1)
        
    # Direnç seviyelerini grafikte gösterme
    for r in rr:
        axlist[0].axhline(y=r, color='red', linestyle='--', linewidth=1)
        
    return fig


def stock_graph(data):
    
        
    plt.style.use("dark_background")

    colors = mpf.make_marketcolors(up="#00ff00", down="#ff0000", wick="inherit", edge="inherit", volume="in")
    mpf_style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=colors)

    fig, axlist = mpf.plot(data, type="candle", style=mpf_style, volume=True, returnfig=True)
    return fig


def show_user_opertation_graph(ticker, data, user_operations):
    ticker_control = ticker.upper()

    is_operations_exist = (user_operations['Hisse Sembolü'] == ticker_control).any()

    if not is_operations_exist:
        return "Bu hisse ile daha önce alım-satım işlemi yapmamışsınız."
    
    df = stock_information(user_operations)
    a_cost = df.loc[df["Hisse Sembolü"] == ticker_control,"Ortalama Maliyet"].values[0]
 

    data.index = data.index.tz_localize(None)
    user_operations.index = user_operations.index.tz_localize(None)
    
    # Normalize the dates to remove time component
    data.index = data.index.normalize()
    user_operations.index = user_operations.index.normalize()
        
    # Check if there are any operations for the given ticker
    is_operations_exist = (user_operations['Hisse Sembolü'] == ticker_control).any()
    
    if not is_operations_exist:
        return "Bu hisse ile daha önce alım-satım işlemi yapmamışsınız."
    
    buy = user_operations[(user_operations['Hisse Sembolü'] == ticker_control) & (user_operations['İşlem Türü'] == 'alım')]
    sell = user_operations[(user_operations['Hisse Sembolü'] == ticker_control) & (user_operations['İşlem Türü'] == 'satım')]

    buy_points=[]
    sell_points=[]
    # DataFrame'lerdeki tarihleri karşılaştır
    for index1, row1 in data.iterrows():
        if index1 in buy.index:
            # Tarih buy DataFrame'inde varsa, Fiyat değerini ekle
           buy_points.append(buy.loc[index1, 'Fiyat'])   
        else:
          # Tarih buy DataFrame'inde yoksa, NaN ekle
         buy_points.append(np.nan)
        if index1 in sell.index:
           sell_points.append(sell.loc[index1,"Fiyat"])
        else:
           sell_points.append(np.nan)
        
    apd = [mpf.make_addplot(buy_points, type="scatter", markersize=50, color='green',label ="Alım yapılan Nokta"),
           mpf.make_addplot(sell_points, type="scatter", markersize=50, color='red', label ="Satış yapılan Nokta")
           ]

    # Plot the data
    plt.style.use("dark_background")
    colors = mpf.make_marketcolors(up="#00ff00", down="#ff0000", wick="inherit", edge="inherit", volume="in")
    mpf_style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=colors)
    
    fig, axlist = mpf.plot(data, addplot = apd, type='candle',style=mpf_style,volume=True, returnfig=True)
    axlist[0].axhline(y=a_cost, color="lightblue", linestyle="--", linewidth =2, label="Ortalama Maliyet")

    return fig
