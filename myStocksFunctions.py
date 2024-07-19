import pandas as pd


df = pd.read_csv('kullanıcı_işlemleri.csv', encoding='utf-8-sig')

def calculate_user_shares(df):
    user_shares = {}
    user_average_cost = {}

    for index, row in df.iterrows():
        user_id = row['Kullanıcı ID']
        stock = row['Hisse Sembolü']
        amount = row['Miktar']
        price = row['Fiyat']

        if user_id not in user_shares:
            user_shares[user_id] = {}
            user_average_cost[user_id] = {}

        if stock not in user_shares[user_id]:
            user_shares[user_id][stock] = 0
            user_average_cost[user_id][stock] = {'toplam_maliyet': 0, 'toplam_miktar': 0}

        if row['İşlem Türü'] == 'alım':
            user_shares[user_id][stock] += amount
            user_average_cost[user_id][stock]['toplam_maliyet'] += price * amount
            user_average_cost[user_id][stock]['toplam_miktar'] += amount
        elif row['İşlem Türü'] == 'satım':
            user_shares[user_id][stock] -= amount

    return user_shares, user_average_cost


def stock_information(df):
    user_shares, user_average_cost = calculate_user_shares(df)
    # Kullanıcının hisse bilgilerini DataFrame'e dönüştürme
    user_shares_list = []

    for user_id, stocks in user_shares.items():
        for stock, amount in stocks.items():
            if amount > 0:
                total_cost = user_average_cost[user_id][stock]['toplam_maliyet']
                total_amount = user_average_cost[user_id][stock]['toplam_miktar']
                average_cost = total_cost / total_amount if total_amount > 0 else 0
            else:
                average_cost = 0  # Eğer hisse miktarı 0 veya negatifse, ortalama maliyeti sıfır yap

            user_shares_list.append({"Kullanıcı ID": user_id, "Hisse Sembolü": stock, "Hisse Miktarı": amount, "Ortalama Maliyet": average_cost})

    user_shares_df = pd.DataFrame(user_shares_list)
    return user_shares_df



