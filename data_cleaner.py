import pandas as pd
data = pd.read_csv('crawler/unclean_data.csv')
for i in range(len(data)):
    data.loc[i, 'area'] = float(data.loc[i, 'area'].replace('m', ''))
    data.loc[i, 'bedroom'] = float(str(data.loc[i, 'bedroom']).replace(' phòng ngủ', ''))
    if 'triệu /\xa0m' in data.loc[i, 'price']:
        data.loc[i, 'price'] = float(data.loc[i, 'price'].replace('triệu /\xa0m', '').replace(',', '.')) * data.loc[i, 'area'] / 1000
    elif 'triệu' in data.loc[i, 'price']:
        data.loc[i, 'price'] = float(data.loc[i, 'price'].replace('triệu', '')) / 1000
    else:
        data.loc[i, 'price'] = float(data.loc[i, 'price'].replace('tỷ', '').replace(',', '.'))
    data.loc[i, 'price'] = round(data.loc[i, 'price'], 2) 
data = data[data['bedroom'] < 5]
data = data[data['price'] < 20]
data.to_csv('dataset.csv')                            
print(data)
