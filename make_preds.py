from server_net import Model
import pandas as pd
from utils import text2canonicals


model = Model()
data_name = 'testdesc'
data = pd.read_csv(data_name + '.csv')

preds = []

for name, cat, desc in zip(data[['name', 'category', 'description']].values):
    feed_data = str(name) + ' ' + str(cat) + ' ' + str(desc) + ' ' + str(cat) + ' ' + str(name)
    feed_data = text2canonicals(feed_data)
    res = model.predict(feed_data, top_n=300)
    res = ','.join(res)
    preds.append(res)

data['Predictions'] = preds

data.to_csv(data_name + '_with_preds.csv', index=False)