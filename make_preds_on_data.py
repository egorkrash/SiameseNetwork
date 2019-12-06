from server_kernel_net import Model
import pandas as pd
from utils import text2canonicals


model = Model(checkpoint_path='./weights/params_9')
data_name = 'testdesc.csv'
data = pd.read_csv(data_name)

preds = []

for name, cat, desc in zip(data[['name', 'category', 'description']].values):
    feed_data = str(name) + ' ' + str(cat) + ' ' + str(desc) + ' ' + str(cat) + ' ' + str(name)
    feed_data = text2canonicals(feed_data)
    res = model.predict(feed_data, top_n=300)
    res = ','.join(res)
    preds.append(res)

data['Predictions'] = preds
data['Keywords'] = data['Keywords'].apply(lambda x: str(x).replace('|', ','))
data.to_csv(data_name + '_with_preds.csv', index=False)