from server_kernel_net import Model
import pandas as pd

# with open('testdesc.txt', 'r') as f:
#     description = f.read().strip()
# description = 'Google Play — магазин приложений, а также игр, книг, музыки и фильмов от компании Google, позволяющий сторонним компаниям предлагать владельцам устройств с операционной системой Android устанавливать и приобретать различные приложения.'
model = Model()
print('Model initialized!')
df = pd.read_csv('data/apps.csv').head(1)
df['Keywords'] = df['Keywords'].apply(lambda x: str(x).replace('|', ','))
kernels = df['kernel'].values
names = df['Name'].values
categories = df['Category'].values
print('Predicting...')
predictions = []
for kernel, name, category in zip(kernels, names, categories):
    prediction = model.predict(kernel, category, name)
    prediction = ','.join(prediction)
    predictions.append(prediction)

print(predictions)