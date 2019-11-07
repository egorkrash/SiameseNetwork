from server_net import Model

with open('testdesc.txt', 'r') as f:
    description = f.read().strip()
#escription = 'Google Play — магазин приложений, а также игр, книг, музыки и фильмов от компании Google, позволяющий сторонним компаниям предлагать владельцам устройств с операционной системой Android устанавливать и приобретать различные приложения.'
model = Model()

predictions = model.predict(description)
for pred in predictions:
    print(pred)
