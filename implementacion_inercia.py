import mnist_loader
import network_inercia
import pickle

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=network_inercia.Network([784,30,10])

net.SGD( training_data, 30, 10, 3.0, 0.3, test_data=test_data)

archivo = open("red_prueba1.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()
#leer el archivo

archivo_lectura = open("red_prueba.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()

net.SGD( training_data, 10, 50, 0.5, 0.3, test_data=test_data)

archivo = open("red_prueba.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()