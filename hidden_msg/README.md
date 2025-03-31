### Reto 1. Descifrando del mensaje secreto.

Un oponente genera mensajes de 32 bits con información confidencial y los cifra utilizando un codificador determinista que reduce cada mensaje a una secuencia de 16 números de punto flotante. De los 10,000 mensajes generados, se ha logrado interceptar solo el 30% de los originales (3000 mensajes) junto con sus correspondientes versiones cifradas (100 de ellos). Su misión es entrenar un decodificador capaz de reconstruir exactamente los mensajes originales a partir de sus versiones cifradas, utilizando únicamente los 3000 mensajes interceptados para el entrenamiento.<br>

Los siguientes son ejemplos de mensajes no-cifrados confidenciales<br>
```
11000110100110011001100110010101
10111010010100010110111001000110
11001100110100000000100001010011
10110001001010101101110100101000
``` 

El siguiente es un ejemplo un mensaje cifrado<br>
```
-0.21161067  0.9819643   0.50920117 -0.10046072  0.0970166   0.47773603
-0.02057732 -0.19037446  0.99944896  0.6825894  -0.1577787   0.3304235
0.73278713 -0.09858703  0.11817224  0.9356269
```

El reto consiste en construir un decodificador capaz de recuperar los mensajes cifrados interceptados sin conocer los detalles de la red neuronal que los generó, incluyendo:
- El número de capas.
- La cantidad de nodos por capa.
- Las funciones de activación utilizadas.
- El número de épocas y el tamaño del batch en su entrenamiento

Vea el notebook para más detalles.