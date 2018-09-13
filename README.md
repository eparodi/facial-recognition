# Reconocimiento Facial

Este programa ejecuta un algoritmo para identificar rostros y luego se pone a prueba con una interfaz de cámara web.

## Integrantes

* Eliseo Parodi Almaraz
* Marcos Abelenda
* Matías Ota
* Ariel Debrouvier

## Requerimientos

* Python 3
* pip3 (manejador de dependencias de Phyton)
   
## Instalación

Estando en la carpeta principal ejecute el siguiente comando para instalar las dependencias

```
pip3 install -r requirements.txt
```

## Ejecución

Para ejecutar el programa, desde la carpeta principal, ejecute el siguiente código

```
python3 main.py
```

## Configuración

La configuración del programa se encuentra en el archivo `config.json`.
Los parametros que toma son:

* path: el directorio donde se encuentran las fotos.
* imageWidth: el ancho de las imágenes en píxeles.
* imageHeight: el alto de las imágenes en píxeles.
* trainImages: la cantidad de imágenes para entrenamiento.
* testImages: la cantidad de imágenes para testeo.
* webcamInterface: indica la fuente de donde sale la cámara web.
  Si es 0 utiliza la cámara web de la laptop, si hay.