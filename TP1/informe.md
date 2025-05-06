# Procesamiento de Imágenes - Trabajo Práctico 1
Integrantes:
- Francisco Devaux
- Agustín Yornet de Rosas

## Introducción
Este trabajo corresponde al Trabajo Práctico 1 de Procesamiento de Imágenes, y consiste en los siguientes documentos:

- Informe hecho en Markdown (informe.md) con detalles sobre la resolución de los ejercicios.
- [Notebook interactivo de Python (PDI_TP1.ipynb)](PDI_TP1.ipynb) con la resolución de los ejercicios y su código fuente. Este puede ejecutarse localmente, celda por celda.
- [Informe hecho en LaTeX (PDI_TP1.ipynb)](TP1_PDI.pdf) con las respuestas teóricas de los ejercicios.
- Carpeta `imágenes/` con las imágenes utilizadas para la resolución de este Trabajo Práctico.

Para poder replicar los resultados, se recomienda crear un entorno virtual, instalar las depedencias contenidas en [requirements.txt](requirements.txt) y correr las celdas una por una, o todas juntas, de forma secuencial. Se ha utilizado Python 3.10.12 como Kernel para ejecutar el [notebook interactivo](PDI_TP1.ipynb).

Se recomienda usar Visual Studio Code para ir ejecutando el Notebook a medida que se va leyendo este informe.

## 1 Modos de Color en Imágenes

### Ejercicio 6

Para reproducir los resultados del Ejercicio 6, se deben ejecutar previamente las primeras tres celdas correspondientes a: 
- Librerías utilizadas.
- Funciones auxiliares `printImg()` y `readImg()`.
- Lectura de `Lenna.png`.

**Inciso a)**

> Ir y ejecutar subsección "Método 1"

Se utilizó `imagen_lenna_gris = cv2.cvtColor(imagen_lenna, cv2.COLOR_RGB2GRAY)` para convertir la imagen a escala de grises. 

```python
imagen_lenna = cv2.imread('./imagenes/Lenna.png', cv2.COLOR_BGR2RGB)

if imagen_lenna is not None:
  imagen_lenna_gris = cv2.cvtColor(imagen_lenna, cv2.COLOR_RGB2GRAY)
  printImg(imagen_lenna_gris,gray = True)
else:
    print("Error: imagen_lenna no se ha cargado correctamente.")
```

**Inciso b)**

> Ir y ejecutar subsección "Método 2"

Se utilizó `cv2.split` para separar los canales de la imagen y la fórmula de luminancia para convertir la imagen. 

```python
if imagen_lenna is not None:
  b, g, r = cv2.split(imagen_lenna)

  imagen_lenna_gris_luminancia = 0.3 * r + 0.59 * g + 0.11 * b
  imagen_lenna_gris_luminancia = imagen_lenna_gris_luminancia.astype(np.uint8)

  printImg(imagen_lenna_gris_luminancia, gray=True)

else:
    print("Error: imagen_lenna no se ha cargado correctamente.")

```

**Inciso c)**
> Ir y ejecutar subsección "Método 3"

Se utilizó `rgb2gray` de `scikit-image` para convertir la imagen en blanco y negro. 

```python
imagen_lenna_rgb = cv2.cvtColor(imagen_lenna, cv2.COLOR_BGR2RGB)
imagen_lenna_gris_scikit = rgb2gray(imagen_lenna_rgb)
plt.imshow(imagen_lenna_gris_scikit, cmap='gray')
plt.show()
```

**Inciso d)**
> Ir y ejecutar subsección "Pregunta 1"

Luego de obtener el `shape` de cada imagen, es posible observar que la imagen original conserva sus tres canales (BGR) mientras que las versiones en escalas en grises poseen un único canal.

**Inciso e)**
> Ir y ejecutar subsección "Pregunta 2"

Luego de obtener el `dtype` de cada imagen, es posible observar que la imagen a color tiene una profundidad de 24 bits, 8 por cada canal.Las dos primeras imágenes en escala de grises tienen una profundidad de 8 bits, un único canal. La última imagen está en formato `float64`, lo que indica una profundidad de 64 bits en punto flotante.

### Ejercicio 7
> Ir y ejecutar subsección "Ejercicio 7"

Se utilizó `cv2.COLOR_RGB2HSV` y `cv2.COLOR_RGB2HLS` para convertir la imagen en perfiles de color HSV y HSL. Para CMYK, se calculó cada componente por separado, y luego fueron unidos. 

```python
# HSV
imagen_hsv = cv2.cvtColor(imagen_lenna, cv2.COLOR_RGB2HSV)

# HLS
imagen_hsl = cv2.cvtColor(imagen_lenna, cv2.COLOR_RGB2HLS)

# CMYK
bgr_normalized = imagen_lenna.astype(float) / 255.0
b = bgr_normalized[:, :, 0]
g = bgr_normalized[:, :, 1]
r = bgr_normalized[:, :, 2]

k = 1 - np.max(bgr_normalized, axis=2)
c = (1 - r - k) / (1 - k + 1e-10)
m = (1 - g - k) / (1 - k + 1e-10)
y = (1 - b - k) / (1 - k + 1e-10)

cmyk = np.stack((c, m, y, k), axis=2)
cmyk_image = (cmyk*255).astype(np.uint8)
```

### Ejercicio 8
> Ir y ejecutar subsección "Ejercicio 8"

Se utilizó `cv2.cvtColor(imagen_lenna_gris, cv2.COLOR_GRAY2RGB)` para convertir la imagen en escala de grises a color, `imagen_lenna_gris_a_rgb.shape` para verificar su forma, y `np.array_equal(b, g) and np.array_equal(g, r)` para verificar si los canales de la imagen tienen los mismos valores.

El resultado del ejercicio fue que la conversión replicó el canal de la imagen en escala de grises tres veces, y tienen los mismos valores.

## 2 Compresión de Imágenes
### Ejercicio 2

**PSNR (Peak Signal-to-Noise Ratio)**

Métrica que mide la diferencia promedio entre la imagen original y la comprimida. Basada en el error cuadrático medio.
Tiene la siguiente fórmula:

$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right)$$

$$\text{MAX es el valor máximo posible de un píxel (255 en imágenes de 8 bits).}$$

$$\text{MSE es el error cuadrático medio entre las dos imágenes.}$$

Si se tienen valores mayores de 40, se considera una calidad de compresión muy buena, casi sin pérdida. Si se encuentran entre 30 y 40, se la considera buena. Si esta entre 20 y 30, regular con una visible pérdida. Si es menor de 20 se la considera mala.

Tiene el inconveniente de que no considera como percibe el ojo humano la imagen.

**SSIM (Structural Similarity Index)**

Métrica que evalúa la similitud estructural entre dos imágenes, teniendo en cuenta la luz y el contraste. Se aproxima mucho mejor a cómo percibimos la calidad bisual las personas.

Si el valor es cercano a 1 significa que las imágenes son muy similares, si es cercano a 0 implica que son muy distintas.

Su fórmula es mucho más compleja que la anterior:

$$ SSIM(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

$$\mu_x, \mu_y: \text{medias de las imágenes x e y}$$
$$\sigma_x^2, \sigma_y^2: \text{varianzas}$$
$$\sigma_{xy}: \text{covarianza entre x e y}$$
$$C_1, C_2: \text{constantes pequeñas para evitar división por cero}$$

### Ejercicio 6
> Ir y ejecutar subsección "Ejercicio 6"

Se construyeron las siguientes funciones para la resolución del Ejercicio 6:
- `rle_encode(img)`: Codifica una imagen en escala de grises usando Run-Length Encoding (RLE), representando secuencias de píxeles repetidos como pares (valor, cantidad).
- `rle_decode(encoded, shape)`: Reconstruye una imagen a partir de una lista RLE de pares (valor, cantidad), devolviéndola con la forma (shape) original.

- `procesar_imagen(path)`: Carga una imagen, la convierte a escala de grises, aplica compresión y descompresión RLE, calcula el PSNR, estima el tamaño comprimido y muestra los resultados y comparaciones visuales.

A partir de la implementación del algoritmo de compresión RLE, se puede observar que su eficacia varía significativamente según las características de la imagen. En el caso de la imagen `img_color1.png`, se logró una reducción considerable del tamaño, pasando de 61.48 KB a un estimado de 28.12 KB, lo que indica que contenía muchas secuencias repetidas de píxeles, haciendo que la compresión sea efectiva. En cambio, las imágenes `Lenna.png` y `paisaje2.jpg` mostraron un comportamiento opuesto: el tamaño estimado tras la compresión fue igual o incluso mayor al tamaño en disco original, debido a la falta de patrones repetitivos evidentes, lo que demuestra que RLE no es adecuado para imágenes con alta variabilidad tonal o detalles complejos.

Por otro lado, el valor de PSNR (infinito) entre la imagen original y la reconstruida en todos los casos confirma que la reconstrucción es perfecta, es decir, no se pierde calidad en el proceso.