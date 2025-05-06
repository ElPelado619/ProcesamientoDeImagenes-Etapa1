# Procesamiento de Imágenes - Trabajo Práctico 2
Integrantes:
- Francisco Devaux
- Agustín Yornet de Rosas

## Introducción
Este trabajo corresponde al Trabajo Práctico 2 de Procesamiento de Imágenes, y consiste en los siguientes documentos:

- Informe hecho en Markdown (informe.md) con detalles sobre la resolución de los ejercicios.
- [Notebook interactivo de Python (PDI_TP2.ipynb)](PDI_TP2.ipynb) con la resolución de los ejercicios y su código fuente. Este puede ejecutarse localmente, celda por celda.
- [Informe hecho en LaTeX (PDI_TP2.ipynb)](TP2_PDI.pdf) con las respuestas teóricas de los ejercicios.
- Carpeta `imágenes/` con las imágenes utilizadas para la resolución de este Trabajo Práctico.

Para poder replicar los resultados, se recomienda crear un entorno virtual, instalar las depedencias contenidas en [requirements.txt](requirements.txt) y correr las celdas una por una, o todas juntas, de forma secuencial. Se ha utilizado Python 3.10.12 como Kernel para ejecutar el [notebook interactivo](PDI_TP2.ipynb).

Se recomienda usar Visual Studio Code para ir ejecutando el Notebook a medida que se va leyendo este informe.

## 1 Histogramas

### Ejercicio 7

> Ir y ejecutar subsección "Ejercicio 7"

Para realizar el ajuste de histogramas, se hizo uso de `skimage.exposure.match_histograms(img1, img4)`.

```python
matched_img = match_histograms(img1, img4)

hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
hist4 = cv2.calcHist([img4], [0], None, [256], [0, 256])
hist_matched = cv2.calcHist([matched_img.astype(np.uint8)], [0], None, [256], [0, 256]) 

hist1 = hist1 / np.sum(hist1)
hist4 = hist4 / np.sum(hist4)
hist_matched = hist_matched / np.sum(hist_matched)
```

Luego de transformar la distribución de intensidades de `paisaje1.jpg` para que se parezca a la de `paisaje4.jpg` por medio de `skimage.exposure.match histograms()`, es posible ver que los histogramas, luego del ajuste, presentan mayor cantidad de similitudes en el rango de valores entre $0$ y $110$ (aproximadamente) con pequeñas oscilaciones. Para valores superiores a $110$, la versión ajustada de `paisaje1.jpg` contiene frecuencias muy altas para valores específicos, cercanos a $110$ y $190$.

### Ejercicio 8

> Ir y ejecutar subsección "Ejercicio 8"

Para realizar la ecualización de histograma, se hizo uso de `cv2.equalizeHist(img)`.

```python
equ = cv2.equalizeHist(img4)

hist_original = cv2.calcHist([img4], [0], None, [256], [0, 256])
hist_ecualizado = cv2.calcHist([equ], [0], None, [256], [0, 256])

hist_original = hist_original / np.sum(hist_original)
hist_ecualizado = hist_ecualizado / np.sum(hist_ecualizado)
```

Al comparar las imágenes y sus histogramas, es posible ver que el contraste de la imagen original ha aumentado de forma notable en su versión ecualizada. Como es posible ver en el histograma de la versión ecualizada, las intensidades se distribuyen de forma más equitativa, aumentando las frecuencias de los valores más altos que eran menos frecuentes en la imagen original.

### Ejercicio 9

> Ir y ejecutar subsección "Ejercicio 9"

Para realizar la umbralización manual, se hizo uso de `cv2.threshold(img1, threshold_value, 255, cv2.THRESH_BINARY)`. Para el método de Otsu, se hizo uso de `cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`.

```python
threshold_value = 240  # Valor de umbral manual

_, img_umbralizada_manual = cv2.threshold(img1, threshold_value, 255, cv2.THRESH_BINARY)

_, img_umbralizada_otsu = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

Fragmento extraido de [https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html]:

> Con la umbralización global, usamos un valor elegido arbitrariamente como umbral. En cambio, el método de Otsu evita tener que elegir un valor y lo determina automáticamente.

> Considera una imagen con solo dos valores de intensidad distintos (una imagen bimodal), donde el histograma consistiría únicamente en dos picos. Un buen umbral estaría en el medio de esos dos valores. De manera similar, el método de Otsu determina un valor de umbral global óptimo a partir del histograma de la imagen.

> Para hacerlo, se utiliza la función cv.threshold(), donde se pasa cv.THRESH_OTSU como una bandera adicional. El valor del umbral puede elegirse arbitrariamente. Luego, el algoritmo encuentra el valor de umbral óptimo, el cual se devuelve como la primera salida.

### Ejercicio 11

> Ir y ejecutar subsección "Ejercicio 11"

En este ejercicio, aplicamos corrección gamma a una imagen en escala de grises, dividiéndola en cuatro regiones y aplicando un valor de gamma diferente a cada una para modificar su brillo y contraste de manera no lineal. La función `gamma_correction()` normaliza los valores de píxeles al rango [0, 1], aplica la transformación 
$I'=I^γ$, y luego los reescala a [0, 255].

```python
def gamma_correction(image, gamma):
    image_normalized = image / 255.0
    image_corrected = np.power(image_normalized, gamma)
    return (image_corrected * 255).astype(np.uint8)
```

## 2 Combinación de Imágenes

### Ejercicio 3
> Ir y ejecutar subsección "Ejercicio 3"

En este ejercicio, aplicamos `cv2.multiply()` y `cv2.divide()` sobre dos imágenes para luego visualizar los resultados 

```python
img3 = cv2.imread('./imagenes/paisaje3.jpg')
img4 = cv2.imread('./imagenes/paisaje4.jpg')

img_multiplicada = cv2.multiply(img3, img4)
img_dividida_1 = cv2.divide(img3, img4)
img_dividida_2 = cv2.divide(img4, img3)
```

### Ejercicio 5
> Ir y ejecutar subsección "Ejercicio 5"

En este ejercicio, aplicamos `cv2.bitwise_and()`, `cv2.bitwise_or()` y `cv2.bitwise_xor()` sobre dos imágenes con `cv2.circle` como máscara para fusionarlas. 

```python
mask = np.zeros(img3.shape[:2], dtype=np.uint8)
cv2.circle(mask, (img3.shape[1] // 2, img3.shape[0] // 2), 100, 255, -1)

img_and = cv2.bitwise_and(img3, img4, mask=mask)
img_or = cv2.bitwise_or(img3, img4, mask=mask)
img_xor = cv2.bitwise_xor(img3, img4, mask=mask)
```

En los tres casos, cuando se provee la máscara circular, las operaciones AND, OR y XOR se realizan entre las dos imágenes sólo donde los valores de la máscara no son iguales a cero, es decir, en el interior del círculo.

Las operaciones bitwise AND, OR y XOR en imágenes BGR (Blue, Green, Red) funcionan canal por canal y pixel por pixel, comparando directamente los valores binarios de cada componente del color. Como los valores están en el rango [0, 255], cada componente tiene una representación binaria de 8 bits.

Viendo nuestras imágenes, podemos ver que aplicar el operador AND resulta en una imagen oscura porque el operador sólo mantiene un bit en 1 si los bits también son 1, y cero en caso contrario. Por ejemplo:

$$
pixel_{img3}(x,y) = 11001010  \\
pixel_{img4}(x,y) = 10101100  \\
resultado  = 10001000
$$

Aplicar el operador OR, por otro lado, resulta en una imagen brillante, ya que el operador mantiene un bit en 1 si al menos uno de los bits es 1. Por ejemplo:

$$
pixel_{img3}(x,y)  = 11001010  \\
pixel_{img4}(x,y)  = 10101100  \\
resultado = 11101110
$$

Aplicar el operador XOR, finalmente, resalta las diferencias entre las dos imágenes, ya que el operador mantiene un bit en 1 si los bits son diferentes. Por ejemplo:

$$
pixel_{img3}(x,y)  = 11001010  \\
pixel_{img4}(x,y)  = 10101100  \\
resultado = 01100110
$$

### Ejercicio 8
> Ir y ejecutar subsección "Ejercicio 8"

En este ejercicio:
- Se obtiene la región de interés de `paisaje3.jpg`. 
- Se crea la máscara de la mariposa por medio de una umbralización y se invierte.
- Se superpone la máscara invertida sobre la región de interés por medio de `bitwise_and` y la máscara invertida.
- Se obtiene el foreground de la mariposa con `bitwise_and` y la máscara original.
- Se suma el fondo de la región de interés enmascarada con el foreground de la mariposa.
- Se reemplaza la suma anterior en la imagen original.


```python
# Paso 1: Región de interés
roi = img2[y_start:y_end, x_start:x_end]

# Paso 2: Crear máscara ignorando fondo blanco
gray_mariposa = cv2.cvtColor(mariposa, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_mariposa, 240, 255, cv2.THRESH_BINARY_INV)

# Paso 3: Invertir máscara
mask_inv = cv2.bitwise_not(mask)

# Paso 4: Fondo de la ROI enmascarado
img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Paso 5: Parte de la mariposa sin fondo blanco
mariposa_fg = cv2.bitwise_and(mariposa, mariposa, mask=mask)

# Paso 6: Suma de ambas partes
dst = cv2.add(img2_bg, mariposa_fg)

# Paso 7: Reemplazo en img2
img2_final = img2.copy()
img2_final[y_start:y_end, x_start:x_end] = dst
```

## 3 Dominio Espacial

### Ejercicio 8
> Ir y ejecutar subsección "Ejercicio 8"

En este ejercicio, aplicamos `cv2.GaussianBlur()` y `cv2.Sobel()` sobre una imagen para luego visualizar los resultados y analizar las diferencias en la detección de bordes. 

```python
# Aplicar filtro gaussiano
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# Operador Sobel sin suavizado
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
sobel_img = cv2.magnitude(sobelx, sobely)

# Operador Sobel con suavizado
sobelx_blurred = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=5)
sobely_blurred = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=5)
sobel_blurred_img = cv2.magnitude(sobelx_blurred, sobely_blurred)
```

### Ejercicio 12
> Ir y ejecutar subsección "Ejercicio 12"

En este ejercicio, se aplican diferentes métodos de detección de bordes a imágenes en escala de grises para compararlos visualmente.

- Sobel: Calcula la derivada de primer orden en las direcciones horizontal (x) y vertical (y) utilizando el operador Sobel. Luego, combina ambas con `cv2.magnitude()` para obtener la magnitud del gradiente total, resaltando los bordes donde hay cambios bruscos de intensidad.

- Prewitt: Similar al Sobel, pero con un kernel más simple. Se aplica mediante `cv2.filter2D()` con kernels definidos manualmente para x e y. También se calcula la magnitud del gradiente para detectar bordes.

- Laplace (Laplacian): Calcula la segunda derivada de la imagen. Detecta bordes donde hay un cambio rápido en la tasa de cambio de intensidad. Es más sensible al ruido, pero efectivo para encontrar todos los bordes (sin distinguir dirección).

- Canny:
Un método más sofisticado que aplica suavizado (ya aplicado con GaussianBlur), cálculo del gradiente, supresión de no máximos, y umbralización con histéresis.

```python
# Aplicar suavizado Gaussiano
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# Sobel
sobelx = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=5)
sobel_img = cv2.magnitude(sobelx, sobely)

# Prewitt
kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
prewittx = cv2.filter2D(blurred_img, -1, kernelx).astype(np.float32)
prewitty = cv2.filter2D(blurred_img, -1, kernely).astype(np.float32)
prewitt_img = cv2.magnitude(prewittx, prewitty)

# Laplace
laplacian_img = cv2.Laplacian(blurred_img, cv2.CV_64F)

# Canny
canny_img = cv2.Canny(blurred_img, 50, 150)
```

### Ejercicio 13
> Ir y ejecutar subsección "Ejercicio 13"

En este ejercicio, aplicamos `cv2.GaussianBlur()` y dos operaciones para obtener el filtro de paso alto y la imagen con mayor nivel de detalle.

```python
blurred = cv2.GaussianBlur(img, (5, 5), 0)
high_pass = img - blurred
sharpened = img + high_pass
```

### Ejercicio 15
> Ir y ejecutar subsección "Ejercicio 15"

En este ejercicio, aplicamos `cv2.GaussianBlur()`, con dos valores de sigma distintos, y una resta para obtener la diferencia de gaussianos.

```python
blurred1 = cv2.GaussianBlur(img, (ksize1, ksize1), sigma1)
blurred2 = cv2.GaussianBlur(img, (ksize2, ksize2), sigma2)
dog_image = blurred1 - blurred2
```