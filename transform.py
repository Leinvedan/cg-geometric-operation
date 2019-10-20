import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from collections import namedtuple
from utils import bilinear_interpolation


# Dimensões da imagem =  1456 x 1448 pixels
source_img = cv2.imread("trooper.jpg", 0)
s_height, s_width = source_img.shape

# Criando o suporte que ira armazenar a imagem final
# Tamanho folha A4 = 2480 x 3508 pixels
r_height, r_width = 3508, 2480
result_img = np.zeros((r_height, r_width), np.uint8)


### ------ DESCOBRINDO OS PARAMETROS DA TRANSFORMACAO

# Sabendo que os vértices do papel são
#
# s2 -- s3
# |     |
# s1 -- s4

Pixel = namedtuple('Pixel', 'x y')

s1 = Pixel(1071, 30) #    --> (3508, 0)
s2 = Pixel(117, 531) #    --> (0, 0)
s3 = Pixel(225, 1410) #   --> (0, 2480)
s4 = Pixel(1302, 1239) # --> (3508,2480)

source_pixels = [s1,s2,s3,s4]

# Queremos transforma-los em:

r1 = Pixel(3507,0)
r2 = Pixel(0,0)
r3 = Pixel(0,2479)
r4 = Pixel(3507, 2479)

result_pixels = [r1,r2,r3,r4]


# Construindo A
def generate_A_and_L_matrix(source_pixels, result_pixels):
  A = []
  L = []
  for pixel in list(zip(source_pixels, result_pixels)):
    L.append([pixel[1].x])
    L.append([pixel[1].y])

    u = [pixel[0].x, pixel[0].y, 1, 0, 0, 0, (-pixel[0].x * pixel[1].x), (-pixel[0].y * pixel[1].x)]
    v = [0, 0, 0, pixel[0].x, pixel[0].y, 1, (-pixel[0].x * pixel[1].y), (-pixel[0].y * pixel[1].y)]
    A.append(u)
    A.append(v)

  A = np.array(A)
  L = np.array(L)
  return A, L

A, L = generate_A_and_L_matrix(source_pixels, result_pixels) 

def calculate_transformation_matrix(A):
  # Agora que temos A e L, queremos:
  # x^ = (A^t.A)^−1 * A^t * L
  At = np.transpose(A) # A^t
  AtA = np.matmul(At, A) # A^t * A
  AtA_inv = np.linalg.inv(AtA)# (A^t * A)^−1

  Xtemp = np.matmul(AtA_inv, At) # (A^t.A)^−1 * A^t
  Xfinal = np.matmul(Xtemp, L) # (A^t * A)^−1 * A^t * L
  Xfinal = np.append(Xfinal, [1])# Adicionando 1 ao final para poder
                                # converter numa matriz 3x3 de transformacao

  # Convertendo para matriz de transformacao 3x3
  transformation_matrix = np.zeros((3, 3), np.float)
  t_width = len(transformation_matrix)
  t_height = len(transformation_matrix[0])
  counter = 0

  for x in range(t_height):
    for y in range(t_width):
      transformation_matrix[x][y] = Xfinal[counter]
      counter += 1
  return transformation_matrix

transformation_matrix = calculate_transformation_matrix(A)

### ------- APLICANDO A TRANSFORMACAO 

# invertendo a matriz para o cálculo do pixel na matriz resultante
inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

# Para cada pixel na imagem de saída

index_matrix = np.indices((r_height,r_width))

# Aplicar a matriz de transformacao nas coordenadas normalizadas
point = np.matmul(
  inverse_transformation_matrix,
  np.array( [[index_matrix[0]],
            [index_matrix[1]],
            [1]
          ]))
point = point/point[2] # normalizar o resultado

x_array = point[0][0].astype(int)
y_array = point[1][0].astype(int)

# Aplicar a interpolação bilinear
result_img = bilinear_interpolation(x_array,y_array, source_img)
#result_img = source_img[x_array, y_array]

cv2.imwrite('result.jpg', result_img)

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(source_img, cmap="gray")
f.add_subplot(1, 2, 2)
plt.imshow(result_img, cmap="gray")
plt.show(block=True)
