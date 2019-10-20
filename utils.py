import numpy as np

def bilinear_interpolation(x,y, img):
    height, width = img.shape

    # transformando em um array numpy para utilizar os metodos
    x = np.asarray(x)
    y = np.asarray(y)

    # pegando os vertices dos 4 pontos (x0,y0) e (x1,y1)
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # truncando os valores entre 0 e altura ou largura
    x0, x1 = np.clip(x0, 0, height-1), np.clip(x1, 0, height-1)
    y0, y1 = np.clip(y0, 0, width-1), np.clip(y1, 0, width-1)

    # pegamos os 4 valores dos 4 pontos próximos do desejado
    pa, pb, pc, pd = img[ x0, y0 ], img[ x1, y0 ], img[ x0, y1 ], img[ x1, y1 ]

    # Calculamos o produto entre as distâncias das coordenadas
    wa, wb, wc, wd = (x1-x) * (y1-y), (x1-x) * (y-y0), (x-x0) * (y1-y), (x-x0) * (y-y0)

    return wa*pa + wb*pb + wc*pc + wd*pd

# reference:
# https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
