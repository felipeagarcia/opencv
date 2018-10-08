import numpy as np
from numpy import cos, sin
import cv2
from Projecoes import Projecoes

if __name__ == '__main__':
    proj = Projecoes()

    # parametros da camera
    f = 16
    sx = sy = 0.01
    ox = 320
    oy = 240
    # definindo as posicoes da camera
    posicoes = [(0, 0, 500), (0, 0, 1000),
                (0, 0, 1000), (0, 0, 1500),
                (0, 0, 2000)]
    angulos = [(180, 0, 0), (180, 0, 0),
               (180, 0, 0), (180, 0, 0),
               (180, 0, 0)]
    count = 0
    for pos_camera, angulos in zip(posicoes, angulos):
        # posicao da camera
        dx, dy, dz = pos_camera
        # rotacao da camera
        rotx, roty, rotz = angulos
        # definindo o plano
        n = m = 11
        d = 10
        Pp = proj.pontos_plano(n, m, d)
        if count > 1:
            Rx = [[1, 0, 0, 0],
                  [0, cos(np.pi/4), -sin(np.pi/4), 0],
                  [0, sin(np.pi/4), cos(np.pi/4), 0],
                  [0, 0, 0, 1]]
            Pp = np.append(Pp, [np.ones(len(Pp[0]))], axis=0)
            Pp = np.matmul(Rx, Pp)
            Pp = np.delete(Pp, -1, 0)
        # convertendo os pontos do plano para a imagem
        H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
        Pc = proj.mundo_para_camera(Pp, H)
        P = proj.proj_perspectiva_mm(Pc, f)
        p_im = proj.proj_perspectiva_pixel(P, sx, sy, ox, oy)
        p_im = np.array(list(map(abs, p_im)))
        # criando uma imagem preta
        img = np.zeros((512, 512, 3), np.uint8)
        img.fill(255)
        # desenhando o retangulo
        for i in range(len(p_im[0])):
            img = cv2.circle(img, tuple(map(int, p_im[:, i])), 4, (0, 0, 0), 7)
        cv2.imwrite('rectangle' + str(count) + '.jpg', img)
        count += 1
