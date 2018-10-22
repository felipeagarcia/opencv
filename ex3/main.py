import numpy as np
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
    pos_camera = (100, 200, 550)
    angulos = (180, 12, 60)
    count = 0
    # posicao da camera
    dx, dy, dz = pos_camera
    # rotacao da camera
    rotx, roty, rotz = angulos
    # definindo o plano
    n = m = 11
    d = 10
    Pp = proj.pontos_calibracao(n, m, d)
    # convertendo os pontos do plano para a imagem
    H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
    Pc = proj.mundo_para_camera(Pp, H)
    P = proj.proj_perspectiva_mm(Pc, f)
    p_im = proj.proj_perspectiva_pixel(P, sx, sy, ox, oy)
    p_im = np.array(list(map(abs, p_im)))
    # criando uma imagem preta
    img = np.zeros((512, 512, 3), np.uint8)
    # img.fill(255)
    color = (255, 255, 255)
    # desenhando o retangulo
    print(count)
    for i in range(len(p_im[0])):
        img = cv2.circle(img, tuple(map(int, p_im[:, i])), 3, color, 4)
    cv2.imwrite('rectangle' + str(count) + '.jpg', img)
    count += 1
    print(f, sx, sy, ox, oy)
    f, sx, sy, ox, oy, R, T = proj.calibration(Pp, p_im)
    print(f, sx, sy, ox, oy)
    print(H)
    T = np.expand_dims(T, axis=1)
    H = np.append(R, T, axis=1)
    H = np.append(H, [[0, 0, 0, 1]], axis=0)
    print(H)
    Pp = proj.pontos_calibracao(n, m, d)
    Pc = proj.mundo_para_camera(Pp, H)
    P = proj.proj_perspectiva_mm(Pc, f)
    p_im = proj.proj_perspectiva_pixel(P, sx, sy, ox, oy)
    p_im = np.array(list(map(abs, p_im)))
    # criando uma imagem preta
    img = np.zeros((512, 512, 3), np.uint8)
    # img.fill(255)
    color = (255, 255, 255)
    # desenhando o retangulo
    print(count)
    for i in range(len(p_im[0])):
        img = cv2.circle(img, tuple(map(int, p_im[:, i])), 3, color, 4)
    cv2.imwrite('rectangle' + str(count) + '.jpg', img)
    count += 1
