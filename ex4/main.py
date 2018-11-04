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
    pos_camera1 = (100, 200, 550)
    angulos1 = (180, 12, 60)
    pos_camera2 = (200, 250, 550)
    angulos2 = (30, 10, 60)
    count = 0
    # posicao da camera
    dx, dy, dz = pos_camera1
    # rotacao da camera
    rotx, roty, rotz = angulos1
    # definindo os planos
    n = m = 11
    d = 10
    Pp = proj.pontos_calibracao(n, m, d)
    # convertendo os pontos do plano para a imagem
    H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
    Pc = proj.mundo_para_camera(Pp, H)
    P = proj.proj_perspectiva_mm(Pc, f)
    p_im1 = proj.proj_perspectiva_pixel(P, sx, sy, ox, oy)
    p_im1 = np.array(list(map(abs, p_im1)))
    # criando uma imagem preta
    img1 = np.zeros((512, 512, 3), np.uint8)
    # img.fill(255)
    color = (255, 255, 255)
    # desenhando o objeto
    print(count)
    for i in range(len(p_im1[0])):
        img1 = cv2.circle(img1, tuple(map(int, p_im1[:, i])), 3, color, 4)
    cv2.imwrite('rectangle' + str(count) + '.jpg', img1)
    count += 1
    # posicao da camera
    dx, dy, dz = pos_camera2
    # rotacao da camera
    rotx, roty, rotz = angulos2
    # definindo os planos
    n = m = 11
    d = 10
    Pp = proj.pontos_calibracao(n, m, d)
    # convertendo os pontos do plano para a imagem
    H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
    Pc = proj.mundo_para_camera(Pp, H)
    P = proj.proj_perspectiva_mm(Pc, f)
    p_im2 = proj.proj_perspectiva_pixel(P, sx, sy, ox, oy)
    p_im2 = np.array(list(map(abs, p_im2)))
    # criando uma imagem preta
    img2 = np.zeros((512, 512, 3), np.uint8)
    # img.fill(255)
    color = (255, 255, 255)
    # desenhando o objeto
    print(count)
    for i in range(len(p_im2[0])):
        img2 = cv2.circle(img2, tuple(map(int, p_im2[:, i])), 3, color, 4)
    cv2.imwrite('rectangle' + str(count) + '.jpg', img2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    pos_camera1 = np.array(pos_camera1)
    pos_camera2 = np.array(pos_camera2)
    E = proj.get_essential_matrix(img1, img2, pos_camera1, pos_camera2)
    print('E', E)
    F = proj.get_fundamental_matrix(Pp, Pp, img1, img2,
                                    pos_camera1, pos_camera2)
    print('F', F)
