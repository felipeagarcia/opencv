import numpy as np
import cv2
from Projecoes import Projecoes


def draw_full_line(point1, point2, img):
    slope = (point2[1] - point1[1])/(point2[0] - point1[0])
    b = point2[1] - (slope * point2[0])
    p = (0, b)
    q = (len(img), int(slope*len(img) + b))
    img = cv2.line(img, p, q, (0, 0, 255), 5)
    return img


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
    H1 = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
    Pc1 = proj.mundo_para_camera(Pp, H1)
    P = proj.proj_perspectiva_mm(Pc1, f)
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
    H2 = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
    Pc2 = proj.mundo_para_camera(Pp, H2)
    P = proj.proj_perspectiva_mm(Pc2, f)
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
    pos_camera1 = np.array(pos_camera1)
    pos_camera2 = np.array(pos_camera2)
    angulos1 = np.array(angulos1)
    angulos2 = np.array(angulos2)
    rotx, roty, rotz = angulos1-angulos2
    R = proj.homogenea(rotx, roty, rotz, 0, 0, 0)
    E = proj.get_essential_matrix(R, pos_camera1, pos_camera2)
    print('E', E)
    F = proj.get_fundamental_matrix(Pc1, Pc2, p_im1, p_im2,
                                    pos_camera1, pos_camera2, Pp, R)
    print('F', F)
    point1, point2 = proj.get_commum_point(img1, img2)
    point1 = np.reshape(point1, (2, 1))
    point2 = np.reshape(point2, (2, 1))
    point1 = proj.pixel_to_mm(point1, sx, sy, ox, oy, f)
    point2 = proj.pixel_to_mm(point2, sx, sy, ox, oy, f)
    d = point1[0] - point2[0]
    p1, p2 = proj.get_commum_points(img1, img2, threshold=0.65)
    p1 = proj.pixel_to_mm(p1, sx, sy, ox, oy, f)
    p2 = proj.pixel_to_mm(p2, sx, sy, ox, oy, f)
    F = proj.get_fundamental_matrix_svd(p1, p2, d)
    print('F_svd', F)
    epl, epr = proj.get_epipoles(F)

    epl = np.reshape(epl, (3, 1))
    epl = proj.proj_perspectiva_pixel(epl, sx, sy, ox, oy)
    img1 = np.zeros((512, 512, 3), np.uint8)
    # img.fill(255)
    color = (255, 255, 255)
    # desenhando o objeto
    for i in range(len(p_im1[0])):
        img1 = cv2.circle(img1, tuple(map(int, p_im1[:, i])), 3, color, 4)
    for i in range(3):
        img1 = draw_full_line(epl, p_im1[:, 50*(i+1)], img1)
    cv2.imwrite('rectangle' + str(3) + '.jpg', img1)
    epr = np.reshape(epr, (3, 1))
    epr = proj.proj_perspectiva_pixel(epr, sx, sy, ox, oy)
    img2 = np.zeros((512, 512, 3), np.uint8)
    # img.fill(255)
    color = (255, 255, 255)
    # desenhando o objeto
    print("left epipole:", epl)
    print("right epipole:", epr)
    for i in range(len(p_im2[0])):
        img2 = cv2.circle(img2, tuple(map(int, p_im2[:, i])), 3, color, 4)
    for i in range(3):
        img2 = draw_full_line(epr, p_im2[:, 50*(i+1)], img2)
    cv2.imwrite('rectangle' + str(4) + '.jpg', img2)
    #################################
    R, T = proj.triangulacao(H1, H2)
    #Pw = proj.camera_para_mundo(Pc1, H)
    # projetando na camera esquerda para visualizar
    a, b, c = proj.get_abc(p1, p2, R, T)
    w = proj.get_w(p1, p2, R)
    H = np.identity(4)
    H[:3, :3] = R
    Et = np.transpose(E)
    N = (np.linalg.norm(np.matmul(Et, E)))
    T /= N
    H[:3, 3] = T[:, 0]
    Pp = proj.camera_para_mundo(Pp, H)
    Pc = proj.mundo_para_camera(Pp, H1)
    P = proj.proj_perspectiva_mm(Pc, f)
    p_im = proj.proj_perspectiva_pixel(P, sx, sy, ox, oy)
    p_im = np.array(list(map(abs, p_im)))
    img = np.zeros((512, 512, 3), np.uint8)
    for i in range(len(p_im[0])):
        img = cv2.circle(img, tuple(map(int, p_im[:, i])), 3, color, 4)
    cv2.imwrite('rectangle' + str(5) + '.jpg', img)
