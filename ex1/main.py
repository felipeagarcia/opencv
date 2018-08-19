import cv2
import numpy as np

colors = [
          (np.array([50, 50, 50]), np.array([60, 255, 255]), (60, 255, 255)),  # green
          (np.array([100, 50, 50]), np.array([140, 255, 255]), (120, 255, 255)),  # blue
          (np.array([27, 50, 50]), np.array([29, 255, 255]), (29, 255, 255)),   # yellow
          (np.array([7, 50, 50]), np.array([12, 255, 255]), (12, 255, 255)),  # orange
          (np.array([2, 0, 0]), np.array([4, 255, 255]), (145, 255, 255)),   # pink
          (np.array([4, 0, 0]), np.array([7, 255, 255]), (0, 225, 143))  
         ]

if __name__ == '__main__':
    img_name = 'cores.jpeg'
    img = cv2.imread(img_name)
    print(img[300,480])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(hsv[462,160])
    temp = []
    i = 0
    for (lower, upper, color) in colors:
        aux = img
        aux = cv2.cvtColor(aux, cv2.COLOR_BGR2HSV)
        cv2.rectangle(aux, (0, 0), (479, 639), color, 1000)
        # Threshold the HSV image
        mask = cv2.inRange(hsv, lower, upper)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(aux, aux, mask=mask)
        temp.append(res)
        cv2.imwrite('temp' + str(i) + '.jpg', cv2.cvtColor(res, cv2.COLOR_HSV2BGR))
        i += 1
    out = temp[0]
    for image in temp:
        out = cv2.bitwise_or(out, image)
    img = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    cv2.imwrite('cores2.jpg', img)
