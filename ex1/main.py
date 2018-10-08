import cv2
import numpy as np

# defining the colors range and the value to be painted
colors = [
          (np.array([50, 50, 50]), np.array([60, 255, 255]), (60, 255, 255)),  # green
          (np.array([100, 50, 50]), np.array([140, 255, 255]), (120, 255, 255)),  # blue
          (np.array([27, 190, 100]), np.array([29, 255, 255]), (29, 255, 255)),   # yellow
          (np.array([7, 200, 200]), np.array([12, 255, 255]), (12, 255, 255)),  # orange
          (np.array([2, 150, 150]), np.array([4, 255, 255]), (145, 255, 255)),   # pink
          (np.array([5, 170, 130]), np.array([7, 255, 255]), (0, 225, 143))  # brown
         ]

if __name__ == '__main__':
    img_name = 'cores.jpeg'
    # loading the image
    img = cv2.imread(img_name)
    # converting the color space from bgr to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    temp = []
    i = 0
    for (lower, upper, color) in colors:
        aux = img
        # creating and painting a temp image
        aux = cv2.cvtColor(aux, cv2.COLOR_BGR2HSV)
        cv2.rectangle(aux, (0, 0), (479, 639), color, 1000)
        # Threshold the HSV image
        mask = cv2.inRange(hsv, lower, upper)
        # painting the area selected by the mask
        res = cv2.bitwise_and(aux, aux, mask=mask)
        temp.append(res)
        # writing the partial results to a file
        cv2.imwrite('temp' + str(i) + '.jpg', cv2.cvtColor(res,
                                                           cv2.COLOR_HSV2BGR))
        i += 1
    out = temp[0]
    # joining all the partial results
    for image in temp:
        out = cv2.bitwise_or(out, image)
    kernel = np.ones((3, 3), np.uint8)
    # improving the final image with topological operations
    out = cv2.dilate(out, kernel, iterations=1)
    out = cv2.erode(out, kernel, iterations=3)
    out = cv2.dilate(out, kernel, iterations=6)
    img = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    # writing the final result to a file
    cv2.imwrite('cores2.jpg', img)
