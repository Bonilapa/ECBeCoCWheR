import cv2
import matplotlib.pyplot as plt

image = cv2.imread("/home/wil/BCR/docker-containers/project/ECBeCoCWheR/src/envs/ents/icons/robot.png")
m = cv2.getRotationMatrix2D((45, 45), 45, 1)
image = cv2.copyMakeBorder(image, 13, 13, 13, 13, cv2.BORDER_CONSTANT, value=(255,255,255))
image = cv2.warpAffine(image, m, (90, 90), borderValue=(255,255,255))
print(image)
plt.imshow(image, cmap='gray')
plt.show()