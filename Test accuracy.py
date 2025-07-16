def main_4(Original_ImajeJ_img, segmented_img):
  import numpy as np
  import cv2
  
  image_1 = cv2.cvtColor(Original_ImajeJ_img, cv2.COLOR_BGR2GRAY)

  image_1 = cv2.bitwise_not(image_1)
  image_2 = cv2.bitwise_not(segmented_img)

  image_array_1 = np.array(image_1)
  image_array_2 = np.array(image_2)

  intersection = np.logical_and(image_array_1, image_array_2)
  union = np.logical_or(image_array_1, image_array_2)
  iou = np.sum(intersection) / np.sum(union)
  return iou*100