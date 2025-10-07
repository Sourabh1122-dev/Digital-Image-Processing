def main_1(img):

  import numpy as np
  import cv2
  import skfuzzy as fuzz


  def increase_constrast(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a, b))

    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img


  def sharpening(image):

    gauss_mask = cv2.GaussianBlur(image, (11, 11), 2)
    sharp = cv2.addWeighted(image, 1.5, gauss_mask, -0.5, 10)
    return sharp


  image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  image_copy = image.copy()
  image = image.astype(np.uint8)
  image = cv2.medianBlur(image, 11)
  image = cv2.edgePreservingFilter(image, sigma_s=5)
  image = increase_constrast(image)
  image = sharpening(image)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  image = (image - np.min(image)) / (np.max(image) - np.min(image))

  image_width = image.shape[0]
  image_height = image.shape[1]
  channels = image.shape[2]

  image = np.reshape(image, (image_width * image_height, channels))

  num_clusters = 3 
  m = 1.5  
  error = 0.0001  
  max_iter = 300

  cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(image.T, num_clusters, m, error, max_iter, seed=0)

  cluster_labels = np.argmax(u, axis=0)

  class_0 = [0 if c else 1 for c in (cluster_labels == 0)]
  class_1 = [0 if c else 1 for c in (cluster_labels == 1)]
  class_2 = [0 if c else 1 for c in (cluster_labels == 2)]

  class_0 = np.reshape(class_0, (image_width, image_height))
  class_1 = np.reshape(class_1, (image_width, image_height))
  class_2 = np.reshape(class_2, (image_width, image_height))

  class_0 = (class_0*255).astype(np.uint8)
  class_1 = (class_1*255).astype(np.uint8)
  class_2 = (class_2*255).astype(np.uint8)

  classes = [class_0, class_1, class_2]

  class_arrays = []
  for i in range(num_clusters):
    colored_image = cv2.cvtColor(classes[i], cv2.COLOR_GRAY2BGR)
    mask = np.all(image_copy == (255, 255, 255), axis=2)
    colored_image[mask] = (255, 255, 255)
    binary_image_i = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
    class_arrays.append(binary_image_i)
  

  return class_arrays, cluster_labels   



def main_2(image, cluster_labels, color_array):

    import numpy as np
    import cv2

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
  
    image_width = image.shape[0]
    image_height = image.shape[1]
    channels = image.shape[2]
    
    combined_labels = np.zeros((image_width * image_height, channels))

    
   
    for i, label in enumerate(cluster_labels): 
       match label:  
         case 0:
            combined_labels[i] = color_array[0]
         case 1:
            combined_labels[i] = color_array[1]
         case 2:
            combined_labels[i] = color_array[2]
    
    combined_labels = np.reshape(combined_labels, (image_width, image_height, channels))

    mask = np.all(image == (255, 255, 255), axis=2)
    combined_labels[mask] = (255, 255, 255)

    return combined_labels



def main_3(binary_image):

  import cv2

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

  # Invert the closed image to obtain the removed outer boundary
  removed_boundary_image = cv2.bitwise_not(closed_image)
  inverted_image = cv2.bitwise_not(removed_boundary_image)

  return inverted_image

