import numpy as np  # For handling arrays and numeric operations
from PIL import Image  # To convert NumPy arrays to PIL images
import cv2
import os
import torch  # For PyTorch functionalities like tensors and inference
import torchvision.transforms as transforms  # For image preprocessing (ToTensor, Normalize)
import time  # To measure processing time for preprocessing and inference
from scipy.ndimage.filters import gaussian_filter

from WebRTC_Client_And_Signaling_Server.cuboid import Cuboid3d
from WebRTC_Client_And_Signaling_Server.pnp_solver import CuboidPNPSolver

def detect_objects(model, data):
	
    # Extract relevant data
    height = data.height
    width = data.width
    encoding = data.encoding
    step = data.step
    img_arr = data.data
    
    	
    image_data = np.array(img_arr, dtype=np.uint8)

    expected_size = height * step
    
    
    if image_data.size != expected_size:
        print(f"Error: Size of image_data ({image_data.size}) does not match expected size ({expected_size})")
    else:
    # Reshape the image data array according to height and step (bytes per row)
        try:
            image_data = image_data.reshape((height, step // 4, 4))  # Dividing by 4 since RGBA has 4 channels
        except ValueError as e:
            print(f"Error: {e}")

        # If the image is stored in big endian format, swap bytes to little endian
        if data.is_bigendian:
            image_data = image_data.byteswap().newbyteorder()

        # Convert RGBA to BGRA (OpenCV uses BGR or BGRA format)
        image_data_rgba = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB)
        rotated_clockwise = cv2.rotate(image_data_rgba, cv2.ROTATE_90_CLOCKWISE)
        
        cv2.imwrite("output_640_480_rgb_image_test.png", rotated_clockwise)
        
    image_path = os.path.join(os.getcwd(), "output_640_480_rgb_image_test.png")
    pil_image = Image.open(image_path)

        
    # Preprocess the image using the same preprocessing function
    preprocess = transforms.Compose([
       transforms.ToTensor(),				
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    preprocess_start_time = time.time()
    image_tensor = preprocess(pil_image).unsqueeze(0)  # Add batch dimension
    preprocess_end_time = time.time()
    preprocessing_time = preprocess_end_time - preprocess_start_time
    print("Preprocessing time:", preprocessing_time, "seconds")

    # Perform inference
    inference_start_time = time.time()
    with torch.no_grad():
        outputs_belief, outputs_affinity = model(image_tensor)
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time
    print("Inference time:", inference_time, "seconds")
    print("Finish running the model")

    objects, all_peaks = process_image(outputs_belief[-1], outputs_affinity[-1])

    print(objects)
    print(all_peaks)

    # Instantiate the CuboidPNPSolver class
    pnp_solver = CuboidPNPSolver(
        model,
        cuboid3d=Cuboid3d([9.6024150848388672, 19.130100250244141, 5.824894905090332])
    )

    pnp_solver_start_time = time.time()
    detected_objects = []
    for obj in objects:
        # Run PNP
        points = obj[1] + [(obj[0][0]*8, obj[0][1]*8)]
        print(points)
        # Filter out None values from points
        points_filtered = [point for point in points if point is not None]
        cuboid2d = np.copy(points_filtered)
        location, quaternion, projected_points = pnp_solver.solve_pnp(points)

        if not location is None:
            detected_objects.append({
                'name': pnp_solver.object_name,
                'location': location,
                'quaternion': quaternion,
                'cuboid2d': cuboid2d,
                'projected_points': projected_points,
                'confidence': obj[-1],
                'raw_points': points
            })

    pnp_solver_end_time = time.time()
    pnp_solve_time = pnp_solver_end_time - pnp_solver_start_time
    print("Pnp Solve Time:", pnp_solve_time, "seconds")
    print(detected_objects)

    return detected_objects
    
def process_image(tensor_belief, tensor_affinity):
  all_peaks = []
  peak_counter = 0

  vertex2 = tensor_belief[0]
  aff = tensor_affinity[0]

  object_detector_start_time = time.time()

  for j in range(vertex2.size()[0]):
      belief = vertex2[j].clone()
      map_ori = belief.cpu().data.numpy()

      map = gaussian_filter(belief.cpu().data.numpy(), sigma=3)
      p = 1
      map_left = np.zeros(map.shape)
      map_left[p:,:] = map[:-p,:]
      map_right = np.zeros(map.shape)
      map_right[:-p,:] = map[p:,:]
      map_up = np.zeros(map.shape)
      map_up[:,p:] = map[:,:-p]
      map_down = np.zeros(map.shape)
      map_down[:,:-p] = map[:,p:]

      peaks_binary = np.logical_and.reduce(
                          (
                              map >= map_left,
                              map >= map_right,
                              map >= map_up,
                              map >= map_down,
                              map > 0.01)
                          )
      peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])

      # Computing the weigthed average for localizing the peaks
      peaks = list(peaks)
      win = 5
      ran = win // 2
      peaks_avg = []
      for p_value in range(len(peaks)):
          p = peaks[p_value]
          weights = np.zeros((win,win))
          i_values = np.zeros((win,win))
          j_values = np.zeros((win,win))
          for i in range(-ran,ran+1):
              for j in range(-ran,ran+1):
                  if p[1]+i < 0 \
                          or p[1]+i >= map_ori.shape[0] \
                          or p[0]+j < 0 \
                          or p[0]+j >= map_ori.shape[1]:
                      continue

                  i_values[j+ran, i+ran] = p[1] + i
                  j_values[j+ran, i+ran] = p[0] + j
                  weights[j+ran, i+ran] = (map_ori[p[1]+i, p[0]+j])

          # if the weights are all zeros
          # then add the none continuous points
          OFFSET_DUE_TO_UPSAMPLING = 0.4395
          try:
              peaks_avg.append(
                  (np.average(j_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING, \
                    np.average(i_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING))
          except:
              peaks_avg.append((p[0] + OFFSET_DUE_TO_UPSAMPLING, p[1] + OFFSET_DUE_TO_UPSAMPLING))
      # Note: Python3 doesn't support len for zip object
      peaks_len = min(len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0]))

      peaks_with_score = [peaks_avg[x_] + (map_ori[peaks[x_][1],peaks[x_][0]],) for x_ in range(len(peaks))]

      id = range(peak_counter, peak_counter + peaks_len)

      peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

      all_peaks.append(peaks_with_score_and_id)
      peak_counter += peaks_len

  objects = []

  # Check object centroid and build the objects if the centroid is found
  for nb_object in range(len(all_peaks[-1])):
      if all_peaks[-1][nb_object][2] > 0.1:
          objects.append([
              [all_peaks[-1][nb_object][:2][0],all_peaks[-1][nb_object][:2][1]],
              [None for i in range(8)],
              [None for i in range(8)],
              all_peaks[-1][nb_object][2]
          ])

  # Working with an output that only has belief maps
  if aff is None:
      if len (objects) > 0 and len(all_peaks)>0 and len(all_peaks[0])>0:
          for i_points in range(8):
              if  len(all_peaks[i_points])>0 and all_peaks[i_points][0][2] > 0.5:
                  objects[0][1][i_points] = (all_peaks[i_points][0][0], all_peaks[i_points][0][1])
  else:
      # For all points found
      for i_lists in range(len(all_peaks[:-1])):
          lists = all_peaks[i_lists]

          for candidate in lists:
              if candidate[2] < 0.1:
                  continue

              i_best = -1
              best_dist = 10000
              best_angle = 100
              for i_obj in range(len(objects)):
                  center = [objects[i_obj][0][0], objects[i_obj][0][1]]

                  # integer is used to look into the affinity map,
                  # but the float version is used to run
                  point_int = [int(candidate[0]), int(candidate[1])]
                  point = [candidate[0], candidate[1]]

                  # look at the distance to the vector field.
                  v_aff = np.array([
                                  aff[i_lists*2,
                                  point_int[1],
                                  point_int[0]].data.item(),
                                  aff[i_lists*2+1,
                                      point_int[1],
                                      point_int[0]].data.item()]) * 10

                  # normalize the vector
                  xvec = v_aff[0]
                  yvec = v_aff[1]

                  norms = np.sqrt(xvec * xvec + yvec * yvec)

                  xvec/=norms
                  yvec/=norms

                  v_aff = np.concatenate([[xvec],[yvec]])

                  v_center = np.array(center) - np.array(point)
                  xvec = v_center[0]
                  yvec = v_center[1]

                  norms = np.sqrt(xvec * xvec + yvec * yvec)

                  xvec /= norms
                  yvec /= norms

                  v_center = np.concatenate([[xvec],[yvec]])

                  # vector affinity
                  dist_angle = np.linalg.norm(v_center - v_aff)

                  # distance between vertexes
                  dist_point = np.linalg.norm(np.array(point) - np.array(center))

                  if dist_angle < 0.5 and (best_dist > 1000 or best_dist > dist_point):
                      i_best = i_obj
                      best_angle = dist_angle
                      best_dist = dist_point

              if i_best == -1:
                  continue

              if objects[i_best][1][i_lists] is None \
                      or best_angle < 0.5 \
                      and best_dist < objects[i_best][2][i_lists][1]:
                  objects[i_best][1][i_lists] = ((candidate[0])*8, (candidate[1])*8)
                  objects[i_best][2][i_lists] = (best_angle, best_dist)

  object_detector_end_time = time.time()
  object_detection_time = object_detector_end_time - object_detector_start_time
  print("Object Detection Time:",object_detection_time)

  return objects, all_peaks
