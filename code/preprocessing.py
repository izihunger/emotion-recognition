import os
import csv
import cv2
import numpy as np
import random
import tqdm

rotation_range = (-15, 15)  # rotation range in degrees
scale_range = (0.8, 1.2)  # scaling factor range
translation_range = (-10, 10)  # translation range in pixels

def augment_image(image, landmarks):
    """Allows you to perform data augmentation on images."""
    h, w = image.shape[:2]

    # rotation
    angle = random.uniform(*rotation_range)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image_rot = cv2.warpAffine(image, M_rot, (w, h))
    landmarks_rot = [np.dot(M_rot, np.array([x, y, 1])) for x, y in landmarks]
    # scaling
    scale = random.uniform(*scale_range)
    M_scale = np.array([[scale, 0, (1 - scale) * w / 2],
                        [0, scale, (1 - scale) * h / 2]])
    image_scaled = cv2.warpAffine(image_rot, M_scale, (w, h))
    landmarks_scaled = [np.dot(M_scale, np.array([x, y, 1])) for x, y in landmarks_rot]
    # translation
    tx = random.uniform(*translation_range)
    ty = random.uniform(*translation_range)
    M_trans = np.array([[1, 0, tx], [0, 1, ty]])
    image_trans = cv2.warpAffine(image_scaled, M_trans, (w, h))
    landmarks_trans = [(x + tx, y + ty) for x, y in landmarks_scaled]
    return image_trans, landmarks_trans

def draw_landmarks(image, landmarks):
    """Allows you to draw landmarks and join each point on images."""
    ranges_to_connect = [
        list(range(0, 17)), list(range(17, 22)), list(range(22, 27)), 
        list(range(27, 31)), list(range(31, 36)), list(range(36, 42)),
        list(range(42, 48)), list(range(48, 60)), list(range(60, 68))
    ]
    
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    for points_range in ranges_to_connect:
        for i in range(len(points_range) - 1):
            start_point = landmarks[points_range[i]]
            end_point = landmarks[points_range[i + 1]]
            cv2.line(image, (int(start_point[0]), int(start_point[1])), 
                           (int(end_point[0]), int(end_point[1])), (255, 0, 0), 1)
    
    return image

def preprocessing(csv_file, image_dir, output_dir, mode="train", augmentations=5):
    """Preprocess images before training or validation."""
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        augmented_data = []
        for row in tqdm.tqdm(reader):
            image_name = row[0]
            x_coords = list(map(float, row[2:70]))
            y_coords = list(map(float, row[70:138]))
            landmarks = list(zip(x_coords, y_coords))

            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)

            if image is not None:
                original_with_landmarks = image.copy()
                # draw_landmarks(original_with_landmarks, landmarks)
                # save the original image with landmarks
                original_image_name = f"{os.path.splitext(image_name)[0]}.jpg"
                original_image_path = os.path.join(output_dir, original_image_name)
                cv2.imwrite(original_image_path, original_with_landmarks)
                # save the original data (used for train and test)
                augmented_data.append([original_image_name, row[1]] + x_coords + y_coords)
                # perform augmentations only for training data
                if mode == "train":
                    for i in range(augmentations):
                        aug_image, aug_landmarks = augment_image(image, landmarks)
                        # draw_landmarks(aug_image, aug_landmarks)
                        aug_image_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.jpg"
                        aug_image_path = os.path.join(output_dir, aug_image_name)
                        cv2.imwrite(aug_image_path, aug_image)
                        aug_x_coords, aug_y_coords = zip(*aug_landmarks)
                        augmented_data.append([aug_image_name, row[1]] + list(aug_x_coords) + list(aug_y_coords))
            else:
                print(f"Image {image_name} not found in {image_dir}")

    # save augmented landmarks to a new CSV file
    augmented_csv_file = os.path.join(output_dir, "augmented_data.csv")
    with open(augmented_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(augmented_data)
    print(f"Processed images with landmarks have been saved to {output_dir}")