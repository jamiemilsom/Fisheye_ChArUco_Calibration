import cv2
import os


input_dir =  '/home/jamie/Pictures/skymasks'
output_dir = '/home/jamie/Pictures/skymasks_resized'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
        # Read the image
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        
        # # Get new dimensions (half the original size)
        # new_width = int(image.shape[1] / 2)
        # new_height = int(image.shape[0] / 2)
        
        # # Resize the image
        # resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert the resized image to grayscale
        resized_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Save the resized image to the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized_image)

print("All images have been resized.")