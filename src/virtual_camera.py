import os
import numpy as np
import cv2

class VirtualCamera:
    def __init__(self, input_folder, output_folder):
        
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.input_image_list = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.endswith(".jpg")]

    @staticmethod
    def load_image(image_path):
        
        """Loads an image from the given path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read the image: {image_path}")
        return image

    @staticmethod
    def circular_mask(center, radius, height, width):
        
        """Creates a circular mask with the given center and radius."""
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        return mask

class RectangularCamera(VirtualCamera):
    def __init__(self, input_folder, output_folder, sections = 4, overlap_ratio = 0.1):
        super().__init__(input_folder, output_folder)
        self.sections = sections
        self.overlap_ratio = overlap_ratio

    def split_image(self, image_path, output_path=None):

        image = VirtualCamera.load_image(image_path)
        height, width = image.shape[:2]
        half_height, half_width = height // 2, width // 2
        third_height, third_width = height // 3, width // 3
        center = (width // 2, height // 2)
        radius = min(width, height) // 2
        mask = VirtualCamera.circular_mask(center, radius, height, width)
        
        height_multipliers = {
            4: 0.5,
            5: 0.5,
            9: 1/3
        }

        try:
            overlap_pixels = int(self.overlap_ratio * height * height_multipliers[self.sections])
        except KeyError:
            raise ValueError(f"Invalid number of sections: {self.sections}, must be 4, 5, or 9")
        
        sections = {}

        if self.sections == 4 or self.sections == 5:
            sections['top_left'] = image[:half_height + overlap_pixels, :half_width + overlap_pixels]
            sections['top_right'] = image[:half_height + overlap_pixels, half_width - overlap_pixels:]
            sections['bottom_left'] = image[half_height - overlap_pixels:, :half_width + overlap_pixels]
            sections['bottom_right'] = image[half_height - overlap_pixels:, half_width - overlap_pixels:]
            
            if self.sections == 5:
                center_size = sections['top_left'].shape
                center_start_y = half_height - center_size[0] // 2
                center_start_x = half_width - center_size[1] // 2
                sections['center'] = image[center_start_y:center_start_y + center_size[0], center_start_x:center_start_x + center_size[1]]

        if self.sections == 9:
            sections['top_left'] = image[:third_height + overlap_pixels, :third_width + overlap_pixels]
            sections['top_center'] = image[:third_height + overlap_pixels, third_width - overlap_pixels // 2:2*third_width + overlap_pixels // 2]
            sections['top_right'] = image[:third_height + overlap_pixels, 2*third_width - overlap_pixels:]
            sections['middle_left'] = image[third_height - overlap_pixels // 2:2*third_height + overlap_pixels // 2, :third_width + overlap_pixels]
            sections['middle_center'] = image[third_height - overlap_pixels // 2:2*third_height + overlap_pixels // 2, third_width - overlap_pixels // 2:2*third_width + overlap_pixels // 2]
            sections['middle_right'] = image[third_height - overlap_pixels // 2:2*third_height + overlap_pixels // 2, 2*third_width - overlap_pixels:]
            sections['bottom_left'] = image[2*third_height - overlap_pixels:, :third_width + overlap_pixels]
            sections['bottom_center'] = image[2*third_height - overlap_pixels:, third_width - overlap_pixels // 2:2*third_width + overlap_pixels // 2]
            sections['bottom_right'] = image[2*third_height - overlap_pixels:, 2*third_width - overlap_pixels:]

        for section_name, section_image in sections.items():

            section_folder = os.path.join(self.output_folder, section_name)
            os.makedirs(section_folder, exist_ok=True)

            base_filename = os.path.basename(image_path)
            output_filename = os.path.join(section_folder, base_filename)

            cv2.imwrite(output_filename, section_image)
            
        
            
        
        
        


class ConcentricCamera(VirtualCamera):
    def __init__(self, input_folder, output_folder):
        super().__init__(input_folder, output_folder)

    def split_image(self, image_path, output_path=None):
        image = VirtualCamera.load_image(image_path)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        radius = min(width, height) // 2

        mask_outer = VirtualCamera.circular_mask(center, radius, height, width)
        mask_inner = VirtualCamera.circular_mask(center, int(radius * ratio), height, width)
        mask = mask_outer - mask_inner
        concentric_image = cv2.bitwise_and(image, image, mask=mask)
        output_path = output_path or 'concentric.jpg'
        cv2.imwrite(output_path, concentric_image)

if __name__ == "__main__":
    image_path = '/home/jamie/Documents/reconstruction/data/calibration/jpg/IMG_20231011_151438_00_003.jpg'
    input_folder = '/home/jamie/Documents/reconstruction/data/calibration/jpg'
    output_folder = '/home/jamie/Documents/reconstruction/data/calibration/testing_balance/processed'

    rectangular_cam = RectangularCamera(input_folder, output_folder, sections=9, overlap_ratio=0.1)
    concentric_cam = ConcentricCamera(input_folder, output_folder)

    for image_path in rectangular_cam.input_image_list:
        rectangular_cam.split_image(image_path)


