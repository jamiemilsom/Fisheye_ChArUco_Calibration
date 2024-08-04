import os
import numpy as np
import cv2

class VirtualCamera:
    """
    A base class for virtual camera implementations.
    
    This class provides basic functionality for loading images and creating masks.
    """
    
    def __init__(self, input_folder, output_folder):
        """
        Initialize the VirtualCamera.

        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder where processed images will be saved.
        """
        
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.input_image_list =  [os.path.join(self.input_folder, f) for f in sorted(os.listdir(self.input_folder)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]



    @staticmethod
    def load_image(image_path):
        
        """
        Load an image from the given path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            numpy.ndarray: The loaded image.

        Raises:
            FileNotFoundError: If the image file is not found.
            ValueError: If the image cannot be read.
        """
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read the image: {image_path}")
        return image



    @staticmethod
    def circular_mask(center, radius, height, width):
        """
        Create a circular mask with the given center and radius.

        Args:
            center (tuple): The (x, y) coordinates of the circle's center.
            radius (int): The radius of the circle.
            height (int): The height of the mask.
            width (int): The width of the mask.

        Returns:
            numpy.ndarray: A binary mask with the circular region set to 255 and the rest to 0.
        """
        
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        return mask



class RectangularCamera(VirtualCamera):
    """
    A virtual camera that splits images into rectangular sections.
    """
    
    def __init__(self, input_folder, output_folder, sections = 4, overlap_ratio = 0.1):
        """
        Initialize the RectangularCamera.

        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder where processed images will be saved.
            sections (int): Number of sections to split the image into (4, 5, or 9).
            overlap_ratio (float): The ratio of overlap between adjacent sections.
        """
        super().__init__(input_folder, output_folder)
        self.sections = sections
        self.overlap_ratio = overlap_ratio
        
        

    def split_image(self, image_path, output_path=None):
        """
        Split an image into rectangular sections.

        Args:
            image_path (str): Path to the input image.
            output_path (str, optional): Path to save the output image (not used in this implementation).

        Returns:
            dict: A dictionary containing the split image sections.

        Raises:
            ValueError: If an invalid number of sections is specified.
        """
        image = VirtualCamera.load_image(image_path)
        height, width = image.shape[:2]
        half_height, half_width = height // 2, width // 2
        third_height, third_width = height // 3, width // 3
        
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
    """
    A virtual camera that splits images into concentric circular sections.
    """
    
    def __init__(self, input_folder, output_folder, splits, overlap_ratio=0):
        """
        Initialize the ConcentricCamera.

        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder where processed images will be saved.
            splits (list): List of radii ratios for splitting the image.
            overlap_ratio (float): The ratio of overlap between adjacent sections (not used in this implementation).
        """
        super().__init__(input_folder, output_folder)
        self.splits = splits
        self.overlap_ratio = overlap_ratio



    def split_image(self, image_path, output_path=None):
        """
        Split an image into concentric circular sections.

        Args:
            image_path (str): Path to the input image.
            output_path (str, optional): Path to save the output image (not used in this implementation).

        Returns:
            dict: A dictionary containing the split image sections.
        """
        image = VirtualCamera.load_image(image_path)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        radius = min(width, height) // 2
        overlap = int(self.overlap_ratio * radius)
        
        sections = {}
        
        for num, split in enumerate(self.splits):
            
            if num == 0:
                
                outer_mask = VirtualCamera.circular_mask(center, int(radius * split), height, width)
                sections[f'camera_{num}'] = cv2.bitwise_and(image, image, mask=outer_mask)
                
            else:
                
                inner_mask = VirtualCamera.circular_mask(center, int(radius * self.splits[num - 1] - overlap), height, width)
                outer_mask = VirtualCamera.circular_mask(center, int(radius * split), height, width)
                mask = cv2.bitwise_xor(outer_mask, inner_mask)
                sections[f'camera_{num}'] = cv2.bitwise_and(image, image, mask=mask)

            
        outermost_mask = VirtualCamera.circular_mask(center, int(radius * self.splits[-1] - overlap), height, width)
        inverse_outermost_mask = cv2.bitwise_not(outermost_mask)
        sections[f'camera_{len(self.splits)}'] = cv2.bitwise_and(image, image, mask=inverse_outermost_mask)

        
        for section_name, section_image in sections.items():

            section_folder = os.path.join(self.output_folder, section_name)
            os.makedirs(section_folder, exist_ok=True)

            base_filename = os.path.basename(image_path)
            output_filename = os.path.join(section_folder, base_filename)

            cv2.imwrite(output_filename, section_image)

        

if __name__ == "__main__":
    
    input_folder = os.path.join(os.path.dirname(__file__), "../../data/raw_images/")
    rectangular_output_folder = os.path.join(os.path.dirname(__file__), "../../data/virtual_cameras/rectangular_images/")
    concentric_output_folder = os.path.join(os.path.dirname(__file__), "../../data/virtual_cameras/concentric_images/")
    
    rectangular_cam = RectangularCamera(input_folder, rectangular_output_folder, sections=9, overlap_ratio=0.1)
    concentric_cam = ConcentricCamera(input_folder, concentric_output_folder, splits=[0.6], overlap_ratio=0.1)
    
    
    for image_path in concentric_cam.input_image_list:
        concentric_cam.split_image(image_path)

    for image_path in rectangular_cam.input_image_list:
        rectangular_cam.split_image(image_path)


