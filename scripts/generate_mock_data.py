import os
import random
from PIL import Image

def create_dummy_image(path, size=(300, 300), color=None):
    """
    Creates a solid colored image and saves it to path.
    """
    if color is None:
        # Generate a random RGB color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = Image.new("RGB", size, color)
    img.save(path, "JPEG")

def main():
    base_dir = "./mock_data"
    splits = ["train", "val"]
    classes = ["Reuse", "Refurbish", "Repair", "Recycle"]
    
    # Configure number of sample folders per class
    samples_per_split = {
        "train": 6,  # 6 phone samples per class for training
        "val": 3     # 3 phone samples per class for validation
    }
    
    print(f"Generating mock dataset structure under: {base_dir}")
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        num_samples = samples_per_split[split]
        
        for cls in classes:
            class_dir = os.path.join(split_dir, cls)
            os.makedirs(class_dir, exist_ok=True)
            
            for s_idx in range(1, num_samples + 1):
                sample_folder_name = f"phone_{s_idx:03d}"
                sample_path = os.path.join(class_dir, sample_folder_name)
                os.makedirs(sample_path, exist_ok=True)
                
                # Randomly choose between 4 and 5 images per phone sample
                num_images = random.choice([4, 5])
                
                # Generate unique solid color for this phone's images to represent similar look
                base_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
                
                for img_idx in range(1, num_images + 1):
                    img_name = f"view_{img_idx}.jpg"
                    img_path = os.path.join(sample_path, img_name)
                    
                    # Apply a tiny random color shift to simulate different views/angles
                    shifted_color = (
                        max(0, min(255, base_color[0] + random.randint(-20, 20))),
                        max(0, min(255, base_color[1] + random.randint(-20, 20))),
                        max(0, min(255, base_color[2] + random.randint(-20, 20))),
                    )
                    
                    create_dummy_image(img_path, color=shifted_color)
                    
                print(f"  Created {num_images} images for {split}/{cls}/{sample_folder_name}")
                
    print("\nMock dataset generated successfully!")
    print(f"Path to mock dataset: {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    main()
