import os
import shutil
from tqdm import tqdm

def prepare_dataset():
    """Prepare the dataset by combining image parts and organizing the structure."""
    # Create main images directory
    os.makedirs('data/HAM10000_images', exist_ok=True)
    
    # Copy images from part 1
    print("Copying images from part 1...")
    for img in tqdm(os.listdir('data/HAM10000_images_part_1')):
        src = os.path.join('data/HAM10000_images_part_1', img)
        dst = os.path.join('data/HAM10000_images', img)
        shutil.copy2(src, dst)
    
    # Copy images from part 2
    print("Copying images from part 2...")
    for img in tqdm(os.listdir('data/HAM10000_images_part_2')):
        src = os.path.join('data/HAM10000_images_part_2', img)
        dst = os.path.join('data/HAM10000_images', img)
        shutil.copy2(src, dst)
    
    print("Dataset preparation completed!")

if __name__ == '__main__':
    prepare_dataset() 