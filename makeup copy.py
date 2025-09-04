import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img-path', default='imgs/116.jpg')
    parse.add_argument('--hair-color', default='red', choices=['red', 'blonde', 'brown', 'black', 'purple', 'blue', 'pink'])
    parse.add_argument('--lip-color', default='pink', choices=['pink', 'red', 'coral', 'purple', 'nude', 'dark_red'])
    parse.add_argument('--eyebrow-color', default='brown', choices=['brown', 'black', 'dark_brown', 'light_brown', 'auburn'])
    parse.add_argument('--blush-color', default='pink', choices=['pink', 'coral', 'peach', 'rose', 'red'])
    parse.add_argument('--eyeshadow-color', default='brown', choices=['brown', 'gold', 'purple', 'blue', 'green', 'pink', 'neutral'])
    return parse.parse_args()


def sharpen(img):
    img = img * 1.0
    # Use channel_axis instead of multichannel for newer scikit-image versions
    gauss_out = gaussian(img, sigma=5, channel_axis=2)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def get_color_scheme(hair_color, lip_color, eyebrow_color, blush_color, eyeshadow_color):
    """Get color schemes for hair, lips, eyebrows, blush, and eyeshadow"""
    hair_colors = {
        'red': [20, 50, 230],      # Bright red
        'blonde': [30, 150, 255],   # Golden blonde
        'brown': [40, 80, 150],     # Medium brown
        'black': [10, 10, 10],      # Black
        'purple': [150, 50, 150],   # Purple
        'blue': [150, 100, 50],     # Blue
        'pink': [100, 50, 200]      # Pink
    }
    
    lip_colors = {
        'pink': [180, 70, 20],      # Pink
        'red': [50, 50, 200],       # Red
        'coral': [100, 100, 200],   # Coral
        'purple': [150, 70, 150],   # Purple
        'nude': [150, 120, 100],    # Nude
        'dark_red': [30, 30, 120]   # Dark red
    }
    
    eyebrow_colors = {
        'brown': [40, 80, 150],     # Medium brown
        'black': [10, 10, 10],      # Black
        'dark_brown': [30, 60, 120], # Dark brown
        'light_brown': [60, 100, 180], # Light brown
        'auburn': [20, 50, 130]     # Auburn
    }
    
    blush_colors = {
        'pink': [180, 70, 20],      # Pink
        'coral': [100, 100, 200],   # Coral
        'peach': [120, 120, 220],   # Peach
        'rose': [150, 80, 30],      # Rose
        'red': [50, 50, 200]        # Red
    }
    
    eyeshadow_colors = {
        'brown': [40, 80, 150],     # Brown
        'gold': [30, 150, 255],     # Gold
        'purple': [150, 70, 150],   # Purple
        'blue': [150, 100, 50],     # Blue
        'green': [100, 150, 50],    # Green
        'pink': [100, 50, 200],     # Pink
        'neutral': [120, 120, 120]  # Neutral
    }
    
    return (hair_colors[hair_color], lip_colors[lip_color], 
            eyebrow_colors[eyebrow_color], blush_colors[blush_color], 
            eyeshadow_colors[eyeshadow_color])

def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    return changed


if __name__ == '__main__':
    # 1  face
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair

    args = parse_args()

    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13
    }

    image_path = args.img_path
    cp = 'cp/79999_iter.pth'
    
    # Get color scheme based on user preferences
    hair_color, lip_color, eyebrow_color, blush_color, eyeshadow_color = get_color_scheme(
        args.hair_color, args.lip_color, args.eyebrow_color, args.blush_color, args.eyeshadow_color)
    print(f"Using hair color: {args.hair_color}")
    print(f"Using lip color: {args.lip_color}")
    print(f"Using eyebrow color: {args.eyebrow_color}")
    print(f"Using blush color: {args.blush_color}")
    print(f"Using eyeshadow color: {args.eyeshadow_color}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        print("Please check if the file exists and the path is correct.")
        exit(1)
    
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    
    # Ensure parsing has the same dimensions as the image
    if parsing.shape != image.shape[:2]:
        parsing = cv2.resize(parsing, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    print(f"Image shape: {image.shape}")
    print(f"Parsing shape: {parsing.shape}")

    # Extended parts table for more makeup features
    parts = [
        table['hair'],      # 17 - hair
        table['upper_lip'], # 12 - upper lip
        table['lower_lip'], # 13 - lower lip
        2,                  # 2 - left eyebrow
        3,                  # 3 - right eyebrow
        4,                  # 4 - left eye
        5,                  # 5 - right eye
        6,                  # 6 - nose
        7,                  # 7 - left ear
        8,                  # 8 - right ear
        9,                  # 9 - left cheek
        10                  # 10 - right cheek
    ]

    # Use the selected colors for each part
    colors = [
        hair_color,         # Hair
        lip_color,          # Upper lip
        lip_color,          # Lower lip
        eyebrow_color,      # Left eyebrow
        eyebrow_color,      # Right eyebrow
        eyeshadow_color,    # Left eye (eyeshadow)
        eyeshadow_color,    # Right eye (eyeshadow)
        blush_color,        # Nose (subtle blush)
        blush_color,        # Left ear (subtle blush)
        blush_color,        # Right ear (subtle blush)
        blush_color,        # Left cheek (main blush area)
        blush_color         # Right cheek (main blush area)
    ]

    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)

    # Save the results automatically (no display needed)
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save original and modified images
    cv2.imwrite(f'{output_dir}/{base_name}_original.jpg', ori)
    cv2.imwrite(f'{output_dir}/{base_name}_makeup.jpg', image)
    
    print(f"Images saved to {output_dir}/")
    print(f"Original: {base_name}_original.jpg")
    print(f"With makeup: {base_name}_makeup.jpg")
    print("Check the output folder to see your results!")















