#!/usr/bin/env python3
"""
Flask API for Beauty AI Platform - Makeup and Hair Styling
Based on the original working makeup system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from PIL import Image
import io

# Import your actual working models directly (no path issues!)
from test import evaluate
from makeup import get_color_scheme, hair

# Helper: robustly decode base64 image strings that may or may not include a data URI prefix
def decode_base64_image(image_value: str) -> bytes:
    if not isinstance(image_value, str) or not image_value:
        raise ValueError("Invalid image payload: expected non-empty string")
    # Support both formats: "data:image/jpeg;base64,AAAA..." and plain "AAAA..."
    base64_str = image_value.split(',', 1)[1] if ',' in image_value else image_value
    try:
        return base64.b64decode(base64_str)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image: {exc}")

# Helpers: flexible color parsing (accept #RRGGBB, rgb(r,g,b), [r,g,b], or named)
def _hex_to_bgr(hex_str: str):
    hs = hex_str.strip()
    if hs.startswith('#'):
        hs = hs[1:]
    if len(hs) != 6:
        raise ValueError("Hex must be #RRGGBB")
    r = int(hs[0:2], 16)
    g = int(hs[2:4], 16)
    b = int(hs[4:6], 16)
    return [b, g, r]

def _rgb_func_to_bgr(rgb_str: str):
    s = rgb_str.strip().lower()
    if not s.startswith('rgb(') or not s.endswith(')'):
        raise ValueError("Invalid rgb() format")
    nums = s[4:-1].split(',')
    if len(nums) != 3:
        raise ValueError("rgb() must have 3 numbers")
    r, g, b = [int(n.strip()) for n in nums]
    return [b, g, r]

def parse_color_value(value, default_name: str, category_defaults: dict):
    # If explicit list/tuple
    if isinstance(value, (list, tuple)) and len(value) == 3:
        r, g, b = value
        return [b, g, r]
    # If string hex or rgb()
    if isinstance(value, str):
        v = value.strip()
        if v.startswith('#'):
            return _hex_to_bgr(v)
        if v.lower().startswith('rgb('):
            return _rgb_func_to_bgr(v)
        # Named color in defaults (BGR already)
        if v in category_defaults:
            return category_defaults[v]
        # Fall back to default name
        if default_name in category_defaults:
            return category_defaults[default_name]
    # Fallback
    return category_defaults.get(default_name, [40, 80, 150])

def select_parts_for_style(style: str):
    s = (style or '').strip().lower()
    if s in ('dramatic', 'bold', 'glam', 'glamorous'):
        # Lips, brows, eyes, cheeks, nose (no hair)
        return [12, 13, 2, 3, 4, 5, 9, 10, 6]
    if s in ('minimal', 'simple', 'light'):
        # Mostly lips (+light cheeks)
        return [12, 13, 9, 10]
    # Natural default: lips, brows, eyes, cheeks (no nose by default)
    return [12, 13, 2, 3, 4, 5, 9, 10]

app = Flask(__name__)
CORS(app)

# Path to your actual working model (relative to this folder)
MODEL_PATH = 'cp/79999_iter.pth'

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Beauty AI Platform Backend is running',
        'model_path': MODEL_PATH
    })

@app.route('/api/apply-makeup', methods=['POST'])
def apply_makeup():
    """Apply makeup to face image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Get makeup style/mode (accept style or makeupType)
        makeup_style = data.get('style') or data.get('makeupType') or 'natural'
        
        # Decode base64 image (robust)
        image_bytes = decode_base64_image(data['image'])
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Save temporary image for your model
        temp_path = 'temp_image.jpg'
        cv2.imwrite(temp_path, opencv_image)
        
        # Use your ACTUAL working model
        print("Running your actual BiSeNet model...")
        parsing = evaluate(temp_path, MODEL_PATH)
        print(f"Image shape: {opencv_image.shape}")
        print(f"Parsing shape: {parsing.shape}")
        print(f"Parsing unique values: {np.unique(parsing)}")
        
        # Get color scheme based on style for defaults (used if user doesn't override)
        style_key = (makeup_style or '').lower()
        if style_key == 'natural':
            hair_color, lip_color, eyebrow_color, blush_color, eyeshadow_color = get_color_scheme(
                'brown', 'pink', 'brown', 'pink', 'neutral'
            )
        elif style_key in ('dramatic', 'bold', 'glam', 'glamorous'):
            hair_color, lip_color, eyebrow_color, blush_color, eyeshadow_color = get_color_scheme(
                'black', 'red', 'black', 'red', 'purple'
            )
        else:  # minimal
            hair_color, lip_color, eyebrow_color, blush_color, eyeshadow_color = get_color_scheme(
                'brown', 'nude', 'brown', 'peach', 'neutral'
            )
        
        # Apply makeup using your actual working functions
        result_image = opencv_image.copy()
        
        # Ensure parsing has the same dimensions as the image
        if parsing.shape != opencv_image.shape[:2]:
            parsing = cv2.resize(parsing, (opencv_image.shape[1], opencv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Determine parts for this style (EXCLUDING HAIR)
        parts = select_parts_for_style(makeup_style)
        
        # Map each part to the appropriate default color
        part_to_color = {
            12: lip_color,
            13: lip_color,
            2: eyebrow_color,
            3: eyebrow_color,
            4: eyeshadow_color,
            5: eyeshadow_color,
            6: blush_color,   # subtle nose
            7: blush_color,   # subtle ear tone
            8: blush_color,
            9: blush_color,
            10: blush_color,
        }
        colors = [part_to_color[p] for p in parts]
        
        # Apply makeup to each part
        for part, color in zip(parts, colors):
            result_image = hair(result_image, parsing, part, color)
        
        # Convert back to PIL and encode
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        
        # Encode result
        buffer = io.BytesIO()
        result_pil.save(buffer, format='JPEG')
        result_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'resultImage': f'data:image/jpeg;base64,{result_b64}'
        })
        
    except Exception as e:
        print(f"Error in apply_makeup: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/style-hair', methods=['POST'])
def style_hair():
    """Apply hair color to face image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Get hair color
        hair_color = data.get('hairColor', 'brown')
        
        # Decode base64 image (robust)
        image_bytes = decode_base64_image(data['image'])
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Save temporary image for your model
        temp_path = 'temp_image.jpg'
        cv2.imwrite(temp_path, opencv_image)
        
        # Use your ACTUAL working model
        print("Running your actual BiSeNet model for hair styling...")
        parsing = evaluate(temp_path, MODEL_PATH)
        
        # Get hair color from your actual color scheme
        hair_color_rgb, _, _, _, _ = get_color_scheme(
            hair_color, 'pink', 'brown', 'pink', 'neutral'
        )
        
        # Apply hair color using your actual working function
        result_image = hair(opencv_image, parsing, 17, hair_color_rgb)  # Part 17 is hair
        
        # Convert back to PIL and encode
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        
        # Encode result
        buffer = io.BytesIO()
        result_pil.save(buffer, format='JPEG')
        result_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'resultImage': f'data:image/jpeg;base64,{result_b64}'
        })
        
    except Exception as e:
        print(f"Error in style_hair: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cosmetic-adjustments', methods=['POST'])
def cosmetic_adjustments():
    """Apply comprehensive cosmetic adjustments"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Get adjustment parameters (allow custom colors)
        hair_color_input = data.get('hairColor', 'brown')
        lip_color_input = data.get('lipColor', 'pink')
        eyebrow_color_input = data.get('eyebrowColor', 'brown')
        blush_color_input = data.get('blushColor', 'pink')
        eyeshadow_color_input = data.get('eyeshadowColor', data.get('eyeColor', 'neutral'))
        nose_color_input = data.get('noseColor')
        skin_color_input = data.get('skinColor')
        eyeliner = data.get('eyeliner', 'natural')
        
        # Decode base64 image (robust)
        image_bytes = decode_base64_image(data['image'])
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Save temporary image for your model
        temp_path = 'temp_image.jpg'
        cv2.imwrite(temp_path, opencv_image)
        
        # Use your ACTUAL working model
        print("Running your actual BiSeNet model for cosmetic adjustments...")
        parsing = evaluate(temp_path, MODEL_PATH)
        
        # Get default named colors from scheme (for fallback)
        def_hair, def_lip, def_brow, def_blush, def_eye = get_color_scheme(
            'brown', 'pink', 'brown', 'pink', 'neutral'
        )

        # Build name -> BGR maps from defaults
        hair_map = { 'red': [20,50,230], 'blonde':[30,150,255], 'brown':[40,80,150], 'black':[10,10,10], 'purple':[150,50,150], 'blue':[150,100,50], 'pink':[100,50,200] }
        lip_map = { 'pink':[180,70,20], 'red':[50,50,200], 'coral':[100,100,200], 'purple':[150,70,150], 'nude':[150,120,100], 'dark_red':[30,30,120] }
        brow_map = { 'brown':[40,80,150], 'black':[10,10,10], 'dark_brown':[30,60,120], 'light_brown':[60,100,180], 'auburn':[20,50,130] }
        blush_map = { 'pink':[180,70,20], 'coral':[100,100,200], 'peach':[120,120,220], 'rose':[150,80,30], 'red':[50,50,200] }
        eye_map = { 'brown':[40,80,150], 'gold':[30,150,255], 'purple':[150,70,150], 'blue':[150,100,50], 'green':[100,150,50], 'pink':[100,50,200], 'neutral':[120,120,120] }
        skin_map = { 'light':[220,200,180], 'medium':[180,160,140], 'dark':[120,100,80] }

        hair_color_rgb = parse_color_value(hair_color_input, 'brown', hair_map)
        lip_color_rgb = parse_color_value(lip_color_input, 'pink', lip_map)
        eyebrow_color_rgb = parse_color_value(eyebrow_color_input, 'brown', brow_map)
        blush_color_rgb = parse_color_value(blush_color_input, 'pink', blush_map)
        eyeshadow_color_rgb = parse_color_value(eyeshadow_color_input, 'neutral', eye_map)
        nose_color_rgb = parse_color_value(nose_color_input, 'pink', blush_map) if nose_color_input else blush_color_rgb
        skin_color_rgb = parse_color_value(skin_color_input, 'medium', skin_map) if skin_color_input else None
        
        # Apply all cosmetic adjustments
        result_image = opencv_image.copy()
        
        # Ensure parsing has the same dimensions as the image
        if parsing.shape != opencv_image.shape[:2]:
            parsing = cv2.resize(parsing, (opencv_image.shape[1], opencv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply all cosmetic adjustments using the original working logic
        parts = [17, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Optionally include full face if skinColor provided
        if skin_color_rgb is not None:
            parts = [1] + parts
        
        colors = []
        if skin_color_rgb is not None:
            colors.append(skin_color_rgb)            # Face
        colors.extend([
            hair_color_rgb,         # Hair
            lip_color_rgb,          # Upper lip
            lip_color_rgb,          # Lower lip
            eyebrow_color_rgb,      # Left eyebrow
            eyebrow_color_rgb,      # Right eyebrow
            eyeshadow_color_rgb,    # Left eye (eyeshadow)
            eyeshadow_color_rgb,    # Right eye (eyeshadow)
            nose_color_rgb,         # Nose
            blush_color_rgb,        # Left ear
            blush_color_rgb,        # Right ear
            blush_color_rgb,        # Left cheek
            blush_color_rgb         # Right cheek
        ])
        
        # Apply makeup to each part
        for part, color in zip(parts, colors):
            result_image = hair(result_image, parsing, part, color)
        
        # Convert back to PIL and encode
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        
        # Encode result
        buffer = io.BytesIO()
        result_pil.save(buffer, format='JPEG')
        result_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'resultImage': f'data:image/jpeg;base64,{result_b64}'
        })
        
    except Exception as e:
        print(f"Error in cosmetic_adjustments: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Beauty AI Platform Backend...")
    print(f"Model path: {MODEL_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=True)