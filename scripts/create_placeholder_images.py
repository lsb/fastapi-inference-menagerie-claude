#!/usr/bin/env python3
"""Create placeholder test images for the remaining cases."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder_image(text: str, filename: str, size: tuple = (512, 512)):
    """Create a placeholder image with text."""
    
    # Create image with gradient background
    img = Image.new('RGB', size, color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Try to use a system font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position to center it
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill='black', font=font)
    
    # Save image
    output_dir = Path("tests/data/images")
    output_path = output_dir / filename
    img.save(output_path)
    print(f"Created placeholder: {output_path}")

def main():
    """Create placeholder images for testing."""
    
    # Create remaining placeholder images
    create_placeholder_image(
        "Cat on Mountain\n(Outdoor Scene)",
        "cat_mountain_sunrise.png"
    )
    
    create_placeholder_image(
        "Dog on Mountain\n(Outdoor Scene)", 
        "dog_mountain_sunrise.png"
    )
    
    print("Placeholder images created successfully!")

if __name__ == "__main__":
    main()