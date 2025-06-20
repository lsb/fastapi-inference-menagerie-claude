#!/usr/bin/env python3
"""Generate mountain test images at 1024x1024 with 20 steps."""

import os
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

def main():
    """Generate mountain test images."""
    
    output_dir = Path("tests/data/images")
    
    print("Loading Stable Diffusion XL pipeline...")
    
    # Load SDXL pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    
    # Force CPU usage
    device = "cpu"
    pipe = pipe.to(device)
    
    print(f"Using device: {device}")
    
    # Enable memory efficient settings for CPU
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    # Mountain test cases only
    mountain_cases = [
        {
            "prompt": "photorealistic, hyper detailed, a beautiful tabby cat sitting on a rocky mountain peak during golden hour sunrise, majestic mountain landscape, natural lighting, sharp focus, 8k resolution, realistic fur texture, dramatic sky with clouds",
            "filename": "cat_mountain_sunrise.png", 
            "description": "Cat on mountain (outdoor)"
        },
        {
            "prompt": "photorealistic, hyper detailed, a german shepherd dog sitting on a rocky mountain peak during golden hour sunrise, majestic mountain landscape, natural lighting, sharp focus, 8k resolution, realistic fur texture, dramatic sky with clouds",
            "filename": "dog_mountain_sunrise.png",
            "description": "Dog on mountain (outdoor)"
        }
    ]
    
    # Generate mountain images
    for i, case in enumerate(mountain_cases, 3):  # Start from 3 since we have 2 already
        print(f"\nGenerating image {i}/4: {case['description']}")
        print(f"Prompt: {case['prompt']}")
        
        try:
            # Generate image with 20 diffusion steps at 1024x1024 resolution
            image = pipe(
                case["prompt"],
                num_inference_steps=20,  # Use 20 diffusion steps as requested
                guidance_scale=5.0,
                height=1024,
                width=1024
            ).images[0]
            
            # Save image
            output_path = output_dir / case["filename"]
            image.save(output_path)
            print(f"Saved: {output_path}")
            
            # Verify dimensions
            print(f"Image size: {image.size[0]}x{image.size[1]}")
            
        except Exception as e:
            print(f"Error generating image {i}: {e}")
            continue
    
    print(f"\nCompleted generating mountain images in {output_dir}")
    
    # Final summary
    print("\nFinal image set:")
    for filename in ["cat_office_typing.png", "dog_office_typing.png", "cat_mountain_sunrise.png", "dog_mountain_sunrise.png"]:
        image_path = output_dir / filename
        if image_path.exists():
            from PIL import Image
            img = Image.open(image_path)
            size_mb = image_path.stat().st_size / (1024*1024)
            print(f"  ✓ {filename}: {img.size[0]}x{img.size[1]}, {size_mb:.1f}MB")
        else:
            print(f"  ✗ {filename}: Missing")

if __name__ == "__main__":
    main()