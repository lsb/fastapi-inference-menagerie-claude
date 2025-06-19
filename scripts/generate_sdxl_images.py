#!/usr/bin/env python3
"""Generate test images using Stable Diffusion XL on CPU."""

import os
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

def main():
    """Generate 4 test images for CLIP testing using SDXL."""
    
    # Create output directory
    output_dir = Path("tests/data/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Stable Diffusion XL pipeline...")
    
    # Load SDXL pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,  # Use float32 for CPU
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None
    )
    
    # Force CPU usage as requested
    device = "cpu"
    pipe = pipe.to(device)
    
    print(f"Using device: {device}")
    
    # Enable memory efficient settings for CPU
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    # Test prompts and filenames - photorealistic style
    test_cases = [
        {
            "prompt": "photorealistic, hyper detailed, a tabby cat with paws on a computer keyboard in a modern office, professional lighting, sharp focus, 8k resolution, realistic fur texture, office environment with desk and monitor",
            "filename": "cat_office_typing.png",
            "description": "Cat in office (indoor)"
        },
        {
            "prompt": "photorealistic, hyper detailed, a golden retriever dog with paws on a computer keyboard in a modern office, professional lighting, sharp focus, 8k resolution, realistic fur texture, office environment with desk and monitor", 
            "filename": "dog_office_typing.png",
            "description": "Dog in office (indoor)"
        },
        {
            "prompt": "photorealistic, hyper detailed, a beautiful cat sitting on a rocky mountain peak during golden hour sunrise, majestic mountain landscape, natural lighting, sharp focus, 8k resolution, realistic fur texture, dramatic sky",
            "filename": "cat_mountain_sunrise.png", 
            "description": "Cat on mountain (outdoor)"
        },
        {
            "prompt": "photorealistic, hyper detailed, a german shepherd dog sitting on a rocky mountain peak during golden hour sunrise, majestic mountain landscape, natural lighting, sharp focus, 8k resolution, realistic fur texture, dramatic sky",
            "filename": "dog_mountain_sunrise.png",
            "description": "Dog on mountain (outdoor)"
        }
    ]
    
    # Generate images
    for i, case in enumerate(test_cases, 1):
        print(f"\nGenerating image {i}/4: {case['description']}")
        print(f"Prompt: {case['prompt']}")
        
        try:
            # Generate image with minimal steps for faster CPU generation
            image = pipe(
                case["prompt"],
                num_inference_steps=10,  # Very few steps for CPU speed
                guidance_scale=5.0,      # Lower guidance for speed
                height=512,
                width=512
            ).images[0]
            
            # Save image
            output_path = output_dir / case["filename"]
            image.save(output_path)
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Error generating image {i}: {e}")
            # Continue with next image
            continue
    
    print(f"\nCompleted generating images in {output_dir}")
    print("\nGenerated files:")
    for case in test_cases:
        output_path = output_dir / case["filename"]
        if output_path.exists():
            print(f"  ✓ {case['filename']}: {case['description']}")
        else:
            print(f"  ✗ {case['filename']}: Failed to generate")

if __name__ == "__main__":
    main()