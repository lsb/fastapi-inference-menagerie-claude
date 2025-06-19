#!/usr/bin/env python3
"""Generate test images using Stable Diffusion 1.5 for CLIP testing."""

import os
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline

def main():
    """Generate 4 test images for CLIP model testing."""
    
    # Create output directory
    output_dir = Path("tests/data/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Stable Diffusion 1.5 pipeline...")
    
    # Load the pipeline with memory optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,  # Use float32 for stability on CPU/MPS
        low_cpu_mem_usage=True
    )
    
    # Use the best available device
    if torch.backends.mps.is_available():
        device = "mps"
        pipe = pipe.to(device)
    elif torch.cuda.is_available():
        device = "cuda"
        pipe = pipe.to(device)
    else:
        device = "cpu"
        pipe = pipe.to(device)
    
    print(f"Using device: {device}")
    
    # Enable memory efficient settings
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    # Only enable CPU offload for CUDA devices
    if hasattr(pipe, "enable_sequential_cpu_offload") and device == "cuda":
        pipe.enable_sequential_cpu_offload()
    
    # Test prompts and filenames
    test_cases = [
        {
            "prompt": "a cat typing at a computer in an office, professional lighting, detailed",
            "filename": "cat_office_typing.png",
            "description": "Cat in office (indoor)"
        },
        {
            "prompt": "a dog typing at a computer in an office, professional lighting, detailed", 
            "filename": "dog_office_typing.png",
            "description": "Dog in office (indoor)"
        },
        {
            "prompt": "a cat sitting on a mountaintop at sunrise, beautiful landscape, detailed",
            "filename": "cat_mountain_sunrise.png", 
            "description": "Cat on mountain (outdoor)"
        },
        {
            "prompt": "a dog sitting on a mountaintop at sunrise, beautiful landscape, detailed",
            "filename": "dog_mountain_sunrise.png",
            "description": "Dog on mountain (outdoor)"
        }
    ]
    
    # Generate images
    for i, case in enumerate(test_cases, 1):
        print(f"\nGenerating image {i}/4: {case['description']}")
        print(f"Prompt: {case['prompt']}")
        
        try:
            # Generate image
            image = pipe(
                case["prompt"],
                num_inference_steps=20,  # Faster generation
                guidance_scale=7.5,      # Default guidance
                height=512,
                width=512
            ).images[0]
            
            # Save image
            output_path = output_dir / case["filename"]
            image.save(output_path)
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Error generating image {i}: {e}")
            # Try to free memory and continue
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            continue
    
    print(f"\nAll test images generated successfully in {output_dir}")
    print("\nGenerated files:")
    for case in test_cases:
        print(f"  - {case['filename']}: {case['description']}")

if __name__ == "__main__":
    main()