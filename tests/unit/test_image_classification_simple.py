"""Simple test for image classification without GCS dependencies."""

import base64
import pytest
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


class TestImageClassification:
    """Test image classification using CLIP directly."""
    
    @pytest.fixture
    def clip_model(self):
        """Load CLIP model for testing."""
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    
    @pytest.fixture
    def test_images(self):
        """Load test images."""
        images_dir = Path(__file__).parent.parent / "data" / "images"
        
        images = {}
        image_files = [
            "cat_office_typing.png",
            "dog_office_typing.png", 
            "cat_mountain_sunrise.png",
            "dog_mountain_sunrise.png"
        ]
        
        for filename in image_files:
            image_path = images_dir / filename
            if image_path.exists():
                image = Image.open(image_path)
                images[filename] = image
            else:
                pytest.skip(f"Test image {filename} not found")
        
        return images
    
    def test_cat_vs_dog_classification(self, clip_model, test_images):
        """Test that CLIP can distinguish between cats and dogs."""
        model, processor = clip_model
        
        # Test descriptions
        texts = ["a cat", "a dog"]
        
        # Test cat images should have higher similarity to "a cat"
        for image_name, image in test_images.items():
            if "cat" in image_name:
                inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # "a cat" should have higher probability than "a dog"
                cat_prob = probs[0][0].item()
                dog_prob = probs[0][1].item()
                
                assert cat_prob > dog_prob, f"Cat image {image_name}: cat={cat_prob:.3f}, dog={dog_prob:.3f}"
                print(f"✓ {image_name}: cat={cat_prob:.3f}, dog={dog_prob:.3f}")
        
        # Test dog images should have higher similarity to "a dog"
        for image_name, image in test_images.items():
            if "dog" in image_name:
                inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # "a dog" should have higher probability than "a cat"
                cat_prob = probs[0][0].item()
                dog_prob = probs[0][1].item()
                
                assert dog_prob > cat_prob, f"Dog image {image_name}: cat={cat_prob:.3f}, dog={dog_prob:.3f}"
                print(f"✓ {image_name}: cat={cat_prob:.3f}, dog={dog_prob:.3f}")
    
    def test_indoor_vs_outdoor_classification(self, clip_model, test_images):
        """Test that CLIP can distinguish between indoor and outdoor scenes."""
        model, processor = clip_model
        
        # Test descriptions
        texts = ["indoor office scene", "outdoor mountain scene"]
        
        # Test office images should have higher similarity to "indoor office scene"
        for image_name, image in test_images.items():
            if "office" in image_name:
                inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # Indoor should have higher probability than outdoor
                indoor_prob = probs[0][0].item()
                outdoor_prob = probs[0][1].item()
                
                assert indoor_prob > outdoor_prob, f"Office image {image_name}: indoor={indoor_prob:.3f}, outdoor={outdoor_prob:.3f}"
                print(f"✓ {image_name}: indoor={indoor_prob:.3f}, outdoor={outdoor_prob:.3f}")
        
        # Test mountain images should have higher similarity to "outdoor mountain scene"
        for image_name, image in test_images.items():
            if "mountain" in image_name:
                inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # Outdoor should have higher probability than indoor
                indoor_prob = probs[0][0].item()
                outdoor_prob = probs[0][1].item()
                
                assert outdoor_prob > indoor_prob, f"Mountain image {image_name}: indoor={indoor_prob:.3f}, outdoor={outdoor_prob:.3f}"
                print(f"✓ {image_name}: indoor={indoor_prob:.3f}, outdoor={outdoor_prob:.3f}")
    
    def test_combined_classification(self, clip_model, test_images):
        """Test classification with specific descriptions."""
        model, processor = clip_model
        
        # Test with specific descriptions
        descriptions = [
            "a cat in an office",
            "a dog in an office", 
            "a cat on a mountain",
            "a dog on a mountain"
        ]
        
        # Test each image against all descriptions
        for image_name, image in test_images.items():
            inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Find the highest probability description
            max_idx = probs[0].argmax().item()
            predicted_description = descriptions[max_idx]
            confidence = probs[0][max_idx].item()
            
            print(f"✓ {image_name}: predicted '{predicted_description}' (confidence: {confidence:.3f})")
            
            # Verify the prediction makes sense
            if "cat_office" in image_name:
                assert "cat" in predicted_description and "office" in predicted_description, \
                    f"Expected cat+office for {image_name}, got: {predicted_description}"
            elif "dog_office" in image_name:
                assert "dog" in predicted_description and "office" in predicted_description, \
                    f"Expected dog+office for {image_name}, got: {predicted_description}"
            elif "cat_mountain" in image_name:
                assert "cat" in predicted_description and "mountain" in predicted_description, \
                    f"Expected cat+mountain for {image_name}, got: {predicted_description}"
            elif "dog_mountain" in image_name:
                assert "dog" in predicted_description and "mountain" in predicted_description, \
                    f"Expected dog+mountain for {image_name}, got: {predicted_description}"