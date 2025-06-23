"""Test CLIP model classification capabilities."""

import base64
import pytest
from pathlib import Path
from PIL import Image
import io

from services.clip.adapter import CLIPAdapter


class TestCLIPClassification:
    """Test CLIP classification for cats vs dogs and indoor vs outdoor scenes."""
    
    @pytest.fixture
    def clip_adapter(self, tmp_path):
        """Initialize CLIP adapter for testing."""
        # Set cache directory to a temporary path for testing
        import os
        os.environ["CACHE_DIR"] = str(tmp_path / "cache")
        
        # Reset the GCS loader singleton to pick up new environment
        import services.common.gcs_loader as gcs_loader
        gcs_loader._gcs_loader = None
        
        # Use a smaller model for faster testing
        adapter = CLIPAdapter(
            gcs_path=None,  # Use HuggingFace Hub
            device="cpu"    # Use CPU for testing
        )
        return adapter
    
    @pytest.fixture
    def test_images(self):
        """Load test images as base64 encoded strings."""
        images_dir = Path(__file__).parent.parent / "data" / "images"
        
        images = {}
        image_files = [
            "cat_office_typing.jpg",
            "dog_office_typing.jpg", 
            "cat_mountain_sunrise.jpg",
            "dog_mountain_sunrise.jpg"
        ]
        
        for filename in image_files:
            image_path = images_dir / filename
            if image_path.exists():
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    base64_str = base64.b64encode(image_data).decode()
                    images[filename] = f"data:image/jpeg;base64,{base64_str}"
            else:
                pytest.skip(f"Test image {filename} not found")
        
        return images
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, clip_adapter):
        """Test that CLIP adapter initializes correctly."""
        assert clip_adapter is not None
        assert clip_adapter.device == "cpu"
        assert not clip_adapter.is_loaded
    
    @pytest.mark.asyncio
    async def test_adapter_loading(self, clip_adapter):
        """Test that CLIP adapter loads the model."""
        await clip_adapter.load_model()
        assert clip_adapter.is_loaded
        assert clip_adapter.model is not None
        assert clip_adapter.processor is not None
    
    @pytest.mark.asyncio
    async def test_cat_vs_dog_classification(self, clip_adapter, test_images):
        """Test that CLIP can distinguish between cats and dogs."""
        await clip_adapter.load_model()
        
        # Test cat images
        cat_office_result = await clip_adapter.predict({
            "task": "similarity",
            "texts": ["a cat", "a dog"],
            "images": [test_images["cat_office_typing.jpg"]]
        })
        
        cat_mountain_result = await clip_adapter.predict({
            "task": "similarity",
            "texts": ["a cat", "a dog"], 
            "images": [test_images["cat_mountain_sunrise.jpg"]]
        })
        
        # Test dog images
        dog_office_result = await clip_adapter.predict({
            "task": "similarity",
            "texts": ["a cat", "a dog"],
            "images": [test_images["dog_office_typing.jpg"]]
        })
        
        dog_mountain_result = await clip_adapter.predict({
            "task": "similarity",
            "texts": ["a cat", "a dog"],
            "images": [test_images["dog_mountain_sunrise.jpg"]]
        })
        
        # Verify results structure
        assert "similarity_matrix" in cat_office_result
        assert "similarity_matrix" in cat_mountain_result
        assert "similarity_matrix" in dog_office_result
        assert "similarity_matrix" in dog_mountain_result
        
        # For cat images, "a cat" should have higher similarity than "a dog"
        cat_office_sim = cat_office_result["similarity_matrix"][0]  # First image
        print(f"\nCat office typing - Similarities: cat={cat_office_sim[0]:.4f}, dog={cat_office_sim[1]:.4f}")
        assert cat_office_sim[0] > cat_office_sim[1], f"Cat office: cat={cat_office_sim[0]}, dog={cat_office_sim[1]}"
        
        cat_mountain_sim = cat_mountain_result["similarity_matrix"][0]
        print(f"Cat mountain sunrise - Similarities: cat={cat_mountain_sim[0]:.4f}, dog={cat_mountain_sim[1]:.4f}")
        assert cat_mountain_sim[0] > cat_mountain_sim[1], f"Cat mountain: cat={cat_mountain_sim[0]}, dog={cat_mountain_sim[1]}"
        
        # For dog images, "a dog" should have higher similarity than "a cat" 
        dog_office_sim = dog_office_result["similarity_matrix"][0]
        print(f"Dog office typing - Similarities: cat={dog_office_sim[0]:.4f}, dog={dog_office_sim[1]:.4f}")
        assert dog_office_sim[1] > dog_office_sim[0], f"Dog office: cat={dog_office_sim[0]}, dog={dog_office_sim[1]}"
        
        dog_mountain_sim = dog_mountain_result["similarity_matrix"][0]
        print(f"Dog mountain sunrise - Similarities: cat={dog_mountain_sim[0]:.4f}, dog={dog_mountain_sim[1]:.4f}")
        assert dog_mountain_sim[1] > dog_mountain_sim[0], f"Dog mountain: cat={dog_mountain_sim[0]}, dog={dog_mountain_sim[1]}"
    
    @pytest.mark.asyncio
    async def test_indoor_vs_outdoor_classification(self, clip_adapter, test_images):
        """Test that CLIP can distinguish between indoor and outdoor scenes."""
        await clip_adapter.load_model()
        
        # Test indoor images (office scenes)
        cat_office_result = await clip_adapter.predict({
            "task": "similarity",
            "texts": ["indoor office scene", "outdoor mountain scene"],
            "images": [test_images["cat_office_typing.jpg"]]
        })
        
        dog_office_result = await clip_adapter.predict({
            "task": "similarity",
            "texts": ["indoor office scene", "outdoor mountain scene"],
            "images": [test_images["dog_office_typing.jpg"]]
        })
        
        # Test outdoor images (mountain scenes)
        cat_mountain_result = await clip_adapter.predict({
            "task": "similarity",
            "texts": ["indoor office scene", "outdoor mountain scene"],
            "images": [test_images["cat_mountain_sunrise.jpg"]]
        })
        
        dog_mountain_result = await clip_adapter.predict({
            "task": "similarity",
            "texts": ["indoor office scene", "outdoor mountain scene"],
            "images": [test_images["dog_mountain_sunrise.jpg"]]
        })
        
        # For office images, "indoor office scene" should have higher similarity
        cat_office_sim = cat_office_result["similarity_matrix"][0]
        print(f"\nCat office - Scene similarities: indoor={cat_office_sim[0]:.4f}, outdoor={cat_office_sim[1]:.4f}")
        assert cat_office_sim[0] > cat_office_sim[1], f"Cat office: indoor={cat_office_sim[0]}, outdoor={cat_office_sim[1]}"
        
        dog_office_sim = dog_office_result["similarity_matrix"][0]
        print(f"Dog office - Scene similarities: indoor={dog_office_sim[0]:.4f}, outdoor={dog_office_sim[1]:.4f}")
        assert dog_office_sim[0] > dog_office_sim[1], f"Dog office: indoor={dog_office_sim[0]}, outdoor={dog_office_sim[1]}"
        
        # For mountain images, "outdoor mountain scene" should have higher similarity
        cat_mountain_sim = cat_mountain_result["similarity_matrix"][0]
        print(f"Cat mountain - Scene similarities: indoor={cat_mountain_sim[0]:.4f}, outdoor={cat_mountain_sim[1]:.4f}")
        assert cat_mountain_sim[1] > cat_mountain_sim[0], f"Cat mountain: indoor={cat_mountain_sim[0]}, outdoor={cat_mountain_sim[1]}"
        
        dog_mountain_sim = dog_mountain_result["similarity_matrix"][0]
        print(f"Dog mountain - Scene similarities: indoor={dog_mountain_sim[0]:.4f}, outdoor={dog_mountain_sim[1]:.4f}")
        assert dog_mountain_sim[1] > dog_mountain_sim[0], f"Dog mountain: indoor={dog_mountain_sim[0]}, outdoor={dog_mountain_sim[1]}"
    
    @pytest.mark.asyncio
    async def test_combined_classification(self, clip_adapter, test_images):
        """Test classification with multiple descriptive texts."""
        await clip_adapter.load_model()
        
        # Test with more specific descriptions
        descriptions = [
            "a cat in an office",
            "a dog in an office", 
            "a cat on a mountain",
            "a dog on a mountain"
        ]
        
        # Test each image against all descriptions
        for image_name, image_data in test_images.items():
            result = await clip_adapter.predict({
                "task": "similarity",
                "texts": descriptions,
                "images": [image_data]
            })
            
            similarities = result["similarity_matrix"][0]
            max_idx = similarities.index(max(similarities))
            predicted_description = descriptions[max_idx]
            
            # Print similarity scores for inspection
            print(f"\n{image_name} - Similarities:")
            for i, desc in enumerate(descriptions):
                print(f"  {desc}: {similarities[i]:.4f}")
            print(f"  Predicted: {predicted_description}")
            
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
    
    @pytest.mark.asyncio
    async def test_text_encoding_consistency(self, clip_adapter):
        """Test that text encoding produces consistent results."""
        await clip_adapter.load_model()
        
        # Test same text multiple times
        texts = ["a cat", "a dog", "indoor scene", "outdoor scene"]
        
        result1 = await clip_adapter.predict({"texts": texts})
        result2 = await clip_adapter.predict({"texts": texts})
        
        # Results should be identical for same inputs
        embeddings1 = result1["embeddings"]
        embeddings2 = result2["embeddings"]
        
        assert len(embeddings1) == len(embeddings2)
        
        # Check that embeddings are very similar (allowing for minor floating point differences)
        for i, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
            assert len(emb1) == len(emb2), f"Embedding {i} length mismatch"
            # Check first few values for approximate equality
            for j in range(min(10, len(emb1))):
                assert abs(emb1[j] - emb2[j]) < 1e-6, f"Embedding {i} value {j} differs significantly"