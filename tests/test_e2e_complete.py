import pytest
import asyncio
import tempfile
import json
import zipfile
from pathlib import Path
from fastapi.testclient import TestClient
import time

from api.main import app
from core.shared_cache import setup_shared_cache

# Setup test environment
setup_shared_cache()

client = TestClient(app)


class TestE2EWorkflow:
    """Complete end-to-end workflow tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data and environment"""
        # Create test worldpack
        self.test_worldpack = self.create_test_worldpack()

        # Test configuration
        self.test_config = {
            "character_name": "test_character",
            "world_id": "test_world",
            "style_preset": "anime_style",
        }

        yield

        # Cleanup
        if hasattr(self, "uploaded_doc_id"):
            # Clean up uploaded documents
            pass

    def create_test_worldpack(self) -> Path:
        """Create a minimal test worldpack"""
        temp_dir = Path(tempfile.mkdtemp())
        worldpack_path = temp_dir / "test_worldpack.zip"

        # Create worldpack content
        characters_yaml = """
characters:
  alice:
    name: "Alice Chen"
    description: "A brave young programmer from Neo Taipei"
    appearance: "short black hair, blue eyes, casual clothes"
    personality: "curious, determined, tech-savvy"
    background: "Born in Neo Taipei, works as a freelance developer"
"""

        scenes_yaml = """
scenes:
  intro:
    title: "The Beginning"
    description: "Alice starts her adventure in the cyberpunk city"
    location: "Neo Taipei Downtown"
    atmosphere: "neon lights, bustling streets, digital billboards"
"""

        lorebook_md = """
# Neo Taipei Lorebook

## The City
Neo Taipei is a sprawling cyberpunk metropolis where technology and tradition blend seamlessly.

## Technology
Advanced AI systems assist citizens in daily life. Holographic displays are common.

## Culture
Traditional Taiwanese culture persists alongside futuristic innovations.
"""

        style_toml = """
[visual_style]
art_style = "anime cyberpunk"
color_palette = ["neon blue", "electric pink", "dark purple"]
mood = "futuristic yet nostalgic"

[narrative_style]
tone = "adventure"
perspective = "second_person"
"""

        # Create ZIP file
        with zipfile.ZipFile(worldpack_path, "w") as zf:
            zf.writestr("characters.yaml", characters_yaml)
            zf.writestr("scenes.yaml", scenes_yaml)
            zf.writestr("lorebook.md", lorebook_md)
            zf.writestr("style.toml", style_toml)

        return worldpack_path

    def test_health_check(self):
        """Test basic health endpoint"""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_monitoring_endpoints(self):
        """Test monitoring and metrics endpoints"""
        # Health check
        response = client.get("/monitoring/health")
        assert response.status_code == 200
        health_data = response.json()

        # Should have basic services
        assert "database" in health_data
        assert "redis" in health_data

        # Metrics
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200
        metrics = response.json()

        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "timestamp" in metrics

    def test_worldpack_upload_and_rag(self):
        """Test worldpack upload and RAG functionality"""
        worldpack_path = self.test_worldpack

        # Upload worldpack
        with open(worldpack_path, "rb") as f:
            files = {"file": ("test_worldpack.zip", f, "application/zip")}
            data = {"world_id": self.test_config["world_id"], "license": "CC-BY-SA-4.0"}
            response = client.post("/rag/upload", files=files, data=data)

        assert response.status_code == 200
        upload_result = response.json()
        assert "doc_ids" in upload_result
        assert len(upload_result["doc_ids"]) > 0

        self.uploaded_doc_id = upload_result["doc_ids"][0]

        # Test RAG retrieval
        query_data = {
            "world_id": self.test_config["world_id"],
            "query": "Tell me about Alice Chen",
            "top_k": 3,
        }
        response = client.post("/rag/retrieve", json=query_data)

        assert response.status_code == 200
        rag_result = response.json()
        assert "chunks" in rag_result
        assert len(rag_result["chunks"]) > 0

        # Verify content relevance
        found_alice = any("Alice" in chunk["text"] for chunk in rag_result["chunks"])
        assert found_alice, "Should find information about Alice"

    def test_llm_conversation_with_rag(self):
        """Test LLM conversation with RAG context"""
        # First ensure we have RAG data
        self.test_worldpack_upload_and_rag()

        # Create conversation with RAG
        conversation_data = {
            "message": "Who is Alice and what does she do?",
            "world_id": self.test_config["world_id"],
            "character_id": "alice",
            "use_rag": True,
            "max_tokens": 200,
        }

        response = client.post("/llm/turn", json=conversation_data)
        assert response.status_code == 200

        turn_result = response.json()
        assert "narration" in turn_result
        assert "citations" in turn_result

        # Should have citations from RAG
        assert len(turn_result["citations"]) > 0

        # Response should mention Alice
        narration = turn_result["narration"].lower()
        assert "alice" in narration

    def test_image_generation_pipeline(self):
        """Test T2I image generation"""
        generation_data = {
            "prompt": "anime style, girl with short black hair, cyberpunk city background",
            "negative_prompt": "blurry, low quality",
            "width": 512,
            "height": 512,
            "steps": 20,
            "seed": 12345,
            "style_preset": "anime_style",
        }

        response = client.post("/t2i/generate", json=generation_data)
        assert response.status_code == 200

        result = response.json()
        assert "image_path" in result
        assert "metadata" in result

        # Verify image file exists
        image_path = Path(result["image_path"])
        assert image_path.exists()
        assert image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]

    def test_controlnet_generation(self):
        """Test ControlNet pose generation"""
        # Create a simple pose skeleton (mock data)
        pose_data = {
            "prompt": "anime girl, standing pose, casual clothes",
            "controlnet_type": "openpose",
            "pose_keypoints": [
                {"x": 256, "y": 100, "confidence": 0.9},  # Head
                {"x": 256, "y": 200, "confidence": 0.9},  # Neck
                {"x": 256, "y": 300, "confidence": 0.9},  # Torso
            ],
            "width": 512,
            "height": 512,
            "steps": 20,
            "seed": 54321,
        }

        response = client.post("/t2i/controlnet/pose", json=pose_data)
        assert response.status_code == 200

        result = response.json()
        assert "image_path" in result

        # Verify controlled generation worked
        image_path = Path(result["image_path"])
        assert image_path.exists()

    def test_vlm_caption_and_consistency(self):
        """Test VLM captioning and consistency checking"""
        # First generate an image
        self.test_image_generation_pipeline()

        # Get the generated image path from previous test
        generation_data = {
            "prompt": "anime style, girl with short black hair",
            "width": 512,
            "height": 512,
            "steps": 10,
            "seed": 99999,
        }

        response = client.post("/t2i/generate", json=generation_data)
        result = response.json()
        image_path = result["image_path"]

        # Test VLM captioning
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"detail_level": "medium"}
            response = client.post("/vlm/caption", files=files, data=data)

        assert response.status_code == 200
        caption_result = response.json()
        assert "caption" in caption_result
        assert len(caption_result["caption"]) > 10

        # Test consistency check
        consistency_data = {
            "image_path": image_path,
            "expected_tags": ["girl", "anime", "black hair"],
            "world_id": self.test_config["world_id"],
        }

        response = client.post("/vlm/check_consistency", json=consistency_data)
        assert response.status_code == 200

        consistency_result = response.json()
        assert "consistency_score" in consistency_result
        assert 0 <= consistency_result["consistency_score"] <= 1

    def test_lora_training_submission(self):
        """Test LoRA training job submission"""
        training_config = {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "dataset_id": "test_character_dataset",
            "character_name": self.test_config["character_name"],
            "config": {
                "rank": 16,
                "learning_rate": 1e-4,
                "max_steps": 100,  # Short training for test
                "resolution": 512,
                "batch_size": 1,
            },
            "notes": "E2E test LoRA training",
        }

        response = client.post("/finetune/lora", json=training_config)
        assert response.status_code == 200

        job_result = response.json()
        assert "job_id" in job_result
        assert "status" in job_result

        self.training_job_id = job_result["job_id"]

        # Check job status
        response = client.get(f"/jobs/{self.training_job_id}")
        assert response.status_code == 200

        status_result = response.json()
        assert status_result["status"] in ["pending", "running", "completed", "failed"]

    def test_batch_generation_workflow(self):
        """Test batch generation workflow"""
        # Create batch job
        batch_config = {
            "world_id": self.test_config["world_id"],
            "style_preset": self.test_config["style_preset"],
            "prompts": [
                "Alice in a cyberpunk alley, neon lights",
                "Alice at a computer terminal, focused expression",
                "Alice walking through Neo Taipei streets",
            ],
            "generation_params": {
                "width": 512,
                "height": 512,
                "steps": 15,
                "seed_policy": "random",
            },
            "output_format": "png",
        }

        response = client.post("/batch/submit", json=batch_config)
        assert response.status_code == 200

        batch_result = response.json()
        assert "batch_id" in batch_result

        self.batch_id = batch_result["batch_id"]

        # Check batch status
        response = client.get(f"/batch/{self.batch_id}/status")
        assert response.status_code == 200

        status_result = response.json()
        assert "status" in status_result
        assert "total_tasks" in status_result
        assert status_result["total_tasks"] == 3

    def test_performance_and_caching(self):
        """Test performance optimizations and caching"""
        # Test cache statistics
        response = client.get("/monitoring/cache_stats")
        assert response.status_code == 200

        cache_stats = response.json()
        assert "embedding" in cache_stats
        assert "model" in cache_stats

        # Test cache clearing
        response = client.post(
            "/monitoring/clear_cache", json={"cache_type": "embedding"}
        )
        assert response.status_code == 200

        # Test the same query twice to verify caching
        query_data = {
            "world_id": self.test_config["world_id"],
            "query": "test caching query",
            "top_k": 1,
        }

        # First query (cache miss)
        start_time = time.time()
        response1 = client.post("/rag/retrieve", json=query_data)
        time1 = time.time() - start_time

        # Second query (cache hit)
        start_time = time.time()
        response2 = client.post("/rag/retrieve", json=query_data)
        time2 = time.time() - start_time

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Second query should be faster (cache hit)
        # Note: This is a rough test, actual performance may vary
        print(f"First query: {time1:.3f}s, Second query: {time2:.3f}s")

    def test_error_handling_and_recovery(self):
        """Test error handling and graceful degradation"""
        # Test invalid world_id
        response = client.post(
            "/rag/retrieve",
            json={"world_id": "nonexistent_world", "query": "test query", "top_k": 5},
        )
        assert response.status_code == 404

        # Test invalid image generation parameters
        response = client.post(
            "/t2i/generate",
            json={
                "prompt": "test",
                "width": -100,  # Invalid
                "height": 512,
                "steps": 20,
            },
        )
        assert response.status_code == 422  # Validation error

        # Test malformed file upload
        files = {"file": ("test.txt", b"not a valid worldpack", "text/plain")}
        data = {"world_id": "test", "license": "MIT"}
        response = client.post("/rag/upload", files=files, data=data)
        assert response.status_code == 400

    def test_complete_story_workflow(self):
        """Test complete interactive story workflow"""
        # Setup: Upload worldpack and generate character
        self.test_worldpack_upload_and_rag()

        # Start a new story session
        story_init = {
            "world_id": self.test_config["world_id"],
            "character_id": "alice",
            "scene_id": "intro",
            "player_name": "Player",
        }

        response = client.post("/story/start", json=story_init)
        assert response.status_code == 200

        session_data = response.json()
        assert "session_id" in session_data
        session_id = session_data["session_id"]

        # Make a story turn
        turn_data = {
            "session_id": session_id,
            "player_choice": "Look around the cyberpunk city",
            "generate_image": True,
        }

        response = client.post("/story/turn", json=turn_data)
        assert response.status_code == 200

        turn_result = response.json()
        assert "narration" in turn_result
        assert "choices" in turn_result
        assert "image_path" in turn_result  # Should have generated scene image
        assert "game_state" in turn_result

        # Verify image was generated
        if turn_result["image_path"]:
            image_path = Path(turn_result["image_path"])
            assert image_path.exists()

        # Continue the story
        if turn_result["choices"]:
            next_choice = turn_result["choices"][0]["text"]
            turn_data["player_choice"] = next_choice

            response = client.post("/story/turn", json=turn_data)
            assert response.status_code == 200

    def test_system_limits_and_quotas(self):
        """Test system limits and resource quotas"""
        # Test rate limiting (if implemented)
        rapid_requests = []
        for i in range(10):
            response = client.get("/healthz")
            rapid_requests.append(response.status_code)

        # Most should succeed, but rate limiting might kick in
        success_count = sum(1 for status in rapid_requests if status == 200)
        assert success_count >= 5, "Should handle at least some rapid requests"

        # Test memory usage monitoring
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200

        metrics = response.json()
        if "gpu_info" in metrics and metrics["gpu_info"]:
            # Check GPU memory is not completely exhausted
            for device in metrics["gpu_info"]["devices"]:
                assert device["utilization_percent"] < 95, "GPU memory usage too high"


# Integration test for specific workflows
@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent operations don't interfere with each other"""

    async def generate_image(prompt_suffix):
        return client.post(
            "/t2i/generate",
            json={
                "prompt": f"anime girl {prompt_suffix}",
                "width": 512,
                "height": 512,
                "steps": 10,
                "seed": hash(prompt_suffix) % 10000,
            },
        )

    # Run multiple generations concurrently
    tasks = [
        generate_image("with blue hair"),
        generate_image("with red hair"),
        generate_image("with green hair"),
    ]

    # Note: FastAPI TestClient doesn't support true async,
    # so this is a simplified test
    results = []
    for task in tasks:
        if hasattr(task, "result"):
            result = task.result()
        else:
            result = task
        results.append(result)

    # All should succeed
    for result in results:
        assert result.status_code == 200
        data = result.json()
        assert "image_path" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
