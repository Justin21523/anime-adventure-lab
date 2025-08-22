#!/usr/bin/env python3
"""
SagaForge Complete Workflow Demo
Demonstrates the full end-to-end pipeline: RAG → LLM → T2I → VLM → LoRA
"""

import os
import sys
import time
import requests
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Any

# Shared cache bootstrap
AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    Path(v).mkdir(parents=True, exist_ok=True)


class SagaForgeDemo:
    """Complete SagaForge workflow demonstration"""

    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
        self.session = requests.Session()
        self.session.timeout = 60

        # Demo data
        self.world_id = "demo_neo_taipei"
        self.character_id = "alice_demo"

        print(f"🎮 SagaForge Demo initialized")
        print(f"📡 API Base: {api_base}")
        print(f"🌍 World ID: {self.world_id}")
        print(f"👤 Character: {self.character_id}")

    def check_health(self) -> bool:
        """Check if SagaForge API is healthy"""
        print("\n🏥 Health Check")
        print("-" * 30)

        try:
            response = self.session.get(f"{self.api_base}/healthz")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Status: {data.get('status', 'unknown')}")
                print(f"📊 Version: {data.get('version', 'unknown')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

    def create_demo_worldpack(self) -> Path:
        """Create a demo worldpack for testing"""
        print("\n📦 Creating Demo Worldpack")
        print("-" * 30)

        temp_dir = Path(tempfile.mkdtemp())
        worldpack_path = temp_dir / "neo_taipei_demo.zip"

        # Demo world content
        characters_yaml = f"""
characters:
  {self.character_id}:
    name: "Alice Chen"
    description: "一位勇敢的年輕程式設計師，來自新台北市"
    appearance: "短黑髮，藍色眼睛，休閒服裝，背著筆記型電腦包"
    personality: "好奇心強，意志堅定，精通科技"
    background: |
      Alice 出生於新台北市，是一名自由工作的軟體開發者。
      她專精於 AI 系統開發，夢想創造能改變世界的技術。
      在賽博龐克的城市中，她尋找著真相與正義。
    relationships:
      mentor: "Dr. Lin - 她的 AI 研究導師"
      friend: "Bobby - 駭客夥伴"
    goals:
      - "揭發企業的 AI 陰謀"
      - "保護市民免受 AI 監控"
      - "開發開源 AI 工具"
"""

        scenes_yaml = """
scenes:
  downtown:
    title: "新台北市中心區"
    description: "霓虹燈閃爍的商業區，到處都是全息廣告和 AI 助手"
    location: "新台北市中心"
    atmosphere: "賽博龐克，未來感，科技與傳統交融"
    details: |
      高聳的摩天大樓直插雲霄，全息投影的廣告牌懸浮在空中。
      街道上行人匆忙，每個人都帶著 AR 眼鏡或神經接口設備。
      傳統的夜市小攤和高科技商店並肩而立。
    npcs:
      - "街頭小販 - 販賣改裝過的 AI 晶片"
      - "企業保安 - 監視可疑活動"
      - "地下駭客 - 在暗網中交易資訊"

  lab:
    title: "地下實驗室"
    description: "隱藏在城市地下的秘密 AI 研究設施"
    location: "新台北地下城"
    atmosphere: "神秘，高科技，略顯危險"
    details: |
      這裡是 Alice 和她的團隊進行 AI 研究的秘密基地。
      牆上掛滿了量子計算機和神經網路圖表。
      空氣中瀰漫著服務器散熱的嗡嗡聲。
"""

        lorebook_md = """
# 新台北市世界觀

## 城市背景

新台北市是 2089 年的賽博龐克大都會，科技與傳統文化交融。人工智慧已經深入日常生活的每個角落。

### 技術設定

- **AI 助手**: 每個市民都有個人 AI 陪伴，處理日常事務
- **神經接口**: 直接大腦-電腦介面技術普及
- **全息技術**: 立體投影廣告和娛樂隨處可見
- **量子網路**: 超高速資料傳輸網路

### 社會結構

- **企業財團**: 控制大部分 AI 技術的巨型企業
- **自由駭客**: 對抗企業監控的地下組織
- **普通市民**: 在高科技社會中努力生存的人們

### 衝突點

- **隱私 vs 便利**: AI 監控帶來便利但侵犯隱私
- **人類 vs 機器**: AI 逐漸取代人類工作
- **企業 vs 個人**: 大企業壟斷技術資源

## 重要組織

### TechCorp 企業集團
- 最大的 AI 技術公司
- 控制城市大部分基礎設施
- 被指控進行非法 AI 實驗

### 自由代碼聯盟
- Alice 所屬的駭客組織
- 致力於開源 AI 技術
- 對抗企業壟斷

## 關鍵道具

### 藍色徽章
- 神秘的量子加密設備
- 可以突破企業 AI 防火牆
- 只有在特定時間才能啟動

### 神經接口頭盔
- 增強人類認知能力
- 允許直接控制 AI 系統
- 存在被駭客入侵的風險
"""

        style_toml = """
[visual_style]
art_style = "cyberpunk anime"
color_palette = ["neon blue", "electric pink", "dark purple", "silver"]
lighting = "neon lighting, dramatic shadows"
mood = "futuristic, mysterious, tech-noir"

[narrative_style]
tone = "adventure with mystery elements"
perspective = "second_person"
language = "traditional_chinese"
pacing = "medium, with action scenes"

[generation_presets]
character_portrait = "anime style, cyberpunk character, detailed face, neon lighting"
scene_background = "cyberpunk cityscape, neon lights, futuristic architecture"
action_scene = "dynamic pose, dramatic lighting, motion blur effects"
"""

        # Create worldpack ZIP
        with zipfile.ZipFile(worldpack_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("characters.yaml", characters_yaml)
            zf.writestr("scenes.yaml", scenes_yaml)
            zf.writestr("lorebook.md", lorebook_md)
            zf.writestr("style.toml", style_toml)

        print(f"✅ Demo worldpack created: {worldpack_path}")
        print(f"📁 Size: {worldpack_path.stat().st_size / 1024:.1f} KB")

        return worldpack_path

    def upload_worldpack(self) -> bool:
        """Upload worldpack and test RAG ingestion"""
        print("\n📚 RAG Ingestion Test")
        print("-" * 30)

        worldpack_path = self.create_demo_worldpack()

        try:
            with open(worldpack_path, "rb") as f:
                files = {"file": ("neo_taipei_demo.zip", f, "application/zip")}
                data = {
                    "world_id": self.world_id,
                    "license": "CC-BY-SA-4.0",
                    "overwrite": "true",
                }

                print("📤 Uploading worldpack...")
                response = self.session.post(
                    f"{self.api_base}/rag/upload", files=files, data=data
                )

            if response.status_code == 200:
                result = response.json()
                doc_count = len(result.get("doc_ids", []))
                chunk_count = result.get("total_chunks", 0)

                print(f"✅ Upload successful")
                print(f"📄 Documents: {doc_count}")
                print(f"🧩 Chunks: {chunk_count}")

                return True
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"📋 Response: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False

    def test_rag_retrieval(self) -> Dict[str, Any]:
        """Test RAG retrieval with various queries"""
        print("\n🔍 RAG Retrieval Test")
        print("-" * 30)

        test_queries = [
            "告訴我關於 Alice Chen 的背景",
            "新台北市的科技設定是什麼？",
            "藍色徽章有什麼特殊功能？",
            "TechCorp 企業集團的角色是什麼？",
        ]

        retrieval_results = {}

        for query in test_queries:
            print(f"❓ Query: {query}")

            try:
                payload = {
                    "world_id": self.world_id,
                    "query": query,
                    "top_k": 3,
                    "rerank": True,
                }

                response = self.session.post(
                    f"{self.api_base}/rag/retrieve", json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    chunks = result.get("chunks", [])

                    print(f"✅ Found {len(chunks)} relevant chunks")
                    if chunks:
                        top_chunk = chunks[0]
                        print(f"📝 Top result: {top_chunk['text'][:100]}...")
                        print(f"🎯 Score: {top_chunk.get('score', 0):.3f}")

                    retrieval_results[query] = result
                else:
                    print(f"❌ Retrieval failed: {response.status_code}")

            except Exception as e:
                print(f"❌ Retrieval error: {e}")

            print()

        return retrieval_results

    def test_llm_conversation(self) -> Dict[str, Any]:
        """Test LLM conversation with RAG context"""
        print("\n💬 LLM Conversation Test")
        print("-" * 30)

        conversation_scenarios = [
            {
                "message": "我想了解 Alice 的故事，她現在在哪裡？",
                "context": "story_introduction",
            },
            {"message": "我想探索新台北市中心區", "context": "scene_exploration"},
            {"message": "告訴我更多關於藍色徽章的資訊", "context": "item_inquiry"},
        ]

        conversation_results = {}

        for i, scenario in enumerate(conversation_scenarios, 1):
            print(f"💭 Scenario {i}: {scenario['context']}")
            print(f"👤 Player: {scenario['message']}")

            try:
                payload = {
                    "message": scenario["message"],
                    "world_id": self.world_id,
                    "character_id": self.character_id,
                    "use_rag": True,
                    "max_tokens": 300,
                    "temperature": 0.7,
                }

                response = self.session.post(f"{self.api_base}/llm/turn", json=payload)

                if response.status_code == 200:
                    result = response.json()

                    print(f"🤖 Narration: {result.get('narration', '')}")

                    choices = result.get("choices", [])
                    if choices:
                        print(f"🎮 Choices:")
                        for j, choice in enumerate(choices[:3], 1):
                            print(f"   {j}. {choice.get('text', '')}")

                    citations = result.get("citations", [])
                    print(f"📚 Citations: {len(citations)} sources referenced")

                    conversation_results[scenario["context"]] = result

                else:
                    print(f"❌ Conversation failed: {response.status_code}")

            except Exception as e:
                print(f"❌ Conversation error: {e}")

            print()

        return conversation_results

    def test_image_generation(self) -> Dict[str, Any]:
        """Test T2I image generation"""
        print("\n🎨 Image Generation Test")
        print("-" * 30)

        generation_scenarios = [
            {
                "name": "character_portrait",
                "prompt": "anime style portrait of Alice Chen, short black hair, blue eyes, casual clothes, cyberpunk background",
                "style": "anime_portrait",
            },
            {
                "name": "scene_background",
                "prompt": "cyberpunk cityscape, neon lights, futuristic buildings, new taipei city, night scene",
                "style": "cyberpunk_scene",
            },
            {
                "name": "action_scene",
                "prompt": "anime girl hacking computer, dramatic lighting, neon glow, focused expression",
                "style": "action_anime",
            },
        ]

        generation_results = {}

        for scenario in generation_scenarios:
            print(f"🖼️ Generating: {scenario['name']}")
            print(f"📝 Prompt: {scenario['prompt']}")

            try:
                payload = {
                    "prompt": scenario["prompt"],
                    "negative_prompt": "blurry, low quality, distorted, extra limbs",
                    "width": 768,
                    "height": 768,
                    "steps": 25,
                    "cfg_scale": 7.5,
                    "seed": 42,  # Fixed seed for reproducibility
                    "style_preset": scenario.get("style"),
                }

                start_time = time.time()
                response = self.session.post(
                    f"{self.api_base}/t2i/generate", json=payload
                )
                generation_time = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    image_path = result.get("image_path")

                    print(f"✅ Generated successfully in {generation_time:.1f}s")
                    print(f"💾 Image saved: {image_path}")

                    # Verify image file exists
                    if image_path and Path(image_path).exists():
                        file_size = Path(image_path).stat().st_size / 1024
                        print(f"📏 File size: {file_size:.1f} KB")

                    generation_results[scenario["name"]] = result

                else:
                    print(f"❌ Generation failed: {response.status_code}")
                    print(f"📋 Response: {response.text}")

            except Exception as e:
                print(f"❌ Generation error: {e}")

            print()

        return generation_results

    def test_controlnet_pose(self) -> Dict[str, Any]:
        """Test ControlNet pose generation"""
        print("\n🕺 ControlNet Pose Test")
        print("-" * 30)

        # Simple pose keypoints (standing pose)
        pose_keypoints = [
            {"x": 384, "y": 150, "confidence": 0.9, "name": "head"},
            {"x": 384, "y": 200, "confidence": 0.9, "name": "neck"},
            {"x": 384, "y": 350, "confidence": 0.9, "name": "torso"},
            {"x": 300, "y": 280, "confidence": 0.8, "name": "left_shoulder"},
            {"x": 468, "y": 280, "confidence": 0.8, "name": "right_shoulder"},
            {"x": 250, "y": 400, "confidence": 0.8, "name": "left_hand"},
            {"x": 518, "y": 400, "confidence": 0.8, "name": "right_hand"},
            {"x": 350, "y": 500, "confidence": 0.9, "name": "left_hip"},
            {"x": 418, "y": 500, "confidence": 0.9, "name": "right_hip"},
            {"x": 350, "y": 650, "confidence": 0.9, "name": "left_knee"},
            {"x": 418, "y": 650, "confidence": 0.9, "name": "right_knee"},
            {"x": 350, "y": 750, "confidence": 0.9, "name": "left_foot"},
            {"x": 418, "y": 750, "confidence": 0.9, "name": "right_foot"},
        ]

        try:
            payload = {
                "prompt": "anime style Alice Chen, confident pose, cyberpunk outfit",
                "negative_prompt": "blurry, distorted anatomy",
                "controlnet_type": "openpose",
                "pose_keypoints": pose_keypoints,
                "width": 768,
                "height": 768,
                "steps": 20,
                "cfg_scale": 7.0,
                "seed": 123,
            }

            print("🎯 Generating pose-controlled image...")
            start_time = time.time()

            response = self.session.post(
                f"{self.api_base}/t2i/controlnet/pose", json=payload
            )

            generation_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                image_path = result.get("image_path")

                print(f"✅ ControlNet generation successful in {generation_time:.1f}s")
                print(f"💾 Controlled image saved: {image_path}")

                return result
            else:
                print(f"❌ ControlNet generation failed: {response.status_code}")
                return {}

        except Exception as e:
            print(f"❌ ControlNet error: {e}")
            return {}

    def test_vlm_analysis(self, image_paths: list) -> Dict[str, Any]:
        """Test VLM image analysis"""
        print("\n👁️ VLM Analysis Test")
        print("-" * 30)

        vlm_results = {}

        for i, image_path in enumerate(image_paths[:2]):  # Limit to 2 images
            if not image_path or not Path(image_path).exists():
                continue

            print(f"🔍 Analyzing image {i+1}: {Path(image_path).name}")

            try:
                # Caption generation
                with open(image_path, "rb") as f:
                    files = {"image": f}
                    data = {"detail_level": "medium"}

                    response = self.session.post(
                        f"{self.api_base}/vlm/caption", files=files, data=data
                    )

                if response.status_code == 200:
                    caption_result = response.json()
                    caption = caption_result.get("caption", "")

                    print(f"📝 Caption: {caption}")

                    # Consistency check
                    consistency_payload = {
                        "image_path": image_path,
                        "expected_tags": ["alice", "anime", "cyberpunk", "girl"],
                        "world_id": self.world_id,
                    }

                    response = self.session.post(
                        f"{self.api_base}/vlm/check_consistency",
                        json=consistency_payload,
                    )

                    if response.status_code == 200:
                        consistency_result = response.json()
                        score = consistency_result.get("consistency_score", 0)

                        print(f"🎯 Consistency score: {score:.3f}")

                        vlm_results[f"image_{i+1}"] = {
                            "caption": caption,
                            "consistency_score": score,
                        }

                else:
                    print(f"❌ VLM analysis failed: {response.status_code}")

            except Exception as e:
                print(f"❌ VLM error: {e}")

            print()

        return vlm_results

    def test_batch_generation(self) -> Dict[str, Any]:
        """Test batch generation workflow"""
        print("\n⚡ Batch Generation Test")
        print("-" * 30)

        # Create batch job
        batch_prompts = [
            "Alice in cyberpunk alley, neon signs, rain",
            "Alice at computer terminal, focused, hacking",
            "Alice meeting with underground hackers, secret meeting",
        ]

        try:
            payload = {
                "world_id": self.world_id,
                "job_type": "image_generation",
                "config": {
                    "prompts": batch_prompts,
                    "style_preset": "cyberpunk_anime",
                    "generation_params": {
                        "width": 512,
                        "height": 512,
                        "steps": 15,
                        "seed_policy": "random",
                    },
                },
            }

            print("📤 Submitting batch job...")
            response = self.session.post(f"{self.api_base}/batch/submit", json=payload)

            if response.status_code == 200:
                result = response.json()
                batch_id = result.get("batch_id")

                print(f"✅ Batch job submitted: {batch_id}")
                print(f"📊 Total tasks: {result.get('total_tasks', 0)}")

                # Monitor progress
                return self.monitor_batch_progress(batch_id)
            else:
                print(f"❌ Batch submission failed: {response.status_code}")
                return {}

        except Exception as e:
            print(f"❌ Batch error: {e}")
            return {}

    def monitor_batch_progress(self, batch_id: str) -> Dict[str, Any]:
        """Monitor batch job progress"""
        print(f"⏳ Monitoring batch progress: {batch_id}")

        max_wait_time = 300  # 5 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                response = self.session.get(f"{self.api_base}/batch/{batch_id}/status")

                if response.status_code == 200:
                    status = response.json()

                    progress = status.get("progress", {})
                    completed = progress.get("completed", 0)
                    total = progress.get("total", 0)
                    failed = progress.get("failed", 0)

                    print(f"📈 Progress: {completed}/{total} (failed: {failed})")

                    if status.get("status") == "completed":
                        print("🎉 Batch job completed!")
                        return status
                    elif status.get("status") == "failed":
                        print("💥 Batch job failed!")
                        return status

                    time.sleep(10)  # Wait 10 seconds before next check
                else:
                    print(f"❌ Status check failed: {response.status_code}")
                    break

            except Exception as e:
                print(f"❌ Status check error: {e}")
                break

        print("⏰ Batch monitoring timeout")
        return {}

    def test_lora_training_submission(self) -> Dict[str, Any]:
        """Test LoRA training job submission"""
        print("\n🎓 LoRA Training Test")
        print("-" * 30)

        try:
            training_config = {
                "base_model": "runwayml/stable-diffusion-v1-5",
                "character_name": self.character_id,
                "dataset_config": {
                    "data_source": "demo_dataset",
                    "character_tags": ["alice_chen", "cyberpunk", "programmer"],
                    "min_images": 10,
                },
                "training_config": {
                    "rank": 16,
                    "learning_rate": 1e-4,
                    "max_steps": 500,  # Short training for demo
                    "batch_size": 1,
                    "resolution": 512,
                    "save_every": 100,
                },
                "output_name": f"{self.character_id}_demo_v1",
                "notes": "Demo LoRA training run",
            }

            print("🚀 Submitting LoRA training job...")
            response = self.session.post(
                f"{self.api_base}/finetune/lora", json=training_config
            )

            if response.status_code == 200:
                result = response.json()
                job_id = result.get("job_id")

                print(f"✅ Training job submitted: {job_id}")
                print(
                    f"⏱️ Estimated time: {result.get('estimated_time', 0)/60:.1f} minutes"
                )

                return result
            else:
                print(f"❌ Training submission failed: {response.status_code}")
                return {}

        except Exception as e:
            print(f"❌ Training submission error: {e}")
            return {}

    def test_monitoring_endpoints(self) -> Dict[str, Any]:
        """Test system monitoring endpoints"""
        print("\n📊 System Monitoring Test")
        print("-" * 30)

        monitoring_results = {}

        # Test health endpoint
        try:
            response = self.session.get(f"{self.api_base}/monitoring/health")
            if response.status_code == 200:
                health_data = response.json()

                print("🏥 Service Health:")
                for service, status in health_data.items():
                    status_icon = "✅" if status.get("status") == "healthy" else "❌"
                    latency = status.get("latency_ms", 0)
                    print(
                        f"   {status_icon} {service}: {status.get('status')} ({latency:.1f}ms)"
                    )

                monitoring_results["health"] = health_data

        except Exception as e:
            print(f"❌ Health monitoring error: {e}")

        # Test metrics endpoint
        try:
            response = self.session.get(f"{self.api_base}/monitoring/metrics")
            if response.status_code == 200:
                metrics = response.json()

                print("\n📈 System Metrics:")
                print(f"   💻 CPU: {metrics.get('cpu_percent', 0):.1f}%")
                print(f"   🧠 Memory: {metrics.get('memory_percent', 0):.1f}%")
                print(f"   💾 Disk: {metrics.get('disk_percent', 0):.1f}%")

                gpu_info = metrics.get("gpu_info")
                if gpu_info and gpu_info["devices"]:
                    gpu = gpu_info["devices"][0]
                    print(
                        f"   🎮 GPU: {gpu.get('utilization_percent', 0):.1f}% ({gpu.get('name', 'Unknown')})"
                    )

                print(f"   👷 Active workers: {metrics.get('active_workers', 0)}")
                print(f"   📋 Queue depth: {metrics.get('queue_depth', 0)}")

                monitoring_results["metrics"] = metrics

        except Exception as e:
            print(f"❌ Metrics monitoring error: {e}")

        return monitoring_results

    def generate_demo_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive demo report"""
        report = f"""
# SagaForge Demo Report
**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**World ID:** {self.world_id}
**Character:** {self.character_id}

## 🎯 Demo Summary

"""

        # Check which tests passed
        passed_tests = []
        failed_tests = []

        for test_name, result in results.items():
            if result and isinstance(result, dict) and result.get("success", True):
                passed_tests.append(test_name)
            else:
                failed_tests.append(test_name)

        report += f"✅ **Passed Tests:** {len(passed_tests)}\n"
        report += f"❌ **Failed Tests:** {len(failed_tests)}\n\n"

        # Detailed results
        if "rag_retrieval" in results:
            report += "## 📚 RAG Retrieval Results\n"
            rag_data = results["rag_retrieval"]
            report += f"- Tested {len(rag_data)} queries\n"
            report += f"- Average chunks retrieved: {sum(len(r.get('chunks', [])) for r in rag_data.values()) / len(rag_data):.1f}\n\n"

        if "image_generation" in results:
            report += "## 🎨 Image Generation Results\n"
            img_data = results["image_generation"]
            report += f"- Generated {len(img_data)} images\n"
            for name, data in img_data.items():
                if data.get("image_path"):
                    report += f"- {name}: ✅ Success\n"
            report += "\n"

        if "vlm_analysis" in results:
            report += "## 👁️ VLM Analysis Results\n"
            vlm_data = results["vlm_analysis"]
            avg_score = (
                sum(data.get("consistency_score", 0) for data in vlm_data.values())
                / len(vlm_data)
                if vlm_data
                else 0
            )
            report += f"- Analyzed {len(vlm_data)} images\n"
            report += f"- Average consistency score: {avg_score:.3f}\n\n"

        if "monitoring" in results:
            report += "## 📊 System Performance\n"
            monitoring = results["monitoring"]
            metrics = monitoring.get("metrics", {})
            report += f"- CPU Usage: {metrics.get('cpu_percent', 0):.1f}%\n"
            report += f"- Memory Usage: {metrics.get('memory_percent', 0):.1f}%\n"
            report += f"- Active Workers: {metrics.get('active_workers', 0)}\n\n"

        report += "## 🔧 Recommendations\n\n"

        if failed_tests:
            report += "### Issues Found:\n"
            for test in failed_tests:
                report += f"- {test}: Please check logs for details\n"

        if len(passed_tests) == len(results):
            report += "🎉 All tests passed! SagaForge is working perfectly.\n"

        report += "\n---\n*Generated by SagaForge Demo Script*"

        return report

    def run_complete_demo(self) -> None:
        """Run the complete SagaForge demonstration workflow"""
        print("🚀 Starting SagaForge Complete Demo")
        print("=" * 50)

        start_time = time.time()
        demo_results = {}

        # Step 1: Health Check
        if not self.check_health():
            print("💥 Demo aborted: API not healthy")
            return

        # Step 2: RAG Pipeline Test
        if self.upload_worldpack():
            demo_results["rag_upload"] = {"success": True}
            demo_results["rag_retrieval"] = self.test_rag_retrieval()
        else:
            demo_results["rag_upload"] = {"success": False}

        # Step 3: LLM Conversation Test
        demo_results["llm_conversation"] = self.test_llm_conversation()

        # Step 4: Image Generation Test
        demo_results["image_generation"] = self.test_image_generation()

        # Step 5: ControlNet Test
        demo_results["controlnet"] = self.test_controlnet_pose()

        # Step 6: VLM Analysis Test
        image_paths = []
        if demo_results["image_generation"]:
            image_paths = [
                data.get("image_path")
                for data in demo_results["image_generation"].values()
            ]
        demo_results["vlm_analysis"] = self.test_vlm_analysis(image_paths)

        # Step 7: Batch Generation Test
        demo_results["batch_generation"] = self.test_batch_generation()

        # Step 8: LoRA Training Test (submission only)
        demo_results["lora_training"] = self.test_lora_training_submission()

        # Step 9: System Monitoring Test
        demo_results["monitoring"] = self.test_monitoring_endpoints()

        # Generate report
        total_time = time.time() - start_time

        print("\n" + "=" * 50)
        print("🎉 Demo Complete!")
        print("=" * 50)
        print(f"⏱️ Total time: {total_time:.1f} seconds")

        # Save report
        report = self.generate_demo_report(demo_results)
        report_path = Path(f"sagaforge_demo_report_{int(time.time())}.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"📄 Demo report saved: {report_path}")
        print("\n" + report)


def main():
    """Main demo function"""
    import argparse

    parser = argparse.ArgumentParser(description="SagaForge Complete Workflow Demo")
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick demo (skip batch and training tests)",
    )

    args = parser.parse_args()

    demo = SagaForgeDemo(api_base=args.api_base)

    if args.quick:
        print("🏃 Running quick demo...")
        # Run essential tests only
        demo.check_health()
        demo.upload_worldpack()
        demo.test_rag_retrieval()
        demo.test_llm_conversation()
        demo.test_image_generation()
    else:
        # Run complete demo
        demo.run_complete_demo()


if __name__ == "__main__":
    main()
