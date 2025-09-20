#!/usr/bin/env python3
# scripts/health_check.py
"""
系統健康檢查腳本
在正式測試前驗證環境配置
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import torch


# Colors
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def check_python_version():
    """檢查 Python 版本"""
    version = sys.version_info
    print(f"🐍 Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 9:
        print(f"   {Colors.GREEN}✅ Python版本符合要求 (>=3.9){Colors.ENDC}")
        return True
    else:
        print(f"   {Colors.RED}❌ Python版本過舊，需要 >=3.9{Colors.ENDC}")
        return False


def check_dependencies():
    """檢查關鍵依賴"""
    dependencies = [
        ("fastapi", "FastAPI"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("pytest", "Pytest"),
        ("numpy", "NumPy"),
        ("faiss", "FAISS (optional)", True),
    ]

    print("📦 依賴檢查:")
    all_ok = True

    for import_name, display_name, *optional in dependencies:
        is_optional = len(optional) > 0 and optional[0]
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"   {Colors.GREEN}✅ {display_name}: {version}{Colors.ENDC}")
        except ImportError:
            if is_optional:
                print(
                    f"   {Colors.YELLOW}⚠️  {display_name}: 未安裝 (可選){Colors.ENDC}"
                )
            else:
                print(f"   {Colors.RED}❌ {display_name}: 未安裝{Colors.ENDC}")
                all_ok = False

    return all_ok


def check_cache_directory():
    """檢查快取目錄"""
    cache_root = os.getenv("AI_CACHE_ROOT", "/tmp/test_cache")
    cache_path = Path(cache_root)

    print(f"📁 快取目錄: {cache_root}")

    if not cache_path.exists():
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            print(f"   {Colors.GREEN}✅ 快取目錄已創建{Colors.ENDC}")
        except Exception as e:
            print(f"   {Colors.RED}❌ 無法創建快取目錄: {e}{Colors.ENDC}")
            return False
    else:
        print(f"   {Colors.GREEN}✅ 快取目錄存在{Colors.ENDC}")

    # Check write permissions
    test_file = cache_path / "test_write.tmp"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print(f"   {Colors.GREEN}✅ 快取目錄可寫入{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"   {Colors.RED}❌ 快取目錄無寫入權限: {e}{Colors.ENDC}")
        return False


def check_gpu_availability():
    """檢查 GPU 可用性"""
    print("🖥️  GPU 檢查:")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"   {Colors.GREEN}✅ GPU可用: {gpu_name}{Colors.ENDC}")
        print(f"   {Colors.BLUE}ℹ️  GPU數量: {gpu_count}{Colors.ENDC}")
        print(f"   {Colors.BLUE}ℹ️  GPU記憶體: {gpu_memory:.1f}GB{Colors.ENDC}")
        return True
    else:
        print(f"   {Colors.YELLOW}⚠️  GPU不可用，將使用CPU{Colors.ENDC}")
        return False


def check_api_port():
    """檢查 API 端口可用性"""
    import socket

    host = "localhost"
    port = int(os.getenv("API_PORT", "8000"))

    print(f"🔌 端口檢查: {host}:{port}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex((host, port))
    sock.close()

    if result == 0:
        print(f"   {Colors.YELLOW}⚠️  端口 {port} 已被占用{Colors.ENDC}")
        return False
    else:
        print(f"   {Colors.GREEN}✅ 端口 {port} 可用{Colors.ENDC}")
        return True


def run_quick_import_test():
    """快速導入測試"""
    print("🔬 快速導入測試:")

    try:
        # Test backend imports
        sys.path.append(str(Path(__file__).parent.parent / "backend"))

        from core.config import get_config
        from core.shared_cache import bootstrap_cache

        print(f"   {Colors.GREEN}✅ 核心模組導入成功{Colors.ENDC}")

        # Test config loading
        config = get_config()
        print(f"   {Colors.GREEN}✅ 配置載入成功{Colors.ENDC}")

        # Test cache bootstrap
        cache = bootstrap_cache()
        print(f"   {Colors.GREEN}✅ 快取初始化成功{Colors.ENDC}")

        return True

    except Exception as e:
        print(f"   {Colors.RED}❌ 導入測試失败: {e}{Colors.ENDC}")
        return False


def main():
    """主檢查函數"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("🏥 Multi-Modal Lab Backend Health Check")
    print("=" * 50)
    print(f"{Colors.ENDC}")

    checks = [
        ("Python版本", check_python_version),
        ("依賴套件", check_dependencies),
        ("快取目錄", check_cache_directory),
        ("GPU可用性", check_gpu_availability),
        ("API端口", check_api_port),
        ("模組導入", run_quick_import_test),
    ]

    results = []
    for name, check_func in checks:
        print()
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   {Colors.RED}❌ {name} 檢查出錯: {e}{Colors.ENDC}")
            results.append((name, False))

    # Summary
    print(f"\n{Colors.BOLD}📋 檢查摘要:{Colors.ENDC}")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = f"{Colors.GREEN}✅" if result else f"{Colors.RED}❌"
        print(f"   {status} {name}{Colors.ENDC}")

    print(f"\n{Colors.BOLD}結果: {passed}/{total} 項檢查通過{Colors.ENDC}")

    if passed == total:
        print(
            f"{Colors.GREEN}{Colors.BOLD}🎉 系統健康檢查通過！可以開始測試。{Colors.ENDC}"
        )
        return 0
    else:
        print(
            f"{Colors.RED}{Colors.BOLD}⚠️  發現 {total-passed} 個問題，請修復後再測試。{Colors.ENDC}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
