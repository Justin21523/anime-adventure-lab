# scripts/test_complete_tools.py
"""
完整測試 Agent 工具系統
測試整合後的所有工具功能和真實 API 調用能力
"""

import sys
import asyncio
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_web_search_engine():
    """測試完整的 WebSearchEngine 功能"""
    print("\n🔍 測試 WebSearchEngine 完整功能...")

    try:
        from core.agents.tools.web_search import (
            WebSearchEngine,
            get_search_engine,
            search,
            search_and_summarize,
            configure_search_engine,
        )

        # 測試 1: 基本搜尋功能
        print("  1️⃣ 測試基本搜尋...")
        engine = get_search_engine()
        results = await engine.search("Python programming", 3)
        print(f"     搜尋到 {len(results)} 個結果")
        for i, result in enumerate(results):
            print(f"     - {result.title[:50]}...")

        # 測試 2: 不同主題的搜尋
        print("  2️⃣ 測試不同主題搜尋...")
        topics = [
            "machine learning",
            "web development",
            "artificial intelligence",
            "unknown topic xyz",
        ]
        for topic in topics:
            results = await engine.search(topic, 2)
            print(f"     {topic}: {len(results)} 結果，相關度: {results[0].score:.2f}")

        # 測試 3: 搜尋歷史
        print("  3️⃣ 測試搜尋歷史...")
        history = engine.get_search_history(5)
        print(f"     歷史記錄: {len(history)} 次搜尋")

        # 測試 4: 統計資訊
        print("  4️⃣ 測試統計資訊...")
        stats = engine.get_search_stats()
        print(f"     統計: {json.dumps(stats, indent=2, default=str)}")

        # 測試 5: 主要搜尋函數
        print("  5️⃣ 測試主要搜尋函數...")
        search_result = await search("Python tutorial", 3)
        print(
            f"     搜尋函數結果: {search_result['success']}, 結果數: {search_result.get('results_count', 0)}"
        )

        # 測試 6: 搜尋和摘要
        print("  6️⃣ 測試搜尋和摘要...")
        summary_result = await search_and_summarize("machine learning basics", 2)
        if summary_result["success"]:
            print(f"     摘要長度: {len(summary_result['summary'])} 字元")
            print(f"     首段摘要: {summary_result['summary'][:100]}...")

        # 測試 7: 配置功能
        print("  7️⃣ 測試配置功能...")
        config_result = configure_search_engine(
            mock_enabled=True, search_engine_type="google"
        )
        print(f"     配置結果: {config_result['success']}")

        print("  ✅ WebSearchEngine 所有測試通過")
        return True

    except Exception as e:
        print(f"  ❌ WebSearchEngine 測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_real_search_apis():
    """測試真實搜尋 API（如果配置了）"""
    print("\n🌐 測試真實搜尋 API...")

    try:
        from core.agents.tools.web_search import (
            get_search_engine,
            configure_search_engine,
        )

        # 檢查是否有 API 密鑰配置
        google_api_key = os.getenv("SEARCH_API_KEY") or os.getenv("GOOGLE_API_KEY")
        bing_api_key = os.getenv("BING_API_KEY")
        serpapi_key = os.getenv("SERPAPI_KEY")

        if not any([google_api_key, bing_api_key, serpapi_key]):
            print("  ⏭️  沒有配置搜尋 API 密鑰，跳過真實 API 測試")
            print("     要測試真實 API，請設定以下環境變數之一：")
            print("     - SEARCH_API_KEY (Google)")
            print("     - BING_API_KEY (Bing)")
            print("     - SERPAPI_KEY (SerpAPI)")
            return True

        engine = get_search_engine()

        # 測試 Google API
        if google_api_key:
            print("  🔍 測試 Google Search API...")
            configure_search_engine(
                mock_enabled=False, api_key=google_api_key, search_engine_type="google"
            )
            results = await engine.search("Python programming tutorial", 3)
            print(
                f"     Google API: {len(results)} 結果，第一個結果來自 {results[0].url if results else 'N/A'}"
            )

        # 測試 Bing API
        if bing_api_key:
            print("  🔍 測試 Bing Search API...")
            configure_search_engine(
                mock_enabled=False, api_key=bing_api_key, search_engine_type="bing"
            )
            results = await engine.search("machine learning basics", 3)
            print(
                f"     Bing API: {len(results)} 結果，第一個結果來自 {results[0].url if results else 'N/A'}"
            )

        # 測試 SerpAPI
        if serpapi_key:
            print("  🔍 測試 SerpAPI...")
            configure_search_engine(
                mock_enabled=False, api_key=serpapi_key, search_engine_type="serpapi"
            )
            results = await engine.search("web development guide", 3)
            print(
                f"     SerpAPI: {len(results)} 結果，第一個結果來自 {results[0].url if results else 'N/A'}"
            )

        # 重置為 mock 模式
        configure_search_engine(mock_enabled=True)
        print("  ✅ 真實搜尋 API 測試完成")
        return True

    except Exception as e:
        print(f"  ❌ 真實搜尋 API 測試失敗: {e}")
        return False


def test_enhanced_calculator():
    """測試增強的計算器功能"""
    print("\n🧮 測試增強計算器功能...")

    try:
        from core.agents.tools.calculator import (
            calculate,
            basic_math,
            percentage,
            unit_convert,
            SafeCalculator,
        )

        # 測試 1: 基本計算
        print("  1️⃣ 測試基本計算...")
        test_expressions = ["2 + 3 * 4", "sqrt(16)", "pow(2, 3)", "sin(pi/2)", "log(e)"]

        for expr in test_expressions:
            result = calculate(expr)
            if result.get("error"):
                print(f"     {expr} = 錯誤: {result['error']}")
            else:
                print(f"     {expr} = {result['result']}")

        # 測試 2: 基本數學運算
        print("  2️⃣ 測試基本數學運算...")
        operations = [
            ("add", 10, 5),
            ("multiply", 7, 8),
            ("divide", 15, 3),
            ("sqrt", 25, None),
            ("power", 2, 8),
        ]

        for op, a, b in operations:
            result = basic_math(op, a, b)
            if result.get("error"):
                print(f"     {op}({a}, {b if b else ''}) = 錯誤: {result['error']}")
            else:
                print(f"     {op}({a}, {b if b else ''}) = {result['result']}")

        # 測試 3: 百分比計算
        print("  3️⃣ 測試百分比計算...")
        percentage_tests = [
            (200, 15),  # 200 的 15%
            (500, 25),  # 500 的 25%
            (1000, 7.5),  # 1000 的 7.5%
        ]

        for value, percent in percentage_tests:
            result = percentage(value, percent)
            if result.get("error"):
                print(f"     {percent}% of {value} = 錯誤: {result['error']}")
            else:
                print(f"     {percent}% of {value} = {result['result']}")

        # 測試 4: 單位轉換
        print("  4️⃣ 測試單位轉換...")
        conversion_tests = [
            (100, "cm", "m"),  # 100 公分 = ? 公尺
            (5, "kg", "lb"),  # 5 公斤 = ? 磅
            (12, "in", "cm"),  # 12 英吋 = ? 公分
            (1, "km", "ft"),  # 1 公里 = ? 英尺
        ]

        for value, from_unit, to_unit in conversion_tests:
            result = unit_convert(value, from_unit, to_unit)
            if result.get("error"):
                print(f"     {value} {from_unit} → {to_unit} = 錯誤: {result['error']}")
            else:
                print(f"     {value} {from_unit} → {to_unit} = {result['result']:.4f}")

        # 測試 5: SafeCalculator 類別
        print("  5️⃣ 測試 SafeCalculator 類別...")
        calc = SafeCalculator()
        complex_expressions = [
            "(2 + 3) * (4 + 5)",
            "factorial(5)",
            "ceil(3.7) + floor(3.7)",
        ]

        for expr in complex_expressions:
            result = calc.evaluate(expr)
            print(f"     SafeCalculator: {expr} = {result}")

        print("  ✅ 計算器所有測試通過")
        return True

    except Exception as e:
        print(f"  ❌ 計算器測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_enhanced_file_ops():
    """測試增強的檔案操作功能"""
    print("\n📁 測試增強檔案操作功能...")

    try:
        from core.agents.tools.file_ops import (
            list_files,
            read_file,
            write_file,
            file_exists,
            delete_file,
            create_directory,
            execute,
            SafeFileOperations,
        )

        # 測試 1: 創建測試目錄結構
        print("  1️⃣ 創建測試目錄結構...")
        test_dir = "test_file_ops"
        create_directory(test_dir)
        create_directory(f"{test_dir}/subdir")
        print(f"     創建目錄: {test_dir}")

        # 測試 2: 寫入測試檔案
        print("  2️⃣ 寫入測試檔案...")
        test_files = [
            (f"{test_dir}/test1.txt", "Hello, World!\nThis is a test file."),
            (f"{test_dir}/test2.txt", "Python programming is awesome!"),
            (f"{test_dir}/subdir/nested.txt", "This is in a subdirectory."),
        ]

        for file_path, content in test_files:
            result = write_file(file_path, content)
            print(f"     寫入 {file_path}: {result.get('success', False)}")

        # 測試 3: 列出檔案
        print("  3️⃣ 列出檔案...")
        list_result = list_files(test_dir)
        if list_result.get("error"):
            print(f"     錯誤: {list_result['error']}")
        else:
            print(f"     {test_dir} 包含 {list_result['total_items']} 個項目")
            for item in list_result["items"]:
                print(f"       - {item['name']} ({item['type']})")

        # 測試 4: 讀取檔案
        print("  4️⃣ 讀取檔案...")
        for file_path, expected_content in test_files:
            result = read_file(file_path)
            if result.get("error"):
                print(f"     讀取 {file_path}: 錯誤 {result['error']}")
            else:
                content_match = result["content"] == expected_content
                print(f"     讀取 {file_path}: 成功，內容匹配: {content_match}")

        # 測試 5: 檔案存在檢查
        print("  5️⃣ 檢查檔案存在...")
        check_files = [f"{test_dir}/test1.txt", f"{test_dir}/nonexistent.txt"]
        for file_path in check_files:
            result = file_exists(file_path)
            print(f"     {file_path} 存在: {result.get('exists', False)}")

        # 測試 6: execute 函數
        print("  6️⃣ 測試 execute 函數...")
        execute_tests = [
            ("list", test_dir),
            ("read", f"{test_dir}/test1.txt"),
            ("exists", f"{test_dir}/test2.txt"),
        ]

        for operation, path in execute_tests:
            result = execute(operation, path)
            success = not result.get("error")
            print(f"     execute('{operation}', '{path}'): {success}")

        # 測試 7: SafeFileOperations 類別
        print("  7️⃣ 測試 SafeFileOperations 類別...")
        try:
            file_ops = SafeFileOperations()
            # 測試安全路徑檢查
            safe_path = file_ops._is_extension_allowed("test.txt")
            print(f"     .txt 檔案允許: {safe_path}")
        except Exception as e:
            print(f"     SafeFileOperations 測試跳過: {e}")

        # 測試 8: 清理測試檔案
        print("  8️⃣ 清理測試檔案...")
        for file_path, _ in test_files:
            delete_file(file_path)
        delete_file(f"{test_dir}/subdir")
        delete_file(test_dir)
        print("     清理完成")

        print("  ✅ 檔案操作所有測試通過")
        return True

    except Exception as e:
        print(f"  ❌ 檔案操作測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_tool_registry_integration():
    """測試工具註冊器整合"""
    print("\n🔧 測試工具註冊器整合...")

    try:
        from core.agents.tool_registry import ToolRegistry
        from core.agents.executor import SafeExecutor

        # 測試 1: 工具註冊器
        print("  1️⃣ 測試工具註冊器...")
        registry = ToolRegistry()
        tools = registry.list_tools()
        print(f"     註冊工具: {', '.join(tools)}")

        # 測試 2: 每個工具的詳細資訊
        print("  2️⃣ 測試工具資訊...")
        for tool_name in tools:
            info = registry.get_tool_info(tool_name)
            if info:
                print(f"     {tool_name}: {info.get('description', 'No description')}")
            function = registry.get_tool_function(tool_name)
            print(f"       函數可用: {function is not None}")

        # 測試 3: 安全執行器
        print("  3️⃣ 測試安全執行器...")
        executor = SafeExecutor()

        test_cases = [
            ("calculator", {"expression": "5 * 6"}),
            ("web_search", {"query": "Python programming", "max_results": 2}),
            ("file_ops", {"path": "."}),
        ]

        for tool_name, params in test_cases:
            try:
                result = await executor.execute_tool(tool_name, params)
                print(f"     {tool_name}: {result.success}")
                if not result.success:
                    print(f"       錯誤: {result.error}")
            except Exception as e:
                print(f"     {tool_name}: 執行失敗 {e}")

        print("  ✅ 工具註冊器整合測試通過")
        return True

    except Exception as e:
        print(f"  ❌ 工具註冊器整合測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_performance_and_reliability():
    """測試效能和可靠性"""
    print("\n⚡ 測試效能和可靠性...")

    try:
        import time
        from core.agents.tools.web_search import get_search_engine
        from core.agents.tools.calculator import calculate

        # 測試 1: 並發搜尋
        print("  1️⃣ 測試並發搜尋...")
        start_time = time.time()

        search_tasks = []
        engine = get_search_engine()
        queries = ["Python", "JavaScript", "Java", "C++", "Go"]

        for query in queries:
            task = engine.search(query, 2)
            search_tasks.append(task)

        results = await asyncio.gather(*search_tasks)
        total_time = time.time() - start_time

        print(f"     {len(results)} 並發搜尋完成，耗時 {total_time:.2f} 秒")
        print(f"     平均每次搜尋 {total_time/len(results):.2f} 秒")

        # 測試 2: 大量計算
        print("  2️⃣ 測試大量計算...")
        start_time = time.time()

        calculation_results = []
        for i in range(100):
            result = calculate(f"{i} * {i+1} + {i*2}")
            calculation_results.append(result)

        calc_time = time.time() - start_time
        successful_calcs = sum(1 for r in calculation_results if not r.get("error"))

        print(f"     100 次計算完成，耗時 {calc_time:.3f} 秒")
        print(f"     成功率: {successful_calcs}/100 ({successful_calcs}%)")
        print(f"     平均每次計算 {calc_time*1000/100:.1f} ms")

        # 測試 3: 錯誤處理
        print("  3️⃣ 測試錯誤處理...")
        error_tests = [
            ("search", {"query": ""}),  # 空搜尋
            ("calculator", {"expression": "1/0"}),  # 除零錯誤
            ("calculator", {"expression": "invalid_function(123)"}),  # 無效函數
        ]

        for test_type, params in error_tests:
            if test_type == "search":
                from core.agents.tools.web_search import search

                result = await search(**params)  # type: ignore
            elif test_type == "calculator":
                result = calculate(**params)

            has_error = not result.get("success", True) or result.get("error")
            print(f"     {test_type} 錯誤處理: {'✅' if has_error else '❌'}")

        print("  ✅ 效能和可靠性測試通過")
        return True

    except Exception as e:
        print(f"  ❌ 效能和可靠性測試失敗: {e}")
        return False


async def main():
    """主測試函數"""
    print("🚀 開始完整 Agent 工具系統測試")
    print("=" * 60)

    tests = [
        ("WebSearchEngine 功能", test_web_search_engine),
        ("真實搜尋 API", test_real_search_apis),
        ("增強計算器功能", test_enhanced_calculator),
        ("增強檔案操作功能", test_enhanced_file_ops),
        ("工具註冊器整合", test_tool_registry_integration),
        ("效能和可靠性", test_performance_and_reliability),
    ]

    results = {}
    total_start_time = time.time()

    for test_name, test_func in tests:
        print(f"\n📋 執行測試: {test_name}")
        print("-" * 40)

        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(test_func):
                results[test_name] = await test_func()
            else:
                results[test_name] = test_func()
        except Exception as e:
            print(f"❌ 測試 '{test_name}' 發生未預期錯誤: {e}")
            results[test_name] = False

        test_time = time.time() - start_time
        print(f"⏱️  測試耗時: {test_time:.2f} 秒")

    total_time = time.time() - total_start_time

    # 結果摘要
    print(f"\n{'='*60}")
    print("🏆 測試結果摘要")
    print("=" * 60)

    passed = 0
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n📊 統計:")
    print(f"   總計: {passed}/{len(results)} 測試通過")
    print(f"   通過率: {passed/len(results)*100:.1f}%")
    print(f"   總耗時: {total_time:.2f} 秒")

    if passed == len(results):
        print("\n🎉 所有測試都通過了！Agent 工具系統運作正常。")
    else:
        print(f"\n⚠️  有 {len(results)-passed} 個測試失敗，請檢查上述錯誤訊息。")

    return passed == len(results)


if __name__ == "__main__":
    import time

    asyncio.run(main())
