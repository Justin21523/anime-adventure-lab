# core/performance/batch_optimizer.py
"""
Batch Processing Optimizer

優化批次處理以提升吞吐量：
- 動態批次大小調整
- 請求合併和去重
- 自適應調度
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchStrategy(str, Enum):
    """批次處理策略"""

    FIXED = "fixed"  # 固定批次大小
    DYNAMIC = "dynamic"  # 動態調整批次大小
    ADAPTIVE = "adaptive"  # 自適應（根據延遲和吞吐量）


@dataclass
class BatchConfig:
    """批次處理配置"""

    # 批次大小
    min_batch_size: int = 1
    max_batch_size: int = 32
    optimal_batch_size: int = 8

    # 等待時間
    max_wait_time_ms: float = 100.0  # 最大等待時間（毫秒）
    min_wait_time_ms: float = 10.0  # 最小等待時間（毫秒）

    # 策略
    strategy: BatchStrategy = BatchStrategy.DYNAMIC

    # 自適應調整
    enable_deduplication: bool = True  # 啟用請求去重
    target_latency_ms: float = 500.0  # 目標延遲
    throughput_window_size: int = 10  # 吞吐量統計窗口


@dataclass
class BatchRequest(Generic[T]):
    """批次請求"""

    request_id: str
    data: T
    timestamp: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None
    hash: Optional[str] = None


@dataclass
class BatchMetrics:
    """批次處理指標"""

    total_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_wait_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    deduplication_hits: int = 0
    throughput_per_second: float = 0.0


class BatchProcessor(Generic[T, R]):
    """
    批次處理器

    將多個請求合併成批次以提升效率
    """

    def __init__(
        self,
        process_function: Callable[[List[T]], List[R]],
        config: Optional[BatchConfig] = None,
    ):
        """
        初始化批次處理器

        Args:
            process_function: 批次處理函數，接收 List[T]，返回 List[R]
            config: 批次配置
        """
        self.process_function = process_function
        self.config = config or BatchConfig()

        # 請求隊列
        self.pending_requests: List[BatchRequest[T]] = []
        self.request_lock = asyncio.Lock()

        # 去重緩存
        self.dedup_cache: Dict[str, R] = {}

        # 指標
        self.metrics = BatchMetrics()
        self.recent_batch_sizes: List[int] = []
        self.recent_latencies: List[float] = []

        # 狀態
        self.is_processing = False
        self.current_batch_size = self.config.optimal_batch_size

    def _compute_hash(self, data: T) -> str:
        """計算請求數據的哈希值用於去重"""
        try:
            # 簡化版本，實際使用可能需要更複雜的序列化
            data_str = str(data)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash: {e}")
            return ""

    async def add_request(self, data: T, request_id: Optional[str] = None) -> R:
        """
        添加請求到批次隊列

        Args:
            data: 請求數據
            request_id: 請求 ID（可選）

        Returns:
            處理結果
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"

        # 檢查去重
        if self.config.enable_deduplication:
            data_hash = self._compute_hash(data)
            if data_hash in self.dedup_cache:
                self.metrics.deduplication_hits += 1
                logger.debug(f"Deduplication hit for request {request_id}")
                return self.dedup_cache[data_hash]
        else:
            data_hash = None

        # 創建請求
        future: asyncio.Future[R] = asyncio.Future()
        request = BatchRequest(
            request_id=request_id,
            data=data,
            future=future,
            hash=data_hash,
        )

        # 添加到隊列
        async with self.request_lock:
            self.pending_requests.append(request)
            self.metrics.total_requests += 1

        # 觸發處理（如果還沒在處理）
        if not self.is_processing:
            asyncio.create_task(self._process_loop())

        # 等待結果
        result = await future
        return result

    async def _process_loop(self):
        """處理循環"""
        if self.is_processing:
            return

        self.is_processing = True

        try:
            while True:
                # 等待請求累積
                wait_time = self._get_wait_time()
                await asyncio.sleep(wait_time / 1000.0)  # 轉換為秒

                # 檢查是否有待處理請求
                async with self.request_lock:
                    if not self.pending_requests:
                        break

                    # 取出批次
                    batch = self.pending_requests[: self.current_batch_size]
                    self.pending_requests = self.pending_requests[self.current_batch_size :]

                if batch:
                    await self._process_batch(batch)

        finally:
            self.is_processing = False

    async def _process_batch(self, batch: List[BatchRequest[T]]):
        """處理一個批次"""
        start_time = time.time()

        try:
            # 提取數據
            batch_data = [req.data for req in batch]

            # 執行批次處理
            results = await asyncio.to_thread(self.process_function, batch_data)

            # 分發結果
            for request, result in zip(batch, results):
                if request.future and not request.future.done():
                    request.future.set_result(result)

                # 更新去重緩存
                if self.config.enable_deduplication and request.hash:
                    self.dedup_cache[request.hash] = result

            # 更新指標
            processing_time = (time.time() - start_time) * 1000  # 毫秒
            self._update_metrics(len(batch), processing_time)

            logger.debug(
                f"Processed batch of {len(batch)} requests in {processing_time:.2f}ms"
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)

            # 設置錯誤結果
            for request in batch:
                if request.future and not request.future.done():
                    request.future.set_exception(e)

    def _get_wait_time(self) -> float:
        """獲取當前應該等待的時間"""
        if self.config.strategy == BatchStrategy.FIXED:
            return self.config.max_wait_time_ms

        elif self.config.strategy == BatchStrategy.DYNAMIC:
            # 根據隊列長度動態調整
            queue_length = len(self.pending_requests)

            if queue_length >= self.current_batch_size:
                return self.config.min_wait_time_ms
            elif queue_length == 0:
                return self.config.max_wait_time_ms
            else:
                # 線性插值
                ratio = queue_length / self.current_batch_size
                return self.config.min_wait_time_ms + (
                    self.config.max_wait_time_ms - self.config.min_wait_time_ms
                ) * (1 - ratio)

        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            # 根據延遲自適應調整
            if self.recent_latencies:
                avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)

                if avg_latency > self.config.target_latency_ms:
                    # 延遲過高，減少等待時間
                    return self.config.min_wait_time_ms
                else:
                    # 延遲正常，可以等待更多請求
                    return self.config.max_wait_time_ms

            return self.config.max_wait_time_ms

        return self.config.max_wait_time_ms

    def _update_metrics(self, batch_size: int, processing_time_ms: float):
        """更新性能指標"""
        self.metrics.total_batches += 1

        # 更新平均批次大小
        self.metrics.avg_batch_size = (
            self.metrics.avg_batch_size * (self.metrics.total_batches - 1)
            + batch_size
        ) / self.metrics.total_batches

        # 更新平均處理時間
        self.metrics.avg_processing_time_ms = (
            self.metrics.avg_processing_time_ms * (self.metrics.total_batches - 1)
            + processing_time_ms
        ) / self.metrics.total_batches

        # 記錄最近的批次大小和延遲
        self.recent_batch_sizes.append(batch_size)
        self.recent_latencies.append(processing_time_ms)

        # 保持窗口大小
        if len(self.recent_batch_sizes) > self.config.throughput_window_size:
            self.recent_batch_sizes.pop(0)
        if len(self.recent_latencies) > self.config.throughput_window_size:
            self.recent_latencies.pop(0)

        # 自適應調整批次大小
        if self.config.strategy == BatchStrategy.ADAPTIVE:
            self._adjust_batch_size()

        # 計算吞吐量
        if self.recent_latencies:
            avg_latency_seconds = sum(self.recent_latencies) / len(self.recent_latencies) / 1000
            avg_batch = sum(self.recent_batch_sizes) / len(self.recent_batch_sizes)
            self.metrics.throughput_per_second = avg_batch / avg_latency_seconds

    def _adjust_batch_size(self):
        """自適應調整批次大小"""
        if len(self.recent_latencies) < 3:
            return

        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)

        # 延遲過高，減小批次
        if avg_latency > self.config.target_latency_ms * 1.2:
            new_size = max(
                self.config.min_batch_size, int(self.current_batch_size * 0.8)
            )
            if new_size != self.current_batch_size:
                logger.info(
                    f"Reducing batch size: {self.current_batch_size} -> {new_size}"
                )
                self.current_batch_size = new_size

        # 延遲較低，增大批次
        elif avg_latency < self.config.target_latency_ms * 0.5:
            new_size = min(
                self.config.max_batch_size, int(self.current_batch_size * 1.2)
            )
            if new_size != self.current_batch_size:
                logger.info(
                    f"Increasing batch size: {self.current_batch_size} -> {new_size}"
                )
                self.current_batch_size = new_size

    def get_metrics(self) -> BatchMetrics:
        """獲取批次處理指標"""
        return self.metrics

    def clear_dedup_cache(self):
        """清空去重緩存"""
        self.dedup_cache.clear()
        logger.info("Deduplication cache cleared")

    def reset_metrics(self):
        """重置指標"""
        self.metrics = BatchMetrics()
        self.recent_batch_sizes.clear()
        self.recent_latencies.clear()
        logger.info("Metrics reset")


# Example usage
async def example_batch_function(texts: List[str]) -> List[int]:
    """示例批次處理函數：計算文本長度"""
    await asyncio.sleep(0.1)  # 模擬處理時間
    return [len(text) for text in texts]


async def main():
    """測試示例"""
    logging.basicConfig(level=logging.INFO)

    # 創建批次處理器
    config = BatchConfig(
        min_batch_size=1,
        max_batch_size=8,
        optimal_batch_size=4,
        strategy=BatchStrategy.ADAPTIVE,
    )

    processor = BatchProcessor(example_batch_function, config)

    # 模擬並發請求
    async def send_request(text: str, delay: float = 0):
        await asyncio.sleep(delay)
        result = await processor.add_request(text)
        logger.info(f"Result for '{text}': {result}")

    # 發送多個請求
    tasks = [
        send_request("Hello", 0),
        send_request("World", 0.01),
        send_request("Batch", 0.02),
        send_request("Processing", 0.03),
        send_request("Test", 0.2),
        send_request("Example", 0.21),
    ]

    await asyncio.gather(*tasks)

    # 顯示指標
    metrics = processor.get_metrics()
    logger.info(f"\nBatch Processing Metrics:")
    logger.info(f"  Total requests: {metrics.total_requests}")
    logger.info(f"  Total batches: {metrics.total_batches}")
    logger.info(f"  Avg batch size: {metrics.avg_batch_size:.2f}")
    logger.info(f"  Avg processing time: {metrics.avg_processing_time_ms:.2f}ms")
    logger.info(f"  Throughput: {metrics.throughput_per_second:.2f} req/s")


if __name__ == "__main__":
    asyncio.run(main())
