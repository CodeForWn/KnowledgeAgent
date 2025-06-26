# -*- coding: utf-8 -*-
import queue
import threading
import time
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from enum import Enum
from typing import Dict, Any, Optional, Callable
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"

class Priority(Enum):
    HIGH = 1      # 简单查询任务
    MEDIUM = 2    # 一般生成任务
    LOW = 3       # 复杂生成任务

class TaskResult:
    def __init__(self, task_id: str, status: TaskStatus, result: Any = None, error: str = None):
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error
        self.created_at = time.time()
        self.updated_at = time.time()
    
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

class Task:
    def __init__(self, task_id: str, func: Callable, args: tuple, kwargs: dict, 
                 priority: Priority, timeout: int = 300):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.priority = priority
        self.timeout = timeout
        self.created_at = time.time()
    
    def __lt__(self, other):
        return self.priority.value < other.priority.value

class QueueManager:
    def __init__(self, max_workers: int = 4, result_ttl: int = 3600):
        """
        初始化队列管理器
        max_workers: 最大工作线程数，根据服务器CPU核心数调整
        result_ttl: 结果缓存时间（秒）
        """
        # 优先级队列
        self.task_queues = {
            Priority.HIGH: queue.PriorityQueue(),
            Priority.MEDIUM: queue.PriorityQueue(), 
            Priority.LOW: queue.PriorityQueue()
        }
        
        # 任务结果存储
        self.results: Dict[str, TaskResult] = {}
        self.result_ttl = result_ttl
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        
        # 运行状态
        self.running = False
        self.worker_threads = []
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'active_tasks': 0
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_results, daemon=True)
        self.cleanup_thread.start()
    
    def start(self):
        """启动队列处理"""
        if self.running:
            return
            
        self.running = True
        
        # 为每个优先级启动工作线程
        for priority in Priority:
            thread = threading.Thread(
                target=self._worker,
                args=(priority,),
                daemon=True,
                name=f"QueueWorker-{priority.name}"
            )
            thread.start()
            self.worker_threads.append(thread)
        
        self.logger.info(f"队列管理器启动，工作线程数: {len(self.worker_threads)}")
    
    def stop(self):
        """停止队列处理"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.logger.info("队列管理器已停止")
    
    def submit_task(self, func: Callable, args: tuple = (), kwargs: dict = None, 
                   priority: Priority = Priority.MEDIUM, timeout: int = 300) -> str:
        """提交任务到队列"""
        if kwargs is None:
            kwargs = {}
            
        task_id = str(uuid.uuid4())
        task = Task(task_id, func, args, kwargs, priority, timeout)
        
        # 添加到对应优先级队列
        self.task_queues[priority].put((priority.value, time.time(), task))
        
        # 创建待处理结果
        self.results[task_id] = TaskResult(task_id, TaskStatus.PENDING)
        
        with self.lock:
            self.stats['total_tasks'] += 1
        
        self.logger.info(f"任务已提交: {task_id}, 优先级: {priority.name}")
        return task_id
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        return self.results.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        result = self.results.get(task_id)
        return result.status if result else None
    
    def _worker(self, priority: Priority):
        """工作线程处理函数"""
        task_queue = self.task_queues[priority]
        
        while self.running:
            try:
                # 从队列获取任务，超时1秒避免无限阻塞
                _, _, task = task_queue.get(timeout=1.0)
                
                with self.lock:
                    self.stats['active_tasks'] += 1
                
                self._execute_task(task)
                
                with self.lock:
                    self.stats['active_tasks'] -= 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"工作线程异常: {e}")
    
    def _execute_task(self, task: Task):
        """执行单个任务"""
        try:
            # 更新状态为运行中
            self.results[task.task_id].status = TaskStatus.RUNNING
            self.results[task.task_id].updated_at = time.time()
            
            # 提交到线程池执行
            future = self.executor.submit(task.func, *task.args, **task.kwargs)
            
            try:
                # 等待结果，带超时
                result = future.result(timeout=task.timeout)
                
                # 更新成功结果
                self.results[task.task_id].status = TaskStatus.SUCCESS
                self.results[task.task_id].result = result
                self.results[task.task_id].updated_at = time.time()
                
                with self.lock:
                    self.stats['completed_tasks'] += 1
                
                self.logger.info(f"任务执行成功: {task.task_id}")
                
            except TimeoutError:
                # 超时处理
                future.cancel()
                self.results[task.task_id].status = TaskStatus.TIMEOUT
                self.results[task.task_id].error = f"任务超时 ({task.timeout}秒)"
                self.results[task.task_id].updated_at = time.time()
                
                with self.lock:
                    self.stats['failed_tasks'] += 1
                
                self.logger.warning(f"任务超时: {task.task_id}")
                
        except Exception as e:
            # 执行异常
            self.results[task.task_id].status = TaskStatus.FAILED
            self.results[task.task_id].error = str(e)
            self.results[task.task_id].updated_at = time.time()
            
            with self.lock:
                self.stats['failed_tasks'] += 1
            
            self.logger.error(f"任务执行失败: {task.task_id}, 错误: {e}")
    
    def _cleanup_results(self):
        """定期清理过期结果"""
        while True:
            try:
                current_time = time.time()
                expired_tasks = []
                
                for task_id, result in self.results.items():
                    if current_time - result.created_at > self.result_ttl:
                        expired_tasks.append(task_id)
                
                for task_id in expired_tasks:
                    del self.results[task_id]
                
                if expired_tasks:
                    self.logger.info(f"清理过期结果: {len(expired_tasks)} 个")
                
                # 每5分钟清理一次
                time.sleep(300)
                
            except Exception as e:
                self.logger.error(f"清理结果异常: {e}")
                time.sleep(60)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.lock:
            stats = self.stats.copy()
        
        # 添加队列信息
        queue_info = {}
        for priority in Priority:
            queue_info[priority.name.lower()] = self.task_queues[priority].qsize()
        
        stats['queue_sizes'] = queue_info
        stats['result_count'] = len(self.results)
        stats['worker_count'] = len(self.worker_threads)
        
        return stats

# 全局队列管理器实例
queue_manager = QueueManager(max_workers=4)

def async_task(priority: Priority = Priority.MEDIUM, timeout: int = 300):
    """装饰器：将函数转换为异步任务"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return queue_manager.submit_task(func, args, kwargs, priority, timeout)
        
        # 保留原函数的同步调用能力
        wrapper.sync = func
        return wrapper
    return decorator 