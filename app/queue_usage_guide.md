# 队列系统使用指南

## 概述

本系统实现了纯Python的队列机制，不依赖外部服务（如Redis），使用内置的`queue`、`threading`和`concurrent.futures`模块。

## 队列特性

### 优先级设置
- **HIGH**: 简单查询任务（知识点提取等）
- **MEDIUM**: 一般生成任务（题目生成、题目解析）
- **LOW**: 复杂生成任务（PPT生成等）

### 性能配置
- **最大工作线程数**: 4（根据服务器CPU核心数）
- **任务超时时间**: 可配置（60秒-900秒）
- **结果缓存时间**: 1小时
- **内存限制**: 每个worker 512MB

## API接口

### 1. 异步题目生成

**接口**: `POST /api/question_agent_pro_async`

**请求参数**:
```json
{
    "knowledge_points": ["地球自转", "洋流"],
    "difficulty_level": "困难",
    "query": null,
    "kb_id": "1911603842693210113",
    "question_type": "单选题",
    "question_count": 3
}
```

**响应**:
```json
{
    "code": 200,
    "msg": "任务已提交到队列",
    "data": {
        "task_id": "12345678-1234-1234-1234-123456789abc"
    }
}
```

### 2. 异步题目解析

**接口**: `POST /api/question_explanation_agent_async`

**请求参数**:
```json
{
    "knowledge_point": "地球自转",
    "question_content": "地球自转的方向是？",
    "difficulty_level": "困难",
    "question_type": "单选题",
    "llm": "qwen",
    "kb_id": "1911603842693210113"
}
```

**响应**:
```json
{
    "code": 200,
    "msg": "任务已提交到队列",
    "data": {
        "task_id": "12345678-1234-1234-1234-123456789abc"
    }
}
```

### 3. 查询任务状态

**接口**: `GET /api/task/status/{task_id}`

**响应**:
```json
{
    "code": 200,
    "msg": "success",
    "data": {
        "task_id": "12345678-1234-1234-1234-123456789abc",
        "status": "running",
        "result": null,
        "error": null,
        "created_at": 1640995200.0,
        "updated_at": 1640995230.0
    }
}
```

### 4. 获取任务结果

**接口**: `GET /api/task/result/{task_id}`

**任务进行中**:
```json
{
    "code": 202,
    "msg": "任务正在处理中，状态: running",
    "data": {
        "status": "running"
    }
}
```

**任务完成**:
```json
{
    "code": 200,
    "msg": "success", 
    "data": {
        "question_prompt": "...",
        "graph2text": "...",
        "questions": [...]
    }
}
```

**任务失败**:
```json
{
    "code": 500,
    "msg": "任务执行失败: 错误信息",
    "data": null
}
```

### 5. 队列统计信息

**接口**: `GET /api/queue/stats`

**响应**:
```json
{
    "code": 200,
    "msg": "success",
    "data": {
        "total_tasks": 150,
        "completed_tasks": 140,
        "failed_tasks": 5,
        "active_tasks": 5,
        "queue_sizes": {
            "high": 2,
            "medium": 8,
            "low": 3
        },
        "result_count": 145,
        "worker_count": 3
    }
}
```

## 客户端使用示例

### JavaScript/TypeScript

```javascript
// 提交任务
async function submitQuestionTask(data) {
    const response = await fetch('/api/question_agent_pro_async', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    const result = await response.json();
    return result.data.task_id;
}

// 轮询获取结果
async function pollTaskResult(taskId, timeout = 300000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
        const response = await fetch(`/api/task/result/${taskId}`);
        const result = await response.json();
        
        if (response.status === 200) {
            return result;  // 任务完成
        } else if (response.status === 500) {
            throw new Error(result.msg);  // 任务失败
        }
        
        // 任务进行中，等待2秒后重试
        await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    throw new Error('任务超时');
}

// 使用示例
async function generateQuestion() {
    try {
        const taskId = await submitQuestionTask({
            knowledge_points: ["地球自转"],
            difficulty_level: "困难",
            question_type: "单选题",
            question_count: 3
        });
        
        console.log('任务已提交:', taskId);
        const result = await pollTaskResult(taskId);
        console.log('任务完成:', result);
        
    } catch (error) {
        console.error('任务失败:', error);
    }
}
```

### Python

```python
import requests
import time
import json

def submit_question_task(data):
    response = requests.post(
        'http://localhost:7777/api/question_agent_pro_async',
        json=data
    )
    result = response.json()
    return result['data']['task_id']

def poll_task_result(task_id, timeout=300):
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = requests.get(f'http://localhost:7777/api/task/result/{task_id}')
        
        if response.status_code == 200:
            return response.json()  # 任务完成
        elif response.status_code == 500:
            raise Exception(response.json()['msg'])  # 任务失败
        
        # 任务进行中，等待2秒后重试
        time.sleep(2)
    
    raise Exception('任务超时')

# 使用示例
def generate_question():
    try:
        task_id = submit_question_task({
            'knowledge_points': ['地球自转'],
            'difficulty_level': '困难',
            'question_type': '单选题',
            'question_count': 3
        })
        
        print(f'任务已提交: {task_id}')
        result = poll_task_result(task_id)
        print(f'任务完成: {result}')
        
    except Exception as error:
        print(f'任务失败: {error}')

if __name__ == '__main__':
    generate_question()
```

## 性能优化建议

### 1. 服务器配置
```python
# 在 queue_manager.py 中调整配置
queue_manager = QueueManager(
    max_workers=8,  # 根据CPU核心数调整（建议为核心数*2）
    result_ttl=7200  # 结果缓存2小时
)
```

### 2. 监控和日志
- 定期检查 `/api/queue/stats` 监控队列状态
- 关注内存使用情况
- 查看应用日志中的任务执行情况

### 3. 错误处理
- 设置合理的客户端超时时间
- 实现重试机制
- 及时清理过期任务结果

## 注意事项

1. **任务结果缓存**: 结果默认保存1小时，过期自动清理
2. **内存限制**: 每个worker有512MB内存限制，避免内存泄漏
3. **并发限制**: 最大4个并发任务，避免系统过载
4. **超时处理**: 任务有超时机制，防止长时间占用资源
5. **优雅关闭**: 应用关闭时会等待当前任务完成

## 故障排除

### 常见问题

1. **任务一直处于pending状态**: 检查队列是否启动，查看日志
2. **任务超时**: 增加timeout时间或优化任务逻辑
3. **内存不足**: 减少max_workers或增加内存限制
4. **队列堆积**: 检查任务执行效率，考虑增加worker数量 