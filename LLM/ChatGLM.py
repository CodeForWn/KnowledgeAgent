import zhipuai
import time
import requests
# 设置您的API密钥
zhipuai.api_key = "b415a5e9089d4bcae6c287890e3073eb.9BDiJukUgt1KPOmA"


def async_invoke():
    # 发起异步调用
    response = zhipuai.model_api.async_invoke(
        model="chatglm_turbo",
        prompt=[{"role": "user", "content": "请介绍一下同济大学应用物理系"}],
        top_p=0.7,
        temperature=0.9,
    )

    # 检查响应的code和success键来确定调用是否成功
    if response['code'] == 200 and response['success']:
        # 从data字典中获取task_id
        task_id = response['data']['task_id']
        print(f"任务ID: {task_id}")
        return task_id
    else:
        print(f"异步调用失败: {response['msg']}")
        return None


def query_async_invoke_result(task_id):
    # 检查任务ID是否有效
    if task_id is None:
        print("无效的任务ID")
        return

    # 循环查询直到获取结果
    while True:
        result = zhipuai.model_api.query_async_invoke_result(task_id)
        # 检查响应的code和success键来确定调用状态
        if result['code'] == 200 and result['success']:
            # 从data字典中获取任务状态
            task_status = result['data']['task_status']

            if task_status == "SUCCESS":
                # 任务完成，打印结果
                print("异步调用结果：")
                for choice in result['data']['choices']:
                    print(choice['content'])
                break
            elif task_status == "FAILED":
                print("异步调用失败")
                break
            else:
                print(f"当前任务状态: {task_status}")
                print("等待结果...")
                time.sleep(5)  # 每5秒查询一次
        else:
            print(f"查询失败: {result['msg']}")
            break


# 调用异步函数并查询结果
task_id = async_invoke()
query_async_invoke_result(task_id)
