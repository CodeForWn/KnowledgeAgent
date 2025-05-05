import json
import requests
import time

# 配置你的接口地址
API_URL = "http://127.0.0.1:7777/api/question"  # 根据你本地服务实际地址改一下，比如端口不是5000就改掉

# 加载待导入的题目JSON文件
with open("/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理/选必1/选必1 练习/其他练习/processed_自转-地方时、时区、区时-试卷试题.（修改）doc.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# 定义发送单条题目的函数
def send_question(q_data):
    # 处理 knowledge_point，提取每个对象的 "name"
    knowledge_point_list = []
    kp_raw = q_data.get("knowledge_point", [])
    if isinstance(kp_raw, list):
        for kp in kp_raw:
            if isinstance(kp, dict) and "name" in kp:
                knowledge_point_list.append(kp["name"])

    payload = {
        "question": q_data.get("question", ""),
        "answer": q_data.get("answer", ""),
        "analysis": q_data.get("explanation", ""),  # explanation -> analysis
        "type": q_data.get("type", "单选题"),
        "diff_level": q_data.get("difficulty_level", "普通"),
        "status": "on",
        "subject": "高中地理",  # ✅统一加上
        "kb_id": "1911603842693210113",  # ✅统一加上
        "folder_id": "1911604997812920321",  # 题库folder_id
        "knowledge_point": knowledge_point_list
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            res = response.json()
            if res.get("code") == 200:
                print(f"✅ 成功插入：docID={res['data']['docID']}")
            else:
                print(f"❌ 插入失败，返回信息：{res}")
        else:
            print(f"❌ 请求失败，HTTP状态码：{response.status_code}")
    except Exception as e:
        print(f"❌ 出现异常：{str(e)}")


# 批量处理
for idx, q in enumerate(questions):
    send_question(q)
    time.sleep(0.2)  # 每条题目间隔0.2秒，保护服务器

print("✅ 所有题目发送完毕。")
