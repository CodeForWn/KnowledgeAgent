import requests
import os

# Flask 服务地址（本地）
url = "http://127.0.0.1:7777/api/generate_ppt_from_outline_and_render"

# 请求体 JSON 数据
payload = {
    "knowledge_point": "地球自转",
    "outline_json": [
    {
        "content": "地球自转——探索地球自转的原理、影响及实际应用\n\n“地球自转”的定义、基本特征及应用背景。",
        "label": "主题",
        "type": "main"
    },
    {
        "content": "<!-- 教学要求部分，包含考纲目标与学习提示 -->\n- 地球运动的特征：要点提示：方向，周期；自转线速度和角速度\n- 要求示例：结合示意图说出地球运动的方向和周期；说明地球运动速度及其时空变化\n- 地球运动的地理意义：要点提示：地方时和区时；地转偏向现象\n- 要求示例：了解昼夜交替的成因，说明昼夜交替对自然环境和人类活动的影响；指导时区划分及日期变更的规定，学会根据区时计算不同地区的时差；知道地转偏向力及其对地表做水平运动物体运动方向的影响",
        "label": "教学要求",
        "type": "main"
    },
    {
        "children": [
            {
                "content": "- **知识点定义与讲解**：“信风”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“季风”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“大气环流”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“洋流”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“地转偏向力”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            }
        ],
        "content": "- **知识点定义与讲解**：“地转偏向”的定义、概念讲解与背景说明。",
        "label": "知识讲解",
        "type": "main"
    },
    {
        "children": [
            {
                "content": "- **知识点定义与讲解**：“日界线”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“自然日界线”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“人为日界线（国际）”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“时区”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“北京时间”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            }
        ],
        "content": "- **知识点定义与讲解**：“地方时”的定义、概念讲解与背景说明。",
        "label": "知识讲解",
        "type": "main"
    },
    {
        "children": [],
        "content": "- **知识点定义与讲解**：“区时”的定义、概念讲解与背景说明。",
        "label": "知识讲解",
        "type": "main"
    },
    {
        "children": [
            {
                "content": "- **知识点定义与讲解**：“晨昏线”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“夜弧”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“昼弧”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“夜半球”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“昼半球”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            }
        ],
        "content": "- **知识点定义与讲解**：“昼夜更替”的定义、概念讲解与背景说明。",
        "label": "知识讲解",
        "type": "main"
    },
    {
        "children": [
            {
                "content": "- **知识点定义与讲解**：“线速度”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            },
            {
                "content": "- **知识点定义与讲解**：“角速度”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            }
        ],
        "content": "- **知识点定义与讲解**：“自转速度”的定义、概念讲解与背景说明。",
        "label": "知识讲解",
        "type": "main"
    },
    {
        "children": [
            {
                "content": "- **知识点定义与讲解**：“太阳日”的定义、概念讲解与背景说明。",
                "label": "知识讲解",
                "type": "sub"
            }
        ],
        "content": "- **知识点定义与讲解**：“自转周期”的定义、概念讲解与背景说明。",
        "label": "知识讲解",
        "type": "main"
    },
    {
        "children": [],
        "content": "- **知识点定义与讲解**：“自转方向”的定义、概念讲解与背景说明。",
        "label": "知识讲解",
        "type": "main"
    },
    {
        "content": "- 本节内容围绕 **地球自转** 展开，涵盖其相关原理与知识点。\n- 各子知识点之间的关系与逻辑顺序。",
        "label": "小结",
        "type": "main"
    },
    {
        "content": "## 习题 1\n### 地方时单选题（普通）.docx\n2020年11月6日11时19分（北京时间），全球首颗6G试验卫星“电子科技大学号”搭载长征六号遥三运载火箭在山西太原卫星发射中心成功升空，并顺利进入预定轨道，标志着中国航天正式进入6G探索时代。据此完成1-2题。\n1．全球首颗6G试验卫星“电子科技大学号”成功升空时，在伦敦的华人收看电视直播的当地区时是（   ）\nA．11月6日3时19分\tB．11月6日4时19分\nC．11月6日5时19分\tD．11月6日6时19分\n2．我国四大航天发射基地中，地球自转的线速度最快的是（   ）\nA．海南文昌（约20°N）\tB．四川西昌（约28°N）\nC．山西太原（约38°N）\tD．甘肃酒泉（约40°N）",
        "label": "习题",
        "type": "main"
    }
],
    "textbook_pdf_path": "/home/ubuntu/work/kmcGPT/temp/resource/中小学课程/高中 地理/选必1/选必1 教材/第一单元地球自转部分.pdf",  # 替换为你本地真实PDF路径
    "llm": "qwen",  # 或 "qwen"
    "top_p": 0.9,
    "temperature": 0.7
}

# # ========== 保存路径配置 ==========
# output_dir = "/home/ubuntu/work/kmcGPT/temp/resource/测试结果"
# os.makedirs(output_dir, exist_ok=True)
#
# filename = f"{payload['knowledge_point']}.md"
# output_path = os.path.join(output_dir, filename)

# ========== 请求发送 ==========
response = requests.post(url, json=payload, stream=True)
print(response.text)
