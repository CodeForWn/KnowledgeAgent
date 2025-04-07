import time
import hmac
import base64
import hashlib
import requests
import os
import json
import urllib.parse
import textwrap

# === 你的认证信息 ===
API_KEY = "67ecfe70d70c4"
SECRET_KEY = "P0YpttKRvnnWMskbjrB0jK00WoHF5xYO"  # 用于 x-token 认证
UID = "fa1c8fe0-0fa1-11f0-869a-b78c8dfc12b4"
CHANNEL = ""
TOKEN_CACHE_PATH = "./aippt_token_cache.json"

TOKEN_URL = "https://co.aippt.cn/api/grant/token"
TEMPLATE_URL = "https://co.aippt.cn/api/template_component/suit/search"
TASK_URL = "https://co.aippt.cn/api/ai/chat/v2/task"
SAVE_URL = "https://co.aippt.cn/api/design/v2/save"
EXPORT_URL = "https://co.aippt.cn/api/download/export/file"
EXPORT_RESULT_URL = "https://co.aippt.cn/api/download/export/file/result"

# === 获取签名的token（自动缓存） ===
def get_token():
    if os.path.exists(TOKEN_CACHE_PATH):
        with open(TOKEN_CACHE_PATH, "r") as f:
            data = json.load(f)
            if data["expire"] > time.time() + 86400:  # 离过期还有1天就继续用
                return data["token"]

    timestamp = str(int(time.time()))
    string_to_sign = f"GET@/api/grant/token/@{timestamp}"
    signature = base64.b64encode(hmac.new(SECRET_KEY.encode(), string_to_sign.encode(), hashlib.sha1).digest()).decode()

    headers = {
        "x-api-key": API_KEY,
        "x-timestamp": timestamp,
        "x-signature": signature
    }
    params = {"uid": UID, "channel": CHANNEL}
    response = requests.get(TOKEN_URL, headers=headers, params=params)
    res = response.json()
    if res["code"] == 0:
        token = res["data"]["token"]
        expire = int(time.time()) + res["data"]["time_expire"]
        with open(TOKEN_CACHE_PATH, "w") as f:
            json.dump({"token": token, "expire": expire}, f)
        return token
    else:
        raise Exception("获取 token 失败: " + res.get("msg", "未知错误"))

# === 获取模板列表供用户手动选择 ===
def get_template_list():
    token = get_token()
    headers = {
        "x-api-key": API_KEY,
        "x-token": token,
        "x-channel": CHANNEL
    }

    print("📦 正在获取模板列表...")
    response = requests.get(TEMPLATE_URL, headers=headers)
    res = response.json()

    if res["code"] == 0:
        all_data = res["data"]
        template_list = all_data.get("list", [])
        print(f"✅ 共找到 {len(template_list)} 个模板：\n")

        valid_templates = []
        for tpl in template_list:
            tpl_id = tpl.get("id")
            cover = tpl.get("cover_img")
            if tpl_id and cover:
                print(f"[ID: {tpl_id}] 封面：{cover}")
                valid_templates.append({
                    "id": tpl_id,
                    "cover": cover
                })

            if not valid_templates:
                print("⚠️ 没有找到有效模板，请检查返回数据结构。")
            return valid_templates
        else:
            raise Exception("获取模板列表失败：" + res.get("msg", "未知错误"))


def get_template_list_for_markdown(token):
    headers = {
        "x-api-key": API_KEY,
        "x-token": token,
        "x-channel": CHANNEL
    }
    params = {
        "page": 1,
        "page_size": 20
    }

    print("📦 正在获取推荐模板套装列表...")
    response = requests.get("https://co.aippt.cn/api/template_component/suit/search", headers=headers, params=params)

    try:
        data = response.json()
    except Exception as e:
        print("❌ JSON解析失败：", e)
        print("原始响应：\n", response.text)
        raise

    if data["code"] != 0:
        raise Exception("❌ 获取模板失败：" + data.get("msg", "未知错误"))

    templates = data["data"].get("list", [])
    if not templates:
        raise Exception("❌ 没有获取到推荐模板")

    for tpl in templates:
        print(f"[ID: {tpl['id']}] 封面：{tpl.get('cover_img', '无')}")

    return templates[0]["id"]  # ✅ 默认返回第一个


# === 渲染主函数：输入markdown + title + template_id，输出下载链接 ===
def render_markdown_to_ppt(title, markdown_text):
    token = get_token()
    headers = {
        "x-api-key": API_KEY,
        "x-token": token,
        "x-channel": CHANNEL
    }
    print(headers)
    # === 创建任务 ===
    data = {
        "type": "7",  # markdown粘贴生成
        "title": title,
        "content": markdown_text,
        "id": ""
    }
    task_resp = requests.post(TASK_URL, headers=headers, data=data)
    task_data = task_resp.json()
    if task_data["code"] != 0:
        raise Exception("❌ 创建任务失败：" + task_data.get("msg", "未知错误"))
    task_id = task_data["data"]["id"]
    print(f"✅ 任务创建成功，task_id: {task_id}")

    template_id = get_template_list_for_markdown(token)
    print(f"✅ 使用推荐模板 ID：{template_id}")

    # === 保存作品 ===
    payload = {
        "name": title,
        "task_id": task_id,
        "template_id": template_id,
        "template_type": 1
    }
    headers_form = headers.copy()
    headers_form["Content-Type"] = "application/x-www-form-urlencoded"
    encoded_payload = urllib.parse.urlencode(payload)

    print("📤 正在保存作品...")
    print("请求体：", encoded_payload)
    save_resp = requests.post(SAVE_URL, headers=headers_form, data=encoded_payload)
    print("响应状态码：", save_resp.status_code)
    print("请求结果:", save_resp.text)

    if save_resp.headers.get("Content-Type", "").startswith("text/html"):
        print("❌ 返回了 HTML 页面，可能是接口路径或请求方式错误")
        print("HTML 内容如下：\n", save_resp.text)
        raise Exception("返回 HTML 非预期响应，终止解析")

    try:
        save_data = save_resp.json()
    except Exception as e:
        print("❌ JSON 解析失败：", e)
        print("原始响应内容：\n", save_resp.text)
        raise

    if save_data["code"] != 0:
        raise Exception("❌ 保存作品失败：" + save_data.get("msg", "未知错误"))

    user_design_id = save_data["data"]["id"]
    print(f"✅ 作品保存成功，作品ID: {user_design_id}")

    # === 导出 PPT 文件 ===
    export_payload = {
        "id": user_design_id,
        "format": "ppt",
        "edit": "true",
        "files_to_zip": "false"
    }
    export_resp = requests.post(EXPORT_URL, headers=headers, data=export_payload)
    export_data = export_resp.json()
    if export_data["code"] != 0:
        raise Exception("❌ 导出任务提交失败：" + export_data.get("msg", "未知错误"))
    task_key = export_data["data"]
    print(f"📤 导出任务创建成功，task_key: {task_key}")

    # === 轮询导出结果 ===
    print("⏳ 正在轮询导出任务结果...")
    for i in range(30):
        time.sleep(2)
        result_resp = requests.post(EXPORT_RESULT_URL, headers=headers, data={"task_key": task_key})
        result_data = result_resp.json()
        if result_data["code"] == 0 and result_data["data"]:
            download_url = result_data["data"][0]
            print("✅ 导出成功！下载链接：")
            print(download_url)
            return download_url

    raise Exception("❌ 轮询超时，未获取到导出链接。")


# === 测试入口 ===
if __name__ == "__main__":
    title = "地球运动"
    markdown = textwrap.dedent("""\
        # 地球运动

        ## 地球自转
        地球每天自转一圈，产生昼夜交替现象。

        ## 地球公转
        地球一周公转约365天，产生四季变化。

        ## 公转与黄赤交角
        太阳直射点随季节移动，是四季的根本原因。
    """)

    render_markdown_to_ppt(title, markdown)