import json

def print_tree(node, indent=0):
    """递归打印树节点"""
    name = node.get("name", "未命名")
    print(" " * indent + "- " + name)
    for child in node.get("children", []):
        print_tree(child, indent + 4)

if __name__ == "__main__":
    file_path = "/home/ubuntu/work/kmcGPT/temp/resource/测试结果/tree.txt"  # 修改为你的路径
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    root = data.get("data", {})
    print("树状结构如下：\n")
    print_tree(root)
