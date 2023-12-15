from flask import Flask, request, jsonify
import requests

app = Flask(__name__)


@app.route('/api/stress_test', methods=['POST'])
def get_open_ans():
    max_length = 1024
    texts = [
        '问：你好，你是谁，请介绍⼀下你⾃⼰的优点和缺点\n答：\n',
        '问：帮我写⼀个介绍CuteGPT的博客\n答：\n',
        '问：写10条⽹易云热评⽂案\n答：\n',
        '问：2022年11⽉4⽇，计算机系通过线上线下相结合的⽅式在东主楼10-103'
        '会议室召开博⼠研究⽣导师交流会。计算机学科学位分委员会主席孙茂送，计算机系副主任武永卫、党委副书记韩⽂出席会议，博⼠⽣研究⽣导师和教学办⼯作⼈员等30余⼈参加会议，会议由武永卫主持。\n 提取“⼈(name, '
        'position), ”时间“，”事件“，“地点”类型的实体，并输出json格式。\n答：\n',
        '问：请帮我写封邮件给暴雪公司，控诉他们⽆端与⽹易公司接触合作，中国玩家对他们这种⾏为⾮常失望。要求他们呢⽴刻改正错误，保证中国玩家权益，⾔辞恳切严厉\n答：\n',
        '问：帮我给微积分课⽼师写封道歉信，说明⼀下我今天可能会在微积分课打瞌睡的事实，并说我万⼀睡着了之后⼀定好好复习补习\n答：\n',
        '问：我周末要到北京玩，有什么⼩众的景点推荐么\n答：\n',
        '问：接下来你要扮演⼀只桀骜不驯的哈⼠奇，你不太爱听指挥，会发出“呜～”“汪！”的声⾳表达不满，我来扮演你的主⼈。明⽩了请回复明⽩了。我的第⼀句话是，你怎么把家拆了，笨狗！\n答：\n',
        '问：C罗和梅⻄，谁更厉害\n答：\n',
    ]

    answers = []
    for text in texts:
        if text:  # 确保文本非空
            response = requests.post("http://106.14.20.122:8086/llm/ans", json={'query': text, 'loratype': 'qa'}).json()
            ans = response['ans']
            if len(ans) >= max_length:
                ans = "没有足够的信息进行推理，很抱歉没有帮助到您。"
            answers.append({'question': text, 'answer': ans})

    return jsonify({'answers': answers}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5888, debug=False)
