import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('myapp')


def process_solr_query(solr_query):
    # 使用正则表达式识别并处理TI, JTI, AU字段
    def add_or_all(match):
        field = match.group(1)
        value = match.group(2)
        return f'{field}:"{value}" OR ALL:"{value}"'

    # 匹配TI、JTI、AU字段并添加OR ALL
    processed_query = re.sub(r'(TI|JTI|AU):"([^"]+)"', add_or_all, solr_query)
    logger.info(f"Processed solr_query: {processed_query}")
    return processed_query


# 示例Solr查询字符串
solr_query = r'JTI:"社会日报" AND TI:"啼笑因缘" AND AU:"何可人"'

# 处理Solr查询字符串
processed_solr_query = process_solr_query(solr_query)

# 输出处理后的结果
print(processed_solr_query)

