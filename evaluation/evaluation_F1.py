# read lines and calculate F1/EM
import collections
import string
import re


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        # \b 表示单词边界
        # (a|an|the) 匹配 a 或 an 或 the
        # \b 再次表示单词边界
        # re.UNICODE 标志使 \b 能正确处理 Unicode 字符
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        # 将文本中的连续空格替换为单个空格
        return ' '.join(text.split())
    def remove_punc(text):
        # 移除文本中的标点符号
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))





def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))