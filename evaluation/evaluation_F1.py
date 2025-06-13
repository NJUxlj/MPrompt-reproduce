# read lines and calculate F1/EM
import collections
import string
import re


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        # \b ��ʾ���ʱ߽�
        # (a|an|the) ƥ�� a �� an �� the
        # \b �ٴα�ʾ���ʱ߽�
        # re.UNICODE ��־ʹ \b ����ȷ���� Unicode �ַ�
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        # ���ı��е������ո��滻Ϊ�����ո�
        return ' '.join(text.split())
    def remove_punc(text):
        # �Ƴ��ı��еı�����
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