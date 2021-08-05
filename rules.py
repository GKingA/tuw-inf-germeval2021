import pandas as pd
import sys
from read_data import read_toxic, demojify, clean_other


def is_dotdot(text):
    if text.endswith('..'):
        return True
    return False


def is_caps(text):
    cnt = 0
    text = text.replace('@USER', '')
    text = text.replace('@MEDIUM', '')
    text = text.replace('@MODERATOR', '')
    text = demojify(text)
    # Not really tokens
    tokens = text.split()
    for token in tokens:
        if token.isupper() and token.isalpha() and len(token) > 3:
            cnt += 1
    if cnt >= 2:
        return True
    return False


def has_emoji(text):
    if text == demojify(text):
        return False
    return True


def has_angry_emoji(text):
    angry = ['angry_face',
             'anger_symbol',
             'anguished_face',
             'face_with_symbols_on_mouth',
             'face_with_steam_from_nose',
             'no_entry',
             #'face_with_rolling_eyes',
             'face_with_raised_eyebrow',
             #'facepalming'
             ]
    dem = demojify(text, 'en')
    for a in angry:
        if a in dem:
            return True
    return False


def has_link(text):
    cleaned = clean_other(text)
    if '[URL]' in cleaned and '[URL]' != cleaned:
        return True
    return False


def has_qmark(text):
    if '?' in text and '?' not in text[-5:]:
        return True
    return False


def engaging(text):
    return (has_qmark(text)) * 1


def fact_claiming(text):
    return (has_link(text)) * 1


def toxic(text):
    return (is_caps(text)) * 1


def simple_rule(df, out, method):
    method_dict = {'toxic': toxic, 'engaging': engaging, 'fact': fact_claiming}
    with open(out, "w") as pred_file:
        pred_file.write('comment_text\tresult\n')
        for text in df.text:
            res = method_dict[method](text)
            pred_file.write(f'{text}\t{res}\n')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise Exception("Use: python3 rules.py [input file] [category: toxic/engaging/fact] [output file]")
    data, _ = read_toxic(sys.argv[1], split=False)
    simple_rule(pd.concat(data[sys.argv[2].upper()]), sys.argv[3], sys.argv[2])
