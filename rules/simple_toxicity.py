import os
import re
import sys
from collections import Counter

import emoji
import stanza
from cleantext import clean
from tqdm import tqdm
from tuw_nlp.common.eval import print_cat_stats
from tuw_nlp.ml.learn_rules import read_data
from tuw_nlp.text.pipeline import CachedStanzaPipeline


nlp = CachedStanzaPipeline(None, 'cache/nlp_cache.json', init=lambda: stanza.Pipeline('de'))


FILES = {
    "insult": "insult.txt",
    "profanity": "profanity.txt",
    "groups": "groups.txt",
    "emojis": "emojis.txt",
    "artefacts": "artefacts.txt",
    "unsorted": "unsorted.txt",
    "animals": "animals.txt"}

EXPRESSION_FILES = {
    "not_toxic_expression": "not_toxic_expression.txt"
}

FACT_FILES = {
    "facts": "facts.txt",
    "emojis": "fact_emojis.txt"
}

ENGAGING_FILES = {
    "engaging": "engaging.txt",
    "emojis": "engaging_emojis.txt"
}

ENGAGING_EXPRESSIONS_FILES = {
    "not_engaging_expression": "not_toxic_expression.txt"
}


def create_patterns(files, graph=False):
    pattern_dict = {}
    for cat, fn in files.items():
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), fn)
        with open(path) as f:
            if graph:
                pattern_dict[cat] = set([nlp(line.strip()) for line in f if not line.startswith('#')])
            else:
                pattern_dict[cat] = set([line.strip().lower() for line in f if not line.startswith('#')])
    return pattern_dict


EXPRESSIONS = create_patterns(EXPRESSION_FILES, graph=True)
PATTS = create_patterns(FILES)
ALL_PATTS = set.union(*PATTS.values())

ENGAGING_EXPRESSIONS = create_patterns(ENGAGING_EXPRESSIONS_FILES, graph=True)
ENGAGING = create_patterns(ENGAGING_FILES)
ALL_ENGAGING = set.union(*ENGAGING.values())

FACTS = create_patterns(FACT_FILES)
ALL_FACTS = set.union(*FACTS.values())

print(len(ALL_PATTS) + len(ALL_ENGAGING) + len(ALL_FACTS))


def create_node_list(graph):
    tokens = []
    for word in graph.words:
        if word.head == 0:
            token = (word.text, word.lemma, word.deprel, 'ROOT')
        else:
            token = (word.text, word.lemma, word.deprel, graph.words[word.head - 1].lemma)
        tokens.append(token)
    return tokens


def match_graph(sen, token, graph_sets):
    token_tuple = (token.text, token.lemma, token.deprel, sen.words[token.head - 1].lemma) if token.head != 0 else \
        (token.text, token.lemma, token.deprel, 'ROOT')
    sentence_tokens = create_node_list(sen)
    for graph in graph_sets:
        gr = graph.sentences[0]
        if ' '.join([word.lemma for word in gr.words]) in ' '.join([word.lemma for word in sen.words]):
            return True
        tokens = create_node_list(gr)
        if tokens[0][:2] in [s[:2] for s in sentence_tokens] \
                and len(set(sentence_tokens) & set(tokens)) == len(tokens) - 1 \
                and (token_tuple in tokens or token_tuple[:2] == tokens[0][:2]):
            return True
    return False


def preproc(text, lang='de'):
    text = emoji.demojize(text, language=lang)
    text = re.sub('@[a-zA-Z0-9_]*', '[USER]', text)
    text.replace('#', '')
    return clean(
        text, lower=False, no_urls=True, no_numbers=False,
        no_currency_symbols=True, replace_with_url='[URL]',
        replace_with_number='[NUMBER]', replace_with_currency_symbol='[CUR]',
        lang=lang)


def is_toxic(doc, patterns, all_patterns, expressions=None):
    for sen in doc.sentences:
        for tok in sen.words:
            if tok.text.lower() in all_patterns or tok.lemma.lower() in all_patterns:
                if expressions is not None:
                    if not match_graph(sen, tok, set.union(*expressions.values())):
                        return True, f'{tok.text} / {tok.lemma}'
                else:
                    return True, f'{tok.text} / {tok.lemma}'
            else:
                for emoticon in patterns['emojis']:
                    if emoticon in tok.text.lower() and len(sen.words) > 1:
                        return True, f'{tok.text} / {tok.lemma}'
    return False, None


def main(category, output):
    t_stats = Counter()
    o_stats = Counter()
    with open(output, 'w') as pred_file:
        pred_file.write('comment_text\tresult\n')
        with nlp:
            for text, label in tqdm(read_data(sys.stdin)):
                doc = nlp(text)
                if category.lower() == 'toxic':
                    pred, why = is_toxic(doc, PATTS, ALL_PATTS, expressions=EXPRESSIONS)
                elif category.lower() == 'engaging':
                    pred, why = is_toxic(doc, ENGAGING, ALL_ENGAGING, expressions=ENGAGING_EXPRESSIONS)
                else:
                    pred, why = is_toxic(doc, FACTS, ALL_FACTS)
                pred_file.write(f'{text}\t{pred*1}\n')
                if pred:
                    if label:
                        #print(f'TP: {why}: {text}')
                        t_stats['TP'] += 1
                        o_stats['TN'] += 1
                    else:
                        print(f'FP: {why}: {text}')
                        #print(f'{text}\tFalse')
                        t_stats['FP'] += 1
                        o_stats['FN'] += 1
                else:
                    if label:
                        #print(f'{text}\tTrue')
                        t_stats['FN'] += 1
                        o_stats['FP'] += 1
                    else:
                        t_stats['TN'] += 1
                        o_stats['TP'] += 1
            print(t_stats)
            print_cat_stats({'toxicity': t_stats, 'other': o_stats}, print_avgs=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
