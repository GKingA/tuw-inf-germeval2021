import argparse
import logging
import sys

import graphviz
import preprocessor as twp
from sklearn import tree
from brise_nlp.plandok.ml import FeatureExtractor
from tuw_nlp.text.utils import save_parsed


def load_data(stream):
    data = []
    for line in stream:
        fields = line.strip().split('\t')
        text, label = fields[:2]
        data.append((text, label))
    return data


def split_data(data, ratio=0.9):
    split = int(len(data) * ratio)
    return data[:split], data[split:]


def preprocess_tweet(raw_text):
    twp.set_options(twp.OPT.URL, twp.OPT.EMOJI)
    text = twp.clean(raw_text)
    text = text.replace('|LBR|', ' ')
    return text


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-cd", "--cache-dir", default=None, type=str)
    parser.add_argument("-cc", "--conll-cache", default=None, type=str)
    return parser.parse_args()


def main():
    logging.basicConfig(
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    logging.getLogger().setLevel(logging.WARNING)
    args = get_args()

    feat_extractor = FeatureExtractor(args)
    print('generating features...')
    for sen_id, (raw_text, label) in enumerate(load_data(sys.stdin)):
        # print(sen_id, attrs)
        attr_names = [label] if label != 'OTHER' else []
        text = preprocess_tweet(raw_text)
        try:
            feat_extractor.featurize_sen(sen_id, text, attr_names)
        except IndexError:
            print('error on line:', sen_id, text, attr_names)
            feat_extractor.sen_ids.remove(sen_id)

    attrs = list(feat_extractor.label_vocab.word_to_id.keys())

    for attr in attrs:
        print(f'learning {attr}...')
        # print('generating X, y...')
        X, y = feat_extractor.get_x_y(attr)
        # print('X shape:', X.shape)
        # print(X)
        # print('y shape:', X.shape)
        # print(y)

        clf = tree.DecisionTreeClassifier()
        clf.fit(X, y)

        feature_names = feat_extractor.get_feature_graph_strings()

        dot_data = tree.export_graphviz(
            clf, feature_names=feature_names, class_names=['No', 'Yes'],
            out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render(attr)

    if args.conll_cache:
        save_parsed(feat_extractor.parsed_text, args.conll_cache)


if __name__ == "__main__":
    main()
