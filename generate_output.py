from argparse import ArgumentParser
from read_data import read_csv
import pandas as pd


def generate(toxic, engaging, fact, out):
    with open(out, 'w') as submission:
        submission.write('comment_id,Sub1_Toxic,Sub2_Engaging,Sub3_FactClaiming\n')
        toxic_df = read_csv(toxic, names=["comment_id_toxic", "comment_text", "toxic"], force_names=True)
        engaging_df = read_csv(engaging, names=["comment_id_engaging", "comment_text", "engaging"], force_names=True)
        fact_df = read_csv(fact, names=["comment_id_fact", "comment_text", "fact"], force_names=True)
        df = pd.concat([toxic_df, engaging_df, fact_df], axis=1)
        for id_, t, e, f in zip(df.comment_id_toxic, df.toxic, df.engaging, df.fact):
            submission.write(f'{id_},{t},{e},{f}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--toxic", "-t", required=True)
    parser.add_argument("--engaging", "-e", required=True)
    parser.add_argument("--fact", "-f", required=True)
    parser.add_argument("--out", "-o", required=True)
    args = parser.parse_args()
    generate(args.toxic, args.engaging, args.fact, args.out)
