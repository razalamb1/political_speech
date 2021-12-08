from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from numpy.random import choice


def load_data(filepath, number, subreddit):
    df = pd.read_parquet(filepath, engine="fastparquet")
    vectorizer = CountVectorizer(lowercase=True, strip_accents="ascii")
    features = vectorizer.fit_transform(df["total_post"])
    vocabulary = vectorizer.get_feature_names_out()
    counts = pd.DataFrame(data=features.toarray(), columns=vocabulary)
    mean_length = counts.sum(axis=1).mean()
    sd_length = counts.sum(axis=1).std()
    final = counts.sum(axis=0)
    final = final / final.sum()
    post = []
    for i in range(0, number):
        length = np.random.poisson(mean_length)
        draw = list(choice(final.index, length, p=final))
        post.append(" ".join(draw))
        pass
    output = pd.DataFrame()
    output["total_post"] = post
    output["label_type"] = subreddit
    return output


if __name__ == "__main__":
    dem = load_data("10_datasets/democrats.parquet", 4000, "democrat")
    rep = load_data("10_datasets/republican.parquet", 4000, "republican")
    neut = load_data("10_datasets/neutral.parquet", 4000, "neutral")
    final_output = pd.concat([dem, rep, neut])
    final_output.to_parquet("10_datasets/synthetic.parquet", engine="fastparquet")
    pass
