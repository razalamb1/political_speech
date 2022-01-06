# Predicting Political Text Using Reddit Data
## Authors: Erika Fox and Raza Lamb

The goal of this project was to utilize Reddit data from specific subreddit in order to build language models in Python to classify political text into three categories: Repbulican, Democrat, or neutral.

### Project Design and Workflow

For this project, two models were built, trained, and evaluated. First, a probabilistic model: we utilized Naive Bayes text classifier from `scikit-learn`. Second, a neural network: this model utilized pre-trained Bidirectional Encoder Representations from Transformers (BERT), a state-of-the-art transformer architecture used for representing words. This model was implemented in PyTorch and the Huggingface transformers library.

First, in order to obtain test/train/validation data, we utilized the code in the `00_scraping` folder in this repository to scrape all comments and posts from three subreddits: r/democrats, r/Republican, and r/NeutralPolitics. For this specific analysis, posts and comments were limited to those occuring in the calendar year 2020, to posts that did not link to articles or memes, and finally to posts longer than 100 characters. This resulted in approximately 800 posts per subreddit, and 4,000 comments were randomly selected (after selecting on above criteria) in each subreddit.

### Results

Full results of both models are available in the [report](https://github.com/razalamb1/political_speech/blob/main/NLP%20Final%20Project%20-%20Raza%20Lamb%20and%20Erika%20Fox%20.pdf), located in the root directory of this repository.

### Recreation

This project should be fully replicable, by following the workflow: `00_scraping` -> `05_cleaning` -> `15_code`. However, the neural network should **NOT** be trained locally, for efficiency sake. For this project, it was trained on Google CoLab Pro.




