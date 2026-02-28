[Don't Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities](https://aclanthology.org/2020.coling-main.518.pdf).
## The Summary
The authors introduce a new annotated dataset of **PCL** (patronising and condescending language) towards vulnerable communities, and show that identifying PCL is hard for standard NLP models, with language models such as BERT achieving the best results.
**PCL**: Language use which shows a **superior attitude** towards others or depicts them in a **compassionate** way. Not always a conscious effect — the author often intends to help the person or group they refer to.
### Dataset
More than 10K paragraphs extracted from news stories, annotated to indicate the presence of PCL. Paragraphs were retrieved from the News on Web (NoW) corpus based on ten selected keywords: _disabled_, _homeless_, _hopeless_, _immigrant_, _in need_, _migrant_, _poor families_, _refugee_, _vulnerable_, _women_, covering 20 English-speaking countries.
The authors propose a **two-level taxonomy** of PCL categories. At the higher level, there are three groups: _The Saviour_, _The Expert_, _The Poet_.
These are subdivided into **seven** fine-grained subcategories: _Unbalanced power relations_, _Shallow solution_ (under The Saviour); _Presupposition_, _Authority voice_ (under The Expert); _Metaphor_, _Compassion_, _The poorer, the merrier_ (under The Poet).
All PCL text spans are annotated with one of these subcategory labels.
**Annotation** was conducted in two steps.
1. Annotators determined which paragraphs contain PCL, using a three-level scale (0 = no PCL, 1 = borderline, 2 = clearly PCL). Two main annotators labelled the full dataset, with a third annotator resolving total disagreements (0 vs 2). The individual annotator labels were then combined into a **5-point scale** (0 through 4) that captures degrees of agreement and borderline cases. For the experiments, labels 0–1 were treated as negative and 2–4 as positive, yielding 995 positive examples.
2. For paragraphs identified as containing PCL, annotators used the BRAT tool to mark specific text spans and assign subcategory labels.
## Experiments
Some forms of PCL can be detected by even simple baselines, while the considered models also struggle to detect certain categories of PCL. There are several different methods used to provide **baselines** for predicting the presence of PCL (Task 1: binary classification), and predicting PCL categories (Task 2: multi-label classification).
- **SVM-WV**: SVM with averaged 300d Word2Vec Skip-gram embeddings.
- **SVM-BoW**: SVM with TF-IDF weighted Bag-of-Words features.
- **BiLSTM**: Bidirectional LSTM using the same Word2Vec embeddings.
- **Fine-tuned Language Models**: BERT-base-cased, BERT-large-cased, RoBERTa-base, and DistilBERT.
- **Random**: 50% class probability baseline.
All considered methods clearly outperform the random baseline, and the BERT-based methods achieve the best results, with RoBERTa slightly ahead overall (F1 = 70.63 for Task 1).
_Unbalanced power relations_ and _Compassion_ appear relatively easy to detect, likely because they rely on identifiable lexical cues (e.g. _us_, _they_, _help_, _must_, and flowery adjectives).
_The poorer, the merrier_ has poor results explainable by its very small sample size (64 instances).
The _Metaphor_ category also has poor results despite having a comparable number of training examples to _Shallow solution_ and _Authority voice_, suggesting that detecting metaphorical PCL requires forms of world knowledge that current models lack.
## Strengths and Weaknesses
**Originality**: Lots of existing research within language studies, sociolinguistics, politics, and medicine into PCL, but research in NLP has been mainly focused on more explicit and aggressive forms of harmful language (e.g. hate speech, offensive language). More recently, Wang and Potts (2019) modelled condescension in direct communication from an NLP perspective via the Talkdown corpus.
The paper's main innovation is introducing a **two-level taxonomy** of PCL categories and providing a carefully annotated dataset specifically targeting PCL towards vulnerable communities in news media. 
he dataset is a genuinely valuable resource: it addresses a gap in NLP where subtle, well-intentioned but harmful language has been understudied.
However, the **modelling contribution is limited**. The authors apply well-known baselines (SVMs, BiLSTM, BERT variants) without proposing any novel architecture or training strategy tailored to the specific challenges of PCL.
For instance, the authors note that detecting certain categories requires world knowledge, but they do not explore any methods that might inject such knowledge (e.g. knowledge graph embeddings or auxiliary training objectives).
The choice to represent paragraphs using averaged Word2Vec embeddings for the SVM-WV baseline is also not discussed. Averaging discards word order entirely, which seems particularly problematic for detecting subtle language patterns like PCL.
These are not necessarily flaws given the paper's framing as a dataset paper with exploratory baselines, but the experimental component is not itself a significant contribution.
Their proposed future extension of the PCL dataset presents important **future work** that will serve the NLP community, including expanding to social media and NGO campaigns.

## Evaluation and Results
They split the problem into two tasks: detecting PCL as a **binary classification** problem (Task 1), and categorising PCL as a **multi-label classification** problem (Task 2).
This decomposition is well-motivated and gives a broader view of the models' capabilities. 
hey evaluated **precision**, **recall**, and **F1-score** for the positive class in Task 1 and for each individual category label in Task 2. This is appropriate given the class imbalance (only 995 of 10,637 paragraphs are positive): using accuracy alone would be misleading here since a classifier predicting "no PCL" for every input would achieve ~90% accuracy.
### Ablation Studies
The paper does not include ablation studies. For example, they do not investigate whether the keyword used to retrieve a paragraph affects model performance, whether the annotation threshold (labels 2+ as positive vs. other cutoffs) impacts results, or whether providing the keyword as an additional input feature helps or hurts. The absence of ablations makes it harder to understand _why_ the models succeed or fail in specific cases.
### Error Analysis
Well-considered and one of the paper's strengths. The authors provide concrete examples of misclassifications by RoBERTa in both Task 1 (Table 5) and Task 2 (Table 6), and offer specific explanations for failure modes.
They observe that false positives often arise from language that superficially resembles PCL (e.g. flowery adjectives in a political context), while false negatives tend to involve categories requiring world knowledge (e.g. _The poorer, the merrier_).
They also note that _The poorer, the merrier_ has the highest inter-annotator agreement despite models struggling with it, which is an interesting finding suggesting that what is obvious to humans may rely on common-sense reasoning that current models lack.
## Clarity and Reproducibility
A very **accessible** and digestible paper. The sociolinguistics background (Sections 3–3.2) is written clearly enough to be understood without NLP expertise, and the taxonomy definitions with examples are effective.
**Reproducibility** is mixed. On the positive side, the authors report hyperparameters for all models (SVM regularisation parameters, kernel types, BiLSTM units, dropout, epochs, batch sizes) and state they fixed random seeds at 1. The dataset is publicly released.
However, several details needed for exact reproduction are missing: learning rates for BERT fine-tuning are not reported, the specific cross-validation fold splits are not provided or described, and preprocessing steps (e.g. how paragraphs were tokenised for each model) are not detailed.
The result is that *approximate reproduction* should be feasible, but *exact replication* would require some guesswork.
## Recommendation
**Weak accept**. The primary contribution, a carefully annotated dataset with a well-motivated two-level PCL taxonomy, is valuable and fills a genuine gap in NLP research on subtle harmful language.
The annotation methodology is thorough, and the error analysis provides useful insights for future work. 
he modelling contribution is limited to applying existing baselines without novel adaptations, but this is acceptable for a dataset-focused paper.
The main weaknesses are the lack of ablation studies and some missing reproducibility details.