"""Topic representation and quality metrics shared by all scripts.

Stop words are kept for the sentence embeddings but removed when computing the c-TF-IDF topic
keywords, on which coherence is evaluated. The same stop-word list and tokenisation are used for
the BERTopic models and for the LDA baseline.
"""
import re

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer

from settings import TOP_N_WORDS

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn CountVectorizer default token pattern


def french_stopwords():
    """NLTK French stop-word list (157 words)."""
    import nltk
    try:
        from nltk.corpus import stopwords
        words = stopwords.words("french")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        words = stopwords.words("french")
    return sorted(set(words))


def keyword_vectorizer(**kwargs):
    """Vectorizer used for topic keywords (BERTopic c-TF-IDF) and for the LDA document-term matrix."""
    return CountVectorizer(stop_words=french_stopwords(), **kwargs)


def tokenize(docs):
    """Reference-corpus tokenisation for coherence: lower-cased word tokens, stop words kept."""
    return [TOKEN_RE.findall(d.lower()) for d in docs]


def bertopic_topic_words(topic_model, top_n=TOP_N_WORDS):
    """Top words of every non-outlier topic of a fitted BERTopic model."""
    return {t: [w for w, _ in topic_model.get_topic(t)][:top_n]
            for t in sorted(topic_model.get_topics()) if t != -1}


def coherence(topics_words, texts, measure="c_v", top_n=TOP_N_WORDS):
    """gensim coherence ('c_v', 'c_npmi' or 'c_uci') with the corpus itself as reference."""
    topics_words = [t for t in topics_words if len(t) > 0]
    if not topics_words:
        return float("nan")
    model = CoherenceModel(topics=topics_words, texts=texts, dictionary=Dictionary(texts),
                           coherence=measure, topn=top_n, processes=1)
    return float(model.get_coherence())


def topic_diversity(topics_words):
    """Share of unique words among the top words of all topics (Dieng et al., 2020)."""
    words = [w for t in topics_words for w in t]
    return len(set(words)) / len(words) if words else float("nan")


def quality_metrics(topics_words, texts):
    """C_V, C_NPMI, C_UCI and Topic Diversity for a list (or dict) of topic word lists."""
    tw = list(topics_words.values()) if isinstance(topics_words, dict) else list(topics_words)
    return {"c_v": coherence(tw, texts, "c_v"), "c_npmi": coherence(tw, texts, "c_npmi"),
            "c_uci": coherence(tw, texts, "c_uci"), "topic_diversity": topic_diversity(tw)}
