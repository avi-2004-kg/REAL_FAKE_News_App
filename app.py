import re
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fake_news_lr_model.joblib"
VECTORIZER_PATH = BASE_DIR / "fake_news_tfidf_vectorizer.joblib"

st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="centered")


@st.cache_resource
def setup_nltk():
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for package, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


setup_nltk()


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH.name}. Put it beside app.py.")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Missing {VECTORIZER_PATH.name}. Put it beside app.py.")
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


try:
    model, tfidf = load_artifacts()
except Exception as exc:
    st.error("The trained model files could not be loaded.")
    st.exception(exc)
    st.stop()


STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["reuters", "ap", "afp", "said"])
LEMMATIZER = WordNetLemmatizer()


def strip_dateline(text):
    return re.sub(
        r"^.{0,80}?\(reuters\)\s*-\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text):
    text = clean_text(strip_dateline(text))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def predict_news(headline, article):
    combined = f"{headline} {article}".strip()
    if not combined:
        raise ValueError("Please enter a headline or article.")

    cleaned = preprocess(combined)
    if not cleaned:
        raise ValueError("The supplied text became empty after preprocessing.")

    features = tfidf.transform([cleaned])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    label = "REAL" if prediction == 1 else "FAKE"
    confidence = float(probabilities[list(model.classes_).index(prediction)])

    probs = {
        "FAKE": float(probabilities[list(model.classes_).index(0)]) if 0 in model.classes_ else 0.0,
        "REAL": float(probabilities[list(model.classes_).index(1)]) if 1 in model.classes_ else 0.0,
    }
    return label, confidence, probs


st.title("📰 Fake News Classifier")
st.write("Enter a news headline and/or article text to classify it as **REAL** or **FAKE**.")

st.info(
    "⚠️ This is a statistical text classifier. It does not independently "
    "fact-check claims against external sources."
)

st.divider()

headline = st.text_input("News Headline", placeholder="Enter the headline...")
article = st.text_area(
    "News Article",
    placeholder="Paste the news article text here...",
    height=250,
)

if st.button("🔍 Classify News", type="primary", use_container_width=True):
    try:
        label, confidence, probs = predict_news(headline, article)

        st.divider()
        st.subheader("Prediction")

        if label == "REAL":
            st.success(f"### Prediction: {label}")
        else:
            st.error(f"### Prediction: {label}")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Model Confidence", f"{confidence * 100:.2f}%")
        with c2:
            st.metric("Predicted Class", label)

        st.subheader("Class Probabilities")
        st.write(f"**FAKE:** {probs['FAKE'] * 100:.2f}%")
        st.progress(probs["FAKE"])
        st.write(f"**REAL:** {probs['REAL'] * 100:.2f}%")
        st.progress(probs["REAL"])

        st.caption(
            "The probability is the model's estimated class probability, "
            "not a guarantee that the underlying claims are true or false."
        )

    except Exception as exc:
        st.error("Unable to classify the supplied text.")
        st.exception(exc)


with st.sidebar:
    st.header("About the Model")
    st.markdown(
        """
**Model:** Logistic Regression

**Features:** TF-IDF

**N-grams:** Unigrams + Bigrams

**Vocabulary:** 5,000 terms

**Classes:** 0 = FAKE, 1 = REAL
"""
    )
    st.divider()
    st.markdown("**Pipeline:** Text → preprocessing → TF-IDF → Logistic Regression → prediction")
    st.caption("Built with Streamlit and scikit-learn.")
