import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ---------- CONFIG ----------
SEQ_LEN = 4
DEVICE = "cpu"
TOP_K = 5

# ---------- MODEL ----------
class WordLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, vocab_size + 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# ---------- LOAD MODEL ----------
checkpoint = torch.load("autocomplete_model.pth", map_location=DEVICE)

word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]

model = WordLSTM(len(word2idx))
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# ---------- PROGRESSIVE AUTOCOMPLETE ----------
def generate_progressive_suggestions(prompt, max_words=5):
    words = prompt.lower().split()
    suggestions = []

    for _ in range(max_words):
        seq = [word2idx.get(w, 0) for w in words[-SEQ_LEN:]]
        seq = [0] * (SEQ_LEN - len(seq)) + seq
        x = torch.tensor(seq).unsqueeze(0)

        with torch.no_grad():
            probs = F.softmax(model(x), dim=-1)
            top_probs, top_idx = torch.topk(probs, TOP_K)

            next_word_id = random.choices(
                top_idx[0].tolist(),
                weights=top_probs[0].tolist()
            )[0]

        next_word = idx2word.get(next_word_id, "")
        words.append(next_word)
        suggestions.append(" ".join(words))

    return suggestions

# ---------- SAME-LENGTH SUGGESTIONS ----------
def generate_alternative_queries(prompt, num_sentences=5, next_words=4):
    alternatives = []

    for _ in range(num_sentences):
        words = prompt.lower().split()

        for _ in range(next_words):
            seq = [word2idx.get(w, 0) for w in words[-SEQ_LEN:]]
            seq = [0] * (SEQ_LEN - len(seq)) + seq
            x = torch.tensor(seq).unsqueeze(0)

            with torch.no_grad():
                probs = F.softmax(model(x), dim=-1)
                top_probs, top_idx = torch.topk(probs, TOP_K)

                next_word_id = random.choices(
                    top_idx[0].tolist(),
                    weights=top_probs[0].tolist()
                )[0]

            next_word = idx2word.get(next_word_id, "")
            words.append(next_word)

        alternatives.append(" ".join(words))

    return alternatives


# ---------- UI ----------
st.set_page_config(page_title="Next Word Predictor", layout="wide")

st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
    max-width: 1200px;
    margin: auto;
}

.card {
    padding: 16px 20px;
    margin-bottom: 12px;
    border-radius: 10px;
    background-color: #ffffff;
    border: 1px solid #e1e4e8;
    font-size: 16px;
    color: #000000;   /* 👈 FIX */
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #000000;   /* 👈 FIX */
}
</style>
""", unsafe_allow_html=True)


st.title("🔍 Smart Next Word Predictor")
st.caption("LSTM-based next-word prediction trained on 1.4 million words")

# ---------- INPUT ----------
query = st.text_input(
    "Type your query",
    placeholder="how to learn python"
)

# ---------- BUTTON ----------
predict = st.button("🔮 Predict")

# ---------- OUTPUT (PARALLEL) ----------
if predict:
    if query.strip() == "":
        st.warning("Please type a query")
    elif len(query.split()) < 2:
        st.info("Type at least 2 words to get predictions")
    else:
        col1, col2 = st.columns(2, gap="large")

        # Progressive Suggestions
        with col1:
            st.subheader("📈 Progressive Suggestions")
            progressive = generate_progressive_suggestions(query, max_words=5)

            for i, s in enumerate(progressive, 1):
                st.markdown(
                    f'<div class="card"><strong>{i}.</strong> {s}</div>',
                    unsafe_allow_html=True
                )

        # Alternative Queries
        with col2:
            st.subheader("📈 Sentence Suggestions")
            alternatives = generate_alternative_queries(
                query,
                num_sentences=5,
                next_words=4
            )

            for i, s in enumerate(alternatives, 1):
                st.markdown(
                    f'<div class="card"><strong>{i}.</strong> {s}</div>',
                    unsafe_allow_html=True
                )

