import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ---------- CONFIG ----------
SEQ_LEN = 4
DEVICE = "cpu"   # Streamlit runs on CPU
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

# ---------- AUTOCOMPLETE ----------
def generate_suggestions(prompt, max_words=5, temperature=0.7, top_k=5):
    model.eval()

    words = prompt.lower().split()
    suggestions = []

    for _ in range(max_words):
        seq = [word2idx.get(w, 0) for w in words[-SEQ_LEN:]]
        seq = [0] * (SEQ_LEN - len(seq)) + seq

        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x) / temperature
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_idx = torch.topk(probs, top_k)

            next_word = random.choices(
                top_idx[0].tolist(),
                weights=top_probs[0].tolist()
            )[0]

        next_word = idx2word.get(next_word, "")
        words.append(next_word)

        # store progressive suggestion
        suggestions.append(" ".join(words))

    return suggestions

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Next Word Predictor", layout="centered")

st.title("🔍 Search Autocomplete")

user_input = st.text_input("Type your query:", "how to learn")

if st.button("Show suggestions"):
    if user_input.strip() == "":
        st.warning("Please type something")
    else:
        suggestions = generate_suggestions(user_input, max_words=5)

        st.subheader("Suggestions")
        for i, s in enumerate(suggestions, 1):
            st.write(f"{i}. {s}")
