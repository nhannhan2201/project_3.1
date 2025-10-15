
import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Literal

# -------------------------------
# EmbeddingVectorizer class
# -------------------------------
class EmbeddingVectorizer:
    def __init__(
        self,
        model_name: str = 'intfloat/multilingual-e5-base',
        normalize: bool = True
    ):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def _format_inputs(
        self,
        texts: List[str],
        mode: Literal['query', 'passage']
    ) -> List[str]:
        if mode not in {"query", "passage"}:
            raise ValueError("Mode must be either 'query' or 'passage'")
        return [f"{mode}: {text.strip()}" for text in texts]

    def transform_numpy(self, texts, mode: Literal['query', 'passage'] = 'query') -> np.ndarray:
        inputs = self._format_inputs(texts, mode)
        embeddings = self.model.encode(inputs, normalize_embeddings=self.normalize)
        return np.array(embeddings)

# -------------------------------
# Load model và mappings
# -------------------------------
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('cluster_to_label.pkl', 'rb') as f:
        cluster_to_label = pickle.load(f)
    with open('id_to_label.pkl', 'rb') as f:
        id_to_label = pickle.load(f)
    vectorizer = EmbeddingVectorizer()
    return kmeans, cluster_to_label, id_to_label, vectorizer

kmeans, cluster_to_label, id_to_label, vectorizer = load_models()

# -------------------------------
# Giao diện người dùng
# -------------------------------
st.title("🌐 Ứng dụng Dự đoán Chủ Đề Bằng Embedding + KMeans")
st.write("Nhập văn bản, hệ thống sẽ tính embedding và dự đoán chủ đề tương ứng.")

user_input = st.text_area("Nhập văn bản tại đây:")

if st.button("Dự đoán"):
    if not user_input.strip():
        st.warning("⚠️ Vui lòng nhập văn bản trước khi dự đoán.")
    else:
        # Tính embedding
        X_new = vectorizer.transform_numpy([user_input], mode='query')

        # Dự đoán cụm
        cluster_id = kmeans.predict(X_new)[0]

        # Gán nhãn
        label_id = cluster_to_label[cluster_id]
        label_name = id_to_label[label_id]

        st.success(f"✅ Kết quả dự đoán: **{label_name}**")
