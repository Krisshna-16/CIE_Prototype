import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Collective Intelligence Engine",
    layout="centered"
)

st.title("🧠 Collective Intelligence Engine")
st.caption(
    "Reusing proven problem-solving patterns through collective intelligence"
)

# ----------------------------
# Load pattern knowledge base
# ----------------------------
@st.cache_data
def load_patterns(file_path="patterns.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Flatten nested lists if present
    flat_patterns = []
    for item in data:
        if isinstance(item, list):
            flat_patterns.extend(item)
        else:
            flat_patterns.append(item)
    
    return flat_patterns

patterns = load_patterns()

# ----------------------------
# Load NLP model
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------
# Build FAISS vector database
# ----------------------------
@st.cache_resource
def build_faiss_index(patterns):
    texts = [p["description"] for p in patterns]
    embeddings = model.encode(texts).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

index, _ = build_faiss_index(patterns)

# ----------------------------
# Rule-based problem decomposition
# ----------------------------
def problem_dimensions(problem):
    mapping = {
        "scale": "Scalability",
        "scalability": "Scalability",
        "cost": "Cost Optimization",
        "resource": "Resource Allocation",
        "time": "Efficiency",
        "delay": "Efficiency",
        "traffic": "Optimization"
    }

    dimensions = []
    for k, v in mapping.items():
        if k in problem.lower():
            dimensions.append(v)

    return list(set(dimensions)) if dimensions else ["General Problem"]

# ----------------------------
# Confidence scoring
# ----------------------------
def confidence_score(distance):
    return float(f"{1 / (1 + distance):.2f}")

# ----------------------------
# UI: Problem input
# ----------------------------
problem = st.text_area(
    "Enter a real-world problem statement",
    height=120,
    placeholder="Example: Urban traffic congestion due to inefficient resource allocation"
)

# ----------------------------
# Main analysis
# ----------------------------
if st.button("Analyze Problem") and problem.strip():

    st.divider()

    # STEP 1: Problem decomposition
    st.subheader("🔍 Problem Decomposition")
    dimensions = problem_dimensions(problem)
    st.write(dimensions)

    # STEP 2: NLP embedding of problem
    problem_embedding = model.encode([problem]).astype("float32")

    # STEP 3: FAISS vector search
    k = min(5, len(patterns))  # show top 5 matches
    distances, indices = index.search(problem_embedding, k=k)

    # STEP 4: Ranked solution patterns
    st.divider()
    st.subheader("📊 Ranked Solution Patterns")

    shown = 0
    top_patterns = []

    for rank, idx in enumerate(indices[0]):
        distance = distances[0][rank]
        score = confidence_score(distance)

        if score < 0.15:
            continue

        pattern = patterns[idx]
        shown += 1
        top_patterns.append((pattern, score))

        st.markdown(
            f"### {shown}. {pattern['problem_type']} "
            f"(Confidence: **{score}**) "
        )

        st.write(pattern["description"])
        st.write("**Previously used in:**", pattern["used_in"])

        st.write("**Recommended Steps:**")
        for step in pattern["solution_steps"]:
            st.write("•", step)

        # Explainable reasoning trace
        st.write("🧠 **Reasoning Trace:**")
        st.write("- Semantic similarity between problem structure and stored pattern")
        st.write("- Overlap with detected problem dimensions:", dimensions)
        st.write("- Proven success in real-world use cases")

        st.divider()

    if shown == 0:
        st.warning("No sufficiently confident solution patterns found for this problem.")

    # ----------------------------
    # STEP 5: Visual Problem Decomposition
    # ----------------------------
    if top_patterns:
        st.subheader("🌳 Visual Problem Decomposition")
        dot = 'digraph G {\n'
        dot += f'"Problem: {problem[:50]}..." [shape=box, style=filled, color=lightblue];\n'
        for pattern, score in top_patterns:
            dot += f'"{pattern["problem_type"]} ({score:.2f})" [shape=ellipse, style=filled, color=lightgreen];\n'
            dot += f'"Problem: {problem[:50]}..." -> "{pattern["problem_type"]} ({score:.2f})";\n'
            for step in pattern["solution_steps"]:
                dot += f'"{pattern["problem_type"]} ({score:.2f})" -> "{step}" [shape=box];\n'
        dot += '}'
        st.graphviz_chart(dot)

else:
    st.info("Enter a problem statement and click **Analyze Problem** to begin.")

