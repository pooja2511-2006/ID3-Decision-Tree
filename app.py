import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx

# --- Helper Functions ---
def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    return -sum((c/len(col)) * math.log2(c/len(col)) for c in counts)

def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    vals = df[attr].unique()
    weighted_entropy = sum(
        (len(df[df[attr] == v]) / len(df)) * entropy(df[df[attr] == v][target])
        for v in vals
    )
    return total_entropy - weighted_entropy

def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]
    best = max(attrs, key=lambda a: info_gain(df, a, target))
    tree = {best: {}}
    for val in df[best].unique():
        sub_df = df[df[best] == val]
        tree[best][val] = id3(sub_df, target, [a for a in attrs if a != best])
    return tree

def predict(tree, input_data):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    val = input_data.get(root)
    if val in tree[root]:
        return predict(tree[root][val], input_data)
    return "Unknown"

# --- Streamlit UI ---
st.title("ID3 Decision Tree Classifier")

# Sample dataset
data_dict = {
    "outlook": ["sunny", "sunny", "overcast", "rain", "rain", "overcast",
                "sunny", "sunny", "overcast", "rain", "overcast", "overcast",
                "rain", "sunny"],
    "humidity": ["high", "normal", "high", "normal", "high", "high",
                 "normal", "normal", "normal", "normal", "normal", "high",
                 "high", "normal"],
    "playtennis": ["no", "yes", "yes", "yes", "no", "yes", "yes", "yes",
                   "no", "yes", "no", "yes", "yes", "yes"]
}
df = pd.DataFrame(data_dict)

# File upload option
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

# Target column selection
target_col = st.selectbox("Target Column", df.columns, index=len(df.columns)-1)
features = [c for c in df.columns if c != target_col]

# Train button
if st.button("Train"):
    tree = id3(df, target_col, features)
    st.session_state['tree'] = tree
    st.json(tree)

    # --- Visualize Decision Tree ---
    def add_edges(tree, parent=None, graph=None):
        if graph is None:
            graph = nx.DiGraph()
        if isinstance(tree, dict):
            root = next(iter(tree))
            if parent:
                graph.add_edge(parent, root)
            for val, subtree in tree[root].items():
                child = f"{root}={val}"
                graph.add_edge(root, child)
                add_edges(subtree, child, graph)
        else:
            graph.add_edge(parent, tree)
        return graph

    G = add_edges(tree)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue",
            font_size=10, font_weight="bold", edge_color="gray")
    st.pyplot(plt)

# Prediction section
if 'tree' in st.session_state:
    inputs = {col: st.selectbox(col, df[col].unique()) for col in features}
    if st.button("Predict"):
        st.write(f"Result: {predict(st.session_state['tree'], inputs)}")
