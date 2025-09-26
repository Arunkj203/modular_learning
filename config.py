
import os
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import pickle



# Dataset paths
dataset_path = {
    "SVAMP": "ChilleD/SVAMP",
    "ASDIV": "asdiv",   
    "MATH23K": "math23k",
    "GSM8K": "gsm8k"
}




# --- Paths ---
GRAPH_PATH = "primitive_graph.gpickle"
FAISS_PATH = "primitive_vectors.index"
METADATA_PATH = "primitive_maps.pkl"

# --- Globals ---
primitive_graph = nx.DiGraph()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
faiss_index = faiss.IndexFlatL2(embedding_dim)
primitive_id_map = {}      # FAISS index -> primitive ID
primitive_metadata = {}    # primitive ID -> metadata dict

# --- Load Memory Function ---
def load_memory():
    global primitive_graph, faiss_index, primitive_id_map, primitive_metadata

    # Load graph
    if os.path.exists(GRAPH_PATH):
        g = nx.read_gpickle(GRAPH_PATH)
        primitive_graph.clear()
        primitive_graph.update(g)
        print(f"Graph loaded from {GRAPH_PATH}")
    else:
        print(f"{GRAPH_PATH} not found, starting with empty graph")

    # Load FAISS index
    if os.path.exists(FAISS_PATH):
        idx = faiss.read_index(FAISS_PATH)
        # Reconstruct all vectors and add to existing index
        faiss_index.reset()
        for i in range(idx.ntotal):
            vec = idx.reconstruct(i)
            faiss_index.add(vec.reshape(1, -1))
        print(f"FAISS index loaded from {FAISS_PATH}")
    else:
        print(f"{FAISS_PATH} not found, starting with empty FAISS index")

    # Load metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            data = pickle.load(f)
            primitive_id_map.clear()
            primitive_id_map.update(data["id_map"])
            primitive_metadata.clear()
            primitive_metadata.update(data["metadata"])
        print(f"Metadata loaded from {METADATA_PATH}")
    else:
        print(f"{METADATA_PATH} not found, starting with empty metadata")

# --- Save Memory Function ---
def save_memory():
    global primitive_graph, faiss_index, primitive_id_map, primitive_metadata

    # Save graph
    nx.write_gpickle(primitive_graph, GRAPH_PATH)

    # Save FAISS index
    faiss.write_index(faiss_index, FAISS_PATH)

    # Save metadata
    with open(METADATA_PATH, "wb") as f:
        pickle.dump({"id_map": primitive_id_map, "metadata": primitive_metadata}, f)

    print("Memory saved successfully!")
