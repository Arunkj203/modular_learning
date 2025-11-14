
import os
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import pickle



# Constants
Base_dir_path = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
dataset_path = {
    "SVAMP": "ChilleD/SVAMP",
    "ASDIV": "asdiv",   
    "MATH23K": "math23k",
    "GSM8K": "openai/gsm8k"
}



# Path to Datasets folder inside the project
primitve_storage_dir = os.path.join(Base_dir_path, "primitve_storage")
os.makedirs(primitve_storage_dir, exist_ok=True)  # create if missing

# --- Paths inside Datasets ---
# GRAPH_PATH = os.path.join(primitve_storage_dir, "primitive_graph.gpickle")
# FAISS_PATH = os.path.join(primitve_storage_dir, "primitive_vectors.index")
# METADATA_PATH = os.path.join(primitve_storage_dir, "primitive_maps.pkl")

# --- Paths inside Datasets ---
GRAPH_PATH = None
FAISS_PATH = None
METADATA_PATH = None




# --- Globals ---
primitive_graph = nx.DiGraph()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
faiss_index = faiss.IndexFlatL2(embedding_dim)
primitive_id_map = {}      # FAISS index -> primitive ID
primitive_metadata = {}    # primitive ID -> metadata dict

# --- Load Memory Function ---
def load_memory(prims_dir):
    global primitive_graph, faiss_index, primitive_id_map, primitive_metadata
    global GRAPH_PATH, FAISS_PATH, METADATA_PATH
    
    prims_storage_dir = os.path.join(Base_dir_path, "primitve_storage",prims_dir)
    os.makedirs(prims_storage_dir, exist_ok=True)  # create if missing

    GRAPH_PATH = os.path.join(prims_storage_dir, "primitive_graph.gpickle")
    FAISS_PATH = os.path.join(prims_storage_dir, "primitive_vectors.index")
    METADATA_PATH = os.path.join(prims_storage_dir, "primitive_maps.pkl")

    # Load graph
    if os.path.exists(GRAPH_PATH):
        with open(GRAPH_PATH, 'rb') as f:
            primitive_graph = pickle.load(f)
        print(f"Graph loaded from {GRAPH_PATH} with {len(primitive_graph.nodes)} nodes and {len(primitive_graph.edges)} edges.")
    else:
        primitive_graph = nx.DiGraph()
        print(f"{GRAPH_PATH} not found, starting with empty graph")

    # Load FAISS index
    if os.path.exists(FAISS_PATH):
        faiss_index = faiss.read_index(FAISS_PATH)
        print(f"FAISS index loaded from {FAISS_PATH}")
        if faiss_index.d != embedding_dim:
            print(f"⚠️ Warning: FAISS index dimension {faiss_index.d} != expected {embedding_dim}")
    else:
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        print(f"{FAISS_PATH} not found, starting with empty FAISS index (dim={embedding_dim})")

    # Load metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            data = pickle.load(f)
            primitive_id_map = data.get("id_map", {})
            primitive_metadata = data.get("metadata", {})
        print(f"Metadata loaded from {METADATA_PATH}")
    else:
        primitive_id_map = {}
        primitive_metadata = {}
        print(f"{METADATA_PATH} not found, starting with empty metadata")

    # Summary
    print(f"\n  Loaded {len(primitive_metadata)} primitives total.")
    print(f"   - FAISS index entries: {faiss_index.ntotal}")
    print(f"   - ID map entries: {len(primitive_id_map)}\n")


# --- Save Memory Function ---
def save_memory():
    global primitive_graph, faiss_index, primitive_id_map, primitive_metadata

    # Save graph
    with open(GRAPH_PATH, 'wb') as f:
        pickle.dump(primitive_graph, f)

    # Save FAISS index
    faiss.write_index(faiss_index, FAISS_PATH)

    # Save metadata
    with open(METADATA_PATH, "wb") as f:
        pickle.dump({"id_map": primitive_id_map, "metadata": primitive_metadata}, f)

    print(f"\n  Memory saved successfully!")
    print(f"   - Graph: {len(primitive_graph.nodes)} nodes, {len(primitive_graph.edges)} edges")
    print(f"   - Primitives: {len(primitive_metadata)} total")
    print(f"   - FAISS entries: {faiss_index.ntotal}")
    print(f"   - ID map entries: {len(primitive_id_map)}\n")
