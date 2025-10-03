
import os,json ,re
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import pickle



# Constants
Retries = 3
Base_dir_path = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
dataset_path = {
    "SVAMP": "ChilleD/SVAMP",
    "ASDIV": "asdiv",   
    "MATH23K": "math23k",
    "GSM8K": "gsm8k"
}



# Path to Datasets folder inside the project
primitve_storage_dir = os.path.join(Base_dir_path, "primitve_storage")
os.makedirs(primitve_storage_dir, exist_ok=True)  # create if missing

# --- Paths inside Datasets ---
GRAPH_PATH = os.path.join(primitve_storage_dir, "primitive_graph.gpickle")
FAISS_PATH = os.path.join(primitve_storage_dir, "primitive_vectors.index")
METADATA_PATH = os.path.join(primitve_storage_dir, "primitive_maps.pkl")

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


def parse_raw_op_with_markers(raw_text: str):
    """
    Extract JSON array of primitives from raw LLM output wrapped with <start> and <end>
    """
    # Extract text between <start> and <end>
    match = re.search(r"<start>(.*?)<end>", raw_text, flags=re.S)
    if not match:
        raise ValueError("Could not find <start> ... <end> in raw output")

    json_text = match.group(1).strip()

    # Remove trailing commas before } or ]
    json_text = re.sub(r',(\s*[\}\]])', r'\1', json_text)

    # Parse JSON
    return json.loads(json_text)

