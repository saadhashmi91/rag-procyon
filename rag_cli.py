import argparse
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, TextStreamer
import openvino_genai as ov_genai

# Paths
CHUNKS_PATH = "data/faiss_index/chunks.json"
FAISS_INDEX_PATH = "data/faiss_index/procyon.index"
MODEL_DIR = "models/llama3_int4"

chat_history = []

def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def embed_query(query, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    vec = model.encode([query], normalize_embeddings=True)
    return vec

def retrieve_top_k_chunks(query, k=5):
    index, chunks = load_faiss_index()
    embedding = embed_query(query)
    scores, indices = index.search(embedding, k)
    retrieved = [(i, chunks[i]["text"]) for i in indices[0]]
    return retrieved

def build_prompt(query, context_chunks):
    context = "\n\n---\n\n".join([chunk for _, chunk in context_chunks])
    return f"""You are a helpful assistant answering based on the UL Procyon Guide.

Context:
{context}

Question: {query}
Answer:"""

def build_chat_prompt(query, context_chunks, history):
    context = "\n\n---\n\n".join([chunk for _, chunk in context_chunks])
    history_str = ""
    for turn in history:
        history_str += f"User: {turn['query']}\nAssistant: {turn['response']}\n\n"

    return f"""You are a helpful assistant answering based on the UL Procyon Guide.

Context:
{context}

{history_str}User: {query}
Assistant:"""




def stream_response(prompt: str):
    print("Loading quantized LLaMA 3.1 (INT4) model via OpenVINO GenAI...")
    # Properly initialize LLMPipeline with model_path and device
    pipe = ov_genai.LLMPipeline(models_path=str(MODEL_DIR), device="CPU")

    # Ensure stop token works, using stop_strings with EOS
    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
    }

    print("\nAnswer:\n", end="", flush=True)
    # Stream tokens via lambda that prints without newline
    streamer = lambda token: print(token, end="", flush=True)

    # Execute generation streaming response
    pipe.generate(prompt, streamer=streamer, **gen_kwargs)
    print()  # final newline

    
def main():
    parser = argparse.ArgumentParser(description="Query the UL Procyon Guide with local RAG.")
    parser.add_argument("--query", help="Single query to answer")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode with memory")
    parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved chunks")

    args = parser.parse_args()

    if args.chat:
        print("RAG Chat Mode (type 'exit' to quit)")
        while True:
            user_query = input("\nYou: ")
            if user_query.strip().lower() in {"exit", "quit"}:
                break

            retrieved = retrieve_top_k_chunks(user_query, k=args.top_k)
            prompt = build_chat_prompt(user_query, retrieved, chat_history)

            print("\nAssistant:", end=" ", flush=True)
            from transformers import pipeline
            model = OVModelForCausalLM.from_pretrained(MODEL_DIR, compile=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(**inputs, streamer=streamer, max_new_tokens=512)

            # Optional: store response in history
            chat_history.append({"query": user_query, "response": "[streamed above]"})

            print("\nReferences:")
            for i, text in retrieved:
                print(f"[Chunk {i}]\n{text[:300].strip()}...\n")

    elif args.query:
        retrieved = retrieve_top_k_chunks(args.query, k=args.top_k)
        prompt = build_prompt(args.query, retrieved)
        stream_response(prompt)

        print("\nReferences:")
        for i, text in retrieved:
            print(f"[Chunk {i}]\n{text[:300].strip()}...\n")

if __name__ == "__main__":
    main()
