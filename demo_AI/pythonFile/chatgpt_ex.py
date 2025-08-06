import json, os, openai
def generate_embeddings(input_path, output_path, model="text-embedding-ada-002"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            chunk = json.loads(line)
            resp = openai.embeddings.create(
                model=model,
                input=chunk["text"]
            )
            embedding = resp.data[0].embedding
            out_obj = {
                "id": chunk["id"],
                "role": chunk["role"],
                "text": chunk["text"],
                "embedding": embedding
            }
            outfile.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 chatgpt_ex.py ./demo_AI/output/chunk.jsonl ./demo_AI/output/output_embeddings.jsonl")
        sys.exit(1)
    generate_embeddings(sys.argv[1], sys.argv[2])