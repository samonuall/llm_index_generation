import json
from pathlib import Path
from datasets import load_dataset

SPLIT_MAP = {
    "tip_of_the_tongue":              "tip_of_the_tongue",
    "paper_retrieval":                "paper_retrieval",
    "stack_exchange":                 "stack_exchange",
    "clinical_trial":                 "clinical_trial",
    "legal_qa":                       "legal_qa",
    "theorem_retrieval":              "theorem_retrieval",
    "code_retrieval":                 "code_retrieval",
    "set_operation_entity_retrieval": "set_operation_entity_retrieval",
}

def main():
    base_dir = Path("data")
    if not base_dir.exists():
        print("data directory not found")
        return

    for split, crumb_name in SPLIT_MAP.items():
        cache_dir = base_dir / split
        if not cache_dir.exists():
            continue
            
        print(f"\nProcessing {split}...")
        
        # 1. Rename existing queries.jsonl to evaluation_queries.jsonl
        old_queries = cache_dir / "queries.jsonl"
        eval_queries = cache_dir / "evaluation_queries.jsonl"
        
        if old_queries.exists() and not eval_queries.exists():
            print(f"Renaming {old_queries} -> {eval_queries}")
            old_queries.rename(eval_queries)
        elif eval_queries.exists():
            print(f"{eval_queries} already exists")
            
        # 2. Download and save validation_queries.jsonl
        val_queries_file = cache_dir / "validation_queries.jsonl"
        if val_queries_file.exists():
            print(f"{val_queries_file} already exists, skipping download")
            continue
            
        print(f"Downloading validation_queries for {split}...")
        try:
            dataset = load_dataset("jfkback/crumb", "validation_queries", split=crumb_name)
            
            queries = []
            for item in dataset:
                qrels = item.get("full_document_qrels") or item.get("passage_qrels") or []
                relevant_ids = [q["id"] for q in qrels if q.get("label", 0) > 0]
                if relevant_ids:
                    queries.append({
                        "query_id": item["query_id"],
                        "query_content": item["query_content"],
                        "relevant_doc_ids": relevant_ids,
                    })
                    
            print(f"Saving {len(queries)} queries to {val_queries_file}")
            with val_queries_file.open("w") as f:
                for q in queries:
                    f.write(json.dumps(q) + "\n")
        except Exception as e:
            print(f"Failed to process {split}: {e}")

if __name__ == "__main__":
    main()