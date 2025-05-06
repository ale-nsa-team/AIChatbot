import json
from pathlib import Path

HASH_REGISTRY_FILE = Path("processed_hashes.json")

def load_processed_hashes():
    if HASH_REGISTRY_FILE.exists():
        with open(HASH_REGISTRY_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_hashes(hashes: set):
    with open(HASH_REGISTRY_FILE, "w") as f:
        json.dump(list(hashes), f, indent=2)

def is_already_processed(file_hash: str, hashes: set) -> bool:
    return file_hash in hashes

def mark_as_processed(file_hash: str, hashes: set):
    hashes.add(file_hash)
    save_processed_hashes(hashes)
