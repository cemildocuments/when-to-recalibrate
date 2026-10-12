"""Check the archived public input files against their recorded SHA-256 hashes."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for row in json.loads((ROOT/'data/provenance.json').read_text()):
    f=ROOT/'data/upstream'/row['dataset']/row['file']
    assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],str(f)
print('All seven archived source files match their recorded hashes.')
