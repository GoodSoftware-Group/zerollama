# geodesic-patch fixture

`checkpoint/` is a tiny Hugging Face-style directory. `fixture.gpatch` was
written by a Python script that matches DAWE's `patch_id` canonicalization
(`json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`).

Regenerate:

```bash
python3 x/weightedit/gpatch/testdata/generate_fixture.py
```
