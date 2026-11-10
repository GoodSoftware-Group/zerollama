# Object detect

`POST /v1/detect` finds open-vocabulary boxes in one camera frame. The chat eye (`ministral-3` on `/v1/chat/completions`) is a different path. This route does not look at a picture by describing it.

## Wire

```json
{"image":"<base64 or data URL>","queries":["cup","person"],"threshold":0.1}
```

`image_base64`, `frame`, and a `data:` `image_url` are the same image. `labels` and `text` are the same as `queries`. An `http` image URL is not fetched.

`status=ok` means OWLv2 ran. Each detection is `label`, `score`, and `box` as pixel `[x1, y1, x2, y2]` with the origin at the top left.

`status=wire_only` means the route is up and no model ran. `detections` is empty. That happens when `ZEROLLAMA_DETECT_URL` is unset, and also when the sibling process has no weights. The body is JSON. A missing route was gin's `404 page not found`, which is what broke the eye.

## Sibling

```bash
cd detect/server
ZEROLLAMA_OWL_MODEL=/path/to/owlv2 python3 app.py --host 127.0.0.1 --port 18097
```

`ZEROLLAMA_OWL_MODEL` is a local directory. A Hub repo id is refused (`local_files_only`). `ZEROLLAMA_DETECT_DEVICE` defaults to `cpu` so a lab start does not take a GPU from the talk or camera lane. Then:

```bash
ZEROLLAMA_DETECT_URL=http://127.0.0.1:18097
```

`GET /api/version` → `zerollama.capabilities.detect.wired` is true only when that URL is set.
