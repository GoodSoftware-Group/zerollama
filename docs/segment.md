# Segment

`POST /v1/segment` is the mask request that sits next to [detect](./detect.md). The body carries the same image fields plus an optional pixel `box`.

The answer is `status=wire_only`, `mask=null`, and `grasp_ready=false` until a SAM2 sibling is actually loaded. `ZEROLLAMA_SEGMENT_URL` points at `segment/server/app.py` (lab `:18096`). Leaving it unset makes zerollama return that same stub.

`capabilities.segment.grasp_ready` stays false even after a sibling answers `ok`. A mask is not a grasp.

OCR and a GPU COCO detector are not on this route.
