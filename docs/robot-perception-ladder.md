# Robot perception

The camera chat model and these routes are separate. The Pi does not run zerollama. It posts here.

| Step | Route | State |
|---|---|---|
| Detect | `POST /v1/detect` | Registered. `wire_only` until `ZEROLLAMA_DETECT_URL` and a local OWLv2 directory are both set. |
| Segment | `POST /v1/segment` | Registered. Stub. `grasp_ready` is false. |
| OCR, GPU COCO | — | Not on these routes. |

`GET /api/version` advertises `capabilities.detect` and `capabilities.segment`.
