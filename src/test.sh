curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" \
-d '{"model": "ip_adapter_dummy", "pos_prompt": "a cat sitting on a couch", "ng_prompt": "blurry", "ip": null, "inject": null}' \
--output result.png