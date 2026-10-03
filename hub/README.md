---
license: apache-2.0
base_model: google/gemma-4-31B-it
tags:
  - decision
  - text-classification
---

# deck31b

deck31b serves frozen `google/gemma-4-31B-it` at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. This repository does not contain
those weights. The server downloads them and quantizes linear layers to FP8
e4m3 weights and activations at startup.

We did not train the checkpoint. The prompt and letter readout follow the
public Cygnet recipe (`blockbrain-ai/cygnet-recipe` at
`81974de878a7385f878be828205c02064f8bc045`), with temperature `3.4`.

Code: https://github.com/krishna-gogineni-765/deck31b

```bash
git clone https://github.com/krishna-gogineni-765/deck31b
cd deck31b
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
python -m pip install -e .
deck31b-serve --host 0.0.0.0 --port 8090
```

`POST /v1/systemone` supports `noul`, `choice`, and `score`. The base weights
remain under the Gemma Terms of Use. This server code is Apache-2.0.
