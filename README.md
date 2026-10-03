# deck31b

deck31b is a frozen decision model. The weights are
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) at
`842da3794eaa0b77d5f08bae87a17459d91ff475`. We did not train them and this
repository does not redistribute them.

At startup the server quantizes linear layers to FP8 e4m3 weights and
activations with TorchAO (`Float8DynamicActivationFloat8WeightConfig`), then
reads the next-token distribution. It does not generate tokens.

The prompt and letter readout follow the public Cygnet recipe
(`blockbrain-ai/cygnet-recipe` at `81974de878a7385f878be828205c02064f8bc045`):
one letter per option, probability mass summed over every vocabulary token
that decodes to that letter, renormalized over the listed options, then
`p^(1/T)` with `T=3.4`. That temperature is the recipe temperature. We did
not fit it on JevBench.

Served id: `deck31b`. License of this server: Apache-2.0. The base weights
remain under the Gemma Terms of Use. Needs one NVIDIA GPU with FP8 and at
least 80 GB of memory. The first start downloads the base checkpoint.

```bash
git clone https://github.com/krishna-gogineni-765/deck31b
cd deck31b
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
python -m pip install -e .
deck31b-serve --port 8090
curl --fail http://127.0.0.1:8090/health
```

The server has no authentication and listens on `127.0.0.1` unless `--host` is set.

`POST /v1/systemone` accepts `noul`, `choice`, and `score`. Up to 26 options
are scored in one pass. Inputs over 16,384 tokens are rejected.

## Local public reference

This is a local measurement, not an official JevBench result. Same readout
the server uses, serial, one NVIDIA H100 80 GB. JevBench public files, 231
items, zero refusals. `T=3.4` and `T=1` selected the same label on every item.

| Tier | Correct | ECE10 at T=3.4 | p50 | p95 |
|---|---:|---:|---:|---:|
| Easy | 48/48 | 0.002 | | |
| Original | 72/72 | 0.005 | | |
| Hard | 95/111 | 0.097 | | |
| Overall | 215/231 | 0.044 | 1.05 s | 1.76 s |

Mean input length was 703 tokens. Steady-state GPU memory after quantization
was about 44 GB. The public items were scored while checking this serving
configuration, so the sealed run should be treated as the official number.
