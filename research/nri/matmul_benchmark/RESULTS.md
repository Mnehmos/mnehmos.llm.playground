# MM-01 initial CPU screen

Registration: 2a91665c4578895d668c0a3079d1a5254e87a1f4.
Benchmark code: 115ab7e9662d0f8688d9cf2d6cb6dd20ebeebfcf.
Executed locally with Linux x86-64, AMD EPYC 9V74, PyTorch 2.10.0+cpu, one thread, float32, seed 307. No GPU. Seven interleaved observations per eligible method and mode, three warmups. Raw JSON is delivered separately to owner; rerun with:
`python research/nri/matmul_benchmark/benchmark.py --device cpu --output results_cpu.json`

Times in ms, rounded. F = forward, FB = forward and both input gradients. Dash means skipped per registration (odd dimension for Strassen; packing cap for FFT).

| Batch,m,k,n | Native F | Native FB | Strassen F | Strassen FB | FFT F | FFT FB |
|---|---:|---:|---:|---:|---:|---:|
|1,32,32,32|0.0170|0.0823|0.1045|0.5235|1.3271|2.3754|
|1,64,64,64|0.0189|0.1541|0.1733|0.6555|10.3027|20.5000|
|1,256,256,256|0.3559|1.1331|0.4524|1.8035|–|–|
|1,1024,1024,1024|21.6354|65.8269|21.7750|70.1982|–|–|
|1,4096,64,192|1.0434|3.5512|1.5554|5.2163|–|–|
|1,4096,64,64|0.3786|1.2214|0.6719|2.3369|–|–|
|1,512,64,256|0.1850|0.6261|0.3332|1.2303|–|–|
|1,512,256,64|0.1828|0.6448|0.3072|1.3467|–|–|
|1,512,64,13|0.1005|0.3019|–|–|18.6500|34.8375|
|2048,8,16,8|0.3731|1.1704|3.4433|9.2621|–|–|
|2048,8,8,16|0.2325|0.9321|3.5607|8.7658|–|–|

All 24 eligible method/shape combinations passed the relative L2 1e-5 forward/gradient gate. Largest relative L2 error across native tests 2.70e-7, Strassen 5.13e-7, packed FFT 3.32e-7. Native won every forward and forward+backward median. Best candidate speedup on representative NRI shapes: 0.681x (Strassen on 4096,64,192). No candidate met 1.10x promotion threshold. The script emitted a nonfatal PyTorch warning when converting an error tensor requiring grad to a scalar during untimed validation.

## Actual 9/4 implementation: blocked
The October 2 upper-bound paper explicitly does not identify a competitive finite matrix size. Its existence argument is not a usable finite coefficient certificate. The independent reproduction at https://github.com/nickedfrc/matrix-multiplication-paper-reproduction states it has not obtained the required rank decomposition; its polynomial packing uses cubic storage and cubic-logarithmic work. This benchmark independently implements that packing identity, not the claimed 9/4 algorithm. No practical 9/4 implementation has been verified here. Do not infer that the theoretical exponent is false or that every optimized Strassen implementation is slower.

## Decision
Stage C training integration not triggered. Existing networks and ongoing research are untouched. Synthetic contiguous matrix shapes, one hardware environment, one seed and seven repeats do not establish actual training speed or numerical stability across training. No profiling or full training, no hardware counters or peak memory measurements, no independent GPU replication. Next useful action: obtain a verified executable finite decomposition or benchmark a compiled/fused fast multiplication implementation and re-register its evaluation; do not modify the historical trainer without evidence and separate authorization.
