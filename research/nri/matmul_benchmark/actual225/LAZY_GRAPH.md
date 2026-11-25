# MM-01E: lazy computational graph

Exploratory extension of MM-01D. Uses the existing lazy.py Primal, Dual and ExactPower from Pass's elementary polynomial-packing construction, not the full near-9/4 matrix multiplication reduction.

## Two execution paths
- Selected-output: contract each local tensor entry exactly using the two local interpolation evaluations, memoize it, then multiply the resulting entries along the tensor product. Skip remaining factors after a zero. With sparse inputs, sum over their nonzero coordinate pairs for requested outputs only. This is an alternative contraction order, not a new rank certificate. For fixed gadget and d factors, visits scale at most as d * nnz(A) * nnz(B) * number of requested outputs (plus bounded local cache misses, integer index and rational arithmetic costs).
- Full-output: apply local sparse factor matrices axis by axis, sharing equal intermediate coordinates, eliminating zero/cancelled entries, and reusing the left encoding across interpolation nodes. Intersect the nonzero encoded vectors before multiplying. Decode and combine at d+1 nodes exactly. This does not expand coefficient matrices, but may still expand intermediate vectors exponentially. Budget guards stop excessive counted work or live entries. These guards do not guarantee bounds on wall time, total memory or fraction bit lengths.

## Executed local tests
Run `python test_graph.py` with assertions enabled. All 1,982 exact output checks passed, including 120 additional dense, signed and zero-input control comparisons. Oracle computes the direct polynomial maps independently of the rank coefficient contractions. Also compared individual entries with the previous ExactPower.entry enumeration. One low-work-budget guard exercised successfully. Tests are finite, not a verification of the full 9/4 theorem.

Representative sparse inputs (seed 307, at most five left/eight right nonzero entries, all outputs):
| Base parameters | power | alternating | global terms upper bound | actual bilinear products |
|---|---:|---|---:|---:|
|2,2,5,1|1|no|40|35|
|2,2,5,1|2|no|1200|318|
|2,2,5,1|2|yes|1200|337|
|2,3,4,2|2|no|1728|929|
|2,3,4,2|2|yes|1728|1421|

Counts omit linear transforms, interpolation and bookkeeping; skipping products on these particular inputs does not reduce the algorithm's worst-case rank. Single selected tensor entry (0,0,0), base (2,2,5,1), power 2, nonalternating: previous expanded term scan 17.335 ms vs factorized query 0.168 ms, one observation per path, not a reliable throughput estimate.

Large powers use one nonzero left and one nonzero right coordinate, chosen to give a known nonzero output (-2/5); then a separate zero-output control. The first query only was timed with tracemalloc; graph descriptor and inputs were constructed before measurement. 
| power | digits in global term count | query ms | peak traced bytes |
|---:|---:|---:|---:|
|8|12|1.054|2800|
|32|44|0.913|3688|
|128|169|1.290|6120|
|4096|5333|20.613|132344|

At power 4096 the two queries combined visited 8192 factors, with 3 local entry cache misses and 8189 hits. No global term enumeration. These are highly structured sparse probes, not full multiplication of enormous matrices. Times are single local observations; traced bytes exclude process memory and prebuilt objects.

## Remaining work
The next mathematical compiler steps remain balanced target selection, independent-left interpolation, cyclic leg permutations, recursive packing composition and verification of a complete finite square multiplication seed. Composition can require large internal sums; factorization of independent powers alone cannot remove them. No new training or practical 2.25-exponent kernel. No alterations to PRs 32–34.
