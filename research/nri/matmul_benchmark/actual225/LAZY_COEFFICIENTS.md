# MM-01D: Lazy coefficient access

Source: Rafael Pass, https://arxiv.org/html/2610.28627v2 (Theorem 3.1, Lemma 3.2, Section A.3). Implemented the elementary primal map, its transposed dual (with reversed left coordinates), and exact tensor powers using interpolation at d+1 nonzero nodes. A term and coordinate index are decoded into mixed-radix digits, and only the requested factors are multiplied. No dense factor arrays are allocated by lazy.py. The bounded cache stores short interpolation polynomials.

For base a=b=2,c=5,t=1, one approximate gadget has 20 terms and dimensions 2,15,20. For d alternating primal/dual copies, exactification has (d+1)*20^d terms, using weights (-1)^(j-1) binomial(d+1,j), j=1..d+1. The contracted base tensor is affine in the interpolation parameter even though individual factors contain its reciprocal; tensoring raises the polynomial degree to at most d. The coefficient for a given term and input/output index is a product of local factors, not a sum over all terms. This is a representation of the elementary tensor power, not the final matrix multiplication identity.

Run `python test_lazy.py` with assertions enabled. Local Python 3.13.5: 1,480 factor entries matched the previously expanded arrays, plus 504, 2,112, 756 and 342 entries in four extra configurations, 5,194 total. All 700 contracted entries passed (16+256+256 exhaustive tiny tests; 100 sampled tiny power-3 and 72 sampled power-2 actual gadget tests, including known nonzero entries). No complete verification of large powers. Extra factor cases were run separately in the same local environment.

| Power d | Digits in term count | Digits in dense coefficient count | Six query ms | Peak traced bytes |
|---:|---:|---:|---:|---:|
|1|2|4|0.282|2520|
|2|4|6|0.263|2712|
|4|6|12|0.263|2760|
|8|12|22|0.395|2748|
|16|23|43|0.501|2852|
|32|44|84|0.830|2924|
|128|169|328|2.939|3532|
|4096|5333|10407|94.599|29180|

Six fixed boundary queries include descriptor construction and a cold cache; three of six were zero at powers >=2. Not a representative query distribution or a full multiplication benchmark. tracemalloc measures Python allocations inside the timed interval, not process RSS. Dense coefficient count is number of scalar entries, not bytes; term count is a supplied upper bound, not minimal rank. Exact fraction bit lengths and integer indexing costs also matter. No claim of training speed or asymptotic wall time.

Example: `ExactPower(Primal(2,2,5,1),4096,True).coefficient(0,0,0)` returns 1. To contract, use entry(i,j,k) only for small cases; default guard rejects more than 100000 terms. Random access avoids expanding factors but does not eliminate the enormous number of bilinear products. Full evaluation by enumeration is still infeasible.

Next: implement balanced-target selection (Lemma 3.2), independent-left interpolation, cyclic permutations of matrix-product legs, recursive packing composition and full near-9/4 seed as a lazy expression graph. Composition can introduce sums over large intermediate dimensions, so coefficient-query cost must be separately bounded. Do not infer full compiler feasibility from the elementary tensor-power success. No qualifying square-matrix certificate or training integration. No change to original NRI training or PRs 32–34.
