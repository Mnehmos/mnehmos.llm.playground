# MM-01C: coefficient discovery, constructive follow-ups

## Literature
- OpenAI's October 2 upper bound uses spectral duality and existence of a finite rank decomposition, without specifying the coefficient arrays.
- Amir Shpilka, October 27, ECCC TR26-281, https://eccc.weizmann.ac.il/report/2026/281/ : constructive tensor restriction algorithm; section 6 bounds the seed size by a singly exponential function of a polynomial in inverse tolerance.
- Rafael Pass, October 30 revision, https://arxiv.org/html/2610.28627v2 : constructive rational algorithm using polynomial packings, amplification, independent-left interpolation, composition, and exactification by Bini interpolation (Theorem 4.3, Corollary 4.4). This removes the need for an unstructured exhaustive search to prove constructibility, not the finite-size cost.
- https://github.com/SECQUOIA/finite-extraction-matrix-multiplication : alternative AI-produced explicit compressed constructions, not expanded coefficients or practical kernels. The repository reports enormous finite source sizes. Not independently verified here.
- https://github.com/AntoineTilloy/explicit-matmul-schemes : accessible exact rational coefficients for explicit higher-exponent schemes, including 14196 terms at side 32. The repository reports a roughly 2.649 exponent at side 6.18e64, not a qualifying 2.35 scheme. Counts are third-party reported, not verified here.

## Eliminate impossible small-block searches
The target r < n^(9/4+epsilon) is an exact integer comparison for rational epsilon. Use the Alder–Strassen lower bound 2n^2-1 for the simple matrix algebra and Bläser's 5n^2/2-3n bound (https://doi.org/10.1007/PL00001605). At exponent strictly below 2.35, all sizes 2 through 9 are ruled out; n=10, r=223 is the smallest size not excluded by these two bounds, not a claim of feasibility. At 2.30 the excluded prefix ends at 16, at 2.26 at 28, at 2.251 at 33. The earlier height-2 enumeration at epsilon .1 was searching an impossible size; it remains an instrument smoke test only.

## Constructive experiment
Implement Pass Theorem 3.1's same-left packing (common left polynomial) and recover exact outputs using two nonzero formal-parameter evaluations. Exhaustive basis-pair verification:
| a,b,c,t | pairs | target effort | source effort mod parameter | exact via two evaluations |
|---|---:|---:|---:|---:|
|1,1,2,1|4|4|4|8|
|2,2,5,1|30|21|20|40|
|4,4,12,1|236|98|96|192|
|9,9,27,1|2421|493|486|972|
Effort is the paper's arithmetic upper-bound count, not measured runtime. The test implementation uses direct convolution to validate the coefficient maps. It does not yet implement full amplification, independent-left interpolation or recursive composition. Exactification at this small stage loses the savings, as expected; the proof uses tensor powers to amortize that cost.

A bounded parameter scan over c and dyadic d for Pass Theorem 4.3 (conservative analytic bounds evaluated in float64) produced the following candidate plans *before* Bini exact conversion:
| target | c | d | rounds k | approximate lower estimate of log10(base side) |
|---|---:|---:|---:|---:|
|2.35|27|4096|9050|4554725.64|
|2.30|96|32768|44503|52066755.91|
|2.26|484|1048576|1505141|11298272337.28|
|2.251|4599|134217728|196367376|19905192160330.85|
These are illustrative plans from conservative bounds, not optimal sizes, exact-certified parameters, expanded certificates or physical storage estimates. No full compiler executed. Exact conversion further increases size.

## Direct coefficient fit
Float64 regularized alternating least squares on the exact target tensor, with implicit matrix contractions and independent dense endpoint verification. One BLAS thread; 200 complete sweeps, three random initializations, n=10, r=223. Chosen after literature screening; exploratory, no population inference.
| seed | best relative Frobenius residual | max absolute coefficient error at end |
|---|---:|---:|
|307|0.8815551633|0.8970588210|
|311|0.8815212358|0.8925149794|
|305|0.8815255771|0.8868142313|
Random n=2,r=7 controls at 500 sweeps reached .0059648788 and .0159687091 relative residual, not exact recovery. No rational reconstruction attempted because no fit was near exact. Random fits do not establish nonexistence. The target rank is close to the elementary lower bound and this search is tiny.

## Decision
No qualifying finite U,V,W coefficients recovered. Stage C still blocked. Priority is a lazy coefficient compiler for constructive Pass/Shpilka maps and comparison with explicit compressed schemes; do not materialize astronomically large arrays. The practical objective would require a substantially better small-size construction or different hardware optimization. Do not modify original NRI training or PRs 32–34. Local scripts and full raw trajectories provided separately.
