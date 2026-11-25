# MM-01B: actual 9/4 proof-to-code implementation

This is an exploratory continuation of MM-01, not a claim of a fast numerical kernel. The November 24 paper supplies the lower bound. The October 2 paper supplies the matching upper bound over complex arithmetic, and Section 5 obtains recursive algorithms by existence of a finite rank decomposition for each epsilon > 0. It does not supply a competitive block size or its coefficients.

Sources:
- https://github.com/openai/math/blob/main/preprints/Matrix-Multiplication-Nine-Fourths-October-2-2026/build/paper.tex (Proposition 3.1, Section 4, Section 5 including the remark about algebraic coefficients).
- https://github.com/openai/math/tree/main/preprints/The-matrix-multiplication-exponent-is-nine-fourths-November-24-2026
- https://github.com/nickedfrc/matrix-multiplication-paper-reproduction (independent prior proof-to-code work; no qualifying certificate reported).

## Implemented
* proof_components.py: exact Fourier separation over a cyclotomic quotient, determinant-filtration change of basis, three-sector decomposition, rational diagonal-growth recurrence. These are actual finite identities used in the upper-bound proof.
* certificate.py: rational rank decomposition check on every matrix-unit pair and output coefficient; recursive exact block multiplication and padding.
* algebraic_search.py: enumerates monic rational polynomials and rational coefficient vectors representing elements in the quotient algebra. For each candidate block size u and rank r, admits only r < u^(9/4+epsilon) by integer arithmetic, then verifies the full identity. A successful algebraic certificate can be loaded and used for recursive rational matrix products. Reducible polynomials are allowed: an exact identity in the quotient holds after evaluation at any root. The control uses Strassen, including a nonrational rescaling in Q(sqrt(2)); this is not a new rank result.

The mathematical guarantee of eventual discovery in unlimited mode is conditional on the upper-bound theorem and its algebraic-coefficient remark. Every finite description is covered at a finite enumeration height. This is a computability construction, not a usable search strategy; the search order is extremely inefficient. The bound concerns scalar arithmetic operations for fixed epsilon, not bit complexity, finite-size wall time, or training performance. No upper-bound theorem or full Lean proof was independently verified by these finite checks.

## Run
Python 3.10+ and SymPy for proof_components.py; the other scripts use the standard library. From this directory (keep assertions enabled):
```sh
python proof_components.py
python certificate.py
python algebraic_search.py --epsilon 1/10 --max-height 2 --max-candidates 100 --max-seconds 5
```

A successful result file, not the current negative result, can be used as follows:
```python
from algebraic_search import load_found, multiply_exact
u,r,R,U,V,W = load_found("found.json")
C = multiply_exact(u,r,R,U,V,W,[[1,2],[3,4]],[[5,6],[7,8]])
```
Budget checks occur between candidates; one expensive candidate is not preempted. Do not use unlimited enumeration on shared infrastructure.

## Local verification
- Fourier separation: 6 cases, 2,275 index tuples.
- Determinant filtration: 75 monomial slices.
- Three-sector decomposition: 1,743 input terms.
- Diagonal growth: 128 exact rational checks.
- Rational executor: all 64 coefficient identities of the Strassen control, 6 recursive products (including padded dimensions), corrupt certificate rejected.
- Algebraic executor: nonrational coefficient control and recursive product pass, corrupt certificate rejected.
- Bounded search: 100 candidates, height 2, epsilon 0.10 (target exponent strictly below 2.35); no qualifying certificate. This is a smoke test, not a substantive rank search.

The local checks used the same mathematical implementation as the committed code, with added input validation/reverification in the committed version. No practical 9/4 certificate obtained. No GPU kernel, PyTorch integration, performance claim or new training experiment. Stage C remains blocked. Preserve existing research and registration.
