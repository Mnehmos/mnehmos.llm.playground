# Conversion trainee at 6,000 updates

The original trainee (seed 482111) completed 6,000 updates with 61 reviewed checkpoints, including initialization. The initial 300-step run was extended while retaining its model, AdamW state and random streams. The original budget record is included. Each advance used the preceding checkpoint's explicit supervisor review. The step-4,000 advance twice stalled before writing a new snapshot. The last valid step-3,900 state was restored and the interval completed. No intervals were skipped.

## Main observations

| Endpoint | Early trained checkpoint | Step 6,000 |
| --- | ---: | ---: |
| Original practiced conversions | 10/10 at 100 | 10/10 |
| Same-route value swap after layer one | 90/90 at 100 | 60/90 |
| Same-unit comparison swap | 0/90 at 100 | 0/90 |
| Reverse conversion probe | 0/9 at 100 | 7/9 after reverse-route training |
| Third-source conversion probe | 1/8 at 100 | 8/8 after training the route |
| Single held-out forward value | 0/1 at 100 | 0/1 |
| Broader held-out unit–value conversion | 40/384 at 100 | 34/384 |
| New conversion frame, mixed routes | 38/384 at 100 | 0/384 |

The original format reached perfect accuracy by step 100 and remained there. Source-number swap success fell to 34/90 at step 3000, later recovered to 67/90 at 5500, fell to 48/90 at 5600 under the same curriculum, and finished at 60/90. The ordinary answer score concealed this change. These are repeated interventions on ten familiar questions, not separate training replications.

All final held-out numeric hits were mass questions, 34/245. Acceleration yielded 0/139. A best constant output selected on those same labels would match 96/245 mass cases. Acceleration conversion was outside the trained directions. The one withheld forward value 4 still yielded 6 rather than 7.

On 270 discriminating cross-route development pairs, an x1 source-number patch made the answer follow the donor number under the base route on 105 trials. At x2 that count was one, and 266 responses stayed at the base answer. Source-unit-only replacements left the base answer on every trial at both sites. These tests provide partial causal access to the number in one location and context. The unit could be used elsewhere. A route-conditioned lookup remains compatible with the behavior.

## Scientific supervision and provenance

Twelve curriculum decisions added styles, directions, equivalence questions, syntax families and force practice, or changed auxiliary losses. No persistent weight interventions were applied. Of the 60 predictions whose due checkpoint occurred, 45 were confirmed and 15 were falsified. The final review records a conditional prediction beyond the completed budget. The checkpoint JSON contains the scientific assessment and outcome ledger. The accounts are exploratory because the supervisor adapted a single trajectory using development evidence.

The measured successful runs used about 488 seconds of training and 259 seconds of checkpoint instrumentation, excluding supervisor review, archive work and discarded retries. Each interval stores 100 lightweight step records. Each checkpoint keeps weights, optimizer, RNG, deterministic observations, signed token traces and the review. The final artifacts span two archives. Extract both into one parent directory to reconstruct the common conversion-run directory. The original and extended study configurations, the exact implementation patch, development data, derived trajectory table and final diagnostics are supplied. Sealed evaluation was not opened.
