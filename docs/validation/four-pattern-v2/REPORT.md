# Four-pattern v2 numerical sensitivity evidence

Command: `PYTHONPATH=expdoe-dk/src /opt/homebrew/Caskroom/miniconda/base/envs/expdoe-dk-codex/bin/python experiments/four_pattern_v2_sensitivity.py --output-dir docs/validation/four-pattern-v2`

Engine source commit: `dc5e5f8606a8b3a3826b8350e436161e6ec0dc0c` (see JSON for dirty state and dependency versions).

Full required grid: True. Default noiseless gate: **passed** (0 failures / 90 fits).

Signed/normalization/training-isolation invariant failures: 0; exact replay failures: 0.

Changed-units comparison: evaluated; tolerance failures: 0.

| Fixture | Structured runs | Typed failures | Max grid-mean change (physical Y) |
|---|---:|---|---:|
| saturation_increasing | 108 | {} | 0.0641629592 |
| saturation_decreasing | 108 | {} | 0.0641629684 |
| threshold_increasing_decreasing | 108 | {} | 0.00530588163 |
| threshold_increasing_flat | 108 | {} | 0.0284896111 |
| threshold_decreasing_increasing | 108 | {} | 0.00530588163 |
| threshold_decreasing_flat | 108 | {} | 0.0284895997 |
| threshold_flat_increasing | 108 | {} | 0.0198978908 |
| threshold_flat_decreasing | 108 | {} | 0.0198978124 |
| synergy_higher | 108 | {} | 0.601818235 |
| synergy_lower | 108 | {} | 0.816925637 |
| antagonism_higher | 108 | {} | 0.816925637 |
| antagonism_lower | 108 | {} | 0.601818235 |
| additive_only | 108 | {} | 0.000895416809 |
| zero_effect | 108 | {} | 6.77091353e-06 |
| weak_noisy | 108 | {} | 0.0399783103 |
| changed_units | 108 | {} | 6.41634363 |
| missing_transition | 108 | {"INSUFFICIENT_DATA": 108} | not_evaluable |
| composed | 108 | {} | 0.652587757 |
| composed_two_shapes | 108 | {} | 0.036636186 |

## Default-policy pooled comparisons

RMSE / MAE / latent truth 95% coverage; seed 17. Incomplete folds have no pooled score.

| Fixture | Track | RMSE | MAE | Coverage |
|---|---|---:|---:|---:|
| saturation_increasing | correct | 0.01765316 | 0.007943692 | 0.9375 |
| saturation_increasing | wrong | 0.01868696 | 0.008359411 | 0.9375 |
| saturation_increasing | M0 | 0.01868697 | 0.008359437 | 0.9375 |
| saturation_increasing | ridge | 0.09958249 | 0.0748955 | unavailable |
| saturation_increasing | random_forest | 0.08079824 | 0.0476672 | unavailable |
| saturation_decreasing | correct | 0.01765316 | 0.007943692 | 0.9375 |
| saturation_decreasing | wrong | 0.01868696 | 0.008359411 | 0.9375 |
| saturation_decreasing | M0 | 0.01868697 | 0.008359437 | 0.9375 |
| saturation_decreasing | ridge | 0.09958249 | 0.0748955 | unavailable |
| saturation_decreasing | random_forest | 0.08079824 | 0.0476672 | unavailable |
| threshold_increasing_decreasing | correct | 0.006796431 | 0.003613289 | 1 |
| threshold_increasing_decreasing | wrong | 0.006824273 | 0.003689674 | 1 |
| threshold_increasing_decreasing | M0 | 0.006824108 | 0.003689644 | 1 |
| threshold_increasing_decreasing | ridge | 0.124509 | 0.1050937 | unavailable |
| threshold_increasing_decreasing | random_forest | 0.04812253 | 0.03416093 | unavailable |
| threshold_increasing_flat | correct | 0.005824154 | 0.00387125 | 1 |
| threshold_increasing_flat | wrong | 0.005968187 | 0.003943607 | 1 |
| threshold_increasing_flat | M0 | 0.005968459 | 0.003943721 | 1 |
| threshold_increasing_flat | ridge | 0.0783367 | 0.06623054 | unavailable |
| threshold_increasing_flat | random_forest | 0.04161248 | 0.02385056 | unavailable |
| threshold_decreasing_increasing | correct | 0.006796431 | 0.003613289 | 1 |
| threshold_decreasing_increasing | wrong | 0.006824273 | 0.003689674 | 1 |
| threshold_decreasing_increasing | M0 | 0.006824108 | 0.003689644 | 1 |
| threshold_decreasing_increasing | ridge | 0.124509 | 0.1050937 | unavailable |
| threshold_decreasing_increasing | random_forest | 0.04812253 | 0.03416093 | unavailable |
| threshold_decreasing_flat | correct | 0.005824154 | 0.00387125 | 1 |
| threshold_decreasing_flat | wrong | 0.005968187 | 0.003943607 | 1 |
| threshold_decreasing_flat | M0 | 0.005968459 | 0.003943721 | 1 |
| threshold_decreasing_flat | ridge | 0.0783367 | 0.06623054 | unavailable |
| threshold_decreasing_flat | random_forest | 0.04161248 | 0.02385056 | unavailable |
| threshold_flat_increasing | correct | 0.006199814 | 0.004117605 | 1 |
| threshold_flat_increasing | wrong | 0.006155337 | 0.004108325 | 1 |
| threshold_flat_increasing | M0 | 0.006155138 | 0.004108286 | 1 |
| threshold_flat_increasing | ridge | 0.04985062 | 0.04214671 | unavailable |
| threshold_flat_increasing | random_forest | 0.0262878 | 0.01390729 | unavailable |
| threshold_flat_decreasing | correct | 0.006199814 | 0.004117605 | 1 |
| threshold_flat_decreasing | wrong | 0.006155337 | 0.004108325 | 1 |
| threshold_flat_decreasing | M0 | 0.006155138 | 0.004108286 | 1 |
| threshold_flat_decreasing | ridge | 0.04985062 | 0.04214671 | unavailable |
| threshold_flat_decreasing | random_forest | 0.0262878 | 0.01390729 | unavailable |
| synergy_higher | correct | 0.003119856 | 0.001997824 | 0 |
| synergy_higher | wrong | 0.2629498 | 0.175647 | 1 |
| synergy_higher | M0 | 0.09226179 | 0.05578985 | 1 |
| synergy_higher | ridge | 0.191155 | 0.13125 | unavailable |
| synergy_higher | random_forest | 0.1806141 | 0.1428534 | unavailable |
| synergy_lower | correct | 0.1664269 | 0.07997996 | 0.5 |
| synergy_lower | wrong | 0.3921363 | 0.2662627 | 1 |
| synergy_lower | M0 | 0.4038831 | 0.3001656 | 1 |
| synergy_lower | ridge | 0.19103 | 0.13125 | unavailable |
| synergy_lower | random_forest | 0.2742814 | 0.2430284 | unavailable |
| antagonism_higher | correct | 0.1664269 | 0.07997996 | 0.5 |
| antagonism_higher | wrong | 0.3921363 | 0.2662627 | 1 |
| antagonism_higher | M0 | 0.4038831 | 0.3001656 | 1 |
| antagonism_higher | ridge | 0.19103 | 0.13125 | unavailable |
| antagonism_higher | random_forest | 0.2742814 | 0.2430284 | unavailable |
| antagonism_lower | correct | 0.003119856 | 0.001997824 | 0 |
| antagonism_lower | wrong | 0.2629498 | 0.175647 | 1 |
| antagonism_lower | M0 | 0.09226179 | 0.05578985 | 1 |
| antagonism_lower | ridge | 0.191155 | 0.13125 | unavailable |
| antagonism_lower | random_forest | 0.1806141 | 0.1428534 | unavailable |
| additive_only | correct | 0.002027329 | 0.001531366 | 0 |
| additive_only | wrong | 0.001409354 | 0.001184676 | 0 |
| additive_only | M0 | 0.001404302 | 0.0009434833 | 1 |
| additive_only | ridge | 0.009807842 | 0.008235294 | unavailable |
| additive_only | random_forest | 0.06336422 | 0.0600375 | unavailable |
| zero_effect | correct | 2.473334e-06 | 2.281506e-06 | 1 |
| zero_effect | wrong | 1.909013e-06 | 1.34685e-06 | 1 |
| zero_effect | M0 | 0 | 0 | 1 |
| zero_effect | ridge | 0 | 0 | unavailable |
| zero_effect | random_forest | 0 | 0 | unavailable |
| weak_noisy | correct | 0.1021435 | 0.0770266 | 0 |
| weak_noisy | wrong | 0.1057919 | 0.08019085 | 0 |
| weak_noisy | M0 | 0.08946669 | 0.0723404 | 0.9375 |
| weak_noisy | ridge | 0.1033639 | 0.07760097 | unavailable |
| weak_noisy | random_forest | 0.08764626 | 0.07088032 | unavailable |
| changed_units | correct | 1.765315 | 0.7943686 | 0.9375 |
| changed_units | wrong | 1.868702 | 0.8359433 | 0.9375 |
| changed_units | M0 | 1.868697 | 0.8359437 | 0.9375 |
| changed_units | ridge | 9.958249 | 7.48955 | unavailable |
| changed_units | random_forest | 8.079824 | 4.76672 | unavailable |
| missing_transition | correct | unavailable | unavailable | unavailable |
| missing_transition | wrong | unavailable | unavailable | unavailable |
| missing_transition | M0 | 0.002598061 | 0.001575935 | 1 |
| missing_transition | ridge | 0.01500028 | 0.01317483 | unavailable |
| missing_transition | random_forest | 0.02240426 | 0.0161355 | unavailable |
| composed | correct | 0.001140278 | 0.000903336 | 0 |
| composed | wrong | 0.1522377 | 0.08329272 | 1 |
| composed | M0 | 0.03042575 | 0.02488314 | 1 |
| composed | ridge | 0.2156119 | 0.166828 | unavailable |
| composed | random_forest | 0.1805922 | 0.1455996 | unavailable |
| composed_two_shapes | correct | 0.00189574 | 0.001656503 | 0 |
| composed_two_shapes | wrong | 0.005394732 | 0.003869161 | 1 |
| composed_two_shapes | M0 | 0.01422884 | 0.01216721 | 1 |
| composed_two_shapes | ridge | 0.1939604 | 0.1514002 | unavailable |
| composed_two_shapes | random_forest | 0.1488803 | 0.123385 | unavailable |

## Interpretation and gates

- GP intervals are latent-response intervals conditional on point-estimate MAP parameters; parameter uncertainty is not integrated.
- Coverage is against noiseless generating truth, not a calibrated observation-prediction interval.
- Mean and residual GP can be confounded; fitted signs do not establish empirical or causal validity, and the full GP need not preserve component shape.
- Prior sensitivity means a strict maximum pairwise grid-mean difference greater than one training response scale; it is not a significance test.
- Constant training responses use S=1. Confidence and prior constants are engineering policy, not calibrated probabilities.
- Ridge and random forest provide point predictions only; their uncertainty is unavailable. They are never production fallbacks.
- No fixture, fold, generator, start, policy or audit-grid point is selected after observing a fit result.
- Missing-transition INSUFFICIENT_DATA is coverage failure, not default-policy optimizer failure.
- Every full record, policy, fold, physical grid, diagnostic, failed start, and pooled metric is in audit.json.
- E6 is not approved: independent result review, unresolved slow tests, version/compatibility and clean offline wheel gates remain separate.
