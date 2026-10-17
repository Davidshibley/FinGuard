# FinGuard Live — Theme 1 Judging Pitch

**Problem:** fraud is rare: our uploaded historical transaction dataset contains 245,864 records with 271 labeled fraud events (around 0.11%). This creates a precision/recall trade-off for investigators.

**Demonstration:** FinGuard Live replays selected held-out historical payment cases as incoming events, uses their precomputed supervised fraud-ranking scores to recommend normal/monitor/review routes, and records analyst acknowledgments/escalations. The site also shows model evaluation, audit export, searchable cases and real-data analytics. This is a safe, offline replay, **not** a connection to a payment network or a deployed inference endpoint.

**Walkthrough:** (1) Start Replay (2) Increase playback rate (3) Observe flagged payment (4) Acknowledge and escalate cases (5) Change threshold to expose the precision/alert-volume trade-off (6) Export audit (7) Show held-out metrics and explain limitations.

**Evidence:** Baseline test precision 3.46%, recall 27.78%, PR-AUC 0.02545, ROC-AUC 0.93834. Held-out evaluation is separate from the biased replay sample; report no invented production latency.

**Future production design:** payment event bus → secure on-platform model-serving API → policy-driven recommendation → authenticated analyst workflow → durable, auditable action store; validate with temporal splits, calibration, operational latency and fairness reviews.
