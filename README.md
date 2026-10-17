# FinGuard Live — IBM Z Datathon Theme 1

**Theme 1: Real-Time AI for Critical Decisions** — payment-risk replay and human-in-the-loop response.

## Open the completed website (no Terminal needed)

Download and open `FinGuard_Live_Theme1.html` in Chrome or another modern browser. It is self-contained and already embeds analytics computed from the three real CSV uploads (`transactions_history.csv`, `transactions.csv`, `clients.csv`). Start with **Live payment replay** in the left navigation.

The replay uses **3,800 selected, held-out historical cases** rather than a production payment connection or an unbiased random live stream. Events are played in time order, but source labels and precomputed logistic-model scores originate from the original offline evaluation. The replay does **not** perform per-event Python model inference. Client-side event processing time and throughput measure the **browser simulation** only, not IBM LinuxONE or model-serving performance. This distinction is made explicit in the interface.

## Live demo workflow

1. Open **Live payment replay** and click **Start replay**.
2. Change **Events / second** and watch transaction count, review alerts, replay throughput, and browser decision time.
3. Change the review threshold to adjust prioritization. The model scores are unchanged.
4. Select **Acknowledge** or **Escalate** for a pending item in the review queue. These actions update a session-only audit record; nothing is blocked or written to a banking system.
5. Click **Export session CSV** to save an audit of replayed cases and simulated actions.
6. Show **Model performance** for held-out metrics and **Investigation center** for individual transaction explanations and labels.

## Reproduce on LinuxONE / Jupyter

Upload `START_HERE_FinGuard.ipynb`, `finguard_pipeline.py`, and `dashboard_template.html` to the same Jupyter folder as the three source CSVs, or use the notebook with CSVs in `/home/ibm-user`. Execute all cells in the notebook. It regenerates `finguard_report.html` with the updated template and the replay tab; the offline model is retrained/evaluated on server data. The already-generated `FinGuard_Live_Theme1.html` is ready to present without executing anything.

## What is real versus simulated

- **Real source data:** records summarized from uploaded LinuxONE CSVs.
- **Model:** saved logistic-regression baseline evaluated on a held-out dataset, not calibrated. Existing test result: precision 3.46%, recall 27.78%, PR-AUC 0.02545, ROC-AUC 0.93834 at validation-selected threshold (from the earlier reproducible run).
- **Simulation:** historical case arrival, queue decisions, analyst acknowledgments/escalations, browser processing timing and throughput.
- **Not implemented:** live payment-network ingestion, remotely deployed inference service, event persistence, authentication/authorization, real payment holds/declines, production latency or reliability benchmarks.

**Accuracy caution:** The replay's 3,800-case selection overrepresents high-scoring cases; replay alerts and label rates cannot be interpreted as operating rates for the full population. Ground-truth labels are displayed for demonstration, never used to make stream decisions. Do not treat scores as calibrated fraud probabilities or automated financial determinations.
