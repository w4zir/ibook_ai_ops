## Q&A

### Q: Does the system currently detect model drift and trigger training if it drifts?

**A:** The system computes a drift score and a `needs_retrain` flag (for example in the `ml_monitoring_pipeline` DAG), and logs that it *would* trigger the `model_training_pipeline` when drift exceeds the 0.3 threshold. However, the actual trigger is currently a stub (logging only); it does not yet use `TriggerDagRunOperator` or otherwise automatically start retraining.

### Q: In realtime simulator mode, why can `--rps 100` over 60 seconds result in only ~15 transactions?

**A:** The realtime runner sends requests sequentially and each transaction waits for the `/predict` HTTP call (or its timeout) before sending the next one. If the configured `API_BASE_URL` points to a slow or unreachable service, each request can block for several seconds, so a 60-second wall-clock run might only complete a small number of transactions (e.g. ~15) even though the target RPS is 100.

### Q: In realtime simulator mode, are all transactions checked against the BentoML fraud/pricing endpoint?

**A:** Every generated transaction in realtime mode is posted to the configured `API_BASE_URL/predict` endpoint for fraud scoring, and the simulator falls back to a synthetic response only if the HTTP call fails. Dynamic pricing, however, is simulated locally inside the transaction generator and is not currently calling a separate BentoML pricing endpoint.

### Q: What’s the difference between Peak RPS, txns, error rate in the HTML report and the `errors=` counter printed in the terminal?

**A:** The **HTML report metrics** are computed *after* the scenario finishes, from the full list of recorded responses: **Peak RPS** is `total_responses ÷ duration_seconds`, **txns**/transaction counts come from the stored `responses` array, and **error rate** is `(#responses with status ≥ 400) ÷ total_responses`. The **`errors=` counter in the terminal** is a *live, running tally* of how many requests have seen an error so far (HTTP status ≥ 400 or a request failure) during execution; it is useful for real‑time feedback but is not the final, normalized metric used in the report.

