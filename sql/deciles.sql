-- Tie-break by ID for deterministic report buckets (not model features).
WITH ranked AS (
  SELECT p.model, p.probability, b."default" AS outcome,
         NTILE(10) OVER (PARTITION BY p.model ORDER BY p.probability, p.ID) AS risk_decile
  FROM predictions p JOIN borrowers b ON p.ID=b.ID
)
SELECT model, risk_decile, COUNT(*) AS n, SUM(outcome) AS defaults,
       AVG(outcome * 1.0) AS observed_rate, AVG(probability) AS predicted_rate
FROM ranked GROUP BY model, risk_decile ORDER BY model, risk_decile;
