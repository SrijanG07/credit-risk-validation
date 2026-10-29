-- Independently reconcile labeled holdout results with Python aggregates.
-- Each prediction row must match exactly one borrower; integrity enforced by code.
WITH joined AS (
  SELECT p.model, p.probability, b."default" AS outcome,
         CASE WHEN b.LIMIT_BAL < 100000 THEN 'low'
              WHEN b.LIMIT_BAL < 300000 THEN 'medium' ELSE 'high' END AS limit_segment
  FROM predictions p JOIN borrowers b ON p.ID = b.ID
)
SELECT model, COUNT(*) AS n, SUM(outcome) AS defaults,
       AVG(outcome * 1.0) AS observed_rate, AVG(probability) AS predicted_rate,
       AVG((probability-outcome)*(probability-outcome)) AS brier
FROM joined GROUP BY model ORDER BY model;
