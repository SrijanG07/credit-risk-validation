SELECT p.model,
       CASE WHEN b.LIMIT_BAL < 100000 THEN 'low'
            WHEN b.LIMIT_BAL < 300000 THEN 'medium' ELSE 'high' END AS limit_segment,
       COUNT(*) AS n, AVG(b."default" * 1.0) AS observed_rate,
       AVG(p.probability) AS predicted_rate
FROM predictions p JOIN borrowers b ON p.ID=b.ID
GROUP BY p.model, limit_segment ORDER BY p.model, limit_segment;
