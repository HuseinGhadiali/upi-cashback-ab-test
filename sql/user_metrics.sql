-- One row per assigned user with every experiment metric (intention-to-treat population).
-- Runs on DuckDB over the raw CSV logs; PostgreSQL-compatible apart from the read_csv() table functions.
WITH exp_bills AS (
    SELECT user_id,
           COUNT(*)                                                       AS bill_payments,
           SUM(amount)                                                    AS bill_gmv,
           SUM(cashback)                                                  AS cashback_paid,
           SUM(CASE WHEN status = 'Refunded' THEN 1 ELSE 0 END)           AS refunds,
           MIN(payment_ts)                                                AS first_bill_ts
    FROM bill_payments
    WHERE period = 'experiment'
    GROUP BY user_id
),
post_bills AS (
    SELECT user_id, COUNT(*) AS post_bill_payments
    FROM bill_payments
    WHERE period = 'post' AND status = 'Success'
    GROUP BY user_id
),
first_exposure AS (
    SELECT user_id, MIN(exposure_ts) AS first_exposure_ts, MIN(app_version) AS app_version
    FROM exposures GROUP BY user_id
)
SELECT a.user_id,
       a.variant,
       u.city_tier, u.platform, u.age_band,
       CASE WHEN u.tenure_days < 180 THEN 'New (<6 months)' ELSE 'Established' END AS tenure_group,
       u.pre_upi_txn_28d, u.pre_upi_gmv_28d,
       fe.first_exposure_ts, fe.app_version,
       CASE WHEN fe.user_id IS NOT NULL THEN 1 ELSE 0 END                          AS exposure_logged,
       CASE WHEN eb.bill_payments > 0 THEN 1 ELSE 0 END                            AS converted,
       COALESCE(eb.bill_payments, 0)                                               AS bill_payments,
       COALESCE(eb.bill_gmv, 0)                                                    AS bill_gmv,
       COALESCE(eb.cashback_paid, 0)                                               AS cashback_paid,
       COALESCE(eb.refunds, 0)                                                     AS refunds,
       eb.first_bill_ts,
       CASE WHEN pb.post_bill_payments > 0 THEN 1 ELSE 0 END                       AS paid_bill_post_period,
       g.uninstalled_during_test, g.support_tickets, g.p2p_txn_during_test
FROM assignments a
JOIN users u            ON u.user_id = a.user_id
LEFT JOIN exp_bills eb  ON eb.user_id = a.user_id
LEFT JOIN post_bills pb ON pb.user_id = a.user_id
LEFT JOIN first_exposure fe ON fe.user_id = a.user_id
JOIN guardrails g       ON g.user_id = a.user_id
ORDER BY a.user_id;
