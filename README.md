# Daily Indian Stock >3% Prediction + Performance System

Every weekday before the Indian market opens, the system:
1. Scans the Nifty 500.
2. Estimates each stock's probability that its intraday HIGH will reach +3%
   from the previous close.
3. Emails the top candidates to the configured address.
4. Reads yesterday's saved predictions.
5. Downloads the completed trading day's data.
6. Marks each prediction HIT/MISS using the actual intraday high.
7. Includes yesterday's hit rate and average intraday high in today's email.
8. Saves today's predictions in `prediction_history/` for the next report.

A HIT means:
actual intraday high >= prediction reference close * 1.03.

This is a research model, not a guarantee or automated trading strategy.

GitHub repository secrets:
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=your sending Gmail address
SMTP_PASSWORD=Google App Password
ALERT_TO=kishankarma55@gmail.com
