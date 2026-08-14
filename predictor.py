
import os, json, warnings, smtplib
from pathlib import Path
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_score

warnings.filterwarnings("ignore")

TARGET = 0.03
MIN_HISTORY = 250
TOP_N = 12
DATA_DIR = Path("prediction_history")
DATA_DIR.mkdir(exist_ok=True)
FEATURES = ["ret1","ret3","ret5","ret10","ret20","range1","atr14_pct","rsi14",
            "vol_ratio5","vol_ratio20","sma20_dist","sma50_dist","sma200_dist",
            "high20_dist","high60_dist","gap","body"]

def universe():
    url="https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    x=pd.read_csv(url)
    return [s.strip()+".NS" for s in x["Symbol"].dropna().astype(str)]

def get_data(symbol):
    x=yf.download(symbol, period="6y", interval="1d", auto_adjust=True,
                  progress=False, threads=False)
    if x.empty: return None
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x[["Open","High","Low","Close","Volume"]].dropna()

def rsi(s,n=14):
    d=s.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

def make_features(d):
    x=d.copy(); c,h,l,o,v=x.Close,x.High,x.Low,x.Open,x.Volume
    x["ret1"]=c.pct_change(); x["ret3"]=c.pct_change(3); x["ret5"]=c.pct_change(5)
    x["ret10"]=c.pct_change(10); x["ret20"]=c.pct_change(20); x["range1"]=(h-l)/c
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    x["atr14_pct"]=tr.rolling(14).mean()/c; x["rsi14"]=rsi(c)
    x["vol_ratio5"]=v/v.rolling(5).mean(); x["vol_ratio20"]=v/v.rolling(20).mean()
    x["sma20_dist"]=c/c.rolling(20).mean()-1; x["sma50_dist"]=c/c.rolling(50).mean()-1
    x["sma200_dist"]=c/c.rolling(200).mean()-1; x["high20_dist"]=c/c.rolling(20).max()-1
    x["high60_dist"]=c/c.rolling(60).max()-1; x["gap"]=o/c.shift()-1; x["body"]=(c-o)/o
    # Tomorrow's intraday high relative to today's close.
    x["future_high_return"]=h.shift(-1)/c-1
    x["target"]=(x["future_high_return"]>=TARGET).astype(int)
    return x.replace([np.inf,-np.inf],np.nan)

def predict(symbol,d):
    f=make_features(d).dropna(subset=FEATURES)
    if len(f)<MIN_HISTORY:return None
    train=f.iloc[:-1].tail(900)
    cut=max(100,int(.8*len(train)))
    imp=SimpleImputer(strategy="median")
    Xtr=imp.fit_transform(train.iloc[:cut][FEATURES])
    Xv=imp.transform(train.iloc[cut:][FEATURES])
    ytr=train.iloc[:cut].target; yv=train.iloc[cut:].target
    model=HistGradientBoostingClassifier(max_iter=220,learning_rate=.055,
        max_leaf_nodes=15,l2_regularization=1,random_state=42)
    model.fit(Xtr,ytr)
    pv=model.predict_proba(Xv)[:,1]
    auc=roc_auc_score(yv,pv) if yv.nunique()>1 else np.nan
    precision=precision_score(yv,pv>=.60,zero_division=0)
    last=f.iloc[[-1]]
    p=float(model.predict_proba(imp.transform(last[FEATURES]))[0,1])
    return {
        "symbol":symbol.replace(".NS",""), "probability":p,
        "prediction_date":str(last.index[0].date()),
        "reference_close":float(last.Close.iloc[0]),
        "rsi":float(last.rsi14.iloc[0]),
        "relative_volume":float(last.vol_ratio20.iloc[0]),
        "ret5":float(last.ret5.iloc[0]),
        "validation_auc":None if not np.isfinite(auc) else float(auc),
        "validation_precision":float(precision)
    }

def history_files():
    return sorted(DATA_DIR.glob("predictions_*.csv"))

def evaluate_yesterday():
    files=history_files()
    if not files:return []
    latest=files[-1]
    p=pd.read_csv(latest)
    if p.empty:return []
    out=[]
    for sym in p.symbol:
        try:
            d=get_data(sym+".NS")
            if d is None:continue
            pred=p[p.symbol==sym].iloc[0]
            ref=float(pred.reference_close)
            # Find the first trading row after the prediction date.
            after=d[d.index.strftime("%Y-%m-%d") > str(pred.prediction_date)]
            if after.empty:continue
            day=after.iloc[0]
            high_ret=float(day.High/ref-1)
            close_ret=float(day.Close/ref-1)
            out.append({
                "symbol":sym,
                "probability":float(pred.probability),
                "reference_close":ref,
                "actual_high_return":high_ret,
                "actual_close_return":close_ret,
                "hit":high_ret>=TARGET
            })
        except Exception:
            pass
    return out

def send_email(today, yesterday):
    today_rows="".join(
        f"<tr><td><b>{r['symbol']}</b></td><td>{r['probability']*100:.1f}%</td>"
        f"<td>₹{r['reference_close']:.2f}</td><td>{r['rsi']:.1f}</td>"
        f"<td>{r['relative_volume']:.2f}x</td><td>{r['ret5']*100:+.1f}%</td></tr>"
        for r in today)
    y_rows="".join(
        f"<tr><td><b>{r['symbol']}</b></td><td>{r['probability']*100:.1f}%</td>"
        f"<td>{r['actual_high_return']*100:+.2f}%</td>"
        f"<td>{r['actual_close_return']*100:+.2f}%</td>"
        f"<td>{'✅ HIT' if r['hit'] else '❌ MISS'}</td></tr>"
        for r in yesterday)
    hits=sum(r["hit"] for r in yesterday)
    total=len(yesterday)
    hitrate=(hits/total*100) if total else 0
    avg_high=(np.mean([r["actual_high_return"] for r in yesterday])*100) if yesterday else 0

    html=f"""<html><body>
    <h2>🇮🇳 Daily Indian Stock Prediction — {datetime.now():%d %b %Y}</h2>
    <h3>🔮 Today's Predictions</h3>
    <p>Probability that today's intraday HIGH reaches at least <b>+3%</b>
    from the previous close.</p>
    <table border=1 cellpadding=6 cellspacing=0>
    <tr><th>Stock</th><th>Probability</th><th>Reference Close</th>
    <th>RSI</th><th>Relative Volume</th><th>5D Return</th></tr>{today_rows}
    </table>
    <h3>📊 Yesterday's Prediction vs Performance</h3>
    <table border=1 cellpadding=6 cellspacing=0>
    <tr><th>Stock</th><th>Predicted Probability</th><th>Actual Intraday High</th>
    <th>Actual Close</th><th>Result</th></tr>{y_rows}
    </table>
    <p><b>Yesterday hit rate:</b> {hits}/{total} ({hitrate:.1f}%)<br>
    <b>Average actual intraday high:</b> {avg_high:+.2f}%</p>
    <p><small>A hit means the stock's intraday high reached +3% from the
    prediction reference close. Model probabilities are estimates, not guarantees.</small></p>
    </body></html>"""

    msg=MIMEMultipart("alternative")
    msg["Subject"]="🇮🇳 Daily Indian Stock Predictions + Yesterday's Results"
    msg["From"]=os.environ["SMTP_USER"]; msg["To"]=os.environ["ALERT_TO"]
    msg.attach(MIMEText(html,"html"))
    with smtplib.SMTP_SSL(os.environ["SMTP_HOST"],int(os.getenv("SMTP_PORT","465"))) as s:
        s.login(os.environ["SMTP_USER"],os.environ["SMTP_PASSWORD"])
        s.sendmail(os.environ["SMTP_USER"],[os.environ["ALERT_TO"]],msg.as_string())

def main():
    # First evaluate the previous prediction file using the completed session.
    yesterday=evaluate_yesterday()

    results=[]
    for i,s in enumerate(universe(),1):
        try:
            d=get_data(s)
            if d is not None:
                r=predict(s,d)
                if r:r["symbol"]=s.replace(".NS",""); results.append(r)
        except Exception: pass
        print(f"{i}",end="\r")

    results=[r for r in results if r["probability"]>=.55]
    results.sort(key=lambda x:x["probability"],reverse=True)
    results=results[:TOP_N]

    # Persist predictions for tomorrow's evaluation.
    stamp=datetime.now().strftime("%Y-%m-%d")
    pd.DataFrame(results).to_csv(DATA_DIR/f"predictions_{stamp}.csv",index=False)

    send_email(results,yesterday)
    print(f"\nSent {len(results)} today's predictions and {len(yesterday)} yesterday results.")

if __name__=="__main__":
    main()
