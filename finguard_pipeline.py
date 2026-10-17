"""FinGuard: offline fraud analytics and honest holdout evaluation for IBM LinuxONE.

The output is a fully self-contained HTML website with *computed* statistics from
approved, local CSV files. No external API or hosted model is used.
"""
from __future__ import annotations

import json
import re
import html
import math
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import warnings

ALIASES = {
    "amount": ["Amount", "Transaction Amount", "transaction_amount", "amt", "transaction_value", "amount_usd", "total", "TX_TOTAL_AMT"],
    "date": ["Date", "Transaction Date", "timestamp", "datetime", "transaction_timestamp", "trans_date", "date_time", "DATE_CREDIT"],
    "time": ["Time", "Transaction Time", "transaction_time"],
    "period": ["step", "period", "time_step"],
    "target": ["Is Fraud?", "is_fraud", "isFraud", "fraud_label", "fraud", "FRAUD_VER_RESULT", "fraud_result", "is_fraudulent"],
    "category": ["MCC", "merchant_category", "merchant_category_code", "category", "transaction_type", "type", "Transaction Type", "ENV_TYPE"],
    "method": ["Use Chip", "use_chip", "payment_method", "channel", "entry_mode", "transaction_method", "CARD_BRND"],
    "merchant_state": ["Merchant State", "merchant_state", "state"],
    "errors": ["Errors?", "errors", "transaction_errors"],
    "client": ["Client", "client_id", "customer_id", "account_id", "nameOrig", "user_id", "User", "CONT_ID"],
    "merchant": ["Merchant Name", "merchant_name", "merchant_id", "nameDest", "TX_MRCHNT_ID"],
    "transaction_id": ["id", "transaction_id", "transaction id", "txn_id"],
}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def detect_columns(columns):
    lookup = {_norm(s): s for s in columns}
    return {kind: next((lookup[_norm(alias)] for alias in candidates if _norm(alias) in lookup), None)
            for kind, candidates in ALIASES.items()}


def locate_data(directory=None):
    """Prefer the known LinuxONE home directory, without guessing file contents."""
    candidates = []
    for d in [directory, Path.cwd(), Path('/home/ibm-user'), Path.cwd().parent]:
        if d is None: continue
        p = Path(d).expanduser().resolve()
        if p not in candidates and p.is_dir(): candidates.append(p)
    findings = {}
    for name in ('transactions_history.csv', 'transactions.csv', 'clients.csv'):
        findings[name] = next((str(p / name) for p in candidates if (p / name).is_file()), None)
    return findings


def check_training_state(directory=None):
    """Check notebook saved results; cannot introspect unrelated kernel memory."""
    p = Path(directory or Path.cwd()).expanduser()
    paths = [p, p / 'FinGuard_LinuxONE', Path('/home/ibm-user')]
    artifacts = []
    notebooks = []
    for folder in paths:
        if not folder.is_dir(): continue
        for pattern in ('*.joblib', '*.pkl', '*metrics*.json', '*evaluation*.json'):
            for f in folder.glob(pattern):
                if f.name not in {k['name'] for k in artifacts}:
                    artifacts.append({'name': f.name, 'path': str(f), 'size_bytes': f.stat().st_size})
        for f in list(folder.glob('*.ipynb'))[:12]:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                cells = data.get('cells', [])
                training_cells = [c for c in cells if c.get('cell_type') == 'code' and any(
                    text in ''.join(c.get('source', [])) for text in ['LogisticRegression(', 'train_test_split(', 'classification_report('])]
                outputs = sum(bool(c.get('outputs', [])) for c in training_cells)
                executed = sum(c.get('execution_count') is not None for c in training_cells)
                notebooks.append({'name': f.name, 'training_code_cells': len(training_cells), 'executed_training_cells': executed, 'cells_with_saved_output': outputs})
            except Exception:
                notebooks.append({'name': f.name, 'note': 'Could not inspect notebook metadata'})
    return {'saved_artifacts': artifacts, 'notebook_checks': notebooks,
            'interpretation': 'This is a file/notebook metadata check, not proof that a usable trained model is in the current notebook kernel. FinGuard retrains reproducibly and reports fresh metrics.'}


def parse_fraud_labels(series):
    """Only accept explicitly interpretable binary outcomes; never infer labels."""
    pos = {'1', 'true', 'yes', 'y', 'fraud', 'fraudulent', 'confirmed fraud', 'positive', 'fraud confirmed', '1.0'}
    neg = {'0', 'false', 'no', 'n', 'legit', 'legitimate', 'nonfraud', 'non fraud', 'not fraud', 'not fraudulent', 'negative', '0.0', 'no fraud', 'approved non fraud'}
    values = series.astype('string').str.strip().str.lower().str.replace(r'[_-]', ' ', regex=True)
    target = values.map(lambda x: 1 if x in pos else 0 if x in neg else None)
    return target


def read_csv(path):
    import pandas as pd
    df = pd.read_csv(path, low_memory=False, on_bad_lines='skip')
    # Normalize whitespace in headers while preserving visible real source names.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _datetime_features(df, mapping):
    """Derive calendar dates from either an ISO column or Year/Month/Day columns.

    Financial history CSV splits calendar dates across Year/Month/Day and Time,
    whereas transactions.csv provides ISO-8601 DATE_CREDIT timestamps.
    """
    import pandas as pd
    import numpy as np
    dtcol = mapping.get('date')
    year_map = {_norm(c): c for c in df.columns}
    if dtcol:
        date_values = pd.to_datetime(df[dtcol].astype('string'), errors='coerce', utc=True)
        date_values = date_values.dt.tz_convert(None)
    elif all(k in year_map for k in ('year', 'month', 'day')):
        fields = {
            key: pd.to_numeric(df[year_map[key]], errors='coerce')
            for key in ('year', 'month', 'day')
        }
        date_values = pd.to_datetime(fields, errors='coerce')
    else:
        date_values = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    if mapping.get('time') and date_values.notna().any():
        # Add HH:MM only when the source has no meaningful timestamp component.
        if (date_values.dt.hour.fillna(0) == 0).mean() > .98:
            time_text = df[mapping['time']].astype('string').str.strip()
            time_text = time_text.where(time_text.str.count(':') >= 2, time_text + ':00')
            duration = pd.to_timedelta(time_text, errors='coerce')
            merged = date_values + duration
            if merged.notna().sum() > date_values.notna().sum() * .65:
                date_values = merged
    out = pd.DataFrame(index=df.index)
    out['hour'] = date_values.dt.hour.astype(float)
    out['day_of_week'] = date_values.dt.dayofweek.astype(float)
    out['month'] = date_values.dt.month.astype(float)
    out['period'] = pd.to_numeric(df[mapping['period']], errors='coerce') if mapping.get('period') else np.nan
    return date_values, out


def prepare_features(df, mapping, allowed_methods=None, allowed_categories=None):
    import pandas as pd
    import numpy as np
    values = pd.DataFrame(index=df.index)
    ac = mapping.get('amount')
    raw_amount = df[ac].astype('string').str.replace(r'[$,€£\s]', '', regex=True).str.replace(r'^\((.*)\)$', r'-\1', regex=True) if ac else pd.Series(pd.NA, index=df.index)
    amount = pd.to_numeric(raw_amount, errors='coerce')
    values['amount_log'] = np.log1p(amount.abs().clip(lower=0)).astype(float)
    values['is_refund'] = (amount < 0).astype(float)
    dates, dtfeatures = _datetime_features(df, mapping)
    for col in dtfeatures: values[col] = dtfeatures[col]
    # Group rare categories. Identifiers are never model inputs.
    cat_cols = []
    for key, colname, top_n in [('category', 'merchant_category', 35), ('method', 'method', 15), ('merchant_state', 'region', 30), ('errors', 'errors', 15)]:
        col = mapping.get(key)
        if col:
            text = df[col].astype('string').fillna('Unknown').str.strip().str.slice(0, 80).replace('', 'Unknown').fillna('Unknown')
            selected = (allowed_categories or {}).get(key) if allowed_categories else None
            if selected is None: selected = text.value_counts().head(top_n).index.tolist()
            text = text.where(text.isin(selected), 'Other')
        else:
            text = pd.Series('Unknown', index=df.index)
        values[colname] = text.astype(str)
        cat_cols.append(colname)
    numerical_cols = ['amount_log', 'is_refund', 'hour', 'day_of_week', 'month', 'period']
    return values, dates, amount, numerical_cols, cat_cols


def _metrics(y_true, scores, threshold):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {'threshold': round(float(threshold), 5), 'rows': len(y_true), 'positives': int(sum(y_true)), 'precision': round(float(precision_score(y_true, pred, zero_division=0)), 5), 'recall': round(float(recall_score(y_true, pred, zero_division=0)), 5), 'f1': round(float(f1_score(y_true, pred, zero_division=0)), 5), 'accuracy': round(float(accuracy_score(y_true, pred)), 5), 'pr_auc': round(float(average_precision_score(y_true, scores)), 5), 'roc_auc': round(float(roc_auc_score(y_true, scores)), 5) if len(set(y_true)) == 2 else None, 'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn), 'alerts': int(sum(pred)), 'prevalence': round(float(sum(y_true)/len(y_true)), 6)}


def _clean_row_payload(df, indices, scores, labels, amount, dates, mapping):
    """De-identified rows. Never output the original account, card or merchant identifier."""
    import pandas as pd
    vals = []
    categories = df[mapping['category']].astype(str) if mapping.get('category') else pd.Series('Uncategorized', index=df.index)
    methods = df[mapping['method']].astype(str) if mapping.get('method') else pd.Series('Unknown', index=df.index)
    for i in indices:
        dt = dates.loc[i] if i in dates.index else pd.NaT
        iso = '' if pd.isna(dt) else pd.Timestamp(dt).strftime('%Y-%m-%d %H:%M')
        val_amount = amount.loc[i]
        vals.append({'id': f'CASE-{int(i)+1:06d}', 'date': iso or 'Undated', 'amount': None if pd.isna(val_amount) else round(float(val_amount), 2), 'category': str(categories.loc[i])[:58], 'method': str(methods.loc[i])[:44], 'score': round(float(scores.loc[i])*100, 2), 'fraud_label': int(labels.loc[i])})
    return vals


def _aggregate(df, mapping, y, amount, dates):
    import pandas as pd
    n = int(len(df)); valid_amounts = amount.dropna()
    positives = int(y.eq(1).sum()) if y is not None else None
    tagged = int(y.notna().sum()) if y is not None else 0
    amounts = {'gross': round(float(valid_amounts.abs().sum()), 2), 'net': round(float(valid_amounts.sum()), 2), 'median': round(float(valid_amounts.median()), 2) if len(valid_amounts) else None, 'invalid': int(amount.isna().sum())}
    grouping = pd.DataFrame({'amount': amount.abs(), 'fraud': y if y is not None else pd.Series(pd.NA,index=df.index)}, index=df.index)
    if dates.notna().sum() >= n*.5:
        # Multi-year card histories need monthly summaries: a trailing 90-day
        # view could hide years of source-labeled fraud activity.
        span_days = (dates.max() - dates.min()).days if dates.notna().any() else 0
        if span_days > 365:
            grouping['bucket'] = dates.dt.strftime('%Y-%m')
            time_type = 'calendar_month'
        else:
            grouping['bucket'] = dates.dt.strftime('%Y-%m-%d')
            time_type = 'calendar_day'
    elif mapping.get('period'):
        grouping['bucket'] = df[mapping['period']].astype('string').fillna('Unknown')
        time_type = 'sequence_period'
    else:
        grouping['bucket'] = 'Undated'; time_type = 'none'
    trend = grouping.groupby('bucket', dropna=False).agg(count=('amount','size'), volume=('amount','sum'), fraud=('fraud',lambda x: int((x == 1).sum()))).reset_index()
    trend = trend.sort_values('bucket').tail(90)
    time_points = [{'date': str(r.bucket), 'count': int(r['count']), 'volume': round(float(r.volume),2), 'fraud': int(r.fraud)} for _,r in trend.iterrows()]
    if mapping.get('category'):
        grouping['category'] = df[mapping['category']].astype('string').fillna('Unknown').str.slice(0,55)
    else: grouping['category']='Other'
    categories = grouping.groupby('category').agg(count=('amount','size'),volume=('amount','sum'), fraud=('fraud',lambda x: int((x==1).sum()))).reset_index().sort_values('count',ascending=False).head(12)
    cat_points = [{'name':str(r.category), 'count':int(r['count']), 'volume':round(float(r.volume),2),'fraud':int(r.fraud)} for _,r in categories.iterrows()]
    return {'rows': n, 'labeled_rows': tagged, 'fraud_count': positives, 'fraud_rate': round((positives/tagged)*100,5) if tagged and positives is not None else None,'amount': amounts,'trend':time_points,'time_type':time_type,'categories':cat_points,'source_columns':list(df.columns),'mapping':mapping}


def _aggregate_secondary(path):
    import pandas as pd
    df=read_csv(path); mapping=detect_columns(df.columns)
    # Avoid unnecessary preprocessing just to show source metadata.
    if mapping.get('amount'):
        raw=df[mapping['amount']].astype('string').str.replace(r'[$,€£\s]', '', regex=True)
        money=pd.to_numeric(raw,errors='coerce')
    else: money=pd.Series(float('nan'),index=df.index)
    label=parse_fraud_labels(df[mapping['target']]) if mapping.get('target') else None
    return {'file':path.name,'rows':len(df),'amount_gross':round(float(money.abs().sum()),2), 'fraud_count':int(label.eq(1).sum()) if label is not None else None,'labeled_count':int(label.notna().sum()) if label is not None else 0,'detected_fields':mapping, 'note':'Comparison source only; not mixed into the historical training data.'}


def run_pipeline(data_dir=None, output_dir=None, verbose=True):
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.metrics import precision_recall_curve

    paths=locate_data(data_dir)
    source=paths.get('transactions_history.csv') or paths.get('transactions.csv')
    if not source:
        raise FileNotFoundError('No transactions_history.csv or transactions.csv found. Put the FinGuard folder next to the CSVs in /home/ibm-user or pass data_dir=Path("/home/ibm-user").')
    source=Path(source)
    output_dir=Path(output_dir or Path.cwd()).expanduser().resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    if verbose:print('Reading real LinuxONE CSV:',source)
    df=read_csv(source)
    mapping=detect_columns(df.columns)
    if not mapping['amount']:
        raise ValueError(f'Could not detect an amount column in {source.name}. Observed columns: {list(df.columns)[:40]}')
    X, dates, amount, numeric_cols, cat_cols = prepare_features(df,mapping)
    valid_amounts=amount.notna()
    y=parse_fraud_labels(df[mapping['target']]) if mapping.get('target') else None
    statistics=_aggregate(df,mapping,y,amount,dates)
    metrics=None; sampling_info=None; cases=[]; eval_scores=[]; coefficient_features=[]
    errors=[]
    if y is not None:
        eligible=valid_amounts & y.notna()
        count_positive=int(y.loc[eligible].eq(1).sum())
        count_negative=int(y.loc[eligible].eq(0).sum())
        if verbose:print(f'Usable labeled rows: {eligible.sum():,}; fraud: {count_positive:,}; not fraud: {count_negative:,}')
    else:
        eligible=pd.Series(False,index=df.index);count_positive=count_negative=0
    if count_positive >= 25 and count_negative >= 50:
        inds=np.flatnonzero(eligible.to_numpy())
        yall=y.iloc[inds].astype(int)
        train_val, test=train_test_split(inds, test_size=.20, random_state=42, stratify=yall)
        train, val=train_test_split(train_val, test_size=.25, random_state=42, stratify=y.iloc[train_val].astype(int))
        # All feature preparation is non-target; the actual imputers and scalers fit on training only.
        transform=ColumnTransformer([
            ('number',Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),numeric_cols),
            ('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',min_frequency=20))]),cat_cols),
        ],remainder='drop',sparse_threshold=.3)
        model=Pipeline([('prep',transform),('classifier',LogisticRegression(max_iter=500,solver='liblinear',class_weight='balanced',random_state=42))])
        if verbose:print(f'Training model on {len(train):,} rows; validation {len(val):,}; holdout test {len(test):,}...')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            model.fit(X.iloc[train],y.iloc[train].astype(int))
        val_prob=model.predict_proba(X.iloc[val])[:,1]
        precision, recall, thresh=precision_recall_curve(y.iloc[val].astype(int),val_prob)
        f1=2*precision[:-1]*recall[:-1]/np.maximum(1e-12,precision[:-1]+recall[:-1])
        # Optimize *validation* F1; never select threshold on held-out test.
        best=int(np.argmax(f1)); threshold=float(thresh[best]) if len(thresh) else .5
        test_prob=model.predict_proba(X.iloc[test])[:,1]
        actual=y.iloc[test].astype(int).to_numpy()
        metrics=_metrics(actual,test_prob,threshold)
        metrics['validation_threshold_selected_for']='F1'
        metrics['train_rows']=len(train);metrics['validation_rows']=len(val)
        sampling_info={'rows_used':int(eligible.sum()),'rows_excluded':int((~eligible).sum()),'features':numeric_cols+cat_cols,'source_label_column':mapping['target'],'split':'stratified 60/20/20 train/validation/test, random_state=42','warning':'Random splits may overestimate future performance if customer patterns or dates overlap. External or temporal validation is required for real deployment.'}
        # Produce real held-out predictions, never in-sample scores for the case explorer.
        sorted_indices=np.argsort(-test_prob)
        high=sorted_indices[:min(2800,len(test_prob))].tolist()
        low=np.random.default_rng(42).choice(sorted_indices[min(2800,len(test_prob)):],size=min(1000,max(0,len(test_prob)-min(2800,len(test_prob)))),replace=False).tolist() if len(test_prob)>2800 else []
        selected=np.array(high+low,dtype=int)
        index=pd.Index(df.index[test])
        prob_ser=pd.Series(test_prob,index=index);label_ser=pd.Series(actual,index=index)
        cases=_clean_row_payload(df,list(index[selected]),prob_ser,label_ser,amount,dates,mapping)
        eval_scores=[{'p':int(round(p*1000000)),'y':int(label)} for p,label in zip(test_prob,actual)]
        if verbose: print(f"Holdout PR-AUC={metrics['pr_auc']:.4f}; precision={metrics['precision']:.4f}; recall={metrics['recall']:.4f}; threshold={threshold:.4f}")
        # Save freshly fitted pipeline, but never import a pickle from an unknown source.
        try:
            import joblib
            joblib.dump(model,output_dir/'finguard_model.joblib')
        except Exception as exc:
            errors.append('Model serialization unavailable: '+str(exc))
        try:
            names=model.named_steps['prep'].get_feature_names_out()
            coefs=model.named_steps['classifier'].coef_[0]
            inds_order=np.argsort(-np.abs(coefs))[:10]
            coefficient_features=[{'name':str(names[i])[:75], 'coefficient':round(float(coefs[i]),4)} for i in inds_order]
        except Exception:
            pass
    else:
        reason=f'Could not train a supervised model: need at least 25 readable fraud labels and 50 readable legitimate labels. Found {count_positive} positive and {count_negative} negative. Check the historical fraud-label column.'
        errors.append(reason)
        if verbose: print(reason)
    secondary=None
    secondary_file=paths.get('transactions.csv')
    if secondary_file and Path(secondary_file).resolve() != source.resolve():
        try: secondary=_aggregate_secondary(Path(secondary_file))
        except Exception as e: errors.append('Could not summarize transactions.csv: '+str(e))
    clients_path=paths.get('clients.csv')
    clients=None
    if clients_path:
        try:
            cl=read_csv(clients_path)
            clients={'file':Path(clients_path).name,'rows':len(cl),'note':'Aggregate metadata only; customer identifiers, records, and individual attributes are not copied into the report.'}
            if 'CHURN' in cl.columns:
                churn=pd.to_numeric(cl['CHURN'],errors='coerce')
                clients['churn_count']=int(churn.eq(1).sum())
                clients['churn_rate_pct']=round(float(churn.eq(1).mean()*100),2)
            if 'AGE_YEARS' in cl.columns:
                ages=pd.to_numeric(cl['AGE_YEARS'],errors='coerce')
                clients['age_bands']={str(label):int(count) for label,count in zip(
                    ('Under 30','30–44','45–59','60+'),
                    (ages.lt(30).sum(),ages.ge(30).mul(ages.lt(45)).sum(),ages.ge(45).mul(ages.lt(60)).sum(),ages.ge(60).sum()))}
            if 'ACTIVITY_LEVEL' in cl.columns:
                activity=cl['ACTIVITY_LEVEL'].astype('string').value_counts()
                clients['activity_levels']={str(k):int(v) for k,v in activity.items() if str(k).strip()!='<NA>'}

        except Exception as e: errors.append('Could not read client metadata: '+str(e))
    payload={'meta':{'created_utc':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), 'title':'FinGuard — Real-data Fraud Intelligence', 'source':source.name, 'source_folder':str(source.parent),'source_kind':'Actual workspace CSV, not synthetic dashboard data','warning':'Hackathon research prototype; source fraud labels are ground truth only within their original dataset. Risk scores are uncalibrated model rankings, not evidence of wrongdoing.'},'summary':statistics,'metrics':metrics,'sampling':sampling_info,'cases':cases,'eval_scores':eval_scores,'top_features':coefficient_features,'secondary':secondary,'clients':clients,'errors':errors}
    # Protect the script tag boundary from untrusted CSV content.
    safe_json=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=(Path(__file__).parent/'dashboard_template.html').read_text(encoding='utf-8')
    rendered=template.replace('/*__FINGUARD_DATA__*/', 'const FG_DATA = '+safe_json+';')
    dest=output_dir/'finguard_report.html'
    dest.write_text(rendered,encoding='utf-8')
    metrics_file=output_dir/'finguard_metrics.json'
    metrics_file.write_text(json.dumps({'source':source.name,'metrics':metrics,'sampling':sampling_info,'errors':errors},indent=2),encoding='utf-8')
    if verbose:
        print('Created:',dest)
        print('Created:',metrics_file)
        if metrics:print('Saved test cases:',len(cases),'(all evaluation scores included for threshold analysis)')
    return {'report':str(dest),'metrics_json':str(metrics_file),'summary':statistics,'metrics':metrics,'errors':errors,'paths':paths}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default=None)
    ap.add_argument('--output-dir',default=None)
    args=ap.parse_args()
    run_pipeline(data_dir=args.data_dir,output_dir=args.output_dir)
