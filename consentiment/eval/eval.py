
import pandas as pd
from ast import literal_eval
from nervaluate import Evaluator
import matplotlib.pyplot as plt

## this converts those entities to span
## it will try to find the entity in the snippet, if it can't, will ignore that entity
def entity_to_span(text, ent, pol='NEU'):
    # print(text, '\n\t', ent, pol)
    span = {'label':pol, 'start':0, 'end':0}
    try: span['start'] = text.index(ent)
    except ValueError:
        # print(text, '\n\t', ent)
        return None
    span['end'] = span['start'] + len(ent)
    return span

def column_to_span(row, col, text:str="text"):
    l = []
    for ent in row[col]:
        span = entity_to_span(row[text], ent[0], ent[1])
        if span is not None: l.append(span)
    return l

def df_to_spans(df:pd.DataFrame, text:str="text"):
    spans = pd.DataFrame(df.values[:, :1], columns=[text])
    for col in df.columns[1:]:
        if col.endswith("_emb"): continue
        spans[col] = df.apply(lambda row: column_to_span(row, col, text=text), axis=1)
    return spans

def evaluate(spans:pd.DataFrame, true_column:str="annotation"):
    metrics = {}
    for c in spans.columns[1:]:
        metrics[c] = Evaluator(spans[true_column].values, spans[c].values, tags=['POS', 'NEU', 'NEG']).evaluate()

    return metrics

def metrics_to_df(metrics, eval_type:str='strict'):
    data = []
    for model in metrics.keys():
        data.append([])
        data[-1] = []
        data[-1].append(model)
        data[-1].append(metrics[model][0][eval_type]['precision'])
        data[-1].append(metrics[model][0][eval_type]['recall'])
        data[-1].append(metrics[model][0][eval_type]['f1'])

    return pd.DataFrame(data, columns=['model', 'precision', 'recall', 'f1'])

def plot_metrics_df(metrics, eval_type:str='strict'):
    xticks = list(metrics.keys())
    x = list(range(len(xticks)))
    y = [metrics[x][0][eval_type]['f1'] for x in xticks]

    fig, ax = plt.subplots(1, 1)
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation='vertical')
    plt.show()
