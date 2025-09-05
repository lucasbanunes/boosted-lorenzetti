
from pathlib import Path
import mlflow
import torch
from torchmetrics.classification import (
    BinaryConfusionMatrix
)
import pandas as pd
from tempfile import TemporaryDirectory
import numpy as np
import plotly.express as px

from .metrics import sp_index


def evaluate_on_data(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    thresholds: torch.Tensor = None,
    prefix: str = '',
    mlflow_log: bool = True
):
    """
    Evaluates the model on the provided data.
    """

    if not prefix.endswith('_') and prefix != '':
        prefix += '_'

    if thresholds is None:
        thresholds = torch.linspace(0, 1, 300)

    tp = torch.empty((len(thresholds),), dtype=torch.int32)
    fp = torch.empty((len(thresholds),), dtype=torch.int32)
    tn = torch.empty((len(thresholds),), dtype=torch.int32)
    fn = torch.empty((len(thresholds),), dtype=torch.int32)

    for i, thresh in enumerate(thresholds):
        cm = BinaryConfusionMatrix(threshold=float(thresh))(y_pred, y_true)
        tp[i] = cm[1, 1]
        fp[i] = cm[0, 1]
        tn[i] = cm[0, 0]
        fn[i] = cm[1, 0]

    eval_df = pd.DataFrame({
        'thresholds': thresholds.numpy(),
        'tp': tp.numpy(),
        'fp': fp.numpy(),
        'tn': tn.numpy(),
        'fn': fn.numpy()
    })
    eval_df['tpr'] = eval_df['tp'] / (eval_df['tp'] + eval_df['fn'])
    eval_df['fpr'] = eval_df['fp'] / (eval_df['fp'] + eval_df['tn'])
    eval_df['acc'] = (eval_df['tp'] + eval_df['tn']) / len(y_pred)
    eval_df['sp'] = sp_index(
        eval_df['tpr'].values,
        eval_df['fpr'].values,
        backend='numpy')
    sp_max_idx = eval_df['sp'].argmax()

    metrics = {
        col_name: eval_df[col_name].tolist()
        for col_name in eval_df.columns
    }

    metrics['max_sp'] = eval_df['sp'].iloc[sp_max_idx]
    metrics['max_sp_fpr'] = eval_df['fpr'].iloc[sp_max_idx]
    metrics['max_sp_tpr'] = eval_df['tpr'].iloc[sp_max_idx]
    metrics['max_sp_acc'] = eval_df['acc'].iloc[sp_max_idx]
    metrics['max_sp_threshold'] = eval_df['thresholds'].iloc[sp_max_idx]
    roc_auc = np.trapezoid(eval_df['tpr'].values, eval_df['fpr'].values)
    metrics['roc_auc'] = roc_auc

    if mlflow_log:
        with TemporaryDirectory() as tmp_dir:
            df_path = Path(tmp_dir) / f'{prefix}eval_df.csv'
            eval_df.to_csv(df_path, index=False)
            mlflow.log_artifact(str(df_path))
        mlflow.log_metric(f'{prefix}max_sp',
                          eval_df['sp'].iloc[sp_max_idx])
        mlflow.log_metric(f'{prefix}max_sp_threshold',
                          eval_df['thresholds'].iloc[sp_max_idx])
        mlflow.log_metric(f'{prefix}max_sp_fpr',
                          eval_df['fpr'].iloc[sp_max_idx])
        mlflow.log_metric(f'{prefix}max_sp_tpr',
                          eval_df['tpr'].iloc[sp_max_idx])
        mlflow.log_metric(f'{prefix}max_sp_acc',
                          eval_df['acc'].iloc[sp_max_idx])
        mlflow.log_metric(f'{prefix}roc_auc', roc_auc)
        roc_curve_artifact = f'{prefix}roc_curve.html'
        fig = px.line(
            eval_df.sort_values('fpr'),
            x='fpr',
            y='tpr',
        )
        fig.update_layout(
            title=f'ROC Curve (AUC {metrics["roc_auc"]:.2f})',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        mlflow.log_figure(fig, roc_curve_artifact)

        tpr_fpr_artifact = f'{prefix}tpr_fpr.html'
        fig = px.line(
            eval_df.sort_values('thresholds'),
            x='thresholds',
            y=['tpr', 'fpr'],
        )
        fig.update_layout(
            title='TPR and FPR vs Thresholds',
            xaxis_title='Thresholds',
            yaxis_title='Rate',
            legend_title='Rate Type'
        )
        mlflow.log_figure(fig, tpr_fpr_artifact)

    return metrics
