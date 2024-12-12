import argparse
from pathlib import Path
from typing import Iterable, Tuple
from lzt_utils.dataset import LztDataset
from lzt_utils.norms import norm1
from lzt_utils.metrics import sp_index
from lzt_utils.root import rdf_to_pandas, open_vector
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from scipy import integrate
import json

RANDOM_SEED = 42
N_FOLDS = 5
N_RINGS = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sig-dir', type=str,
                        required=True, dest='sig_dir')
    parser.add_argument('--bkg-dir', type=str,
                        required=True, dest='bkg_dir')
    parser.add_argument('--output-dir', type=str,
                        required=True, dest='output_dir')
    parser.add_argument('--dense-layers', type=int, nargs='+',
                        required=True, dest='dense_layers')
    parser.add_argument('--n-folds', type=int,
                        default=N_FOLDS, dest='n_folds')
    parser.add_argument('--random-seed', type=int,
                        default=RANDOM_SEED, dest='random_seed')
    parser.add_argument('--data-percentage', type=float,
                        default=1.0, dest='data_percentage')
    args = parser.parse_args()
    args.sig_dir = Path(args.sig_dir).resolve()
    args.bkg_dir = Path(args.bkg_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    return args.__dict__


def get_model(input_shape: Tuple[int, int, int],
              dense_layers: Iterable[int]):

    model = tf.keras.Sequential()
    for i, n_neurons in enumerate(dense_layers):
        if i:
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
        else:
            # First Layer
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu',
                                            input_shape=input_shape))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(
                          name='accuracy'
                      ),
                  ])
    return model


def get_data(sig_dataset: LztDataset,
             bkg_dataset: LztDataset,
             n_folds: int,
             random_seed: int,
             data_percentage: float):
    datasets = [bkg_dataset, sig_dataset]
    pdfs = []
    for is_signal, dataset in enumerate(datasets):
        ntuple_rdf = dataset.get_ntuple_rdf()
        rings_cols, ntuple_rdf = open_vector('rings', N_RINGS, ntuple_rdf)
        ntuple_pdf = rdf_to_pandas(ntuple_rdf, rings_cols, nrows=-1)
        split_limit = int(np.floor(len(ntuple_pdf)*data_percentage))
        ntuple_pdf = ntuple_pdf.iloc[:split_limit]
        ntuple_pdf['is_signal'] = np.full(len(ntuple_pdf), is_signal)
        pdfs.append(ntuple_pdf)

    dataset = pd.concat(pdfs)
    X = norm1(dataset[rings_cols].values)
    y = dataset['is_signal'].values

    unique_classes = np.array([0, 1])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_seed)
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        x_train, y_train = X[train_idx], y[train_idx]
        class_weights = compute_class_weight('balanced',
                                             classes=unique_classes,
                                             y=y_train)
        train_weights = dict(zip(unique_classes, class_weights))

        x_val, y_val = X[val_idx], y[val_idx]
        class_weights = compute_class_weight('balanced',
                                             classes=unique_classes,
                                             y=y_val)
        val_weights = dict(zip(unique_classes, class_weights))
        yield i, x_train, y_train, train_weights, x_val, y_val, val_weights


def evaluate_model(model, history,
                   x_train, y_train, train_weights,
                   x_val, y_val, val_weights,
                   output_dir: Path, iteration: int):

    history_df = pd.DataFrame.from_dict(history.history)
    history_df.to_csv(output_dir / f'history_{iteration:08d}.csv')
    metrics_list = []

    metrics_train = evaluate_data(model, x_train, y_train, train_weights)
    metrics_train = pd.DataFrame.from_dict(metrics_train)
    metrics_train['dataset'] = 'train'
    metrics_list.append(metrics_train)
    metrics_val = evaluate_data(model, x_val, y_val, val_weights)
    metrics_val = pd.DataFrame.from_dict(metrics_val)
    metrics_val['dataset'] = 'val'
    metrics_list.append(metrics_val)

    metrics = pd.concat(metrics_list, axis=0)
    metrics.to_csv(output_dir / f'metrics_{iteration:08d}.csv')


def evaluate_data(model, x, y, weights):
    y_pred = model.predict(x)
    y = y.reshape(-1,1)
    tpr, fpr, thresholds = roc_curve(y, y_pred)
    acc = np.sum(y == (y_pred > thresholds), axis=0) / len(y)
    metrics = {
        'tpr': tpr.tolist(),
        'fpr': fpr.tolist(),
        'thresholds': thresholds.tolist(),
        'sp': sp_index(tpr, fpr).tolist(),
        'auc': integrate.trapezoid(y=fpr, x=tpr),
        'accuracy': acc
    }
    return metrics


def main(sig_dir: Path,
         bkg_dir: Path,
         output_dir: Path,
         dense_layers: Iterable[int],
         n_folds: int,
         random_seed: int,
         data_percentage: float):
    sig_ds = LztDataset.from_dir(sig_dir)
    bkg_ds = LztDataset.from_dir(bkg_dir)
    iterator = get_data(sig_ds, bkg_ds, n_folds, random_seed, data_percentage)
    output_dir.mkdir(exist_ok=True, parents=True)
    input_args = {
        'sig_dir': str(sig_dir),
        'bkg_dir': str(bkg_dir),
        'output_dir': str(output_dir),
        'dense_layers': list(dense_layers)
    }
    input_args_path = output_dir / 'input_args.json'
    model_arch_json = output_dir / 'model_arch.json'
    with input_args_path.open('w') as f:
        json.dump(input_args, f)
    for (i,
         x_train, y_train, train_weights,
         x_val, y_val, val_weights) in iterator:

        final_model_path = output_dir / f'final_model_{i:08d}.keras'
        if final_model_path.exists():
            continue

        model = get_model(
            x_train.shape[1:],
            dense_layers)

        if i == 0:
            with model_arch_json.open('w') as f:
                f.write(model.to_json())
        model.save(output_dir / f'inital_model_{i:08d}.keras')
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5)
        ]
        history = model.fit(x_train,
                            y_train,
                            callbacks=callbacks,
                            batch_size=32,
                            class_weight=train_weights,
                            epochs=int(1e6),
                            verbose=2,    # Line only
                            validation_data=(x_val, y_val))
        evaluate_model(model, history,
                       x_train, y_train, train_weights,
                       x_val, y_val, val_weights,
                       output_dir=output_dir, iteration=i)
        model.save(final_model_path)


if __name__ == '__main__':
    args = parse_args()

    main(**args)
