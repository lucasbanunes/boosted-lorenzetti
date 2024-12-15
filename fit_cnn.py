import argparse
from pathlib import Path
from typing import Iterable, Tuple, List

from lzt_utils.dataset import LztDataset
from lzt_utils.norms import norm1
from lzt_utils.metrics import sp_index, roc_curve
from lzt_utils.root import rdf_to_pandas, open_vector
from lzt_utils.constants import RINGS_LAYERS
import tensorflow as tf
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
    parser.add_argument('--epochs', type=int,
                        default=1000000, dest='epochs')
    parser.add_argument('--incep-modules', type=int,
                        required=True, dest='incep_modules')
    parser.add_argument('--n-filters', type=int,
                        required=True, dest='n_filters')
    parser.add_argument('--kernel-sizes', type=int, nargs='+',
                        required=True, dest='kernel_sizes')
    args = parser.parse_args()
    args.sig_dir = Path(args.sig_dir).resolve()
    args.bkg_dir = Path(args.bkg_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    return args.__dict__


def get_inception_module(
        kernel_sizes: List[Tuple[int]],
        n_filters: List[int],
        input_shape: Tuple[int],
        module_name: str = 'inception_module',
        summary: bool = False) -> tf.keras.Model:
    """
    This function creates an inception module with the given kernel sizes
    and number of filters.
    The inception module is an architecture defined at:
    [1] C. Szegedy et al., “Going Deeper with Convolutions

    Parameters
    ----------
    kernel_sizes : List[Tuple[int]]
        List of kernel sizes to use in the inception module
    n_filters : List[int]
        List of number of filters to use in the inception module
        The number of filters is equal to every kernel size
    input_shape : Tuple[int]
        Input shape of the inception module
    module_name : str
        Name of the inception module, by default 'inception_module'
    summary : bool, optional
        If True, prints the summary of the inception module, by default False

    Returns
    -------
    tf.keras.Model
        The created Inception module
    """
    kernel_sizes = np.array(kernel_sizes, dtype=np.uint64)
    n_filters = np.array(n_filters, dtype=np.uint64)
    incep_input = tf.keras.layers.Input(
        shape=input_shape,
        name='inception_module_input')
    convs = list()
    for ks, n in zip(kernel_sizes, n_filters):
        ks_name = ''.join([str(i) for i in ks])
        convs.append(
            tf.keras.layers.Conv1D(
                n,
                kernel_size=ks,
                activation='relu',
                padding='same',
                name=f'kernel_{ks_name}_incep'
            )(incep_input)
        )
    concat_layer = tf.keras.layers.Concatenate()(convs)
    incep = tf.keras.Model(incep_input, concat_layer, name=module_name)
    if summary:
        incep.summary()
    return incep


def get_model(
        incep_modules: int,
        n_filters: int,
        dense_neurons: List[int],
        kernel_sizes: List[Tuple[int]],
        model_name: str = 'inception_per_layer',
        summary: bool = False) -> tf.keras.Model:
    """
    This function creates a model with inception modules per layer.
    The model is composed of a set of inception modules, one for each layer
    of the calorimeter rings. The inception modules outputs are concatenated
    to a dense layer that outputs the final prediction.

    Parameters
    ----------
    incep_modules : int
        Number of inception modules to use in each layer
    n_filters : int
        Number of filters to use in the inception modules
    dense_neurons : List[int]
        Number of neurons to use in the dense layers.
        Each element of the list is a layer of the dense network
    kernel_sizes : List[Tuple[int]]
        List of kernel sizes to use in the inception modules
    model_name : str, optional
        Name of the tf.keras.Model, by default 'inception_per_layer'
    summary : bool, optional
        If True, calls tf.keras.Model.summary() at the end of the execution,
        by default False. The summary of the internal call for creation of
        inception modules is not printed.

    Returns
    -------
    tf.keras.Model
        The created model
    """
    branches_inputs = list()
    branches_outputs = list()
    for layer, idxs in RINGS_LAYERS.items():
        n_rings = len(idxs)
        reshape_shape = (n_rings, 1)
        branch_input = tf.keras.layers.Input(
            shape=(n_rings,),
            name=f'{layer}_input')
        branch_reshape = tf.keras.layers.Reshape(
            reshape_shape,
            name=f'{layer}_reshape')(branch_input)

        # Initializing the variables for the loop
        incep_module = branch_reshape
        input_shape = reshape_shape
        for nfactor in range(1, incep_modules+1):
            filters_arr = np.full(len(kernel_sizes), nfactor*n_filters)
            incep_module = get_inception_module(
                [(ks,) for ks in kernel_sizes],
                filters_arr,
                input_shape,
                module_name=f"{layer}_incep_{nfactor}",
                summary=False
            )(incep_module)
            input_shape = incep_module.shape[1:]
        flatten = tf.keras.layers.Flatten(
            name=f'{layer}_flatten')(incep_module)
        branches_inputs.append(branch_input)
        branches_outputs.append(flatten)

    concat = tf.keras.layers.Concatenate()(branches_outputs)
    dense = concat
    for i, n_neurons in enumerate(dense_neurons):
        dense = tf.keras.layers.Dense(
            n_neurons,
            activation='relu',
            name=f'dense_layer_{i}')(dense)
    dense = tf.keras.layers.Dense(
        1, activation='linear',
        name='output_for_inference')(dense)
    model_output = tf.keras.layers.Activation(
        'sigmoid',
        name='output_for_training')(dense)
    model = tf.keras.Model(branches_inputs, model_output, name=model_name)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(
                          name='accuracy'
                      ),
                  ])
    if summary:
        model.summary()
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

        x_train_multiple = list()
        x_val_multiple = list()
        for idxs in RINGS_LAYERS.values():
            x_train_multiple.append(x_train[:, idxs])
            x_val_multiple.append(x_val[:, idxs])
        yield (i,
               x_train_multiple,
               y_train,
               train_weights,
               x_val_multiple,
               y_val,
               val_weights)


def evaluate_model(model, history,
                   x_train, y_train, train_weights,
                   x_val, y_val, val_weights):

    history_df = pd.DataFrame.from_dict(history.history)
    metrics_list = []

    metrics_train, y_pred_train = evaluate_data(
        model, x_train, y_train, train_weights)
    metrics_train = pd.DataFrame.from_dict(metrics_train)
    metrics_train['dataset'] = 'train'
    metrics_list.append(metrics_train)
    metrics_val, y_pred_val = evaluate_data(model, x_val, y_val, val_weights)
    metrics_val = pd.DataFrame.from_dict(metrics_val)
    metrics_val['dataset'] = 'val'
    metrics_list.append(metrics_val)

    metrics_df = pd.concat(metrics_list, axis=0)
    predictions_df = pd.DataFrame({
        'y_true': np.concatenate([y_train, y_val], axis=0),
        'y_pred': np.concatenate([y_pred_train.flatten(),
                                  y_pred_val.flatten()], axis=0),
        'dataset': np.concatenate([np.full(len(y_train), 'train'),
                                   np.full(len(y_val), 'val')])
    })

    return history_df, metrics_df, predictions_df


def evaluate_data(model, x, y, weights):
    y_pred = model.predict(x)
    y = y.reshape(-1, 1)
    thresholds = np.linspace(0, 1, 300).reshape(1, -1)
    tpr, fpr, thresholds = roc_curve(y, y_pred, thresholds=thresholds)
    acc = np.sum(y == (y_pred > thresholds), axis=0) / len(y)
    metrics = {
        'tpr': tpr,
        'fpr': fpr,
        'thresholds': thresholds,
        'sp': sp_index(tpr, fpr),
        'accuracy': acc
    }
    return metrics, y_pred


def main(sig_dir: Path,
         bkg_dir: Path,
         output_dir: Path,
         dense_layers: Iterable[int],
         n_folds: int,
         random_seed: int,
         data_percentage: float,
         epochs: int,
         incep_modules: int,
         n_filters: int,
         kernel_sizes: List[Tuple[int]]):
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
    history_per_fold = []
    metrics_per_fold = []
    for (ifold,
         x_train, y_train, train_weights,
         x_val, y_val, val_weights) in iterator:

        fold_dir = output_dir / f'fold_{ifold:02d}'
        fold_dir.mkdir(exist_ok=True)

        final_model_path = fold_dir / 'final_model.keras'

        model = get_model(
            incep_modules,
            n_filters,
            dense_layers,
            kernel_sizes,
            summary=True)

        if ifold == 0:
            with model_arch_json.open('w') as f:
                f.write(model.to_json())
        model.save(fold_dir / 'inital_model.keras')
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=5,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        ]
        history = model.fit(x_train,
                            y_train,
                            callbacks=callbacks,
                            batch_size=32,
                            class_weight=train_weights,
                            epochs=epochs,
                            verbose=2,    # Line only
                            validation_data=(x_val, y_val))
        history_df, metrics_df, predictions_df = evaluate_model(
            model, history,
            x_train, y_train, train_weights,
            x_val, y_val, val_weights)
        history_df['fold'] = ifold
        metrics_df['fold'] = ifold
        predictions_df['fold'] = ifold
        history_df.to_csv(fold_dir / 'history.csv')
        metrics_df.to_csv(fold_dir / 'metrics.csv')
        predictions_df.to_csv(fold_dir / 'predictions.csv')
        model.save(final_model_path)
        history_per_fold.append(history_df)
        metrics_per_fold.append(metrics_df)

    metrics_per_fold = pd.concat(metrics_per_fold, axis=0)
    metrics_per_fold.to_csv(output_dir / 'metrics_all_folds.csv')
    history_per_fold = pd.concat(history_per_fold, axis=0)
    history_per_fold.to_csv(output_dir / 'history_all_folds.csv')
    auc_per_fold = metrics_per_fold.groupby(['fold', 'dataset']).apply(
        lambda x: integrate.trapezoid(x['tpr'], x['fpr']),
        include_groups=False)
    auc_per_fold = auc_per_fold.reset_index().rename(columns={0: 'auc'})
    auc_per_fold.to_csv(output_dir / 'auc_all_folds.csv')


if __name__ == '__main__':
    args = parse_args()

    main(**args)
