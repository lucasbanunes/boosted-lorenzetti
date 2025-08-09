from pathlib import Path
from datetime import datetime, timezone
import shutil
import logging
from typing import List, Dict, Any, Tuple, Set
import mlflow
from mlflow.models import infer_signature

from . import MLFlowLoggedJob
from .. import types


class TrainingJob(MLFlowLoggedJob):
    db_path: Path
    train_query: str
    dims: types.DimsType
    val_query: str | None = None
    test_query: str | None = None
    predict_query: str | None = None
    label_col: str | None = 'label'
    activation: types.ActivationType = 'relu'
    batch_size: types.BatchSizeType = 32
    run_id: types.MLFlowRunId = None
    name: str = 'MLP Training Job'
    accelerator: types.AcceleratorType = 'cpu'
    patience: types.PatienceType = 10
    checkpoints_dir: types.CheckpointsDirType = Path('checkpoints/')
    max_epochs: types.MaxEpochsType = 3

    def to_mlflow(self,
                  nested: bool = False,
                  tags: List[str] = [],
                  extra_tags: Dict[str, Any] = {}) -> str:
        extra_tags['model'] = 'MLP'
        return super().to_mlflow(nested=nested,
                                 tags=tags,
                                 extra_tags=extra_tags)

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None = None):

        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)
        datamodule = MLPDataset(
            db_path=self.db_path,
            train_query=self.train_query,
            val_query=self.val_query,
            test_query=self.test_query,
            predict_query=self.predict_query,
            label_cols=self.label_col,
            batch_size=self.batch_size)
        datamodule.log_to_mlflow()
        class_weights = datamodule.get_class_weights(
            how='balanced'
        )
        mlflow.log_param("class_weights", class_weights)
        model = MLP(dims=self.dims,
                    class_weights=class_weights,
                    activation=self.activation)
        sample_X, _ = datamodule.get_df_from_query(self.train_query, limit=10)
        with torch.no_grad():
            model.eval()
            output = model(sample_X.to_torch())
            signature = infer_signature(
                model_input=sample_X.to_pandas(
                    use_pyarrow_extension_array=True),
                model_output=output.numpy()  # Convert to numpy for signature
            )
        model.train()
        logging.info('Model initialized and example input processed.')
        logging.info('Data module created from datasets.')

        logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=self.name,
            tracking_uri=tracking_uri,
            run_id=self.run_id
        )

        checkpoint = ModelCheckpoint(
            monitor="val_max_sp",  # Monitor a validation metric
            dirpath=self.checkpoints_dir,  # Directory to save checkpoints
            filename='best-model-{epoch:02d}-{val_max_sp:.2f}',
            save_top_k=3,
            mode="min",  # Save based on minimum validation loss,
            save_on_train_epoch_end=False
        )
        callbacks = [
            EarlyStopping(
                monitor="val_max_sp",
                patience=self.patience,
                mode="min"
            ),
            checkpoint,
        ]

        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=1,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=False,
        )
        logging.info('Starting training process...')
        fit_start = datetime.now(timezone.utc)
        mlflow.log_metric('fit_start', fit_start.timestamp())
        trainer.fit(model, datamodule=datamodule)
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric('fit_end', fit_end.timestamp())
        mlflow.log_metric(
            "fit_duration", (fit_end - fit_start).total_seconds())
        logging.info('Training completed.')

        best_model = MLP.load_from_checkpoint(
            checkpoint.best_model_path
        )
        log_path = tmp_dir / 'model.ckpt'
        shutil.copy(checkpoint.best_model_path, log_path)
        mlflow.log_artifact(log_path)
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            name="ml_model",
            signature=signature,
        )
        onnx_path = tmp_dir / 'model.onnx'
        best_model.to_onnx(onnx_path, export_params=True)
        mlflow.log_artifact(str(onnx_path))
        logging.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(self.checkpoints_dir))

        logging.info('Evaluating best model on train dataset.')
        train_X, train_y = datamodule.train_df()
        model.evaluate_on_data(
            X=train_X,
            y=train_y,
            prefix='train',
            mlflow_log=True
        )
        del train_X, train_y  # Free memory

        if self.val_query:
            logging.info('Evaluating best model on validation dataset.')
            val_X, val_y = datamodule.val_df()
            model.evaluate_on_data(
                X=val_X,
                y=val_y,
                prefix='val',
                mlflow_log=True
            )
            del val_X, val_y  # Free memory

        if self.test_query:
            logging.info('Evaluating best model on test dataset.')
            test_X, test_y = datamodule.test_df()
            model.evaluate_on_data(
                X=test_X,
                y=test_y,
                prefix='test',
                mlflow_log=True
            )
            del test_X, test_y  # Free memory

        if self.predict_query:
            logging.info('Evaluating best model on prediction dataset.')
            predict_X, predict_y = datamodule.predict_df()
            model.evaluate_on_data(
                X=predict_X,
                y=predict_y,
                prefix='predict',
                mlflow_log=True
            )
            del predict_X, predict_y  # Free memory

        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)

    def as_metric_dict(self) -> Dict[str, Any]:
        """
        Converts the metrics of the training job to a dictionary format.
        """
        metric_dict = {
            'run_id': self.run_id,
            **self.metrics
        }
        metric_dict.update(self.tags)
        return metric_dict

    @staticmethod
    def get_metrics_df(
        jobs: List['TrainingJob']
    ) -> Tuple[pd.DataFrame, Set[str]]:
        """
        Collects metrics from a list of TrainingJob instances and returns a DataFrame.
        """
        metric_names = set()
        data = list()
        for job in jobs:
            for metric_name in job.metrics.keys():
                metric_names.add(metric_name)
            job_data = job.as_metric_dict()
            data.append(job_data)
        return pd.DataFrame.from_records(data), metric_names
