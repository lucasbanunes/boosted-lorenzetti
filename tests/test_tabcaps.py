# from boosted_lorenzetti.models import tabcaps
# from pathlib import Path

# from boosted_lorenzetti.constants import N_RINGS


# def test_full_training(test_dataset_path: Path):
#     experiment_name = 'test_full_training'

#     ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]
#     query_cols = ring_cols + ['label']
#     query_cols_str = ', '.join(query_cols)
#     train_query = f"SELECT {query_cols_str} FROM data WHERE fold != 0;"
#     val_query = f"SELECT {query_cols_str} FROM data WHERE fold = 0;"

#     run_id = tabcaps.create_training(
#         init_dim=1,
#         primary_capsule_dim=1,
#         digit_capsule_dim=1,
#         n_leaves=1,
#         db_path=test_dataset_path,
#         train_query=train_query,
#         val_query=val_query,
#         label_col='label',
#         experiment_name=experiment_name
#     )

#     tabcaps.run_training(
#         run_ids=run_id,
#         experiment_name=experiment_name,
#     )


# def test_multiple_trainings(test_dataset_path: Path):
#     experiment_name = 'test_multiple_trainings'

#     ring_cols = [f'cl_rings[{i+1}]' for i in range(N_RINGS)]
#     query_cols = ring_cols + ['label']
#     query_cols_str = ', '.join(query_cols)
#     train_query = f"SELECT {query_cols_str} FROM data WHERE fold != 0;"
#     val_query = f"SELECT {query_cols_str} FROM data WHERE fold = 0;"

#     run_ids = []
#     run_ids.append(
#         tabcaps.create_training(
#             db_path=test_dataset_path,
#             train_query=train_query,
#             val_query=val_query,
#             label_col='label',
#             dims=[N_RINGS, 1],
#             experiment_name=experiment_name
#         )
#     )
#     run_ids.append(
#         tabcaps.create_training(
#             db_path=test_dataset_path,
#             train_query=train_query,
#             val_query=val_query,
#             label_col='label',
#             dims=[N_RINGS, 1],
#             experiment_name=experiment_name
#         )
#     )

#     tabcaps.run_training(
#         run_ids=run_ids,
#         experiment_name=experiment_name,
#     )


# def test_kfold_training(test_dataset_path: Path):
#     experiment_name = 'test_kfold_training'

#     run_id = tabcaps.create_kfold(
#         db_path=test_dataset_path,
#         table_name='data',
#         dims=[N_RINGS, 1],
#         best_metric='val_max_sp',
#         best_metric_mode='max',
#         rings_col='cl_rings',
#         label_col='label',
#         fold_col='fold',
#         folds=5,
#         inits=1,
#         experiment_name=experiment_name,
#         max_epochs=2,
#     )

#     tabcaps.run_kfold(
#         run_id=run_id,
#         experiment_name=experiment_name,
#     )
