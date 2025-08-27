import cyclopts

from boosted_lorenzetti.dataset.duckdb import app as duckdb_app
from boosted_lorenzetti.models.kmeans import app as kmeans_app
from boosted_lorenzetti.models.mlp import app as mlp_app
from boosted_lorenzetti.dataset.ntuple import app as ntuple_app
# from boosted_lorenzetti.models.tabcaps import app as tabcaps_app
from boosted_lorenzetti.dataset.npz import app as npz_app

from boosted_lorenzetti.utils import set_logger

set_logger()


app = cyclopts.App(
    help='CLI utilities for boosted-lorenzetti'
)
app.command(duckdb_app)
app.command(kmeans_app)
app.command(mlp_app)
app.command(npz_app)
app.command(ntuple_app)
# app.add_typer(tabcaps_app)


if __name__ == '__main__':
    app()
