from typer import Typer

from boosted_lorenzetti.models.mlp import app as mlp_app
from boosted_lorenzetti.dataset.ntuple import app as ntuple_app
from boosted_lorenzetti.models.tabcaps import app as tabcaps_app
from boosted_lorenzetti.dataset.npz import app as npz_app

app = Typer(
    help='CLI utilities for boosted-lorenzetti'
)
app.add_typer(mlp_app)
app.add_typer(npz_app)
app.add_typer(ntuple_app)
app.add_typer(tabcaps_app)


if __name__ == '__main__':
    app()
