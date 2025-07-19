from typer import Typer

from boosted_lorenzetti.models.mlp import app as mlp_app
from boosted_lorenzetti.dataset.convert import app as convert_app
from boosted_lorenzetti.dataset.ingest import app as ingest_app

app = Typer(
    help='CLI utilities for boosted-lorenzetti'
)
app.add_typer(convert_app)
app.add_typer(ingest_app)
app.add_typer(mlp_app)


if __name__ == '__main__':
    app()
