from typing import Annotated
import typer
from pathlib import Path
from .jobs import ElectronCutBasedJob

app = typer.Typer(
    name="cutbased",
    help="Cutbased selection utilities",
)

# @app.command("list-selectors")
# def list_maps():
#     """List available cutbased maps"""

#     cutmapdir = Path(__file__).parent / 'cutmaps'

#     from boosted_lorenzetti.cutbased.selectors import get_available_selectors

#     selectors = get_available_selectors()
#     for pid, tags in selectors.items():
#         typer.echo(f"{pid}: {', '.join(tags)}")


@app.command()
def predict(
    config_path: Annotated[
        Path,
        typer.Option(
            help="YAML file describing the cutbased job to be run."
        )
    ]
) -> ElectronCutBasedJob:
    """Run a cutbased job based on a YAML configuration file."""

    job = ElectronCutBasedJob.from_yaml(config_path)
    job.run()
    return job
