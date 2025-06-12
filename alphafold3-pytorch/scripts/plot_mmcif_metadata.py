import os

import plotly.express as px
import polars as pl


def plot_mmcif_metadata(mmcif_metadata_filepath: str, column: str):
    """Plots a distribution of the input mmCIF metadata."""
    df = pl.read_csv(mmcif_metadata_filepath)
    fig = px.histogram(df.to_pandas(), x=column, nbins=250)
    fig.show()


if __name__ == "__main__":
    plot_mmcif_metadata(
        os.path.join("caches", "pdb_data", "metadata", "mmcif.csv"),
        # os.path.join("caches", "afdb_data", "metadata", "mmcif.csv"),
        column="num_tokens",  # NOTE: must be one of the columns in the metadata file
    )
