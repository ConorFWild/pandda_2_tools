import argparse
import pathlib
import pandas as pd
from bokeh.plotting import figure, output_file, show, ColumnDataSource, save
import bokeh.models as bmo
from bokeh.palettes import d3
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap


def get_args():
    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument("-i", "--input_csv",
                        type=str,
                        default="labelled_embedding.csv",
                        help="Path to the config file with uncommon options")
    parser.add_argument("-o", "--output_dir",
                        type=str,
                        help="The directory dir such that dir//<dataset_names>//<pdbs and mtzs>",
                        required=True)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # parse args
    args = get_args()

    # Get csv
    df = pd.read_csv(args.input_csv,
                     index_col=0,
                     )
    df["cluster"] = df["cluster"].apply(str)

    # Get cds
    cds = ColumnDataSource(df)

    # # use whatever palette you want...
    palette = d3['Category20'][(len(df['cluster'].unique()) % 19) + 2]
    color_map = bmo.CategoricalColorMapper(factors=df['cluster'].unique(),
                                           palette=palette)

    # Define tooltipts
    TOOLTIPS = [
        ("dtag", "@dtag"),
        ("(x,y)", "($x, $y)"),
        ("cluster", "@cluster")
    ]

    # Gen figure
    p = figure(plot_width=800,
               plot_height=800,
               tooltips=TOOLTIPS,
               title="Mouse over the dots")

    # Plot data
    p.circle('x', 'y', color={'field': 'cluster', 'transform': color_map}, size=10, source=cds)

    # Save figure
    save(p,
         (pathlib.Path(args.output_dir) / "clustering.html").as_posix()
         )

