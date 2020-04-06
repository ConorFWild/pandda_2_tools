import argparse
import pathlib
import pandas as pd
from bokeh.plotting import figure, output_file, show, ColumnDataSource, save
import bokeh.models as bmo
from bokeh.palettes import d3
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap


def graph(df,
          output_path,
          x="x",
          y="y",
          id="dtag",
          col="cluster",
          ):

    df[col] = df[col].apply(str)

    # Get cds
    cds = ColumnDataSource(df)

    # # use whatever palette you want...
    palette = d3['Category20'][(len(df[col].unique()) % 19) + 2]
    color_map = bmo.CategoricalColorMapper(factors=df[col].unique(),
                                           palette=palette)

    # Define tooltipts
    TOOLTIPS = [("".format(id), "@{}".format(id)),
                ("({},{})".format(x, y), "(${}, ${})".format(x, y)),
                ("{}".format(col), "@{}".format(col))
                ]

    # Gen figure
    p = figure(plot_width=800,
               plot_height=800,
               tooltips=TOOLTIPS,
               title="Mouse over the dots",
               )

    # Plot data
    p.circle('x',
             'y',
             color={'field': '{}'.format(col),
                              'transform': color_map,
                              },
             size=10,
             source=cds,
             )

    # Save figure
    save(p,
         output_path,
         )
