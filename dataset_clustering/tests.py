from dsc.cluster_2 import cluster_datasets, get_args

def test_cluster_datasets():
    args = get_args().parse_args(["--data_dirs",
                                  "/dls/labxchem/data/2018/lb18145-80/processing/analysis/initial_model",
                                  "--out_dir",
                                  "/dls/science/groups/i04-1/conor_dev/untitled/output_mureeca_mpi",
                                  "--num_datasets",
                                  "90",
                                  ]
                                 )
    print(args.data_dirs)

    cluster_datasets(args)