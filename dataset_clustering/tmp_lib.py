
def get_config(args):
    # Config
    config = configparser.ConfigParser()
    config.read(vars(args)["config_path"])

    return config


def get_settings(args, config):
    settings = {}
    settings.update(config)
    settings.update(vars(args))
    settings["args"].update(vars(args))

    return settings


def get_sample_loader(**settings):
    config = settings

    if config["dataset"]["sample_loader"] == "default":
        grid_loader = PanDDAGridSetup(cpus=int(config["args"]["cpus"]),
                                      mask_pdb=config["grid"]["mask_pdb"],
                                      align_mask_to_reference=bool(config["grid"]["align_mask_to_reference"]),
                                      alignment_method=config["grid"]["alignment_method"],
                                      outer_mask=float(config["grid"]["outer_mask"]),
                                      inner_mask=float(config["grid"]["inner_mask"]),
                                      inner_mask_symmetry=float(config["grid"]["inner_mask_symmetry"]),
                                      grid_spacing=float(config["grid"]["grid_spacing"]),
                                      padding=float(config["grid"]["padding"]),
                                      verbose=bool(config["args"]["verbose"]),
                                      mask_selection_string=str(config["grid"]["mask_selection_string"])
                                      )

        sample_loader = DefaultSampleLoader(config["maps"]["resolution_factor"],
                                            config["maps"]["density_scaling"],
                                            int(config["args"]["cpus"]),
                                            bool(config["args"]["verbose"]),
                                            grid_loader,
                                            )

    return sample_loader


def get_dataloader(**settings):
    config = settings

    # # Dataloader
    if config["dataset"]["dataloader"] == "default":
        dataloader = DefaultDataloader(config["args"]["data_dirs"],
                                       config["dataloader"]["pdb_style"],
                                       config["dataloader"]["mtz_style"],
                                       config["dataloader"]["pdb_regex"],
                                       config["dataloader"]["mtz_regex"],
                                       config["dataloader"]["dir_regex"],
                                       config["dataloader"]["only_datasets"],
                                       config["dataloader"]["ignore_datasets"],
                                       config["dataloader"]["dataset_prefix"],
                                       config["args"]["out_dir"],
                                       config["dataloader"]["lig_style"],
                                       )

    return dataloader


def get_grid_loader(**settings):
    config = settings

    grid_loader = PanDDAGridSetup(cpus=int(config["args"]["cpus"]),
                                  mask_pdb=config["grid"]["mask_pdb"],
                                  align_mask_to_reference=bool(config["grid"]["align_mask_to_reference"]),
                                  alignment_method=config["grid"]["alignment_method"],
                                  outer_mask=float(config["grid"]["outer_mask"]),
                                  inner_mask=float(config["grid"]["inner_mask"]),
                                  inner_mask_symmetry=float(config["grid"]["inner_mask_symmetry"]),
                                  grid_spacing=float(config["grid"]["grid_spacing"]),
                                  padding=float(config["grid"]["padding"]),
                                  verbose=bool(config["args"]["verbose"]),
                                  mask_selection_string=str(config["grid"]["mask_selection_string"])
                                  )

    return grid_loader


def get_reference_getter(**settings):
    config = settings

    # # Reference
    if config["dataset"]["get_reference"] == "default":
        get_reference = DefaultReferenceGetter(out_dir=config["args"]["out_dir"],
                                               reference_pdb_path=None,
                                               reference_mtz_path=None,
                                               reference_structure_factors=None,
                                               structure_factors=[config["diffraction_data"][
                                                   "structure_factors"]],
                                               )

    return get_reference


def get_map_loader(**settings):
    config = settings

    # Pull vars
    verbose = bool(config["args"]["verbose"])
    resolution_factor = float(config["maps"]["resolution_factor"])
    density_scaling = str(config["maps"]["density_scaling"])
    map_loader = MapLoaderDask(verbose,
                               resolution_factor,
                               density_scaling)

    return map_loader


def get_reference_map_getter(**settings):
    config = settings
    resolution_factor = config["maps"]["resolution_factor"]
    density_scaling = config["maps"]["density_scaling"]

    reference_map_getter = PanddaReferenceMapLoader(resolution_factor,
                                                    density_scaling)

    return reference_map_getter


def get_transforms(**kargs):
    config = kargs

    cpus = int(config["args"]["cpus"])

    # # Transforms
    dataset_transforms = OrderedDict()
    if config["transforms"]["PanddaDataChecker"] == "True":
        dataset_transforms["data_check"] = PanddaDataChecker([config["diffraction_data"]["structure_factors"]],
                                                             config["diffraction_data"][
                                                                 "low_resolution_completeness"],
                                                             config["diffraction_data"][
                                                                 "all_data_are_valid_values"]
                                                             )

    if config["transforms"]["PanddaDiffractionScaler"] == "True":
        dataset_transforms["scale_diffraction"] = PanddaDiffractionScaler(
            config["diffraction_data"]["apply_b_factor_scaling"]
        )

    if config["transforms"]["PanddaDatasetFilterer"] == "True":
        dataset_transforms["filter_structure"] = PanddaDatasetFilterer(
            config["dataset_filtering"]["similar_models_only"],
            config["dataset_filtering"]["max_rfree"],
            config["dataset_filtering"]["same_space_group_only"]
        )

    if config["transforms"]["PanddaDatasetFiltererWilsonRMSD"] == "True":
        dataset_transforms["filter_wilson"] = PanddaDatasetFiltererWilsonRMSD(
            config["diffraction_data"]["max_wilson_plot_z_score"],
            config["diffraction_data"]["apply_b_factor_scaling"],
            [config["diffraction_data"]["structure_factors"]]
        )

    if config["transforms"]["PanddaDefaultStructureAligner"] == "True":
        method = str(config["transforms"]["align_method"])
        dataset_transforms["align"] = PanddaDefaultStructureAligner(
            method=method,
            cpus=cpus
        )

    return dataset_transforms


def get_dataset(path, getter):
    dataset = getter(path)
    return dataset


def load_map(map_loader, truncated_datasets, ref_map, grid, max_res):
    xmap = map_loader(truncated_datasets, ref_map, grid, max_res)
    return xmap


def process_dask(map_loader, truncated_datasets, ref_map, grid, max_res):
    print("opening cluster")
    cluster = SGECluster(queue="medium.q",
                         project="labxchem",
                         dashboard_address=':9797',
                         cores=10,
                         processes=5,
                         memory="64GB",
                         env_extra=["export BOOST_ADAPTBX_FPE_DEFAULT=1",
                                    "export BOOST_ADAPTBX_SIGNALS_DEFAULT=1"],
                         resource_spec="m_mem_free=64G,redhat_release=rhel7",
                         python="/dls/science/groups/i04-1/conor_dev/ccp4/build/bin/cctbx.python",
                         walltime="03:00:00")

    print("scaling cluster")
    cluster.scale(80)
    time.sleep(60)
    client = Client(cluster)
    # print(client.scheduler_info())

    print("Restarting cluster")
    client.restart()
    print(client)
    # print(client.scheduler_info())
    print(cluster.dashboard_link)
    print(client.get_versions(check=True))
    time.sleep(20)

    print("scattering datas")
    ds = client.scatter([d for dtag, d in truncated_datasets.items()])
    print("Scattered datasets")
    g = client.scatter(grid)
    print("scattered grid")
    # # r = client.scatter(reference)
    rm = client.scatter(ref_map)
    print("scattered refmap")
    # n = len(ds)
    #
    # # job = lambda d: map_loader(d, g, rm, max_res)
    #
    # xmaps_futures_list = client.map(map_loader, ds, [g] * n, [rm] * n, [max_res] * n)
    # print("Got futures")
    # xmaps_list = client.gather(xmaps_futures_list)
    # print("Got maps")

    # delayeds = [dask.delayed(map_loader)(d, g, rm, max_res) for d in ds]
    # print("Got delayed")
    # print(delayeds[0])
    #
    # futures = [client.compute(delayed) for delayed in delayeds]
    # print("Got futures")
    # print(futures[0])
    #
    # xmaps_list = [future.result() for future in futures]
    # print("Got maps")
    # print(xmaps_list[0])

    #     futures = []
    # #     futures = {}
    # #     for dtag, dts in truncated_datasets.items():
    #     for i, dts in enumerate(ds):
    #         print("submitted future {}".format(dtag))
    # #         futures.append(client.submit(load_map, map_loader, dts, g, rm, max_res))
    #         futures.append(client.submit(load_map, map_loader, dts, g, rm, max_res))
    # #         futures[truncated_datasets.items()[i][0]] = client.submit(load_map, map_loader, dts, g, rm, max_res)
    #     print("submitted futures")

    futures = {}
    for dtag, dts in truncated_datasets.items():
        futures[dtag] = client.submit(load_map, map_loader, dts, g, rm, max_res)

    #     xmaps_list = []
    #     for i, ftr in enumerate(futures):
    #         print("getting future {}".format(i))
    #         print("future status: {}".format(ftr))
    #         xmaps_list.append(ftr.result())
    #     print("Get xmaps")

    #     client.gather(futures)
    results = {}
    wait = True
    while True:
        wait = False
        #         for i, future in enumerate(futures):
        for dtag, future in futures.items():

            if future.cancelled():
                print("resubmitting {}!".format(dtag))
                futures[dtag] = client.submit(load_map,
                                              map_loader,
                                              truncated_datasets[dtag],
                                              grid, ref_map, max_res)
                wait = True
            if not future.done():
                wait = True

            if future.done() & (not future.cancelled()) & (not (dtag in results.keys())):
                print("grabbing future {}".format(dtag))
                results[dtag] = future.result()

        time.sleep(10)
        if wait is False:
            print("all ready for collection")
            break
        else:
            print("A future isn't done yet or was resubmited")

    try:
        client.close()
    except:
        print("client didn't close")
    try:
        cluster.close()
    except:
        print("cluster didn't close")

    return results


def process_joblib(map_loader, truncated_datasets, ref_map, grid, max_res):
    xmaps_list = joblib.Parallel(n_jobs=20)(joblib.delayed(map_loader)(d, grid, ref_map, max_res)
                                            for dtag, d
                                            in truncated_datasets.items())
    return xmaps_list


def process_seriel(map_loader, truncated_datasets, ref_map, max_res):
    xmaps_list = map(lambda d: map_loader(d, grid, ref_map, max_res),
                     [d
                      for dtag, d
                      in truncated_datasets])
    return xmaps_list


def transform_dataset(dataset, dataset_transforms, reference):
    # Apply dataset transforms
    print("checking data")
    if "data_check" in dataset_transforms:
        transform = dataset_transforms["data_check"]
        dataset = transform(dataset, reference)
    print("Checked data")
    print(len(dataset.datasets.keys()))

    print("Scaling diffractions")
    if "scale_diffraction" in dataset_transforms:
        transform = dataset_transforms["scale_diffraction"]
        dataset = transform(dataset, reference)
    print("scaled diffractions")
    print(len(dataset.datasets.keys()))


    # if "filter_structure" in dataset_transforms:
    #     transform = dataset_transforms["filter_structure"]
    #     dataset = transform(dataset, reference)
    # print(len(dataset.datasets.keys()))
    # same_spacegroup
    # reasonable_rfree
    # similar_models
    # dataset = filter_dataset(lambda d: same_spacegroup(d, reference), dataset)
    # dataset = filter_dataset(lambda d: reasonable_rfree(d, settings[]), dataset)
    # dataset = filter_dataset(lambda d: similar_models(d, reference), dataset)

    print("Filtering wilson")
    if "filter_wilson" in dataset_transforms:
        transform = dataset_transforms["filter_wilson"]
        dataset = transform(dataset, reference)
    print(len(dataset.datasets.keys()))
    print("Filtered wilson")

    # print("Aligning")
    # if "align" in dataset_transforms:
    #     transform = dataset_transforms["align"]
    #     dataset = transform(dataset, reference)
    # print(len(dataset.datasets.keys()))
    # print("aligned")


    return dataset


def subsample_dataset(dataset, n):

    if n == "all":
        dtags = dataset.datasets.keys()
    else:
        dtags = np.random.choice(dataset.datasets.keys(), size=int(n), replace=False)
    new_datasets = {dtag: dataset.datasets[dtag] for dtag in dtags}
    new_dataset = dataset
    new_dataset.datasets = new_datasets
    return new_dataset


def get_xmaps(dataset, grid, reference, reference_map_getter, map_loader, processer):
    # ###############################################
    # Get resolution
    # ###############################################
    max_res = max([dts.data.summary.high_res for dtag, dts
                   in dataset.datasets.items()])

    # ###############################################
    # Truncate datasets
    # ###############################################
    truncated_reference, truncated_datasets = PanddaDiffractionDataTruncater()(dataset.datasets,
                                                                               reference)

    # ###############################################
    # Generate maps
    # ###############################################
    # Generate reference map for shell
    ref_map = reference_map_getter(reference,
                                   max_res,
                                   grid)

    # Load maps

    xmaps = processer(map_loader, truncated_datasets, ref_map, grid, max_res)

    #     xmaps = {}

    #     for i, dtag in enumerate(truncated_datasets):
    #         # xmaps[dtag] = map_loader(truncated_datasets[dtag],
    #         #                          grid,
    #         #                          ref_map,
    #         #                          max_res)
    #         xmaps[dtag] = xmaps_list[i]

    print(xmaps)

    return xmaps


def truncate_datasets(dataset):
    truncated_reference, truncated_datasets = PanddaDiffractionDataTruncater()(dataset.datasets,
                                                                               reference)
    return truncated_reference, truncated_datasets


def embed_xmaps(numpy_maps):
    #     # Convert to numpy
    #     numpy_maps = [xmap.get_map_data(sparse=False).as_numpy_array()
    #                   for dtag, xmap
    #                   in xmaps.items()]

    # Convert to sample by feature
    sample_by_feature = np.vstack([np_map.flatten()
                                   for np_map
                                   in numpy_maps])

    # Reduce dimension by PCA
    pca = PCA(n_components=50)
    sample_by_feature_pca = pca.fit_transform(sample_by_feature)

    # Reduce dimension by TSNE
    tsne = TSNE(n_components=2,
                method="exact")
    sample_by_feature_tsne = tsne.fit_transform(sample_by_feature_pca)

    return sample_by_feature_tsne


def cluster_embedding(xmap_embedding):
    # Perform initial clustering
    clusterer = hdbscan.HDBSCAN(allow_single_cluster=True,
                                prediction_data=True,
                                min_cluster_size=5)
    labels = clusterer.fit(xmap_embedding.astype(np.float64)).labels_

    return labels





def output_cluster_graph(cluster_df, path):
    # #############################################################################
    # Output plots and csvs
    # #############################################################################

    # TODO: Notebook only feature!
    ### Plot with annotations
    # Plot clusters

    fig, ax = plt.subplots(figsize=(10, 10))

    plot = sns.scatterplot(x=cluster_df["x"],
                           y=cluster_df["y"],
                           hue=cluster_df["cluster"])

    fig = plot.get_figure()
    fig.savefig(path)


def mapdict(func, target_dict):
    items = target_dict.items()
    keys = [item[0] for item in items]
    values = [item[1] for item in items]
    results = map(func, values)
    results_dict = dict(zip(keys, results))
    return results_dict


def mapdictparallel(func, dictionary, executor):
    keys = dictionary.keys()
    values = dictionary.values()

    # results = joblib.Parallel(n_jobs=21, verbose=10)(
    #     joblib.delayed(func)(value) for value in values
    # )
    print("Execing")
    # with MPIPoolExecutor(max_workers=15) as executor:
    results = executor.map(func, values)

    results_dict = dict(zip(keys, results))
    return results_dict


def xmap_to_numpy(xmap):
    return xmap.get_map_data(sparse=False).as_numpy_array()


def flex_map_to_numpy(flex_map):
    return flex_map.as_numpy_array()


# def xmap_to_numpy(xmap):
#     return xmap.get_map_data(sparse=False).as_numpy_array()


def get_res(dataset):
    res = dataset.data.summary.high_res
    return float(res)


def get_min_res(dataset):

    res_list = map(get_res, dataset.datasets.values())
    min_res = min(res_list)

    return min_res


# dataset -> {dtag: numpy}
def get_and_align_xmaps(dataset):

    truncated_reference, truncated_datasets = truncate_datasets(dataset)
    dataset.datasets = truncated_datasets
    print(dataset.datasets.keys())

    # dataset -> float
    min_res = get_min_res(dataset)

    # dataset -> {dtag: flex_map}
    xmaps = mapdictparallel(lambda ds: load_xmap(ds, min_res), dataset.datasets)
    print(xmaps.keys())

    # {dtag: xmaps} -> {dtag: numpy}
    xmaps_np = mapdictparallel(xmap_to_numpy, xmaps)
    print(xmaps_np.keys())

    # {dtag: numpy} -> {dtag: numpy}
    xmaps_aligned = mapdictparallel(lambda xmap_np: align_xmap_np(xmaps_np.values()[0], xmap_np),
                                    xmaps_np)
    print(xmaps_aligned.keys())

    return xmaps_aligned


# def load_xmaps(datasets):
#
#     xmaps = joblib.Parallel(n_jobs=21, verbose=10, batch_size=1)(
#         joblib.delayed(load_xmap)(dts) for dtag, dts in datasets)


def load_xmap_as_flex(pandda_dataset, min_res):

    miller_array_trunc = pandda_dataset.data.miller_arrays['truncated']
    # FFT

    flex_ed_map = miller_array_trunc.fft_map(
        resolution_factor=0.25,
        d_min=min_res,
        symmetry_flags=cctbx.maptbx.use_space_group_symmetry)

    scaled_maps = flex_ed_map.apply_sigma_scaling().real_map()

    return scaled_maps


def load_xmap(pandda_dataset, min_res):

    miller_array_trunc = pandda_dataset.data.miller_arrays['truncated']
    # FFT

    fft_map = miller_array_trunc.fft_map(
        resolution_factor=0.25,
        d_min=min_res,
        symmetry_flags=cctbx.maptbx.use_space_group_symmetry)

    scaled_map = fft_map.apply_sigma_scaling()

    xmap = ElectronDensityMap.from_fft_map(scaled_map)

    return xmap


# def register_maps(numpy_maps):
#
#     # Set up image registration
#     registered_maps = []
#     registered_maps.append(numpy_maps[0])
#
#     static = numpy_maps[0]
#     static_grid2world = np.eye(4)
#     moving_grid2world = np.eye(4)
#
#     # Rectify maps
#     rectified_maps = joblib.Parallel(n_jobs=21, verbose=10, batch_size=1)(
#         joblib.delayed(rectify)(static, moving, static_grid2world, moving_grid2world)
#         for moving
#         in numpy_maps[1:])
#     print(len(rectified_maps))
#
#     return rectified_maps




def xmaps_to_numpy(xmaps):
    numpy_maps = [xmap.get_map_data(sparse=False).as_numpy_array() for dtag, xmap in xmaps.items()]
    return numpy_maps


def get_map_clusters(cluster_df, xmaps):
    map_clusters = {}
    for cluster_num in cluster_df["cluster"].unique():
        cluster_dtags = cluster_df[cluster_df["cluster"] == cluster_num]["dtag"].values
        map_clusters[cluster_num] = [xmap for dtag, xmap in xmaps.items() if (dtag in cluster_dtags)]

    return map_clusters




def make_and_output_mean_maps(xmaps, cluster_df, out_dir, grid):

    # -> {cluster_num : [xmaps]}
    map_clusters = get_map_clusters(cluster_df,
                                    xmaps,
                                    )

    #  -> {cluster_num: mean map}
    mean_maps = map(lambda x: make_mean_map(x[0], x[1]),
                    map_clusters.items()
                    )
    mean_maps = dict(mean_maps)

    # Output the mean maps
    # {dtag: numpy} -> None
    map(lambda x: output_mean_map(x[0], x[1], out_dir, grid.space_group()),
        mean_maps.items()
        )


def mpi_executor(func, values, executor):
    mp_start = time.time()
    print("\tentered ezecutor")
    results = executor.map(func, values)
    print("\tGot results")
    mp_finish = time.time()
    print("Multiprocessed in: {}".format(mp_finish-mp_start))

    return results

