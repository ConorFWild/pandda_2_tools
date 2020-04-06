from dask.distributed import Client

from dask_jobqueue import SGECluster


def process(dict):
    results = {}

    for key, func in dict.items():
        results[key] = func()

    return results


def call(func):
    return func()


def process_dask(funcs,
                 jobs=10,
                 cores=3,
                 processes=3,
                 h_vmem=20,
                 m_mem_free=5,
                 h_rt=3000,
                 ):
    cluster = SGECluster(n_workers=0,
                         job_cls=None,
                         loop=None,
                         security=None,
                         silence_logs='error',
                         name=None,
                         asynchronous=False,
                         interface=None,
                         host=None,
                         protocol='tcp://',
                         dashboard_address=':8787',
                         config_name=None,
                         processes=processes,
                         queue='low.q',
                         project="labxchem",
                         cores=cores,
                         memory="{}GB".format(h_vmem),
                         walltime=h_rt,
                         resource_spec="m_mem_free={}G,h_vmem={}G,h_rt={}".format(m_mem_free, h_vmem, h_rt),
                         )

    cluster.scale(jobs=jobs)

    client = Client(cluster)

    results_futures = client.map(call,
                         funcs,
                         )

    results = client.gather(results_futures)

    return results
