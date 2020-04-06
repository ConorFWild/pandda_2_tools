import luigi
from autobuild.luigi_sge import SGEJobTask


class Task(SGEJobTask):
    func = luigi.Parameter()
    output_path = luigi.Parameter()

    def work(self):
        self.func()

    def output(self):
        return luigi.LocalTarget(str(self.output_path))


class ProcessorLuigi:

    def __init__(self,
                 jobs=10,
                 parallel_env="smp",
                 n_cpu=12,
                 run_locally=False,
                 h_vmem=100,
                 m_mem_free=5,
                 ):
        self.jobs = jobs
        self.parallel_env = parallel_env
        self.n_cpu = n_cpu
        self.run_locally = run_locally
        self.h_vmem = h_vmem
        self.m_mem_free = m_mem_free


    def __call__(self,
                 funcs,
                 output_paths=None,
                 result_loader=None,
                 shared_tmp_dir=None,
                 ):
        tasks = [Task(func=func,
                      output_path=output_path,
                      shared_tmp_dir="/dls/labxchem/data/2017/lb18145-17/processing/analysis/TMP_autobuild/luigi",
                      parallel_env=self.parallel_env,
                      n_cpu=self.n_cpu,
                      run_locally=False,
                      h_vmem=self.h_vmem,
                      m_mem_free=self.m_mem_free,
                      )
                 for func, output_path
                 in zip(funcs, output_paths)
                 ]

        luigi.build(tasks,
                    local_scheduler=True,
                    workers=self.jobs,
                    detailed_summary=False,
                    )

        if result_loader:
            results = [result_loader(output_path)
                       for output_path
                       in output_paths
                       ]

        else:
            results = []

        return results

