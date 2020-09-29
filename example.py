import multiprocessing
import queue
import time
import os
import random

import GPyOpt
import numpy as np


class GPyOptAskTellOptimizer:
    def __init__(self, optimizer_opts, batch_size=8, max_iter=16):
        optimizer_opts["f"] = self.objective_wrapper
        optimizer_opts["batch_size"] = batch_size
        optimizer_opts["num_cores"] = batch_size
        self.optimizer_opts = optimizer_opts
        self.optimizer = None

        self.max_iter = max_iter
        self.batch_size = batch_size
        self.process = None

        self.param_queue = multiprocessing.Queue(maxsize=batch_size)
        self.value_queue = multiprocessing.Queue(maxsize=batch_size)

        self.process = multiprocessing.Process(target=self.job)
        self.process.start()

    def job(self):
        # TODO: 何故かrun_optimizationではなくインスタンス化した直後に f が呼ばれ始めるのでmax_iterがどうなってるか調べる。
        self.optimizer = GPyOpt.methods.BayesianOptimization(**self.optimizer_opts)
        self.optimizer.run_optimization(max_iter=self.max_iter)

    def stop(self):
        if self.process is None:
            return

        self.process.terminate()
        self.process = None

    def objective_wrapper(self, x):
        pid = os.getpid()  # gpyoptはbatch optimizationのときprocessを生成して目的関数を呼ぶので、pidが識別子として使える。
        self.param_queue.put((pid, x))
        while True:
            try:
                sample_pid, value = self.value_queue.get(block=False)
                if pid == sample_pid:
                    break
                # 別のprocessがsampleしたものはもう一度詰め直す
                self.value_queue.put((sample_pid, value))
            except queue.Empty:
                # 同じのを引き続けるのを防ぐためにrandom sleep.
                time.sleep(1 * random.random())
        return value

    def ask(self):
        params = []
        for i in range(self.batch_size):
            pid, x = self.param_queue.get(block=True)
            params.append((pid, x[0, :]))
        return params  # List[Tuple[int, np.ndarray]]

    def tell(self, values):  # List[Tuple[int, float]]
        for i in range(self.batch_size):
            self.value_queue.put(values[i])


def f(x0, x1):
    f0 = np.log(10.5-x0) + 0.1*np.sin(15*x0)
    f1 = np.cos(1.5*x0) + 0.1*x0
    value = (1-x1)*f0 + x1*f1
    print("value={} (x0={}, x1={})".format(value, x0, x1))
    return value


def main():
    bounds = [
        {'name': 'x0', 'type': 'continuous', 'domain': (0, 10)},
        {'name': 'x1', 'type': 'discrete', 'domain': (0, 1)},
    ]
    opts = {
        "domain": bounds,
        "acquisition_type": 'EI',
        "normalize_Y": True,
        "initial_design_numdata": 8,
        "evaluator_type": 'local_penalization',
        "acquisition_jitter": 0
    }
    batch_size = 8
    max_iter = 10
    optimizer = GPyOptAskTellOptimizer(opts, batch_size=batch_size, max_iter=max_iter)

    best_value = None
    best_param = None

    for iteration in range(max_iter):
        print("iteration", iteration)

        params = optimizer.ask()
        y = []
        for i in range(batch_size):
            x_id, x = params[i]
            value = f(x[0], x[1])
            y.append((x_id, value))

            if best_value is None or best_value > value:
                best_value = value
                best_param = x
        optimizer.tell(y)

    optimizer.stop()
    print("best value={}, param={}".format(best_value, best_param))


if __name__ == '__main__':
    main()