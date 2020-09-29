import copy
import threading
import time

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

        self.thread = None
        self.thread = threading.Thread(target=self.job)
        self.thread.start()

        self.lock = threading.Lock()
        self.ask_params = []  # list[ param ]
        self.tell_values = []  # list[tuple[ param value ]]

    def job(self):
        # 何故かrun_optimizationではなくインスタンス化した直後に f が1 iteration実行される
        self.optimizer = GPyOpt.methods.BayesianOptimization(**self.optimizer_opts)
        self.optimizer.run_optimization(max_iter=self.max_iter-1)

    def stop(self):
        if self.thread is None:
            return

        self.thread.join()
        self.thread = None

    def objective_wrapper(self, x):
        param = copy.deepcopy(x[0, :])
        with self.lock:
            self.ask_params.append(param)

        frozen_params = frozenset(param)
        while True:
            with self.lock:
                if frozen_params in self.tell_values:
                    return self.tell_values[frozen_params]
            time.sleep(0.1)

    def ask(self):
        while True:
            with self.lock:
                if len(self.ask_params) == self.batch_size:
                    break
            time.sleep(0.1)

        with self.lock:
            params = copy.deepcopy(self.ask_params)
            self.ask_params = []
        return params  # List[np.ndarray]

    def tell(self, values):  # List[Tuple[np.ndarray, float]]
        with self.lock:
            self.tell_values = copy.deepcopy({
                frozenset(param): v for param, v in values
            })


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
            x = params[i]
            value = f(x[0], x[1])
            y.append((x, value))

            if best_value is None or best_value > value:
                best_value = value
                best_param = x
        optimizer.tell(y)

    optimizer.stop()
    print("best value={}, param={}".format(best_value, best_param))


if __name__ == '__main__':
    main()