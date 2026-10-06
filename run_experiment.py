import json

import numpy as np

from main import main
from utils import seed_everything


SEEDS = [42, 43, 44, 45, 46]


def run_experiment() -> None:
    results = {}

    for run, seed in enumerate(SEEDS, start=1):
        seed_everything(seed)
        print(f'-------- RUN {run} | SEED {seed} --------')
        results[seed] = main()

        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)

    avg = []
    wga = []

    for seed in SEEDS:
        avg.append(results[seed]['Accuracy'])
        wga.append(results[seed]['WGA'])

    avg = np.asarray(avg)
    wga = np.asarray(wga)

    avg_mean = np.mean(avg)
    avg_std = np.std(avg)
    wga_mean = np.mean(wga)
    wga_std = np.std(wga)
    gap_mean = avg_mean - wga_mean

    final_result = {}
    final_result['AVG'] = float(avg_mean)
    final_result['AVG_STD'] = float(avg_std)
    final_result['WGA'] = float(wga_mean)
    final_result['WGA_STD'] = float(wga_std)
    final_result['GAP'] = float(gap_mean)

    results['final'] = final_result
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(
        '\n5-Run Summary\n'
        '-------------\n'
        f'AVG: {final_result["AVG"] * 100:.2f}% ± {final_result["AVG_STD"] * 100:.2f}%\n'
        f'WGA: {final_result["WGA"] * 100:.2f}% ± {final_result["WGA_STD"] * 100:.2f}%\n'
        f'GAP: {final_result["GAP"] * 100:.2f}%'
    )


if __name__ == '__main__':
    run_experiment()