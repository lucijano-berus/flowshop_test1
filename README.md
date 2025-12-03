# Flow-shop scheduling demo

This repository demonstrates a multi-objective flow-shop scheduler using the
[pymoo](https://pymoo.org/) optimization library. The included data file
(`data/flowshop_jobs.tsv`) matches the tab-separated matrix shared in the prompt
and uses comma decimals, which are normalized automatically by the script.

## Running the example

1. Install dependencies (pandas, numpy, matplotlib, pymoo):

   ```bash
   pip install pymoo pandas numpy matplotlib
   ```

2. Execute the scheduler. It will solve the flow-shop problem with NSGA-II,
   NSGA-III, and MOEA/D and save a Gantt chart to `flowshop_gantt.png`:

   ```bash
   python flowshop.py
   ```

The console output lists the objective values for each algorithm, reports the
best makespan that was found, and tells you where the image was saved.
