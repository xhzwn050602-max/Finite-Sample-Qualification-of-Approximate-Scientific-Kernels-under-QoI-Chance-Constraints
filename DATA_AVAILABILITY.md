# Code and data availability

Code, configurations, recorded paired traces, summary files, numerical audits,
and plotting scripts are available in this repository:
https://github.com/xhzwn050602-max/Finite-Sample-Qualification-of-Approximate-Scientific-Kernels-under-QoI-Chance-Constraints

All inputs are generated model data. Raw JSONL traces are stored with lossless
gzip compression in `data/`; run `python prepare_data.py` to restore the exact
original files and verify the data hashes recorded in `DATA_MANIFEST.json`.
The original configuration and observation bytes are unchanged. Timing results
depend on the recorded machine and software environment. No data DOI is assigned.
