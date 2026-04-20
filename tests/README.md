# Test Updates (Short)

## What changed
- Added `tests/bm25_client_test.py` for BM25 client unit contracts (R1-R8).
- Added `tests/bm25_code_agent_integration_test.py` for CodeAgent + BM25 client/server integration:
  - hypothesis index naming (`hyp_{id}`)
  - text routed to the correct index
  - eval path uses correct index names
  - `eval_utils.run_subset_eval` passes the provided `index_name` to `batch_retrieve`

## How to run (main)
- Run client unit tests:
  - `python -m unittest tests/bm25_client_test.py -v`
- Run integration tests:
  - `python -m unittest tests/bm25_code_agent_integration_test.py -v`
- Run both files together:
  - `python -m unittest tests/bm25_client_test.py tests/bm25_code_agent_integration_test.py -v`
