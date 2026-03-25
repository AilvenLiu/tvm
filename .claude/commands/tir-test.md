<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

Run the full TIRX test suite.

## Steps

1. Select the least busy GPU to avoid conflicts:
   ```bash
   export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
   ```

2. Run the full test suite with xdist parallelism:
   ```bash
   pytest tests/python/tirx/ -n 16
   ```

3. Report results: total passed, failed, skipped, errors.

## Failure triage rules

**CRITICAL: Never pipe test output to `tail` or `grep` when diagnosing failures. Always capture and read full logs.**

Classify every failure into one of these categories:

- **A — Environment/import error**: Module not found, missing dependency, collection error. These are not caused by code changes.
- **B — Real kernel correctness regression**: Assertion failures (cosine_sim, numerical diff), `CUDA: unspecified launch failure`, or wrong results. **These MUST be investigated and fixed if caused by current changes.**
- **C — Secondary xdist crash**: `KeyError: <WorkerController gwXX>` after a worker abort. The KeyError itself is noise — find the underlying cause (usually category B in another worker).

**Never dismiss a failure as "pre-existing" without evidence.** If a test fails:
1. Check whether the test touches code you changed.
2. If unclear, verify on the parent commit before claiming pre-existing.
3. All failures caused by current changes MUST be fixed — not deferred.
