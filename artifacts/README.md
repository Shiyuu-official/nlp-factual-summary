# Experiment Artifacts

`server_full_100.tar.gz` contains the complete final server run output for the 500-sample GovReport experiment.

To extract:

```bash
tar -xzf artifacts/server_full_100.tar.gz
```

The archive contains:

- `pipeline_result.json`
- `step1_data_stats.json`
- `step2_summaries.json`
- `step3_consistency.json`
- `step4_corrections.json`
- `step5_comparison.json`
- `step6_length_impact.json`
- `step6_length_impact.png`
- `step6_cases.json`
- partial checkpoint files and `pipeline.log`

