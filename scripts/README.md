# scripts

## `update-pricing.mjs`

Regenerates `src/data/pricing.json` (the cloud GPU offers snapshot) from public sources. Requires Node 22+, no npm dependencies.

```sh
node scripts/update-pricing.mjs            # fetch and write pricing.json (only if prices changed)
node scripts/update-pricing.mjs --dry-run  # fetch and print a summary, write nothing
node scripts/update-pricing.mjs --check    # validate pricing.json + gpus.json offline (CI-safe)
GCP_API_KEY=... node scripts/update-pricing.mjs   # also fetch per-region GCP T4/V100 GPU prices
```

Sources:

| Provider | Data | Source |
| --- | --- | --- |
| AWS | On-demand prices and region availability | Public AWS price files behind the EC2 pricing page (`b0.p.awsstatic.com`) |
| AWS | Spot (average) and 1-year Savings Plan prices | [ec2instances.info](https://instances.vantage.sh) open data (MIT), read as a stream |
| GCP | A2 / A3 / G2 / G4 / N1 prices for every region | [gcloud-compute.com](https://gcloud-compute.com) CSV (Apache-2.0), built from the Cloud Billing API |
| GCP | A4, A3 Ultra, and T4/V100 GPU prices (us-central1) | Official [accelerator-optimized pricing page](https://cloud.google.com/products/compute/pricing/accelerator-optimized) |
| GCP | T4/V100 GPU prices for each region (optional) | Cloud Billing Catalog API, only when `GCP_API_KEY` is set |

Each source runs on its own. When one fails, the script logs `WARN`, keeps that source's offers from the previous `pricing.json`, and still writes everything else. It never writes an empty or invalid file. If every source fails, it exits non-zero and leaves the file as it was. To test the fallbacks, set `PRICING_SKIP=awsOnDemand,vantage,gcpCsv,gcpPage,gcpSkus` (any subset) to simulate an outage.

Instance metadata (which GPU each instance type has, and how many) lives in the tables at the top of the script (`AWS_INSTANCES`, `GCP_BUNDLED`, `GCP_ATTACH_GPUS`). To track a new instance family, add it there, and add its GPU to `src/data/gpus.json` if it is new. `--check` fails when an offer points to a `gpuId` that is not in the catalog.

### CI

`.github/workflows/update-pricing.yml` runs the script every Monday and on manual dispatch. When the data changes, it opens a PR titled "chore: update cloud GPU pricing". Setup:

- Turn on *Settings → Actions → General → Allow GitHub Actions to create and approve pull requests*.
- Optional: add a `GCP_API_KEY` repository secret. This needs an API key with the Cloud Billing API enabled.
