#!/usr/bin/env node
// Regenerates src/data/pricing.json from public cloud pricing sources.
//
//   node scripts/update-pricing.mjs           fetch sources, write pricing.json if offers changed
//   node scripts/update-pricing.mjs --dry-run fetch sources, print a summary, write nothing
//   node scripts/update-pricing.mjs --check   validate pricing.json + gpus.json only (no network)
//
// Dependency-free (Node >= 22). Optional env: GCP_API_KEY (Cloud Billing Catalog API key).
//
// Robustness contract: every source is independent. If a source fails, the offers it is
// responsible for are carried over from the previous pricing.json and a warning is logged.
// The file is never written empty, and is only rewritten when the offers actually change.

import { readFile, writeFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';

const ROOT = new URL('../', import.meta.url);
const PRICING_PATH = new URL('src/data/pricing.json', ROOT);
const GPUS_PATH = new URL('src/data/gpus.json', ROOT);
const UA = 'need-more-vram-pricing-updater/1.0 (open-source GPU calculator)';
const HOURS_PER_MONTH = 730;

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

const REGIONS = [
  { provider: 'aws', id: 'us-east-1', name: 'US East (N. Virginia)' },
  { provider: 'aws', id: 'us-west-2', name: 'US West (Oregon)' },
  { provider: 'aws', id: 'ap-southeast-1', name: 'Asia Pacific (Singapore)' },
  { provider: 'aws', id: 'ap-southeast-3', name: 'Asia Pacific (Jakarta)' },
  { provider: 'gcp', id: 'us-central1', name: 'Iowa (us-central1)' },
  { provider: 'gcp', id: 'us-east4', name: 'Northern Virginia (us-east4)' },
  { provider: 'gcp', id: 'asia-southeast1', name: 'Singapore (asia-southeast1)' },
  { provider: 'gcp', id: 'asia-southeast2', name: 'Jakarta (asia-southeast2)' },
];
const AWS_REGIONS = REGIONS.filter((r) => r.provider === 'aws');
const GCP_REGIONS = REGIONS.filter((r) => r.provider === 'gcp');
// The "Location" string AWS uses in its public price files == our display name.
const awsLocation = (r) => r.name;

// ---------------------------------------------------------------------------
// Instance metadata: instanceType -> [gpuId, gpuCount]
// Fractional-GPU shapes (g6f, gr6f, g4-standard-6/12/24) are intentionally excluded.
// ---------------------------------------------------------------------------

const sizes = (family, gpuId, map) =>
  Object.fromEntries(Object.entries(map).map(([size, n]) => [`${family}.${size}`, [gpuId, n]]));

const AWS_INSTANCES = {
  'p6-b300.48xlarge': ['b300-sxm-288gb', 8],
  'p6-b200.48xlarge': ['b200-sxm-180gb', 8],
  'p5en.48xlarge': ['h200-sxm-141gb', 8],
  'p5e.48xlarge': ['h200-sxm-141gb', 8], // mostly Capacity Blocks only; included when publicly priced
  'p5.48xlarge': ['h100-sxm-80gb', 8],
  'p5.4xlarge': ['h100-sxm-80gb', 1],
  'p4de.24xlarge': ['a100-sxm-80gb', 8],
  'p4d.24xlarge': ['a100-sxm-40gb', 8],
  'p3dn.24xlarge': ['v100-sxm2-32gb', 8],
  ...sizes('p3', 'v100-sxm2-16gb', { '2xlarge': 1, '8xlarge': 4, '16xlarge': 8 }),
  ...sizes('g7e', 'rtx-pro-6000-blackwell-96gb', { '2xlarge': 1, '4xlarge': 1, '8xlarge': 1, '12xlarge': 2, '24xlarge': 4, '48xlarge': 8 }),
  ...sizes('g7', 'rtx-pro-4500-blackwell-32gb', { '2xlarge': 1, '4xlarge': 1, '8xlarge': 1, '12xlarge': 2, '24xlarge': 4, '48xlarge': 8 }),
  ...sizes('g6e', 'l40s-48gb', { xlarge: 1, '2xlarge': 1, '4xlarge': 1, '8xlarge': 1, '16xlarge': 1, '12xlarge': 4, '24xlarge': 4, '48xlarge': 8 }),
  ...sizes('g6', 'l4-24gb', { xlarge: 1, '2xlarge': 1, '4xlarge': 1, '8xlarge': 1, '16xlarge': 1, '12xlarge': 4, '24xlarge': 4, '48xlarge': 8 }),
  ...sizes('gr6', 'l4-24gb', { '4xlarge': 1, '8xlarge': 1 }),
  ...sizes('g5', 'a10g-24gb', { xlarge: 1, '2xlarge': 1, '4xlarge': 1, '8xlarge': 1, '16xlarge': 1, '12xlarge': 4, '24xlarge': 4, '48xlarge': 8 }),
  ...sizes('g4dn', 't4-16gb', { xlarge: 1, '2xlarge': 1, '4xlarge': 1, '8xlarge': 1, '16xlarge': 1, '12xlarge': 4, metal: 8 }),
};

// GCP machine types with bundled GPUs. onDemand:false = Google does not sell this shape
// on-demand (only Spot / Flex-start / reservations), see
// https://cloud.google.com/compute/docs/accelerator-optimized-machines (consumption options).
// commit:false = no 1-year resource CUD path for this shape.
const GCP_BUNDLED = {
  'a4-highgpu-8g': { gpu: ['b200-sxm-180gb', 8], onDemand: false, pageOnly: true },
  'a3-ultragpu-8g': { gpu: ['h200-sxm-141gb', 8], onDemand: false, pageOnly: true },
  'a3-megagpu-8g': { gpu: ['h100-sxm-80gb', 8] },
  'a3-highgpu-8g': { gpu: ['h100-sxm-80gb', 8] },
  'a3-highgpu-4g': { gpu: ['h100-sxm-80gb', 4], onDemand: false, commit: false },
  'a3-highgpu-2g': { gpu: ['h100-sxm-80gb', 2], onDemand: false, commit: false },
  'a3-highgpu-1g': { gpu: ['h100-sxm-80gb', 1], onDemand: false, commit: false },
  'a2-ultragpu-8g': { gpu: ['a100-sxm-80gb', 8], commit: false }, // CUDs via sales only
  'a2-ultragpu-4g': { gpu: ['a100-sxm-80gb', 4], commit: false }, // CUDs via sales only
  'a2-ultragpu-2g': { gpu: ['a100-sxm-80gb', 2], commit: false }, // CUDs via sales only
  'a2-ultragpu-1g': { gpu: ['a100-sxm-80gb', 1], commit: false }, // CUDs via sales only
  'a2-megagpu-16g': { gpu: ['a100-sxm-40gb', 16] },
  'a2-highgpu-8g': { gpu: ['a100-sxm-40gb', 8] },
  'a2-highgpu-4g': { gpu: ['a100-sxm-40gb', 4] },
  'a2-highgpu-2g': { gpu: ['a100-sxm-40gb', 2] },
  'a2-highgpu-1g': { gpu: ['a100-sxm-40gb', 1] },
  'g4-standard-384': { gpu: ['rtx-pro-6000-blackwell-96gb', 8] },
  'g4-standard-192': { gpu: ['rtx-pro-6000-blackwell-96gb', 4] },
  'g4-standard-96': { gpu: ['rtx-pro-6000-blackwell-96gb', 2] },
  'g4-standard-48': { gpu: ['rtx-pro-6000-blackwell-96gb', 1] },
  'g2-standard-96': { gpu: ['l4-24gb', 8] },
  'g2-standard-48': { gpu: ['l4-24gb', 4] },
  'g2-standard-24': { gpu: ['l4-24gb', 2] },
  'g2-standard-32': { gpu: ['l4-24gb', 1] },
  'g2-standard-16': { gpu: ['l4-24gb', 1] },
  'g2-standard-12': { gpu: ['l4-24gb', 1] },
  'g2-standard-8': { gpu: ['l4-24gb', 1] },
  'g2-standard-4': { gpu: ['l4-24gb', 1] },
};

// GCP N1 + attachable GPU combos (VM price + GPU price). Shapes follow Google's
// documented vCPU limits per GPU count. Regions = where the GPU is offered per
// https://cloud.google.com/compute/docs/gpus/gpu-regions-zones (only our tracked regions).
const GCP_ATTACH_GPUS = {
  t4: {
    gpuId: 't4-16gb',
    label: 'nvidia-tesla-t4',
    pageModel: 'NVIDIA T4',
    skuRe: /^Nvidia Tesla T4 GPU\b/i,
    regions: ['us-central1', 'us-east4', 'asia-southeast1', 'asia-southeast2'],
    shapes: [['n1-standard-8', 1], ['n1-standard-16', 2], ['n1-standard-32', 4]],
  },
  v100: {
    gpuId: 'v100-sxm2-16gb',
    label: 'nvidia-tesla-v100',
    pageModel: 'NVIDIA V100',
    skuRe: /^Nvidia Tesla V100 GPU\b/i,
    regions: ['us-central1'],
    shapes: [['n1-standard-8', 1], ['n1-standard-16', 2], ['n1-standard-32', 4], ['n1-standard-64', 8]],
  },
};

// ---------------------------------------------------------------------------
// Source URLs
// ---------------------------------------------------------------------------

const SRC = {
  awsOnDemand: (loc) =>
    `https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/${encodeURIComponent(loc)}/Linux/index.json`,
  vantage: 'https://instances.vantage.sh/instances.json',
  gcpCsv: 'https://gcloud-compute.com/machine-types-regions.csv',
  gcpPage: 'https://cloud.google.com/products/compute/pricing/accelerator-optimized',
  gcpSkus: 'https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus',
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

const warnings = [];
const warn = (msg) => {
  warnings.push(msg);
  console.warn(`WARN ${msg}`);
};
const info = (msg) => console.log(msg);
// PRICING_SKIP=awsOnDemand,vantage,gcpCsv,gcpPage,gcpSkus simulates source outages (for testing fallbacks).
const SKIP = new Set((process.env.PRICING_SKIP ?? '').split(',').filter(Boolean));
const guard = (name) => {
  if (SKIP.has(name)) throw new Error('skipped via PRICING_SKIP');
};

const round = (n, d = 6) => (n == null || !Number.isFinite(n) ? null : Math.round(n * 10 ** d) / 10 ** d);
const num = (v) => {
  if (v == null || v === '') return null;
  const n = typeof v === 'number' ? v : Number(String(v).replace(/[$,\s]/g, ''));
  return Number.isFinite(n) && n > 0 ? n : null;
};
/** A spot / commitment price that is not below on-demand is not a real discount offer. */
const discounted = (price, od) => (price != null && (od == null || price < od) ? price : null);
const offerKey = (o) => `${o.provider}|${o.instanceType}|${o.region}`;

const redact = (url) => String(url).replace(/([?&]key=)[^&]+/, '$1***');

async function fetchWithRetry(url, { timeoutMs = 120_000, retries = 2, headers = {} } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, {
        headers: { 'user-agent': UA, 'accept-encoding': 'gzip, deflate, br', ...headers },
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${redact(url)}`);
      return res;
    } catch (err) {
      lastErr = err;
      if (attempt < retries) await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
    }
  }
  throw lastErr;
}

async function fetchText(url, opts) {
  const res = await fetchWithRetry(url, opts);
  const buf = Buffer.from(await res.arrayBuffer());
  // Some static hosts serve pre-gzipped bodies without Content-Encoding.
  const body = buf[0] === 0x1f && buf[1] === 0x8b ? gunzipSync(buf) : buf;
  return body.toString('utf8');
}

const fetchJson = async (url, opts) => JSON.parse(await fetchText(url, opts));

/** Streams a top-level JSON array and calls onItem(obj) per element, without buffering the whole body. */
async function streamJsonArray(url, onItem, opts) {
  const res = await fetchWithRetry(url, { timeoutMs: 600_000, ...opts });
  const decoder = new TextDecoder();
  let depth = 0;
  let inStr = false;
  let esc = false;
  let parts = [];
  let count = 0;
  for await (const chunk of res.body) {
    const text = decoder.decode(chunk, { stream: true });
    let start = depth >= 2 ? 0 : -1;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inStr) {
        if (esc) esc = false;
        else if (c === '\\') esc = true;
        else if (c === '"') inStr = false;
        continue;
      }
      if (c === '"') inStr = true;
      else if (c === '{' || c === '[') {
        depth++;
        if (depth === 2) start = i;
      } else if (c === '}' || c === ']') {
        depth--;
        if (depth === 1 && start >= 0) {
          parts.push(text.slice(start, i + 1));
          onItem(JSON.parse(parts.join('')));
          count++;
          parts = [];
          start = -1;
        }
      }
    }
    if (depth >= 2 && start >= 0) parts.push(text.slice(start));
  }
  if (depth !== 0) throw new Error(`truncated JSON array from ${url}`);
  return count;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else q = false;
      } else field += c;
    } else if (c === '"') q = true;
    else if (c === ',') {
      row.push(field);
      field = '';
    } else if (c === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else if (c !== '\r') field += c;
  }
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  const [header, ...data] = rows;
  return data.filter((r) => r.length === header.length).map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

const stripTags = (html) =>
  html
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&#39;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/\s+/g, ' ')
    .trim();

/** Returns HTML tables as arrays of rows of cell text. */
function htmlTables(html) {
  const tables = [];
  for (const t of html.matchAll(/<table\b[\s\S]*?<\/table>/gi)) {
    const rows = [];
    for (const r of t[0].matchAll(/<tr\b[\s\S]*?<\/tr>/gi)) {
      rows.push([...r[0].matchAll(/<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi)].map((c) => stripTags(c[1])));
    }
    tables.push(rows);
  }
  return tables;
}

// "$84.806908493 / 1 hour" -> 84.806908493 ; "N/A" -> null
const parseHourly = (s) => {
  if (!s) return null;
  const m = s.replace(/\s+/g, '').match(/\$([0-9][0-9.,]*)\/1hour/i);
  return m ? num(m[1]) : null;
};

// ---------------------------------------------------------------------------
// AWS
// ---------------------------------------------------------------------------

async function fetchAwsOnDemand() {
  const out = {}; // region -> instanceType -> {od, vcpus, ramGB}
  for (const region of AWS_REGIONS) {
    try {
      guard('awsOnDemand');
      const j = await fetchJson(SRC.awsOnDemand(awsLocation(region)));
      const entries = Object.values(j.regions?.[awsLocation(region)] ?? {});
      const map = {};
      for (const e of entries) {
        const type = e['Instance Type'];
        if (!AWS_INSTANCES[type]) continue;
        const od = num(e.price);
        if (!od) continue;
        map[type] = { od, vcpus: num(e.vCPU), ramGB: num(String(e.Memory ?? '').replace(/GiB/i, '')) };
      }
      if (!Object.keys(map).length) throw new Error('no tracked GPU instances found');
      out[region.id] = map;
      info(`aws on-demand ${region.id}: ${Object.keys(map).length} instance types`);
    } catch (err) {
      warn(`AWS on-demand price file failed for ${region.id}: ${err.message}`);
    }
  }
  return out;
}

const AWS_COMMIT_KEYS = ['yrTerm1InstanceSavings.noUpfront', 'yrTerm1Standard.noUpfront', 'yrTerm1Savings.noUpfront'];

async function fetchVantage() {
  const out = {}; // region -> instanceType -> {od, spot, commit, vcpus, ramGB, gpus}
  try {
    guard('vantage');
    const n = await streamJsonArray(SRC.vantage, (it) => {
      if (!AWS_INSTANCES[it.instance_type]) return;
      for (const region of AWS_REGIONS) {
        const p = it.pricing?.[region.id]?.linux;
        if (!p) continue;
        const commitKey = AWS_COMMIT_KEYS.find((k) => num(p.reserved?.[k]));
        (out[region.id] ??= {})[it.instance_type] = {
          od: num(p.ondemand),
          spot: num(p.spot_avg) ?? num(p.spot_min),
          commit: commitKey ? num(p.reserved[commitKey]) : null,
          vcpus: num(it.vCPU),
          ramGB: num(it.memory),
          gpus: num(it.GPU),
        };
      }
    });
    if (!Object.keys(out).length) throw new Error(`parsed ${n} instances but none matched`);
    info(`vantage: parsed ${n} instances`);
    return out;
  } catch (err) {
    warn(`instances.vantage.sh failed (spot / 1y prices will be carried over): ${err.message}`);
    return null;
  }
}

async function buildAws(prevByKey) {
  const [official, vantage] = await Promise.all([fetchAwsOnDemand(), fetchVantage()]);
  const offers = [];
  const okGroups = new Set();
  for (const region of AWS_REGIONS) {
    const off = official[region.id];
    const van = vantage?.[region.id];
    if (!off && !van) continue; // group failed -> carried over by caller
    okGroups.add(`aws:${region.id}`);
    // Availability: the official price file is authoritative when it loaded.
    const types = Object.keys(off ?? van ?? {});
    for (const type of types) {
      const [gpuId, gpuCount] = AWS_INSTANCES[type];
      const o = off?.[type];
      const v = van?.[type];
      const od = o?.od ?? v?.od;
      if (!od) continue;
      if (v?.gpus && v.gpus !== gpuCount) warn(`${type}: vantage reports ${v.gpus} GPUs, table says ${gpuCount}`);
      const prev = prevByKey.get(`aws|${type}|${region.id}`);
      offers.push({
        provider: 'aws',
        instanceType: type,
        gpuId,
        gpuCount,
        vcpus: o?.vcpus ?? v?.vcpus ?? prev?.vcpus ?? null,
        ramGB: o?.ramGB ?? v?.ramGB ?? prev?.ramGB ?? null,
        region: region.id,
        onDemandHourly: round(od),
        spotHourly: round(discounted(vantage ? (v?.spot ?? null) : (prev?.spotHourly ?? null), od)),
        commit1yHourly: round(discounted(vantage ? (v?.commit ?? null) : (prev?.commit1yHourly ?? null), od)),
      });
    }
  }
  const sources = [];
  if (Object.keys(official).length)
    sources.push({
      provider: 'aws',
      url: 'https://aws.amazon.com/ec2/pricing/on-demand/',
      note: 'On-demand Linux prices and region availability from the public AWS price files behind the EC2 pricing page (https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/<Location>/Linux/index.json).',
    });
  if (vantage)
    sources.push({
      provider: 'aws',
      url: SRC.vantage,
      note: 'Spot (average of recent per-AZ spot prices) and 1-year commitment (EC2 Instance Savings Plan, no upfront; falls back to Standard RI / Compute Savings Plan) from the open ec2instances.info dataset (MIT, Vantage).',
    });
  return { offers, okGroups, sources };
}

// ---------------------------------------------------------------------------
// GCP
// ---------------------------------------------------------------------------

async function fetchGcpCsv() {
  try {
    guard('gcpCsv');
    const rows = parseCsv(await fetchText(SRC.gcpCsv, { timeoutMs: 180_000 }));
    const regionIds = new Set(GCP_REGIONS.map((r) => r.id));
    const out = {}; // region -> name -> row
    for (const r of rows) {
      if (!regionIds.has(r.region)) continue;
      if (!GCP_BUNDLED[r.name] && !/^n1-standard-(8|16|32|64)$/.test(r.name)) continue;
      (out[r.region] ??= {})[r.name] = {
        od: num(r.hour),
        spot: r.spot === '0' || r.spot === '0.0' ? null : num(r.hourSpot),
        commit: num(r.month1yCud) ? num(r.month1yCud) / HOURS_PER_MONTH : null,
        vcpus: num(r.vCpus),
        ramGB: num(r.memoryGB),
      };
    }
    if (!Object.keys(out).length) throw new Error('no matching rows (CSV format changed?)');
    info(`gcloud-compute.com CSV: ${Object.values(out).reduce((a, m) => a + Object.keys(m).length, 0)} rows matched`);
    return out;
  } catch (err) {
    warn(`gcloud-compute.com CSV failed: ${err.message}`);
    return null;
  }
}

/** Official accelerator-optimized pricing page (server-rendered; us-central1 default region). */
async function fetchGcpPage() {
  try {
    guard('gcpPage');
    const tables = htmlTables(await fetchText(SRC.gcpPage));
    const machines = {};
    const gpus = {};
    for (const rows of tables) {
      const hi = rows.findIndex((r) => r.some((c) => /^Machine type$/i.test(c)) || r.some((c) => /^Model$/i.test(c)));
      if (hi < 0) continue;
      const h = rows[hi];
      const col = (re) => h.findIndex((c) => re.test(c));
      if (h.some((c) => /^Machine type$/i.test(c))) {
        const iName = col(/^Machine type$/i);
        const iPrice = col(/^Price \(USD\)/i);
        const iSpot = col(/Current Spot/i);
        const iCud1 = col(/Compute Resource CUDs - 1 Year/i);
        const iComp = col(/^Components$/i);
        for (const r of rows.slice(hi + 1)) {
          const name = r[iName];
          if (!GCP_BUNDLED[name] || machines[name] || r.length !== h.length) continue;
          const comp = r[iComp] ?? '';
          const p = {
            od: parseHourly(r[iPrice]),
            spot: iSpot >= 0 ? parseHourly(r[iSpot]) : null,
            commit: iCud1 >= 0 ? parseHourly(r[iCud1]) : null,
            vcpus: num(comp.match(/vCPUs?:\s*([\d,]+)/i)?.[1]),
            ramGB: num(comp.match(/Memory:\s*([\d,]+)/i)?.[1]),
          };
          // Skip monthly-price tables (no "/ 1 hour" cells).
          if (p.od || p.spot || p.commit) machines[name] = p;
        }
      } else {
        const iModel = col(/^Model$/i);
        const iPrice = col(/^GPU price/i);
        const iCud1 = col(/^1 year commitment/i);
        for (const r of rows.slice(hi + 1)) {
          const model = r[iModel];
          const g = Object.values(GCP_ATTACH_GPUS).find((x) => x.pageModel === model);
          if (!g || gpus[g.gpuId]) continue;
          gpus[g.gpuId] = { od: parseHourly(r[iPrice]), commit: parseHourly(r[iCud1]), spot: null };
        }
      }
    }
    if (!Object.keys(machines).length) throw new Error('no machine rows parsed (page layout changed?)');
    info(`GCP pricing page (us-central1): ${Object.keys(machines).length} machine types, ${Object.keys(gpus).length} attachable GPUs`);
    return { machines, gpus };
  } catch (err) {
    warn(`GCP accelerator-optimized pricing page failed: ${err.message}`);
    return null;
  }
}

/** Cloud Billing Catalog API (needs GCP_API_KEY): per-region attachable GPU prices. */
async function fetchGcpSkus(apiKey) {
  if (!apiKey) return null;
  try {
    guard('gcpSkus');
    const out = {}; // gpuId -> region -> {od, spot, commit}
    let pageToken = '';
    let pages = 0;
    do {
      const u = new URL(SRC.gcpSkus);
      u.searchParams.set('key', apiKey);
      u.searchParams.set('currencyCode', 'USD');
      u.searchParams.set('pageSize', '5000');
      if (pageToken) u.searchParams.set('pageToken', pageToken);
      const j = await fetchJson(u.toString());
      for (const sku of j.skus ?? []) {
        if (sku.category?.resourceGroup !== 'GPU') continue;
        const desc = String(sku.description ?? '').replace(/^Commitment v1:\s*/i, '');
        const g = Object.values(GCP_ATTACH_GPUS).find((x) => x.skuRe.test(desc));
        if (!g || /Virtual Workstation|DWS|Calendar|Flex/i.test(desc)) continue;
        const kind = { OnDemand: 'od', Preemptible: 'spot', Commit1Yr: 'commit' }[sku.category?.usageType];
        if (!kind) continue;
        const pe = sku.pricingInfo?.[0]?.pricingExpression;
        if (pe?.usageUnit !== 'h') continue;
        const rate = pe.tieredRates?.at(-1)?.unitPrice;
        const price = rate ? Number(rate.units ?? 0) + (rate.nanos ?? 0) / 1e9 : null;
        if (!price) continue;
        for (const region of sku.serviceRegions ?? []) {
          ((out[g.gpuId] ??= {})[region] ??= {})[kind] = price;
        }
      }
      pageToken = j.nextPageToken ?? '';
      pages++;
    } while (pageToken && pages < 50);
    if (!Object.keys(out).length) throw new Error('no GPU SKUs matched');
    info(`GCP Cloud Billing Catalog: GPU SKUs for ${Object.keys(out).join(', ')}`);
    return out;
  } catch (err) {
    warn(`GCP Cloud Billing Catalog API failed: ${err.message}`);
    return null;
  }
}

function gcpGroupOf(instanceType) {
  if (instanceType.startsWith('n1-')) return 'gcp:n1';
  return GCP_BUNDLED[instanceType]?.pageOnly ? 'gcp:page' : 'gcp:bundled';
}

async function buildGcp(prevOffers) {
  const [csv, page, skus] = await Promise.all([fetchGcpCsv(), fetchGcpPage(), fetchGcpSkus(process.env.GCP_API_KEY)]);
  const offers = [];
  const okGroups = new Set();
  const extraCarry = new Set(); // previous offer keys to keep even though their group succeeded

  const pushBundled = (name, region, p) => {
    const meta = GCP_BUNDLED[name];
    const od = meta.onDemand === false ? null : p.od;
    const spot = discounted(p.spot ?? null, p.od);
    const commit = meta.commit === false ? null : discounted(p.commit ?? null, p.od);
    if (!od && !spot && !commit) return;
    offers.push({
      provider: 'gcp',
      instanceType: name,
      gpuId: meta.gpu[0],
      gpuCount: meta.gpu[1],
      vcpus: p.vcpus ?? null,
      ramGB: p.ramGB ?? null,
      region,
      onDemandHourly: round(od),
      spotHourly: round(spot),
      commit1yHourly: round(commit),
    });
  };

  // Bundled families: CSV for all regions; official page as us-central1 fallback.
  if (csv) {
    okGroups.add('gcp:bundled');
    for (const [region, byName] of Object.entries(csv))
      for (const [name, p] of Object.entries(byName)) if (GCP_BUNDLED[name] && !GCP_BUNDLED[name].pageOnly) pushBundled(name, region, p);
  } else if (page) {
    // Group stays "failed": fresh us-central1 page prices win, everything else is carried over.
    warn('using the official GCP page (us-central1 only) for bundled families; the rest is carried over');
    for (const [name, p] of Object.entries(page.machines)) if (!GCP_BUNDLED[name].pageOnly) pushBundled(name, 'us-central1', p);
  }

  // Reservation/Spot-only families that the CSV does not cover: official page, us-central1.
  if (page) {
    const got = Object.keys(page.machines).filter((n) => GCP_BUNDLED[n].pageOnly);
    if (got.length) {
      okGroups.add('gcp:page');
      for (const name of got) pushBundled(name, 'us-central1', page.machines[name]);
    }
    // The page lists us-central1 prices for every machine it shows; fill CSV gaps there
    // (e.g. a3-megagpu-8g, which the CSV omits for us-central1).
    if (csv)
      for (const [name, p] of Object.entries(page.machines))
        if (!GCP_BUNDLED[name].pageOnly && !csv['us-central1']?.[name]) pushBundled(name, 'us-central1', p);
  } else if (csv) {
    // Page down: keep the us-central1 offers it used to fill in.
    for (const o of prevOffers)
      if (o.provider === 'gcp' && o.region === 'us-central1' && GCP_BUNDLED[o.instanceType] && !csv['us-central1']?.[o.instanceType])
        extraCarry.add(offerKey(o));
  }

  // N1 + attachable GPU: VM from CSV, GPU from Billing API (all regions) or official page (us-central1).
  if (csv) {
    let n1Count = 0;
    for (const g of Object.values(GCP_ATTACH_GPUS)) {
      for (const region of g.regions) {
        const gp = skus?.[g.gpuId]?.[region] ?? (region === 'us-central1' ? page?.gpus?.[g.gpuId] : null);
        if (!gp?.od) continue;
        for (const [vmName, count] of g.shapes) {
          const vm = csv[region]?.[vmName];
          if (!vm?.od) continue;
          offers.push({
            provider: 'gcp',
            instanceType: `${vmName} + ${count}x ${g.label}`,
            gpuId: g.gpuId,
            gpuCount: count,
            vcpus: vm.vcpus,
            ramGB: vm.ramGB,
            region,
            onDemandHourly: round(vm.od + count * gp.od),
            spotHourly: round(vm.spot && gp.spot ? vm.spot + count * gp.spot : null),
            commit1yHourly: round(vm.commit && gp.commit ? vm.commit + count * gp.commit : null),
          });
          n1Count++;
        }
      }
    }
    if (n1Count) okGroups.add('gcp:n1');
  }

  const sources = [];
  if (csv)
    sources.push({
      provider: 'gcp',
      url: SRC.gcpCsv,
      note: 'A2/A3 High+Mega/G2/G4 and N1 VM prices per region (on-demand, current spot, 1-year resource CUD = month1yCud / 730) from the gcloud-compute.com open dataset (Apache-2.0, Nils Knieling), derived from the Cloud Billing Catalog API.',
    });
  if (page)
    sources.push({
      provider: 'gcp',
      url: SRC.gcpPage,
      note: 'A4 (B200) and A3 Ultra (H200) prices plus N1-attachable T4/V100 GPU prices for us-central1 from the official accelerator-optimized pricing page. A4, A3 Ultra and A3 High <8 GPU have no on-demand option (Spot, Flex-start or reservations only), so onDemandHourly is null for them.',
    });
  if (skus)
    sources.push({
      provider: 'gcp',
      url: 'https://cloud.google.com/billing/docs/how-to/get-pricing-information-api',
      note: 'Per-region T4/V100 GPU prices (on-demand, spot, 1-year commitment) from the Cloud Billing Catalog API.',
    });
  return { offers, okGroups, sources, extraCarry };
}

// ---------------------------------------------------------------------------
// Validation (--check)
// ---------------------------------------------------------------------------

function validate(pricing, gpus) {
  const errors = [];
  const isNumOrNull = (v) => v === null || (typeof v === 'number' && Number.isFinite(v) && v > 0);

  if (!Array.isArray(gpus) || !gpus.length) errors.push('gpus.json: must be a non-empty array');
  const gpuIds = new Set();
  for (const g of gpus ?? []) {
    const where = `gpus.json[${g?.id}]`;
    if (!g?.id || typeof g.id !== 'string') errors.push(`${where}: missing id`);
    if (gpuIds.has(g.id)) errors.push(`${where}: duplicate id`);
    gpuIds.add(g.id);
    if (!['nvidia', 'amd', 'intel', 'google'].includes(g.vendor)) errors.push(`${where}: bad vendor ${g.vendor}`);
    if (!['datacenter', 'workstation', 'consumer'].includes(g.segment)) errors.push(`${where}: bad segment ${g.segment}`);
    if (!(g.memoryGB > 0)) errors.push(`${where}: memoryGB must be > 0`);
    if (!isNumOrNull(g.memoryBandwidthGBs)) errors.push(`${where}: bad memoryBandwidthGBs`);
    for (const k of ['fp32', 'bf16', 'fp8', 'int8', 'fp4'])
      if (!g.tflops || !(k in g.tflops) || !isNumOrNull(g.tflops[k])) errors.push(`${where}: bad tflops.${k}`);
    if (!['nvlink', 'pcie', 'infinity-fabric'].includes(g.interconnect?.type)) errors.push(`${where}: bad interconnect.type`);
    if (!isNumOrNull(g.interconnect?.bandwidthGBs)) errors.push(`${where}: bad interconnect.bandwidthGBs`);
    if (!isNumOrNull(g.tdpW)) errors.push(`${where}: bad tdpW`);
    if (typeof g.fp8 !== 'boolean' || typeof g.fp4 !== 'boolean') errors.push(`${where}: fp8/fp4 must be boolean`);
    if (g.fp8 !== (g.tflops?.fp8 != null) && g.tflops?.fp8 != null) errors.push(`${where}: fp8 flag inconsistent with tflops.fp8`);
    if (g.fp4 !== (g.tflops?.fp4 != null) && g.tflops?.fp4 != null) errors.push(`${where}: fp4 flag inconsistent with tflops.fp4`);
    if (!Array.isArray(g.sources) || !g.sources.length) errors.push(`${where}: sources required`);
  }

  if (!/^\d{4}-\d{2}-\d{2}$/.test(pricing?.updatedAt ?? '')) errors.push('pricing.json: bad updatedAt');
  if (pricing?.currency !== 'USD') errors.push('pricing.json: currency must be USD');
  const regionKeys = new Set((pricing?.regions ?? []).map((r) => `${r.provider}|${r.id}`));
  if (!pricing?.offers?.length) errors.push('pricing.json: offers must be non-empty');
  const seen = new Set();
  for (const o of pricing?.offers ?? []) {
    const where = `offer ${offerKey(o)}`;
    if (!['aws', 'gcp'].includes(o.provider)) errors.push(`${where}: bad provider`);
    if (!gpuIds.has(o.gpuId)) errors.push(`${where}: unknown gpuId ${o.gpuId}`);
    if (!Number.isInteger(o.gpuCount) || o.gpuCount < 1) errors.push(`${where}: gpuCount must be a positive integer`);
    if (!regionKeys.has(`${o.provider}|${o.region}`)) errors.push(`${where}: region not declared in regions[]`);
    if (seen.has(offerKey(o))) errors.push(`${where}: duplicate`);
    seen.add(offerKey(o));
    for (const k of ['vcpus', 'ramGB', 'onDemandHourly', 'spotHourly', 'commit1yHourly'])
      if (!isNumOrNull(o[k])) errors.push(`${where}: bad ${k}=${o[k]}`);
    if (o.onDemandHourly == null && o.spotHourly == null && o.commit1yHourly == null) errors.push(`${where}: no price at all`);
  }
  return errors;
}

function summarize(pricing) {
  const counts = {};
  for (const o of pricing.offers) counts[`${o.provider} ${o.region}`] = (counts[`${o.provider} ${o.region}`] ?? 0) + 1;
  return Object.entries(counts)
    .map(([k, v]) => `  ${k.padEnd(22)} ${v}`)
    .join('\n');
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function readJson(url) {
  try {
    return JSON.parse(await readFile(url, 'utf8'));
  } catch {
    return null;
  }
}

async function main() {
  const args = new Set(process.argv.slice(2));
  const gpus = await readJson(GPUS_PATH);

  if (args.has('--check')) {
    const pricing = await readJson(PRICING_PATH);
    const errors = pricing && gpus ? validate(pricing, gpus) : ['pricing.json or gpus.json missing / not valid JSON'];
    if (errors.length) {
      console.error(errors.map((e) => `ERROR ${e}`).join('\n'));
      process.exit(1);
    }
    info(`OK: ${gpus.length} GPUs, ${pricing.offers.length} offers (updatedAt ${pricing.updatedAt})\n${summarize(pricing)}`);
    return;
  }

  const prev = await readJson(PRICING_PATH);
  const prevOffers = prev?.offers ?? [];
  const prevByKey = new Map(prevOffers.map((o) => [offerKey(o), o]));

  const [aws, gcp] = await Promise.all([buildAws(prevByKey), buildGcp(prevOffers)]);
  const okGroups = new Set([...aws.okGroups, ...gcp.okGroups]);
  const groupOf = (o) => (o.provider === 'aws' ? `aws:${o.region}` : gcpGroupOf(o.instanceType));
  const groupOk = (o) => okGroups.has(groupOf(o));

  const fresh = [...aws.offers, ...gcp.offers];
  const freshKeys = new Set(fresh.map(offerKey));
  const carried = prevOffers.filter((o) => (!groupOk(o) || gcp.extraCarry.has(offerKey(o))) && !freshKeys.has(offerKey(o)));
  if (carried.length) warn(`carried over ${carried.length} offers from the previous pricing.json`);

  const offers = [...fresh, ...carried].sort(
    (a, b) =>
      a.provider.localeCompare(b.provider) ||
      a.region.localeCompare(b.region) ||
      a.gpuId.localeCompare(b.gpuId) ||
      a.gpuCount - b.gpuCount ||
      (a.onDemandHourly ?? a.spotHourly ?? 0) - (b.onDemandHourly ?? b.spotHourly ?? 0) ||
      a.instanceType.localeCompare(b.instanceType),
  );

  if (!fresh.length) {
    console.error('ERROR every pricing source failed; leaving pricing.json untouched.');
    process.exit(1);
  }

  // Keep source notes for providers that were entirely carried over.
  const sources = [...aws.sources, ...gcp.sources];
  for (const p of ['aws', 'gcp'])
    if (!sources.some((s) => s.provider === p)) sources.push(...(prev?.sources ?? []).filter((s) => s.provider === p));

  const next = {
    updatedAt: new Date().toISOString().slice(0, 10),
    currency: 'USD',
    sources,
    regions: REGIONS,
    offers,
  };

  const errors = gpus ? validate(next, gpus) : ['gpus.json missing'];
  if (errors.length) {
    console.error(errors.map((e) => `ERROR ${e}`).join('\n'));
    console.error('Refusing to write an invalid pricing.json.');
    process.exit(1);
  }

  info(`\n${offers.length} offers (${fresh.length} fresh, ${carried.length} carried over)\n${summarize(next)}`);
  if (warnings.length) info(`\n${warnings.length} warning(s); see WARN lines above.`);

  const unchanged = prev && JSON.stringify({ ...prev, updatedAt: '' }) === JSON.stringify({ ...next, updatedAt: '' });
  if (args.has('--dry-run')) return info('\n--dry-run: not writing pricing.json');
  if (unchanged) return info('\nNo price changes; pricing.json left untouched.');
  await writeFile(PRICING_PATH, `${JSON.stringify(next, null, 2)}\n`);
  info(`\nWrote ${PRICING_PATH.pathname}`);
}

main().catch((err) => {
  console.error(`ERROR ${err.stack ?? err}`);
  process.exit(1);
});
