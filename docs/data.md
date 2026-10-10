# `quanttro.data`

The `quanttro.data` module contains the functions for fetching data for BIST
(Borsa Istanbul) stocks, macro indicators (such as USD/TRY), and general
(crypto/gold/FX) yfinance tickers, along with file-based caching and basic
data quality validation. The typical sequence when starting a
backtest/research workflow is as follows:

1. Use `resolve_symbols` to convert the symbol set you want to work with
   (BIST30, BIST100, or your own list) into actual yfinance tickers.
2. Use `fetch_ohlcv` (or `fetch_any` for a single symbol) to fetch the OHLCV
   data - the results are automatically cached on disk (see `cache.py`), so
   the same data isn't repeatedly pulled from the internet.
3. Use `validation_report` to scan the fetched data for anomalies (price
   jumps, frozen prices, zero volume) - this step is recommended before
   starting any backtest.

High-level shortcut: `import quanttro as bq; bq.fetch("THYAO.IS")` or
`bq.fetch("BIST30")` combines the most commonly used paths of this module
into a single function (see `quanttro/__init__.py`).

---

## Symbol selection (`quanttro.data.symbols`)

### `BIST30`

The yfinance ticker list (with the `.IS` suffix) of BIST's 30 most liquid
stocks. Identical, unchanged, to the list in the original `symbols.py` at
the project root.

**Type:** `list[str]`, 30 elements.

**Example:**
```python
from quanttro.data import symbols
print(len(symbols.BIST30))
print(symbols.BIST30[:5])
```

**Output:**
```
30
['AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'ASTOR.IS', 'BIMAS.IS']
```

### `BIST100`

The deduplicated, alphabetically sorted combination of `BIST30` with
additional BIST100 stocks that stand out in terms of market
cap/trading volume.

**Type:** `list[str]`, 94 elements (also includes `BIST30`).

**Note:** Index compositions change over time; this list is a static
approximation of "well-known, liquid BIST stocks" - for the official/current
index composition, refer to BIST/KAP sources.

**Example:**
```python
from quanttro.data import symbols
print(len(symbols.BIST100))
```

**Output:**
```
94
```

### `SymbolSet`

An `Enum` for the predefined symbol sets. Values:
- `SymbolSet.BIST30` (value: `"bist30"`)
- `SymbolSet.BIST100` (value: `"bist100"`)
- `SymbolSet.CUSTOM` (value: `"custom"`) - a marker with no predefined list,
  indicating that the user is passing a raw ticker list directly to
  `resolve_symbols` (there is no separate `_SYMBOL_SETS` entry for it).

### `resolve_symbols(selector)`

Converts a symbol selection from the user into an actual ticker list.

**Parameters:**
- `selector` (`SymbolSet | str | list[str]`): can be provided as:
  - a `SymbolSet.BIST30` / `SymbolSet.BIST100` enum value
  - a case-insensitive abbreviation string such as `"BIST30"` / `"bist100"`
  - the user's own custom ticker list (e.g. `["PAXG-USD", "THYAO.IS"]`)
    - in this case the list is returned as-is (copied), which also supports
      non-BIST tickers (gold/crypto/FX).

**Returns:** `list[str]` - the resolved ticker list.

**Errors:** Raises `ValueError` if given an unknown `SymbolSet` or an
unrecognized string (lists the available options in the error message).

**Example:**
```python
from quanttro.data import symbols

r1 = symbols.resolve_symbols(symbols.SymbolSet.BIST30)
r2 = symbols.resolve_symbols("bist100")
r3 = symbols.resolve_symbols(["PAXG-USD", "THYAO.IS"])
print(len(r1), r1[:5])
print(len(r2), r2[:5])
print(r3)
```

**Output:**
```
30 ['AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'ASTOR.IS', 'BIMAS.IS']
94 ['AEFES.IS', 'AGHOL.IS', 'AKBNK.IS', 'AKSA.IS', 'AKSEN.IS']
['PAXG-USD', 'THYAO.IS']
```

---

## Data fetching (`quanttro.data.fetch`)

All fetch functions use the `cache.py` layer below: if the same
`(ticker, interval, period)` combination is requested again within the
TTL period (24 hours by default), it is not pulled from the network but
read from disk instead.

### `TIMEFRAMES`

Dictionary of predefined timeframe definitions (from the original
`fetch_data.py` at the project root):

```python
{
    "daily": {"interval": "1d", "period": "max"},
    "hourly": {"interval": "60m", "period": "730d"},
    "5min": {"interval": "5m", "period": "60d"},
}
```

### `DEFAULT_MACRO_TICKERS`

`["USDTRY=X"]` - the default ticker list for `fetch_macro`.

### `fetch_any(ticker, interval="1d", period="max", auto_adjust=True, cache_config=None)`

Fetches any yfinance ticker (stock, FX, crypto, commodity, ...). Does not
perform any BIST-specific preprocessing; wraps `yfinance.download`, returns
`None` for empty results, and flattens multi-index columns.

**Parameters:**
- `ticker` (str): yfinance ticker (example: `"THYAO.IS"`, `"PAXG-USD"`, `"XAUUSD=X"`)
- `interval` (str): bar interval, default `"1d"`
- `period` (str): lookback range, default `"max"`
- `auto_adjust` (bool): yfinance's automatic price adjustment, default `True`
- `cache_config` (`CacheConfig | None`): custom cache config; if `None`,
  `DEFAULT_CACHE_CONFIG` is used

**Returns:** `pd.DataFrame | None` - `None` if there is no data (empty result).

**Example:** (runs without waiting on the network since THYAO.IS is already
in the local cache)
```python
from quanttro.data import fetch

df = fetch.fetch_any("THYAO.IS", interval="1d", period="max")
print(type(df), df.shape)
print(df.tail(3))
```

**Output:**
```
<class 'pandas.DataFrame'> (6792, 5)
Price        Close    High     Low    Open    Volume
Date                                                
2026-10-06  290.50  294.25  290.50  293.25  27302874
2026-10-07  286.25  290.50  283.75  290.00  27077918
2026-10-08  284.25  287.50  282.00  286.25  20491198
```

### `fetch_ohlcv(symbols, interval="1d", period="max", cache_config=None, sleep_seconds=0.3)`

Fetches OHLCV data for a list of symbols (calls `fetch_any` sequentially
for each symbol).

**Parameters:**
- `symbols` (`list[str]`): ticker list (see `resolve_symbols`)
- `interval` (str): default `"1d"`
- `period` (str): default `"max"`
- `cache_config` (`CacheConfig | None`): default `None` (-> `DEFAULT_CACHE_CONFIG`)
- `sleep_seconds` (float): delay (in seconds) between consecutive requests,
  default `0.3` - if the data comes from the cache, this delay effectively
  has no effect; it only applies between actual network requests

**Returns:** `dict[str, pd.DataFrame]` - `{symbol: DataFrame}`; symbols
whose data could not be fetched (empty result) are excluded from the
result and `"[interval] symbol: no data, skipped"` is printed to the console.

**Example:**
```python
from quanttro.data import fetch

data = fetch.fetch_ohlcv(["THYAO.IS", "AKBNK.IS"], interval="1d", period="max")
print(list(data.keys()))
print(data["THYAO.IS"].shape)
```

**Output:**
```
['THYAO.IS', 'AKBNK.IS']
(6792, 5)
```

### `fetch_macro(tickers=None, interval="1d", period="max", cache_config=None)`

Fetches data for macro indicator(s). If `tickers` is not provided,
`DEFAULT_MACRO_TICKERS` (`["USDTRY=X"]`) is used. Internally calls
`fetch_ohlcv`.

**Parameters:**
- `tickers` (`list[str] | None`): macro ticker list, default `None`
- `interval` (str): default `"1d"`
- `period` (str): default `"max"`
- `cache_config` (`CacheConfig | None`): default `None`

**Returns:** `dict[str, pd.DataFrame]`

**Example:**
```python
from quanttro.data import fetch

macro = fetch.fetch_macro()
print(list(macro.keys()))
print(macro["USDTRY=X"].tail(2))
```

**Output:**
```
['USDTRY=X']
Price           Close       High        Low       Open  Volume
Date                                                          
2026-10-07  49.195599  49.197392  49.180901  49.192600       0
2026-10-08  49.212101  49.216202  49.198898  49.213001       0
```

### `fetch_tefas_fund(fund_code, start=None, end=None)`

A placeholder for fetching TEFAS (Turkey Electronic Fund Trading Platform)
fund data. **Not yet implemented** - since TEFAS has no official yfinance
source, this function always raises `NotImplementedError` until a real
integration (TEFAS's own web service) is written. It does not produce
fabricated/fake data.

**Parameters:**
- `fund_code` (str): TEFAS fund code
- `start` (`str | None`), `end` (`str | None`): date range (not yet used)

**Returns:** `pd.DataFrame` (for documentation purposes only - never
actually returns)

**Example:**
```python
from quanttro.data import fetch

try:
    fetch.fetch_tefas_fund("ABC")
except NotImplementedError as e:
    print("NotImplementedError:", e)
```

**Output:**
```
NotImplementedError: fetch_tefas_fund('ABC') is not yet implemented. A separate scraper/API integration is required for TEFAS fund data (see this function's docstring). This error was left in place instead of producing fake/fabricated data.
```

### `save_ohlcv(data, out_dir)`

Writes the fetched `{symbol: DataFrame}` mapping to disk as CSV files.
File names are generated by stripping the yfinance suffixes (`.IS`, `=X`)
(example: `"THYAO.IS"` -> `THYAO.csv`, `"USDTRY=X"` -> `USDTRY.csv`).

**Parameters:**
- `data` (`dict[str, pd.DataFrame]`): the symbol -> DataFrame mapping to write
- `out_dir` (`Path | str`): output directory (created if it doesn't exist)

**Returns:** `list[Path]` - the paths of the files written.

**Example:**
```python
import tempfile
from quanttro.data import fetch

df = fetch.fetch_any("THYAO.IS")
tmpdir = tempfile.mkdtemp()
paths = fetch.save_ohlcv({"THYAO.IS": df.tail(5)}, tmpdir)
print(paths)
```

**Output:**
```
[PosixPath('/var/folders/.../tmp.../THYAO.csv')]
```
(The file contents are the output of `DataFrame.to_csv` with the standard
`Date,Close,High,Low,Open,Volume` header.)

---

## Cache layer (`quanttro.data.cache`)

A simple file-based (parquet) cache. When the same `(namespace, identifier,
interval, period)` combination is requested again, it is read from disk if
there is a file fresher than the cache TTL; otherwise `None` is returned and
the caller (`fetch.py`) re-fetches the data and writes it to the cache.

### `DEFAULT_TTL_SECONDS`

`24 * 60 * 60` (86400) - the default cache freshness duration (in seconds).

### `CacheConfig`

A `dataclass` that configures cache behavior.

**Fields:**
- `cache_dir` (`Path`): the directory where parquet files will be written,
  default `Path.home() / ".cache" / "quanttro"`
- `ttl_seconds` (`float`): the maximum age for a cache entry to be
  considered "fresh", default `DEFAULT_TTL_SECONDS` (86400, i.e. 24 hours)
- `enabled` (`bool`): if `False`, the cache is entirely disabled (always
  re-fetched, nothing is written to disk), default `True`

**Example:**
```python
from quanttro.data.cache import DEFAULT_CACHE_CONFIG
print(DEFAULT_CACHE_CONFIG)
```

**Output:**
```
CacheConfig(cache_dir=PosixPath('/Users/yagizefe/.cache/quanttro'), ttl_seconds=86400, enabled=True)
```

### `DEFAULT_CACHE_CONFIG`

The default global configuration instance, created with `CacheConfig()`
(the value seen in the example above).

### `read_cache(namespace, identifier, interval, period, config=DEFAULT_CACHE_CONFIG)`

Returns a `DataFrame` if there is a fresh entry in the cache, otherwise
`None`.

**Parameters:**
- `namespace` (str): logical namespace (example: `"raw"`)
- `identifier` (str): the example ticker/symbol name
- `interval` (str), `period` (str): timeframe parameters
- `config` (`CacheConfig`): default `DEFAULT_CACHE_CONFIG`

**Returns:** `pd.DataFrame | None`. Returns `None` if the file doesn't
exist, is older than the TTL, or is corrupted (unreadable).

### `write_cache(df, namespace, identifier, interval, period, config=DEFAULT_CACHE_CONFIG)`

Writes the `DataFrame` as parquet to the cache directory.

**Parameters:** same as `read_cache` + `df` (`pd.DataFrame`): the data to write.

**Returns:** `Path | None` - the path of the file written; if
`config.enabled=False`, writes nothing and returns `None`.

### `clear_cache(config=DEFAULT_CACHE_CONFIG)`

Deletes all `.parquet` files in the cache directory.

**Parameters:**
- `config` (`CacheConfig`): default `DEFAULT_CACHE_CONFIG`

**Returns:** `int` - the number of files deleted.

**Example:** (read/write/clear roundtrip, using a temporary directory)
```python
import tempfile
import pandas as pd
from quanttro.data import cache

tmpconfig = cache.CacheConfig(cache_dir=tempfile.mkdtemp())
test_df = pd.DataFrame({"a": [1, 2, 3]})

p = cache.write_cache(test_df, "ns", "ID1", "1d", "max", tmpconfig)
print("written:", p)

back = cache.read_cache("ns", "ID1", "1d", "max", tmpconfig)
print(back)

n = cache.clear_cache(tmpconfig)
print("deleted:", n)
```

**Output:**
```
written: /var/folders/.../tmp.../ns__ID1__1d_max.parquet
   a
0  1
1  2
2  3
deleted: 1
```

---

## Data quality validation (`quanttro.data.validate`)

The checks in this module are based on REAL anomalies encountered in BIST
OHLCV data (via yfinance):

- Around 2005-01-03, THYAO shows a massive price jump relative to the
  previous bar - not an actual corporate action, but rather caused by
  yfinance incorrectly reflecting Turkey's 2005 TL -> YTL redenomination
  (the removal of six zeros) on the stock. `detect_price_jumps` was written
  to automatically catch anomalies of this kind.
- Low-liquidity stocks may show consecutive zero-volume bars (no trading)
  -> `detect_zero_volume_runs`.
- There can be periods where the data provider returns the same closing
  price repeatedly (frozen/stale price) -> `detect_stale_prices`.

All functions automatically locate the `Close`/`close` and (where needed)
`Volume`/`volume` columns in `df`; if neither is found, a `KeyError` is
raised.

### `detect_price_jumps(df, threshold=0.5)`

Flags rows where `|return| >= threshold` relative to the previous bar.
Direction (up/down) does not matter.

**Parameters:**
- `df` (`pd.DataFrame`): OHLCV data containing at least a `Close`/`close` column
- `threshold` (float): absolute percentage return threshold, default `0.5` (50%)

**Returns:** `pd.DataFrame` - only the flagged rows, with a `pct_change`
column added on top of the original columns.

**Example:** (full THYAO history, catches the redenomination anomaly
mentioned in the source code)
```python
from quanttro.data import fetch, validate

df = fetch.fetch_any("THYAO.IS", interval="1d", period="max")
jumps = validate.detect_price_jumps(df)
print(jumps.shape)
print(jumps)
```

**Output:**
```
(2, 6)
Price          Close     High       Low      Open      Volume  pct_change
Date                                                                     
2000-10-17  0.001684  0.00174  0.001626  0.001740  4249991597    0.520325
2005-01-03  0.901924  0.91334  0.896215  0.901924     3453812  981.131324
```
(The 2005-01-03 row is the redenomination error mentioned in the
docstring - a ~981x jump relative to the previous bar.)

### `detect_zero_volume_runs(df, min_run=3)`

Flags consecutive zero-volume bar sequences of length `min_run` or longer.
Missing (`NaN`) volume is also treated as zero volume.

**Parameters:**
- `df` (`pd.DataFrame`): data containing at least a `Volume`/`volume` column
- `min_run` (int): minimum number of consecutive zero-volume bars to flag, default `3`

**Returns:** `pd.DataFrame` - the flagged rows, with `run_id` and
`run_length` columns added.

**Example:**
```python
from quanttro.data import fetch, validate

df = fetch.fetch_any("THYAO.IS", interval="1d", period="max")
zv = validate.detect_zero_volume_runs(df)
print(zv.shape)
```

**Output:**
```
(130, 7)
```
(Across THYAO's full history, there are 31 distinct zero-volume runs of at
least 3 bars in length, totaling 130 rows.)

### `detect_stale_prices(df, min_run=5)`

Flags consecutive identical-closing-price sequences of length `min_run` or
longer.

**Parameters:**
- `df` (`pd.DataFrame`): data containing at least a `Close`/`close` column
- `min_run` (int): minimum number of consecutive constant-price bars to flag, default `5`

**Returns:** `pd.DataFrame` - the flagged rows, with `run_id` and
`run_length` columns added.

**Example:**
```python
from quanttro.data import fetch, validate

df = fetch.fetch_any("THYAO.IS", interval="1d", period="max")
sp = validate.detect_stale_prices(df)
print(sp.shape)
```

**Output:**
```
(168, 7)
```

### `validation_report(df, jump_threshold=0.5, zero_volume_min_run=3, stale_min_run=5)`

Combines all three checks (`detect_price_jumps`, `detect_zero_volume_runs`,
`detect_stale_prices`) into a single summary dictionary.

**Parameters:**
- `df` (`pd.DataFrame`): OHLCV data to validate
- `jump_threshold` (float): threshold for `detect_price_jumps`, default `0.5`
- `zero_volume_min_run` (int): `min_run` for `detect_zero_volume_runs`, default `3`
- `stale_min_run` (int): `min_run` for `detect_stale_prices`, default `5`

**Returns:** `dict` - schema:
```python
{
    "n_rows": int,
    "price_jumps": {"count": int, "dates": [str, ...]},
    "zero_volume_runs": {"count": int, "n_runs": int, "dates": [str, ...]},
    "stale_price_runs": {"count": int, "n_runs": int, "dates": [str, ...]},
    "is_clean": bool,  # True if there are no anomalies in any of the three
}
```

**Example:**
```python
from quanttro.data import fetch, validate

df = fetch.fetch_any("THYAO.IS", interval="1d", period="max")
rep = validate.validation_report(df)
print(rep["n_rows"], rep["is_clean"])
print(rep["price_jumps"]["count"], rep["price_jumps"]["dates"][:3])
```

**Output:**
```
6792 False
2 ['2000-10-17 00:00:00', '2005-01-03 00:00:00']
```

---

## High-level shortcut: `bq.fetch`

The `fetch(symbols, interval="1d", period="max", **kwargs)` function in
`quanttro.__init__` wraps the `resolve_symbols` + `fetch_any`/`fetch_ohlcv`
pairing above into a single call:

- `bq.fetch("THYAO.IS")` -> a single `DataFrame` (`None` if there is no data)
- `bq.fetch("PAXG-USD")` -> any ticker recognized by yfinance
- `bq.fetch(["THYAO.IS", "AKBNK.IS"])` -> a `{ticker: DataFrame}` dictionary
- `bq.fetch("BIST30")` / `bq.fetch("BIST100")` -> a dictionary for all the respective stocks

**Example:**
```python
import quanttro as bq

df = bq.fetch("THYAO.IS")
print(df.shape)
```

**Output:**
```
(6792, 5)
```

For finer control (cache settings, `sleep_seconds`, etc.), it is recommended
to use `quanttro.data.fetch_any` / `fetch_ohlcv` directly.
