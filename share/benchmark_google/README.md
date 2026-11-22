# Google Benchmark Setup (`share/benchmark_google`)

Dieses Verzeichnis enthält reproduzierbare Performance-Benchmarks für `nm`.

## Targets

- `//share/benchmark_google:bench_nm_relu`
- `//share/benchmark_google:bench_nm_mnist`
- `//share/benchmark_google:bench_nm_seq_conv`

## Runner-Skript

`run_benchmarks.sh` baut die Binaries in `-c opt` und schreibt pro Case:

- `*.json` (maschinenlesbar)
- `*.csv` (Tabellen/Spreadsheet)
- `*.console.txt` (Konsolenoutput)

Ergebnisse landen unter:

- `share/benchmark_google/results/<timestamp>/...`

Zusätzlich werden zentrale Verlaufdateien gepflegt:

- `share/benchmark_google/history/benchmark_history.csv`
- `share/benchmark_google/history/benchmark_history.jsonl`

Und (optional, wenn `matplotlib` installiert ist) Trendplots:

- `share/benchmark_google/plots/trend_*.png`

## Nutzung

Quick-Run (empfohlen für lokale Checks):

- `./share/benchmark_google/run_benchmarks.sh`

Full-Run mit längerer Messdauer:

- `./share/benchmark_google/run_benchmarks.sh full 1.0s`

Retention konfigurieren (verhindert Dateiflut):

- `KEEP_RUNS=30 ./share/benchmark_google/run_benchmarks.sh`

## Hinweise

- Für stabile Vergleiche möglichst gleiche Lastbedingungen verwenden (keine parallelen Heavy-Tasks).
- Trends primär über `history/benchmark_history.csv` vergleichen (nicht über tausende Einzeldateien).
- Rohdaten unter `results/<timestamp>` als Diagnose nutzen; historische Analyse über die aggregierte History.
- Für robuste Aussagen mehrere Läufe pro Commit machen und Median/Streuung betrachten.

## Schlanker Best-Practice-Workflow

1. Lauf starten (`quick` lokal, `full` für Release-Prüfung)
2. Aggregierte History wächst automatisch (append-only)
3. Trendplots werden aktualisiert (wenn `matplotlib` verfügbar)
4. Alte Run-Ordner werden automatisch auf `KEEP_RUNS` begrenzt

So bleibt der Speicherverbrauch kontrolliert, während Performance-Trends langfristig auswertbar bleiben.
