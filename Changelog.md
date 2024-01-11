# Changelog

## v1.0.0 - 2024-01-11
- Initial public release
- Runs epubcheck in parallel, one process per CPU core
- Outputs each file as it is processed, removing the line when the file is done
- Basic progress bar shows count and percentage of files processed
- History is retained so that if a file has already been verified, it won't be verified again unless the `--force` option is used
- Written using epubcheck v5.1.0 but the underlying version shouldn't matter so long as it provides the same command line interface and exit statuses
