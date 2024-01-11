# EpubCheckMate

Pretty parallel processing of epub files using epubcheck.

EpubCheckMate is a fully self-contained script with no external dependencies, making it incredibly easy to use and deploy.

## Installation

EpubCheckMate doesn't require any installation steps due to its self-contained nature. Simply download the script and you're ready to go. Follow these steps:

1. Download the `epubcheckmate.py` file directly or clone the repository: `git clone https://github.com/YodaDaCoda/EpubCheckMate`
2. Navigate to the directory containing `epubcheckmate.py`: `cd EpubCheckMate`
3. (Optional) For easier access, move `epubcheckmate.py` to a directory in your `PATH`.  
For example, if you have a `bin` directory in your home directory that's included in your `PATH`, you can do: `mv epubcheckmate.py ~/bin/epubcheckmate`.  
Now you can run EpubCheckMate directly from your terminal anywhere by typing `epubcheckmate`.

## Usage

To use EpubCheckMate, run the following command:

```bash
epubcheckmate [options]
```

Options:

- `-h, --help`: Show the help message and exit.
- `-v, --version`: Show program's version number and exit.
- `-d, --directory`: Specifies the directory (or individual file) to verify. Can be specified multiple times. Defaults to the current working directory.
- `-t, --threads`: Specifies the number of worker threads to use. Defaults to the number of CPU cores.
- `-f, --force`: Ignores the history and forces verification.
