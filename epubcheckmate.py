#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EpubCheckMate - pretty parallel processing of epub files using epubcheck
Copyright (C) 2023  William Brockhus
"""

__app_name__ = "EpubCheckMate"
__version__ = "1.0.0"
__author__ = "William Brockhus"
__license__ = "GPL-3.0-or-later"
__copyright__ = """
EpubCheckMate - pretty parallel processing of epub files using epubcheck
Copyright (C) 2023  William Brockhus

This file is part of EpubCheckMate.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import sys
import argparse
import re
import subprocess
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from pathlib import Path
import shlex
import shutil
import textwrap
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod


if os.name == "nt":  # Windows
    XDG_CONFIG_HOME = os.environ.get(
        "XDG_CONFIG_HOME", os.path.join(os.environ["APPDATA"])
    )
else:  # Unix/Linux/Mac
    XDG_CONFIG_HOME = os.environ.get(
        "XDG_CONFIG_HOME", os.path.join(os.environ["HOME"], ".config")
    )

APP_CONFIG_HOME = os.path.join(XDG_CONFIG_HOME, __app_name__)
HISTORY_FILE = os.path.join(APP_CONFIG_HOME, "history.json")


class Task:
    def __init__(self, file, success=None, output=None, file_hash=None):
        self.file = file
        self.success = success
        self.output = output
        self.file_hash = file_hash

    def clone(self):
        return Task(self.file, self.success, self.output, self.file_hash)


class Progress:
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.completed_tasks = 0


class HistoryManagerInterface(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def add(self, file, file_hash):
        pass

    @abstractmethod
    def contains(self, file):
        pass

    @abstractmethod
    def remove(self, file):
        pass


class HistoryManager(HistoryManagerInterface):
    def __init__(self):
        self.load()

    def load(self):
        self.history = {}
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                self.history = json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f)

    def add(self, file, file_hash):
        file = str(file.resolve())  # Convert the file path to absolute path
        if file_hash is None:
            file_hash = get_file_hash(file)
        self.history[file] = file_hash
        self.save()

    def remove(self, file):
        self.history.pop(file, None)
        self.save()

    def contains(self, file):
        file = str(file.resolve())  # Convert the file path to absolute path
        result = file in self.history
        return result

    def matches(self, file, file_hash):
        file = str(file.resolve())  # Convert the file path to absolute path
        result = self.history.get(file, None) == file_hash
        return result


class NoOpHistoryManager(HistoryManagerInterface):
    def load(self):
        pass

    def save(self):
        pass

    def add(self, file, file_hash):
        pass

    def contains(self, file):
        return False

    def remove(self, file):
        pass

    def matches(self, file, file_hash):
        return False


def get_file_hash(file):
    hasher = hashlib.sha256()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    hash_value = hasher.hexdigest()
    return hash_value


def verify(task, should_terminate):
    process = subprocess.Popen(
        ["epubcheck", "--failonwarnings", task.file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    while True:
        if should_terminate.is_set():
            process.terminate()
            return

        if process.poll() is not None:  # If the process has finished
            break

        time.sleep(0.1)  # Sleep for a bit to avoid busy waiting

    task.success = process.returncode == 0
    task.output = process.communicate()[0]


def verify_wrapper(queue, file, history_manager, should_terminate):
    task = Task(file)

    # put the task in the queue for the message handler to pick up
    queue.put(task.clone())

    # TODO: add a way for the main thread to signal that we should die

    # early exit if file has already been verified
    if history_manager.contains(file):
        task.file_hash = get_file_hash(file)
        if history_manager.matches(file, task.file_hash):
            task.success = True
            task.output = "File already verified."
            queue.put(task.clone())
            return task

    # verify the file
    verify(task, should_terminate)

    # put the task back in the queue for the message handler to handle status
    queue.put(task.clone())

    # return the task so the main loop can handle the future result
    return task


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def get_files(directory_or_file_list):
    files = []
    visited_paths = set()

    for path_str in directory_or_file_list:
        path = Path(path_str)

        if (
            path in visited_paths
        ):  # Skip processing if path already exists in visited_paths
            continue
        visited_paths.add(path)

        if path.is_dir():
            files.extend(
                sorted(path.rglob("*.epub"), key=lambda x: natural_keys(str(x)))
            )
        elif path.is_file() and path.suffix == ".epub":
            files.append(path)

    if not files:
        print("No files found.")
        sys.exit(1)

    return files


class ConsoleLineManager:
    def __init__(self, total_lines):
        self.total_lines = total_lines
        self.current_line = total_lines

    def init(self):
        # Print out all the lines
        for _ in range(self.total_lines):
            print("")

    def reset(self):
        # move the cursor back to the bottom
        self.move_cursor(self.total_lines - self.current_line)
        # ensure the bottom line is cleared
        self.clear_current_line()
        # walk up, clearing each line
        for _ in range(self.total_lines):
            self.move_cursor(-1)
            self.clear_current_line()

    def move_cursor(self, lines):
        # print(f"moving cursor {abs(lines)} lines {'up' if lines < 0 else 'down'}", end="")
        if lines < 0:
            # Move cursor up
            sys.stdout.write(f"\033[{-lines}A")
        elif lines > 0:
            # Move cursor down
            sys.stdout.write(f"\033[{lines}B")
        self.current_line += lines
        sys.stdout.flush()  # Ensure the output is displayed immediately

    def move_to_line(self, line):
        # Move cursor up or down to the provided line
        self.move_cursor(line - self.current_line)

    def clear_current_line(self):
        # Move cursor to the start of the line
        sys.stdout.write("\r")
        # Clear the line
        sys.stdout.write("\033[K")
        sys.stdout.flush()  # Ensure the output is displayed immediately

    def write_to_current_line(self, message):
        # Get the current width of the terminal
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        # Truncate the message if it's longer than the terminal width
        truncated_message = textwrap.shorten(
            message, width=terminal_width, placeholder="…"
        )
        # Clear the line
        self.clear_current_line()
        # Print the truncated message
        sys.stdout.write(truncated_message)
        # Move cursor to the start of the line
        sys.stdout.write("\r")
        # Ensure the output is displayed immediately
        sys.stdout.flush()

    def write_to_line(self, line, message):
        # Move cursor up to the line
        self.move_to_line(line)
        # Write the message to the line
        self.write_to_current_line(message)
        # Move cursor back to the bottom
        self.move_to_line(self.total_lines)

    def clear_line(self, line):
        self.move_to_line(line)
        self.clear_current_line()
        self.move_to_line(self.total_lines)


class MessageHandler:
    def __init__(self, max_messages, queue) -> None:
        self.max_messages = max_messages
        self.queue = queue
        self.lines = {}
        self.available_lines = list(range(max_messages))
        self.available_lines.reverse()
        self.console = ConsoleLineManager(max_messages)
        self.console.init()

    def handle_task_message(self, task):
        if task.success is None:  # If verification is not complete
            # Assign an available line to this file
            line = self.available_lines.pop()
            self.lines[str(task.file)] = line
            self.console.write_to_line(line, f"Checking '{task.file}'")
        elif task.success == True:  # If verification succeeded
            # we're about to remove a line from the console
            # as a result, all the below lines will be moved up by one
            # grab the last line before doing any of that
            last_line = max(self.lines.values())

            # Move cursor up to the line for this file and clear the line
            line = self.lines.pop(str(task.file))
            self.console.clear_line(line)

            # Shift all lines below the cleared line up by one
            line_items = self.lines.items()
            line_items = sorted(line_items, key=lambda x: x[1])
            line_items = filter(lambda x: x[1] > line, line_items)
            for file, file_line in line_items:
                self.lines[file] = file_line - 1
                self.console.clear_line(file_line)
                self.console.write_to_line(self.lines[file], f"Checking '{file}'")

            # Add the line back to the list of available lines
            self.available_lines.append(last_line)

        elif task.success == False:  # If verification failed
            # Clear all lines
            self.console.reset()
            # Print the stdout from the task
            print(task.output)
            # Print the epubcheck command that was run
            print(f"epubcheck --failonwarnings {shlex.quote(str(task.file))}")
            # Stop the thread

    def handle_progress_message(self, progress):
        # should be no need to move the cursor, except to ensure it's at the beginning of the line
        # just print out the progress bar

        # default max size of the progress bar
        # the entire bar, including the percentage, must fit within this width
        max_message_width = 50
        # shorten the message for a narrow terminal
        max_message_width = min(max_message_width, shutil.get_terminal_size().columns)

        # Calculate the number of characters in the progress bar
        extra_chars = 11  # 2 brackets, 1 space, 1 slash, 7 chars for the percentage
        progress_chars = len(str(progress.total_tasks)) * 2

        # Calculate the number of characters available for the progress bar
        bar_width = max_message_width - extra_chars - progress_chars

        # Calculate the progress percentage
        progress_percentage = progress.completed_tasks / progress.total_tasks

        # Calculate the number of '#' characters in the progress bar
        complete_width = int(bar_width * progress_percentage)
        incomplete_width = bar_width - complete_width

        self.console.write_to_current_line(
            f'[{"⣿" * complete_width}{"⣀" * incomplete_width}] {progress.completed_tasks}/{progress.total_tasks} {progress_percentage:.2%}'
        )

    def message_handler(self):
        while True:
            item = self.queue.get()

            if item is None:  # Sentinel value to stop the thread
                # Clear all lines
                self.console.reset()
                break

            if isinstance(item, Task):
                self.handle_task_message(item)
                if item.success == False:
                    break

            elif isinstance(item, Progress):
                self.handle_progress_message(item)


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"value must be a positive integer: '{value}'")
    return ivalue


def file_or_directory(value):
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(
            f"value must be a file or directory: '{value}'"
        )
    return value


class AppendOrReplaceAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # If the default value is currently set, replace it
        if getattr(namespace, self.dest) == parser.get_default(self.dest):
            setattr(namespace, self.dest, [values])
        else:  # Otherwise, append to the list
            getattr(namespace, self.dest).append(values)


def setup_parser():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Verify files in a directory.")
    # Add arguments
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f'%(prog)s v{__version__}  © {__author__}'
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=file_or_directory,
        action=AppendOrReplaceAction,
        default=["."],
        help="Directory (or individual file) to verify. Can be specified multiple times. Defaults to current working directory.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=positive_int,
        default=multiprocessing.cpu_count(),
        help=f"Number of worker threads to use. Defaults to number of cpu cores ({multiprocessing.cpu_count()}).",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="ignore the history and force verification",
    )
    return parser


def get_config():
    parser = setup_parser()
    config = parser.parse_args()
    return config


def main():
    config = get_config()

    files = get_files(config.directory)

    config.threads = min(config.threads, len(files))

    if config.threads < 1:
        print("No files found.")
        sys.exit(1)

    queue = Queue()

    # Start a separate thread to print messages
    message_wrapper = MessageHandler(config.threads, queue)
    print_thread = threading.Thread(target=message_wrapper.message_handler, args=())
    print_thread.start()

    # Create a Progress instance and put it in the queue
    progress = Progress(len(files))
    queue.put(progress)

    # retrieve the history config
    if config.force:
        history_manager = NoOpHistoryManager()
    else:
        history_manager = HistoryManager()

    # create a thread-safe event to signal that we should terminate
    should_terminate = threading.Event()

    # launch each verification in a thread
    with ThreadPoolExecutor(max_workers=config.threads) as executor:
        futures = {
            executor.submit(
                verify_wrapper, queue, file, history_manager, should_terminate
            )
            for file in files
        }

        # as each verify job completes, test for failure
        for future in as_completed(futures):
            task = future.result()
            if not task.success:
                # Set the flag to terminate the threads
                should_terminate.set()

                # wait for the threads to finish
                executor.shutdown(wait=False, cancel_futures=True)

                # remove this file from the history
                if history_manager.contains(task.file):
                    history_manager.remove(file)

                break

            # add it to the history
            history_manager.add(task.file, task.file_hash)

            # update the progress & put it in the queue
            progress.completed_tasks += 1
            queue.put(progress)

    # stop the print thread gracefully, allowing it to clear its output and avoid a race condition
    queue.put(None)
    print_thread.join()

    if not task.success:
        sys.exit(1)

    print("All files verified successfully.")


if __name__ == "__main__":
    main()
