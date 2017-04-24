# -*- coding: utf-8 -*-

# Configuration file for gunicorn

import multiprocessing
bind = "0.0.0.0:9000"
#workers = multiprocessing.cpu_count() * 2 + 1
# only one worker or we will run out of memory
# TODO: is it possible to share memory between processes
workers = 1
log_level = "DEBUG"
log_file = ""
timeout = 3000
graceful_timeout = 3000
