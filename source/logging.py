# Copyright 2021 Tristan Behrens.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import logging

loggers_dict = {}

def create_logger(name:str):
    global loggers_dict
    if name in loggers_dict:
        return loggers_dict[name]
    else:  
        logger = logging.getLogger(name)
        loggers_dict[name] = logger
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

def set_log_level(name, level):
    logger_names = []
    if name == "all":
        logger_names = list(loggers_dict.keys())
    else:
        logger_names = [name]
    for name in logger_names:
        logger = loggers_dict[name]
        logger.setLevel(level)
