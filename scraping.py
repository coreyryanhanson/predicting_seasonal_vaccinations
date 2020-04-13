#!/usr/bin/env python

import spotipy
import spotipy.oauth2 as oauth2
import config
import pandas as pd
import time
import json
import numpy as np
import requests
import functools
import pickle
from itertools import product

class API_Caller(object):
    """"Object to construct loops of API calls, tracking errors and storing the results within the object which can
    be automatically backed up to a pickle file."""

    def __init__(self, search_keywords, filename, limit=50, seed=None):
        """Initiates the API caller object. It takes a list of search term choices and a filename to save backups to,
         Limit caps the amount of results returned in a single search, while market controls the markets parameter."""

        self.wait_min, self.wait_max = 1, 2
        self.start_index, self.current_index = None, None
        self.filename = filename
        self.search_keywords = search_keywords
        self.limit = limit
        self.skipped, self.missing_features = [], []
        self.removed_duplicates, self.duplicate_count = [], 0
        self.df = pd.DataFrame()
        self.randomizer = np.random.RandomState(seed=seed)
        self.error_count, self.query_times = 0, 0
        self.search_cycles = 0

    def _error_checked(func):
        """Decorator used to check called function's total error count before proceeeding"""

        @functools.wraps(func)
        def wrapper_error_check(self, *args, **kwargs):
            if self.error_count >= 5:
                self.error_count = 0
                self.pickle_dump(self)
                raise ValueError('Too many errors. Aborting the API calls')
            else:
                return func(self, *args, **kwargs)
        return wrapper_error_check

    def _error_increment(self, err_type, func, args):
        """Provides an error message, while adding to the total error count before being routed to the specified function"""

        errors = {"merging":"Retrying last search",
                  "parsing": f"Error in getting data. Cannot parse json. Trying again in {self.wait_time} seconds"}
        print(errors[err_type])
        self.error_count += 1
        return func(*args)

    def _remove_list_dupes(self, seq):
        """Keeps only unique items in a list, while still preserving order when list size is very large.
        Taken from https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order"""

        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def _test_empty_list_2keys(self, query, index, nested_key1, nested_key2):
        """Tests for missing values returning nans if list length is not long enough"""

        try:
            query[index]
        except:
            result = (np.nan, np.nan)
        else:
            result = (query[index][nested_key1], query[index][nested_key2])
        return result

    def _wait_cycle(self):
        """Randomly chooses a wait time based on the min and max values and waits for that duration."""

        self.wait_time = np.random.uniform(self.wait_min, self.wait_max)
        time.sleep(self.wait_time)

    def pickle_dump(self, data):
        """Saves a backup pickle of the scrapped data appending the interval numbers of the data that was taken."""

        with open(self.filename + f"{self.start_index}-{self.current_index}.pickle", 'wb') as f:
            pickle.dump(data, f)

