#!/usr/bin/env python
import psutil
from pprint import pprint
import logging
import time
import datetime


class Logger:

    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)


logger = Logger().logger
user1 = 'anders'
user2 = 'myhsueh'
user3 = 'wchsieh'
while (True):
    user1_T_usage = sum(p.memory_info()[0] for p in psutil.process_iter()
                        if p.username() == user1)

    user2_T_usage = sum(p.memory_info()[0] for p in psutil.process_iter()
                        if p.username() == user2)

    user3_T_usage = sum(p.memory_info()[0] for p in psutil.process_iter()
                        if p.username() == user3)
    virtual_memory = dict(psutil.virtual_memory()._asdict())
    sys_total = virtual_memory['total']
    user1_per = user1_T_usage / sys_total
    user2_per = user2_T_usage / sys_total
    user3_per = user3_T_usage / sys_total
    now = datetime.datetime.now()
    logger.info("Time: {}".format(now))
    logger.info("User {} consumed CPU ram in percentage : {:.2%}".format(
        user1, user1_per))
    logger.info("User {} consumed CPU ram in percentage : {:.2%}".format(
        user2, user2_per))
    logger.info("User {} consumed CPU ram in percentage : {:.2%}".format(
        user3, user3_per))
    time.sleep(10.0)