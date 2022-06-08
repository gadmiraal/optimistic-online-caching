from collections import namedtuple

import numpy as np
from numpy import ndarray

from policies.policy_abc import Policy


# head points to the doubly linked list of Node
Bucket = namedtuple('Bucket', ['counter', 'head'])
# item caches a reference to the bucket_node for quick accss the next bucket
Item = namedtuple('Item', ['key', 'value', 'bucket_node'])


class LFU(Policy):

    def __init__(self, capacity: int, catalog: int, time_window: int):
        super().__init__(capacity, catalog, time_window)
        self.bucket_head = dll_init()
        self.cache = dict()
        self.name = "LFU"

    def remove_node(self, node):
        item = node[2]
        self.cache.pop(item.key)  # remove from cache
        dll_remove(node)  # remove node from the bucket

        bucket = item.bucket_node[2]
        if bucket.head[1] is bucket.head:
            # remove the bucket if empty
            dll_remove(item.bucket_node)

    def add_node(self, key, value, original_bucket_node):
        """Add the (key, value) content pulled from orginal_bucket_node
        to a new bucket"""
        counter = 0 if original_bucket_node is self.bucket_head \
            else original_bucket_node[2].counter
        next_bucket_node = original_bucket_node[1]

        if next_bucket_node is self.bucket_head:
            # No bucket(counter + k) exists, append a new bucket(counter + 1)
            bucket = Bucket(counter + 1, dll_init())
            bucket_node = dll_append(self.bucket_head, bucket)
        elif next_bucket_node[2].counter != counter + 1:
            # bucket(counter + k) exist, insert bucket(counter + 1) BEFORE next_bucket_node
            bucket = Bucket(counter + 1, dll_init())
            bucket_node = dll_insert_before(next_bucket_node, bucket)
        else:
            # bucket(counter + 1) exists, use it
            bucket = next_bucket_node[2]
            bucket_node = next_bucket_node

        # Create the item, append it to the bucket and add to the cache.
        item = Item(key, value, bucket_node)
        self.cache[key] = dll_append(bucket.head, item)

    def put(self, y: ndarray):
        key = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
        value = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
        # special case for capacity <= 0
        if self.k <= 0:
            return

        # Does the key exist in the cache?
        node = self.cache.get(key)
        if node:
            item = node[2]
            self.remove_node(node)
            self.add_node(item.key, value, item.bucket_node)
            return

        if len(self.cache) >= self.k:
            # Apply LRFU alogrithm here!
            bucket = self.bucket_head[1][2]
            self.remove_node(bucket.head[1])

        self.add_node(key, value, self.bucket_head)

    def get(self, y: ndarray) -> float:
        key = np.where(y == 1)[0][0]  # Todo change when multiple requests are made
        node = self.cache.get(key)
        if node is None:
            return 0

        item = node[2]
        self.remove_node(node)
        self.add_node(item.key, item.value, item.bucket_node)
        return 1

    def cost(self, r_t):
        content = np.fromiter(self.cache.keys(), dtype=int)
        x = np.zeros(self.N)
        x[content] = 1
        return np.sum(self.w * r_t * (1 - x))

    def cache_content(self):
        keys = np.arange(self.N)
        zipped = zip(keys, np.zeros(self.N))
        dic = dict(zipped)
        for key in self.cache.keys():
            dic[key] = 1.0

        return dic


def dll_init():
    head = []
    head[:] = [head, head, None]
    return head


def dll_append(head, value):
    last = head[0]
    node = [last, head, value]
    last[1] = head[0] = node
    return node


def dll_remove(node):
    prev_link, next_link, _ = node
    prev_link[1] = next_link
    next_link[0] = prev_link
    return node


def dll_insert_before(succedent, value):
    # Create a node to host value, and insert BEFORE the succedent
    node = [succedent[0], succedent, value]
    succedent[0][1] = node
    succedent[0] = node
    return node


def dll_insert_after(precedent, value):
    # Create a node to host value, and insert AFTER the precedent
    node = [precedent, precedent[1], value]
    precedent[1][0] = node
    precedent[1] = node
    return node


def dll_iter(head):
    curr = head[1]  # start at the first node
    while curr is not head:
        yield curr[2]  # yield the curr[KEY]
        curr = curr[1]  # move to next node
