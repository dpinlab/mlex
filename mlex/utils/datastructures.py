from __future__ import annotations
from typing import Any, Callable, TypeVar, Generic, Optional
import json

T = TypeVar("T")


def _debug_linked_list(dll: DoublyLinkedList):
    nodes = []
    edges = []
    current = dll.head
    while current:
        nodes.append(
            {
                "id": str(current.value["key"]),
                "label": str(current.value["key"]),
                "color": "green",
                "shape": "box",
            }
        )
        current = current.next
    current = dll.head
    while current:
        if current.next is not None:
            edges.append(
                {
                    "from": str(current.value["key"]),
                    "to": str(current.next.value["key"]),
                }
            )
            edges.append(
                {
                    "from": str(current.next.value["key"]),
                    "to": str(current.value["key"]),
                }
            )
        current = current.next
    network = {
        "kind": {"graph": True},
        "nodes": nodes,
        "edges": edges,
    }
    return json.dumps(network)


class Node(Generic[T]):
    def __init__(self, value: T):
        self._value: T = value
        self._next: Optional["Node[T]"] = None
        self._previous: Optional["Node[T]"] = None

    @property
    def next(self):
        return self._next

    @property
    def previous(self):
        return self._previous

    @next.setter
    def next(self, node: Node[T]):
        self._next = node

    @previous.setter
    def previous(self, node: Node[T]):
        self._previous = node

    @property
    def value(self):
        return self._value


class DoublyLinkedList(Generic[T]):
    def __init__(self):
        self._head: Optional["Node[T]"] = None
        self._tail: Optional["Node[T]"] = None
        self._length = 0

    @property
    def length(self):
        return self._length

    @property
    def head(self):
        return self._head

    def append(self, value: T):
        node = Node[T](value)
        if self._head is None:
            self._head = node
            self.tail = node
            self._length = 1
        else:
            self.tail.next = node
            node.previous = self.tail
            self.tail = node
            self._length = self._length + 1

    def search(self, key: Any, key_selector: Callable[[T], Any]):
        current = self._head
        while current is not None:
            if key_selector(current) == key:
                return current
            current = current.next
        return current

    def splice(self, node: Node[T], dll: DoublyLinkedList):
        if dll is None or dll.length == 0:
            return
        if self._head == node and self.tail == node:
            self._head.next = dll.head
            dll.head.previous = self._head
            self.tail = dll.tail
            self._length = self._length + dll.length
        elif node == self.tail:
            self.tail.next = dll.head
            dll.head.previous = self.tail
            self.tail = dll.tail
            self._length = self._length + dll.length
        else:
            dll.tail.next = node.next
            node.next.previous = dll.tail
            node.next = dll.head
            dll.head.previous = node
            self._length = self._length + dll.length

    def remove(self, key, key_selector: Callable[[T], Any]):
        node = self.search(key=key, key_selector=key_selector)
        if node is not None:
            if self._length == 1:
                self._head = None
                self.tail = None
            elif node == self._head:
                self._head = self._head.next
                self._head.previous.next = None
                self._head.previous = None
            elif node == self.tail:
                self.tail = self.tail.previous
                self.tail.next.previous = None
                self.tail.next = None
            else:
                node.previous.next = node.next
                node.next.previous = node.previous
                node.previous = None
                node.next = None
            self._length = self._length - 1

    def right_shift(self):
        if self._length <= 1 :
            return
        if self._length == 2:
            tmp = self._head
            self._head = self.tail
            self.tail = tmp
            self._head.next = tmp
            self.tmp.previous = self.tail
            self._head.previous = None
            self.tail.next = None
        else:
            self._head.previous = self.tail
            self.tail.next = self._head
            self.tail.previous.next = None
            self.tail = self.tail.previous
            self._head.previous.previous = None
            self._head = self._head.previous

    def __len__(self):
        self.length

    def __iter__(self):
        return _DoublyLinkedListIterator(self)

class _DoublyLinkedListIterator:

    def __init__(self, dll:DoublyLinkedList):
        self._dll = dll
        self._current = dll.head

    def __iter__(self):
        return self

    def __next__(self):
        if self._current is None:
            raise StopIteration
        item = self._current.value
        self._current = self._current.next
        return item
