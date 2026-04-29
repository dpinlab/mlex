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
                 "to": str(current.value["key"])}
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
        self.value: T = value
        self.next: Optional["Node[T]"] = None
        self.previous: Optional["Node[T]"] = None


class DoublyLinkedList(Generic[T]):
    def __init__(self):
        self.head: Optional["Node[T]"] = None
        self.tail: Optional["Node[T]"] = None
        self.length = 0

    def append(self, value: T):
        node = Node[T](value)
        if self.head is None:
            self.head = node
            self.tail = node
            self.length = 1
        else:
            self.tail.next = node
            node.previous = self.tail
            self.tail = node
            self.length = self.length + 1

    def search(self, key: Any, key_selector: Callable[[T], Any]):
        current = self.head
        while current is not None:
            if key_selector(current) == key:
                return current
            current = current.next
        return current

    def splice(self, node: Node[T], dll: DoublyLinkedList):
        if dll is None or dll.length == 0:
            return
        if self.head == node and self.tail == node:
            self.head.next = dll.head
            dll.head.previous = self.head
            self.tail = dll.tail
            self.length = self.length + dll.length
        elif node == self.tail:
            self.tail.next = dll.head
            dll.head.previous = self.tail
            self.tail = dll.tail
            self.length = self.length + dll.length
        else:
            dll.tail.next = node.next
            node.next.previous = dll.tail
            node.next = dll.head
            dll.head.previous = node
            self.length = self.length + dll.length

    def remove(self, key, key_selector: Callable[[T], Any]):
        node = self.search(key=key, key_selector=key_selector)
        if node is not None:
            if self.length == 1:
                self.head = None
                self.tail = None
            elif node == self.head:
                self.head = self.head.next
                self.head.previous.next = None
                self.head.previous = None
            elif node == self.tail:
                self.tail = self.tail.previous
                self.tail.next.previous = None
                self.tail.next = None
            else:
                node.previous.next = node.next
                node.next.previous = node.previous
                node.previous = None
                node.next = None
            self.length = self.length - 1

    def right_shift(self):
        if self.length == 1 or self.length == 0:
            return
        if self.length == 2:
            tmp = self.head
            self.head = self.tail
            self.tail = tmp
            self.head.next = tmp
            self.tmp.previous = self.tail
            self.head.previous = None
            self.tail.next = None
        else:
            self.head.previous = self.tail
            self.tail.next = self.head
            self.tail.previous.next = None
            self.tail = self.tail.previous
            self.head.previous.previous = None
            self.head = self.head.previous

    def itemize(self, key_selector):
        items = []
        current = self.head
        while current:
            items.append(key_selector(current))
            current = current.next
        return items
