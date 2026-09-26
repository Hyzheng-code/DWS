# DWS research fork: modified from the imported SGLang 0.5.10 source.
"""Fail-soft TCP client for the external DWS prompt predictor."""

from __future__ import annotations

import logging
import pickle
import socket
import struct
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_HEADER = struct.Struct(">I")  # 4-byte big-endian length prefix


# ----- wire framing (shared with predictor_service.py) ----------------------
def send_msg(sock: socket.socket, obj) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("predictor connection closed")
        got += k
    return bytes(buf)


def recv_msg(sock: socket.socket):
    (length,) = _HEADER.unpack(_recv_exactly(sock, _HEADER.size))
    return pickle.loads(_recv_exactly(sock, length))


class PredictorClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 5.0,
        rid_prefix: str = "",
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.rid_prefix = str(rid_prefix or "")
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self.enabled = True
        self.info: Dict = {}

    # -- connection ----------------------------------------------------------
    def _connect(self) -> None:
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s

    def _ensure(self) -> None:
        if self._sock is None:
            self._connect()

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _rpc(self, msg: Dict):
        """One request/response round-trip. Returns None on any failure (and
        disables further calls so we never spam a dead service)."""
        if not self.enabled:
            return None
        with self._lock:
            for attempt in (0, 1):  # one reconnect retry
                try:
                    self._ensure()
                    send_msg(self._sock, msg)
                    return recv_msg(self._sock)
                except Exception as e:  # noqa: BLE001
                    self._close()
                    if attempt == 1:
                        logger.warning(
                            "[my_predict] predictor RPC failed (%s); disabling client",
                            e,
                        )
                        self.enabled = False
                        return None
        return None

    # -- API -----------------------------------------------------------------
    def _wire_rid(self, rid) -> str:
        return f"{self.rid_prefix}{rid}"

    def _wire_items(self, items: List[Dict]):
        if not self.rid_prefix:
            return items, [str(item["rid"]) for item in items]
        wire_items = []
        local_rids = []
        for item in items:
            local_rid = str(item["rid"])
            wire_item = dict(item)
            wire_item["rid"] = self._wire_rid(local_rid)
            wire_items.append(wire_item)
            local_rids.append(local_rid)
        return wire_items, local_rids

    def _local_results(self, results: Dict, local_rids: List[str]):
        if not self.rid_prefix:
            return results
        return {
            local_rid: results.get(self._wire_rid(local_rid), results.get(local_rid))
            for local_rid in local_rids
            if self._wire_rid(local_rid) in results or local_rid in results
        }

    def ping(self) -> Optional[Dict]:
        out = self._rpc({"cmd": "ping"})
        if out:
            self.info = out
        return out

    def prompt(self, items: List[Dict]) -> Optional[Dict[str, Dict]]:
        """Run the optional out-of-process arrival prediction path.

        Items contain rid, main-model input_ids and prompt text. The server
        predicts the denoising surface once, without per-request model state.
        """
        if not items:
            return {}
        wire_items, local_rids = self._wire_items(items)
        out = self._rpc({"cmd": "prompt", "items": wire_items})
        return (
            None
            if out is None
            else self._local_results(out.get("results", {}), local_rids)
        )


# ----- process-wide prompt clients -----------------------------------------
_CLIENTS: Dict[tuple, PredictorClient] = {}
_CLIENTS_LOCK = threading.Lock()


def get_client(
    host: str,
    port: int,
    channel: str = "default",
    timeout: float = 5.0,
    rid_prefix: str = "",
) -> PredictorClient:
    """Return a process-local client.

    Channels select independent RPC connections. ``rid_prefix``
    isolates request identifiers from servers sharing one predictor service.
    """
    key = (host, int(port), str(channel), float(timeout), str(rid_prefix or ""))
    with _CLIENTS_LOCK:
        c = _CLIENTS.get(key)
        if c is None:
            c = PredictorClient(
                host,
                int(port),
                timeout=float(timeout),
                rid_prefix=str(rid_prefix or ""),
            )
            _CLIENTS[key] = c
        return c
