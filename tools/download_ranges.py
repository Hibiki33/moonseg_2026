"""Parallel HTTP ranges using curl, with optional per-process DNS overrides.

TLS hostname/certificate verification stays enabled. Overrides are obtained from
a HTTPS DNS service and passed only to curl; system DNS settings are untouched.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


def resolve_hosts(endpoint, hosts):
    options = []
    if endpoint:
        for host in hosts:
            query = endpoint + "?name=" + quote(host) + "&type=A"
            with urlopen(Request(query, headers={"User-Agent": "MoonSeg/0.1"}), timeout=30) as response:
                answers = json.load(response)
            addresses = [row["data"] for row in answers.get("Answer", []) if row.get("type") == 1]
            if not addresses:
                raise RuntimeError(f"No IPv4 records returned for {host}")
            options.extend(["--resolve", host + ":443:" + ",".join(addresses)])
            print(f"HTTPS DNS resolved {host}: {', '.join(addresses)}", flush=True)
    return options


def download_ranges(entries, *, jobs=3, connections=8, dns_resolver=None, resolve_overrides=()):
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required for range downloads")
    resolve = resolve_hosts(dns_resolver, ["cas-bridge.xethub.hf.co"])
    for value in resolve_overrides:
        resolve.extend(["--resolve", value])

    def one_file(entry):
        url, destination, size = entry
        destination = Path(destination)
        size = int(size)
        if destination.is_file():
            if destination.stat().st_size == size and not Path(str(destination) + ".aria2").exists():
                print(f"Present: {destination.name}", flush=True)
                return
            raise RuntimeError(f"Existing incomplete file uses a different downloader: {destination}")
        parts = destination.parent / ("." + destination.name + ".parts")
        parts.mkdir(parents=True, exist_ok=True)
        manifest = {"url": url, "bytes": size, "chunk_size": 128 * 1024 * 1024}
        record = parts / "source.json"
        if record.exists() and json.loads(record.read_text()) != manifest:
            raise RuntimeError(f"Range download source changed: {record}")
        record.write_text(json.dumps(manifest, indent=2) + "\n")
        tasks = [(start, min(size - 1, start + manifest["chunk_size"] - 1))
                 for start in range(0, size, manifest["chunk_size"])]
        started = time.monotonic()
        signed = {"url": None, "time": 0.0}
        resolve_lock = threading.Lock()

        def download_url():
            # Resolve the mirror once per file, not once per parallel range.
            # Reuse the signed CDN location while it is valid to avoid mirror rate limits.
            class NoRedirect(HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    return None

            with resolve_lock:
                if signed["url"] and time.monotonic() - signed["time"] < 2400:
                    return signed["url"]
                opener = build_opener(NoRedirect)
                for attempt in range(8):
                    try:
                        with opener.open(Request(url, method="HEAD", headers={"User-Agent": "MoonSeg/0.1"}), timeout=30):
                            location = url
                    except HTTPError as error:
                        if error.code in (301, 302, 303, 307, 308):
                            location = error.headers["Location"]
                        elif error.code == 429:
                            time.sleep(min(30, max(1, int(error.headers.get("Retry-After", "15")))))
                            continue
                        else:
                            raise
                    signed.update(url=location, time=time.monotonic())
                    return location
                raise RuntimeError(f"Mirror rate limit persists for {destination.name}")

        def one_part(bounds):
            start, end = bounds
            target = parts / f"{start:014d}.part"
            required = end - start + 1
            if target.exists() and target.stat().st_size == required:
                return required
            temporary = target.with_suffix(".download")
            headers = target.with_suffix(".headers")
            errors = target.with_suffix(".stderr")
            for attempt in range(5):
                # Every retry replaces this bounded chunk. Completed chunks are resumable.
                with errors.open("wb") as error_log:
                    command = ["curl", "--fail", "--silent", "--show-error", "--location",
                               "--connect-timeout", "20", "--max-time", "600",
                               "--speed-limit", "1024", "--speed-time", "60",
                               "--range", f"{start}-{end}", "--output", str(temporary),
                               "--dump-header", str(headers), *resolve, download_url()]
                    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=error_log)
                content_range = re.findall(r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)",
                                           headers.read_text(errors="replace") if headers.exists() else "")
                correct_range = content_range and tuple(map(int, content_range[-1])) == (start, end, size)
                if result.returncode == 0 and correct_range and temporary.stat().st_size == required:
                    temporary.replace(target)
                    headers.unlink(missing_ok=True)
                    errors.unlink(missing_ok=True)
                    return required
                time.sleep(min(2 * (attempt + 1), 10))
            raise RuntimeError(f"Range download failed: {destination.name} bytes {start}-{end}; see {errors}")

        total = 0
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = [pool.submit(one_part, bounds) for bounds in tasks]
            for future in as_completed(futures):
                total += future.result()
                print(f"{destination.name}: {total / 1e9:.2f}/{size / 1e9:.2f} GB "
                      f"({100 * total / size:.1f}%)", flush=True)
        assembled = destination.with_suffix(destination.suffix + ".assembling")
        with assembled.open("wb") as writer:
            for start, _ in tasks:
                with (parts / f"{start:014d}.part").open("rb") as reader:
                    shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        if assembled.stat().st_size != size:
            raise RuntimeError(f"Assembled size mismatch: {destination}")
        assembled.replace(destination)
        shutil.rmtree(parts)
        print(f"Completed: {destination.name}", flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(one_file, entry) for entry in entries]
        for future in as_completed(futures):
            future.result()
