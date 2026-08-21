import json
import shutil
import struct
from pathlib import Path

SOURCE_DIR = Path(
    r"C:\Users\gkhmyznikov\.cache\huggingface\hub"
    r"\models--unsloth--Qwen3.8-27B-NVFP4\snapshots"
    r"\7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
)
DEST_DIR = Path(r"C:\Dev\models\Qwen3.8-27B-NVFP4-sharded-1g")
MAX_SHARD_BYTES = 1024**3
COPY_BUFFER_BYTES = 16 * 1024**2


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as file:
        header_length = struct.unpack("<Q", file.read(8))[0]
        return json.loads(file.read(header_length)), 8 + header_length


def padded_header(header: dict) -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return encoded + b" " * ((-len(encoded)) % 8)


def copy_range(source, destination, offset: int, length: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(COPY_BUFFER_BYTES, remaining))
        if not chunk:
            raise EOFError(f"Unexpected EOF with {remaining} bytes remaining")
        destination.write(chunk)
        remaining -= len(chunk)


def edge_bytes(path: Path, data_offset: int, offsets: list[int]) -> tuple[bytes, bytes]:
    start, end = offsets
    length = min(64, end - start)
    with path.open("rb") as file:
        file.seek(data_offset + start)
        first = file.read(length)
        file.seek(data_offset + end - length)
        last = file.read(length)
    return first, last


def main() -> None:
    if DEST_DIR.exists():
        raise FileExistsError(DEST_DIR)
    DEST_DIR.mkdir(parents=True)

    published_index = json.loads(
        (SOURCE_DIR / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    source_names = sorted(set(published_index["weight_map"].values()))
    source_headers: dict[str, tuple[Path, dict, int]] = {}
    metadata = None
    tensors: list[tuple[str, Path, dict, int]] = []

    for source_name in source_names:
        source_path = (SOURCE_DIR / source_name).resolve()
        header, data_offset = read_header(source_path)
        source_metadata = header.pop("__metadata__", None)
        metadata = metadata or source_metadata
        source_headers[source_name] = (source_path, header, data_offset)
        ordered = sorted(header.items(), key=lambda item: item[1]["data_offsets"][0])
        tensors.extend(
            (name, source_path, info, data_offset) for name, info in ordered
        )

    if set(published_index["weight_map"]) != {name for name, *_ in tensors}:
        raise ValueError("Published index and safetensors headers disagree")

    for path in SOURCE_DIR.iterdir():
        if path.name not in {*source_names, "model.safetensors.index.json"}:
            shutil.copy2(path.resolve(), DEST_DIR / path.name)

    shards: list[list[tuple[str, Path, dict, int]]] = []
    current: list[tuple[str, Path, dict, int]] = []
    current_size = 0
    for tensor in tensors:
        start, end = tensor[2]["data_offsets"]
        tensor_size = end - start
        if current and current_size + tensor_size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_size = 0
        current.append(tensor)
        current_size += tensor_size
    if current:
        shards.append(current)

    weight_map: dict[str, str] = {}
    total_size = 0
    shard_count = len(shards)
    open_sources = {
        source_path: source_path.open("rb")
        for source_path, _, _ in source_headers.values()
    }
    try:
        for index, shard in enumerate(shards, start=1):
            filename = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            shard_header: dict[str, object] = {}
            if metadata is not None:
                shard_header["__metadata__"] = metadata
            shard_offset = 0
            for name, _, info, _ in shard:
                start, end = info["data_offsets"]
                size = end - start
                shard_header[name] = {
                    "dtype": info["dtype"],
                    "shape": info["shape"],
                    "data_offsets": [shard_offset, shard_offset + size],
                }
                shard_offset += size
                total_size += size
                weight_map[name] = filename

            encoded_header = padded_header(shard_header)
            destination_path = DEST_DIR / filename
            with destination_path.open("wb") as destination:
                destination.write(struct.pack("<Q", len(encoded_header)))
                destination.write(encoded_header)
                for _, source_path, info, data_offset in shard:
                    start, end = info["data_offsets"]
                    copy_range(
                        open_sources[source_path],
                        destination,
                        data_offset + start,
                        end - start,
                    )
            print(filename, round(destination_path.stat().st_size / 2**30, 3))
    finally:
        for source in open_sources.values():
            source.close()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (DEST_DIR / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    shard_tensors: dict[str, tuple[Path, dict, int]] = {}
    for path in sorted(DEST_DIR.glob("model-*.safetensors")):
        header, data_offset = read_header(path)
        header.pop("__metadata__", None)
        shard_tensors.update(
            (name, (path, info, data_offset)) for name, info in header.items()
        )

    if set(weight_map) != set(shard_tensors):
        raise ValueError("Sharded index and shard headers disagree")
    for source_name, (_, header, _) in source_headers.items():
        for name, source_info in header.items():
            shard_path, shard_info, _ = shard_tensors[name]
            if source_info["dtype"] != shard_info["dtype"]:
                raise ValueError(f"Dtype mismatch for {name}")
            if source_info["shape"] != shard_info["shape"]:
                raise ValueError(f"Shape mismatch for {name}")
            if published_index["weight_map"][name] != source_name:
                raise ValueError(f"Published mapping mismatch for {name}")
            if weight_map[name] != shard_path.name:
                raise ValueError(f"Sharded mapping mismatch for {name}")

    sample_names = [
        tensors[0][0],
        tensors[len(tensors) // 2][0],
        tensors[-1][0],
    ]
    for name in sample_names:
        source_name = published_index["weight_map"][name]
        source_path, source_header, source_data_offset = source_headers[source_name]
        shard_path, shard_info, shard_data_offset = shard_tensors[name]
        if edge_bytes(
            source_path,
            source_data_offset,
            source_header[name]["data_offsets"],
        ) != edge_bytes(shard_path, shard_data_offset, shard_info["data_offsets"]):
            raise ValueError(f"Representative bytes mismatch for {name}")
        print("validated", name)

    print(DEST_DIR)
    print("shards", shard_count, "tensors", len(weight_map), "bytes", total_size)


if __name__ == "__main__":
    main()