#!/usr/bin/env python3

from __future__ import annotations

import argparse
import binascii
import io
import struct
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


def be_u16(buf: bytes, off: int) -> int:
    return struct.unpack_from(">H", buf, off)[0]


def be_u24(buf: bytes, off: int) -> int:
    return (buf[off] << 16) | (buf[off + 1] << 8) | buf[off + 2]


def be_u32(buf: bytes, off: int) -> int:
    return struct.unpack_from(">I", buf, off)[0]


def be_u64(buf: bytes, off: int) -> int:
    return struct.unpack_from(">Q", buf, off)[0]


def read_c_string(buf: bytes, off: int) -> tuple[str, int]:
    end = buf.find(b"\x00", off)
    if end < 0:
        return "", len(buf)
    return buf[off:end].decode("utf-8", errors="replace"), end + 1


@dataclass
class Box:
    type: str
    start: int
    header_size: int
    size: int
    payload_start: int
    payload_end: int
    children: list["Box"] = field(default_factory=list)


@dataclass
class ItemInfo:
    item_id: int
    item_type: str
    name: str
    hidden: bool
    content_type: str | None = None


@dataclass
class ItemLocation:
    item_id: int
    construction_method: int
    base_offset: int
    extents: list[tuple[int, int]]


@dataclass
class Property:
    index: int
    kind: str
    summary: str
    payload: bytes


def iter_boxes(buf: bytes, start: int = 0, end: int | None = None) -> list[Box]:
    if end is None:
        end = len(buf)
    out: list[Box] = []
    pos = start
    while pos + 8 <= end:
        size32 = be_u32(buf, pos)
        typ = buf[pos + 4 : pos + 8].decode("latin-1")
        header = 8
        if size32 == 1:
            if pos + 16 > end:
                break
            size = be_u64(buf, pos + 8)
            header = 16
        elif size32 == 0:
            size = end - pos
        else:
            size = size32
        if size < header or pos + size > end:
            break
        payload_start = pos + header
        payload_end = pos + size
        box = Box(
            type=typ,
            start=pos,
            header_size=header,
            size=size,
            payload_start=payload_start,
            payload_end=payload_end,
        )
        out.append(box)
        pos += size
    return out


def find_child(box: Box, typ: str) -> Box | None:
    for child in box.children:
        if child.type == typ:
            return child
    return None


def parse_child_boxes(buf: bytes, box: Box) -> list[Box]:
    container_types = {"meta", "dinf", "dref", "iinf", "iref", "iprp", "ipco", "grpl"}
    if box.type not in container_types:
        return []
    start = box.payload_start
    if box.type in {"meta", "iinf", "iref"}:
        start += 4
    return iter_boxes(buf, start, box.payload_end)


def populate_children(buf: bytes, box: Box) -> None:
    box.children = parse_child_boxes(buf, box)
    for child in box.children:
        populate_children(buf, child)


def parse_infe(buf: bytes, box: Box) -> ItemInfo:
    payload = buf[box.payload_start : box.payload_end]
    version = payload[0]
    off = 4
    if version in (2, 3):
        item_id = be_u16(payload, off)
        off += 2
        _protection = be_u16(payload, off)
        off += 2
        item_type = payload[off : off + 4].decode("latin-1")
        off += 4
        name, off = read_c_string(payload, off)
        content_type = None
        if item_type == "mime":
            content_type, off = read_c_string(payload, off)
            _encoding, off = read_c_string(payload, off)
        hidden = False
        return ItemInfo(item_id=item_id, item_type=item_type, name=name, hidden=hidden, content_type=content_type)
    raise ValueError(f"unsupported infe version {version}")


def parse_iinf(buf: bytes, box: Box) -> dict[int, ItemInfo]:
    payload = buf[box.payload_start : box.payload_end]
    version = payload[0]
    off = 4
    if version == 0:
        entry_count = be_u16(payload, off)
        off += 2
    else:
        entry_count = be_u32(payload, off)
        off += 4
    _ = entry_count
    out: dict[int, ItemInfo] = {}
    for child in iter_boxes(payload, off):
        if child.type != "infe":
            continue
        global_box = Box(
            type=child.type,
            start=box.payload_start + child.start,
            header_size=child.header_size,
            size=child.size,
            payload_start=box.payload_start + child.payload_start,
            payload_end=box.payload_start + child.payload_end,
        )
        info = parse_infe(buf, global_box)
        out[info.item_id] = info
    return out


def parse_iref(buf: bytes, box: Box) -> list[tuple[str, int, list[int]]]:
    payload = buf[box.payload_start : box.payload_end]
    version = payload[0]
    width = 2 if version == 0 else 4
    refs: list[tuple[str, int, list[int]]] = []
    for child in iter_boxes(payload, 4):
        data = payload[child.payload_start : child.payload_end]
        if width == 2:
            from_id = be_u16(data, 0)
            count = be_u16(data, 2)
            offs = 4
            to_ids = [be_u16(data, offs + i * 2) for i in range(count)]
        else:
            from_id = be_u32(data, 0)
            count = be_u16(data, 4)
            offs = 6
            to_ids = [be_u32(data, offs + i * 4) for i in range(count)]
        refs.append((child.type, from_id, to_ids))
    return refs


def parse_colr(data: bytes) -> str:
    if len(data) < 4:
        return "invalid colr"
    kind = data[:4].decode("latin-1")
    if kind == "nclx" and len(data) >= 11:
        prim = be_u16(data, 4)
        transfer = be_u16(data, 6)
        matrix = be_u16(data, 8)
        full = data[10] >> 7
        return f"nclx primaries={prim} transfer={transfer} matrix={matrix} full_range={full}"
    if kind == "prof":
        return f"icc profile ({len(data) - 4} bytes)"
    return kind


def parse_auxc(data: bytes) -> str:
    if len(data) < 4:
        return "invalid auxC"
    off = 4
    end = data.find(b"\x00", off)
    if end < 0:
        end = len(data)
    aux_type = data[off:end].decode("utf-8", errors="replace")
    return f"aux type={aux_type}"


def parse_ispe(data: bytes) -> str:
    if len(data) < 12:
        return "invalid ispe"
    return f"{be_u32(data, 4)}x{be_u32(data, 8)}"


def parse_pixi(data: bytes) -> str:
    if len(data) < 5:
        return "invalid pixi"
    count = data[4]
    bits = ",".join(str(data[5 + i]) for i in range(min(count, len(data) - 5)))
    return f"bits={bits}"


def parse_properties(buf: bytes, iprp: Box) -> tuple[list[Property], dict[int, list[int]]]:
    ipco = find_child(iprp, "ipco")
    ipma = find_child(iprp, "ipma")
    if ipco is None or ipma is None:
        return [], {}

    props: list[Property] = []
    for idx, child in enumerate(ipco.children, start=1):
        data = buf[child.payload_start : child.payload_end]
        summary = ""
        if child.type == "colr":
            summary = parse_colr(data)
        elif child.type == "auxC":
            summary = parse_auxc(data)
        elif child.type == "ispe":
            summary = parse_ispe(data)
        elif child.type == "pixi":
            summary = parse_pixi(data)
        else:
            summary = f"{child.type} ({len(data)} bytes)"
        props.append(Property(index=idx, kind=child.type, summary=summary, payload=data))

    assoc: dict[int, list[int]] = {}
    payload = buf[ipma.payload_start : ipma.payload_end]
    version = payload[0]
    flags = be_u24(payload, 1)
    off = 4
    entry_count = be_u32(payload, off)
    off += 4
    for _ in range(entry_count):
        item_id = be_u32(payload, off) if version >= 1 else be_u16(payload, off)
        off += 4 if version >= 1 else 2
        count = payload[off]
        off += 1
        assoc[item_id] = []
        for _ in range(count):
            if flags & 1:
                raw = be_u16(payload, off)
                off += 2
                prop_index = raw & 0x7FFF
            else:
                raw = payload[off]
                off += 1
                prop_index = raw & 0x7F
            assoc[item_id].append(prop_index)
    return props, assoc


def parse_iloc(buf: bytes, box: Box) -> tuple[dict[int, ItemLocation], bytes]:
    payload = buf[box.payload_start : box.payload_end]
    version = payload[0]
    off = 4
    tmp = payload[off]
    offset_size = tmp >> 4
    length_size = tmp & 0x0F
    tmp = payload[off + 1]
    base_offset_size = tmp >> 4
    index_size = tmp & 0x0F if version in (1, 2) else 0
    off += 2
    if version < 2:
        item_count = be_u16(payload, off)
        off += 2
    else:
        item_count = be_u32(payload, off)
        off += 4
    out: dict[int, ItemLocation] = {}
    for _ in range(item_count):
        item_id = be_u16(payload, off) if version < 2 else be_u32(payload, off)
        off += 2 if version < 2 else 4
        construction_method = 0
        if version in (1, 2):
            construction_method = be_u16(payload, off) & 0x000F
            off += 2
        _data_ref_index = be_u16(payload, off)
        off += 2
        base_offset = int.from_bytes(payload[off : off + base_offset_size], "big") if base_offset_size else 0
        off += base_offset_size
        extent_count = be_u16(payload, off)
        off += 2
        extents: list[tuple[int, int]] = []
        for _ in range(extent_count):
            if index_size:
                off += index_size
            extent_offset = int.from_bytes(payload[off : off + offset_size], "big") if offset_size else 0
            off += offset_size
            extent_length = int.from_bytes(payload[off : off + length_size], "big") if length_size else 0
            off += length_size
            extents.append((extent_offset, extent_length))
        out[item_id] = ItemLocation(item_id=item_id, construction_method=construction_method, base_offset=base_offset, extents=extents)
    meta = box
    parent = None
    return out, payload


def parse_idat(buf: bytes, meta_box: Box) -> bytes:
    idat = find_child(meta_box, "idat")
    if idat is None:
        return b""
    return buf[idat.payload_start : idat.payload_end]


def item_bytes(buf: bytes, meta_box: Box, locations: dict[int, ItemLocation], item_id: int) -> bytes:
    loc = locations[item_id]
    out = bytearray()
    idat = parse_idat(buf, meta_box)
    for extent_offset, extent_length in loc.extents:
        if loc.construction_method == 0:
            start = loc.base_offset + extent_offset
            out.extend(buf[start : start + extent_length])
        elif loc.construction_method == 1:
            start = loc.base_offset + extent_offset
            out.extend(idat[start : start + extent_length])
        else:
            raise ValueError(f"unsupported construction method {loc.construction_method}")
    return bytes(out)


def parse_icc_tags(icc: bytes) -> dict[str, tuple[int, int]]:
    if len(icc) < 132:
        return {}
    tag_count = be_u32(icc, 128)
    out: dict[str, tuple[int, int]] = {}
    off = 132
    for _ in range(tag_count):
        sig = icc[off : off + 4].decode("latin-1")
        data_off = be_u32(icc, off + 4)
        data_len = be_u32(icc, off + 8)
        out[sig] = (data_off, data_len)
        off += 12
    return out


def parse_desc_tag(icc: bytes, tags: dict[str, tuple[int, int]]) -> str | None:
    for key in ("desc", "mluc"):
        if key not in tags:
            continue
        off, size = tags[key]
        chunk = icc[off : off + size]
        if key == "desc" and len(chunk) >= 12:
            n = be_u32(chunk, 8)
            raw = chunk[12 : 12 + max(0, n - 1)]
            return raw.decode("latin-1", errors="replace")
        if key == "mluc" and len(chunk) >= 16:
            count = be_u32(chunk, 8)
            rec_size = be_u32(chunk, 12)
            if count > 0 and rec_size >= 12 and len(chunk) >= 16 + rec_size:
                length = be_u32(chunk, 16 + 4)
                offset = be_u32(chunk, 16 + 8)
                raw = chunk[offset : offset + length]
                return raw.decode("utf-16-be", errors="replace")
    return None


def extract_xmp_fields(xml_bytes: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return fields
    for elem in root.iter():
        if elem.text and elem.text.strip():
            tag = elem.tag.split("}", 1)[-1]
            if "HDRGainMap" in tag or tag in {"HDRGainMapVersion", "HDRGainMapHeadroom"}:
                fields[tag] = elem.text.strip()
        for attr, value in elem.attrib.items():
            name = attr.split("}", 1)[-1]
            if "HDRGainMap" in name:
                fields[name] = value
    return fields


def format_hex(blob: bytes, limit: int = 96) -> str:
    text = binascii.hexlify(blob[:limit]).decode("ascii")
    if len(blob) > limit:
        text += "..."
    return text


def inspect(path: Path) -> int:
    buf = path.read_bytes()
    top = iter_boxes(buf)
    for box in top:
        populate_children(buf, box)
    meta = next((b for b in top if b.type == "meta"), None)
    if meta is None:
        print("no meta box found", file=sys.stderr)
        return 1

    pitm = find_child(meta, "pitm")
    primary_id = None
    if pitm:
        version = buf[pitm.payload_start]
        primary_id = be_u16(buf, pitm.payload_start + 4) if version == 0 else be_u32(buf, pitm.payload_start + 4)
    items = parse_iinf(buf, find_child(meta, "iinf")) if find_child(meta, "iinf") else {}
    refs = parse_iref(buf, find_child(meta, "iref")) if find_child(meta, "iref") else []
    props, assoc = parse_properties(buf, find_child(meta, "iprp")) if find_child(meta, "iprp") else ([], {})
    prop_map = {p.index: p for p in props}
    locations, _ = parse_iloc(buf, find_child(meta, "iloc")) if find_child(meta, "iloc") else ({}, b"")

    print(f"File: {path}")
    print(f"Primary item id: {primary_id}")
    print()

    by_from: dict[int, list[tuple[str, list[int]]]] = {}
    for kind, from_id, to_ids in refs:
        by_from.setdefault(from_id, []).append((kind, to_ids))

    for item_id in sorted(items):
        info = items[item_id]
        flags = []
        if item_id == primary_id:
            flags.append("primary")
        if info.hidden:
            flags.append("hidden")
        print(f"Item {item_id}: type={info.item_type} {' '.join(flags)}".rstrip())
        if info.content_type:
            print(f"  content_type: {info.content_type}")
        if item_id in assoc:
            for idx in assoc[item_id]:
                prop = prop_map.get(idx)
                if prop:
                    print(f"  prop[{idx}] {prop.kind}: {prop.summary}")
        for kind, to_ids in by_from.get(item_id, []):
            print(f"  ref {kind} -> {to_ids}")
        if item_id in locations:
            loc = locations[item_id]
            total = sum(length for _, length in loc.extents)
            print(f"  data: construction={loc.construction_method} extents={len(loc.extents)} bytes={total}")
        if info.item_type == "mime" and info.content_type == "application/rdf+xml" and item_id in locations:
            data = item_bytes(buf, meta, locations, item_id)
            fields = extract_xmp_fields(data)
            if fields:
                for key, value in sorted(fields.items()):
                    print(f"  xmp {key}: {value}")
        if info.item_type == "tmap" and item_id in assoc:
            for idx in assoc[item_id]:
                prop = prop_map.get(idx)
                if prop and prop.kind == "colr" and prop.payload[:4] == b"prof":
                    icc = prop.payload[4:]
                    tags = parse_icc_tags(icc)
                    print(f"  icc desc: {parse_desc_tag(icc, tags)}")
                    print(f"  icc tags: {', '.join(sorted(tags))}")
                    if "hdgm" in tags:
                        off, size = tags["hdgm"]
                        blob = icc[off : off + size]
                        print(f"  hdgm size: {size}")
                        print(f"  hdgm hex: {format_hex(blob)}")
        print()

    if path.suffix.lower() in {".heic", ".heif"}:
        print("Current Pictolab decoder uses:")
        print("  - Apple HDR gain-map auxiliary image")
        print("  - EXIF MakerNote HDRHeadroom/HDRGain")
        print("  - SDR base image decoded to Display P3 RGBA")
        print()
        print("Not currently used from this file:")
        print("  - hidden tmap item")
        print("  - tmap ICC profile / hdgm tag")
        print("  - HDRGainMap XMP fields")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect HDR-related structure inside HEIC/AVIF files.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    return inspect(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
