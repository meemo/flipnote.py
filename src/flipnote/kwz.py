"""
KWZ format parser for Flipnote Studio 3D (.kwz) files.

Matches the C implementation in libugomemo exactly:
- Section-based parsing (KFH, KTN, KMC, KMI, KSN)
- Bitpacked frame decoding with all 8 tile types
- 3-layer compositing with 7-color palette
- Variable-width 2/4-bit ADPCM audio at 16364 Hz
"""

import struct
import numpy as np
from hashlib import md5

from flipnote.schema import convertKWZFSIDToPPM
from flipnote import _native

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KWZ_FRAME_WIDTH = 320
KWZ_FRAME_HEIGHT = 240
KWZ_TILE_SIZE = 8
KWZ_LARGE_TILE = 128
KWZ_TILES_X = KWZ_FRAME_WIDTH // KWZ_TILE_SIZE   # 40
KWZ_TILES_Y = KWZ_FRAME_HEIGHT // KWZ_TILE_SIZE   # 30
KWZ_TILE_COUNT = KWZ_TILES_X * KWZ_TILES_Y        # 1200

KWZ_SIGNATURE_SIZE = 256
KWZ_FSID_LENGTH = 10
KWZ_FILENAME_LENGTH = 28
DSI_EPOCH = 946706400

FRAMERATES = [0.2, 0.5, 1, 2, 4, 6, 8, 12, 20, 24, 30]

PALETTE = [
    (0xFF, 0xFF, 0xFF),  # 0: white
    (0x14, 0x14, 0x14),  # 1: black
    (0xFF, 0x17, 0x17),  # 2: red
    (0xFF, 0xE6, 0x00),  # 3: yellow
    (0x00, 0x82, 0x32),  # 4: green
    (0x06, 0xAE, 0xFF),  # 5: blue
    (0x00, 0x00, 0x00),  # 6: transparent (placeholder)
]

# Commonly occurring line offsets
KWZ_COMMON_LINE_INDEX = [
    0x0000, 0x0CD0, 0x19A0, 0x02D9, 0x088B, 0x0051, 0x00F3, 0x0009,
    0x001B, 0x0001, 0x0003, 0x05B2, 0x1116, 0x00A2, 0x01E6, 0x0012,
    0x0036, 0x0002, 0x0006, 0x0B64, 0x08DC, 0x0144, 0x00FC, 0x0024,
    0x001C, 0x0004, 0x0334, 0x099C, 0x0668, 0x1338, 0x1004, 0x166C,
]

# Commonly occurring line offsets, shifted left by one pixel
KWZ_LINE_INDEX_SHIFTED = [
    0x0000, 0x0CD0, 0x19A0, 0x0003, 0x02D9, 0x088B, 0x0051, 0x00F3,
    0x0009, 0x001B, 0x0001, 0x0006, 0x05B2, 0x1116, 0x00A2, 0x01E6,
    0x0012, 0x0036, 0x0002, 0x02DC, 0x0B64, 0x08DC, 0x0144, 0x00FC,
    0x0024, 0x001C, 0x099C, 0x0334, 0x1338, 0x0668, 0x166C, 0x1004,
]

# ADPCM tables
ADPCM_STEP_TABLE = [
        7,     8,     9,    10,    11,    12,    13,    14,    16,    17,
       19,    21,    23,    25,    28,    31,    34,    37,    41,    45,
       50,    55,    60,    66,    73,    80,    88,    97,   107,   118,
      130,   143,   157,   173,   190,   209,   230,   253,   279,   307,
      337,   371,   408,   449,   494,   544,   598,   658,   724,   796,
      876,   963,  1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,
     2272,  2499,  2749,  3024,  3327,  3660,  4026,  4428,  4871,  5358,
     5894,  6484,  7132,  7845,  8630,  9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
]

ADPCM_INDEX_TABLE_2BIT = [-1, 2, -1, 2]
ADPCM_INDEX_TABLE_4BIT = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]

KWZ_STEP_INDEX_MIN = 0
KWZ_STEP_INDEX_MAX = 79
KWZ_PREDICTOR_MIN = -2048
KWZ_PREDICTOR_MAX = 2047
KWZ_SCALING_FACTOR = 16
KWZ_VARIABLE_THRESHOLD = 18
KWZ_AUDIO_SAMPLE_RATE = 16364

# Tile type 7 patterns: which rows get line_b (True) vs line_a (False)
TILE_TYPE_7_PATTERNS = [
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 0, 1],
]

# Section magic bytes (4th byte is the section type indicator)
SECTION_MAGIC = {
    b"KFH": 0x14,
    b"KTN": 0x02,
    b"KSN": 0x01,
    b"KMI": 0x05,
    b"KMC": 0x02,
}


# ---------------------------------------------------------------------------
# Line table generation
# ---------------------------------------------------------------------------

def _generate_line_table():
    """Generate the 6561-entry line table. Each entry is 8 bytes of pixel data (values 0-2).

    Index formula: pixels[1]*2187 + pixels[0]*729 + pixels[3]*243 + pixels[2]*81
                 + pixels[5]*27 + pixels[4]*9 + pixels[7]*3 + pixels[6]
    """
    table = np.zeros((6561, 8), dtype=np.uint8)
    idx = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for e in range(3):
                        for f in range(3):
                            for g in range(3):
                                for h in range(3):
                                    # Swizzled order: b,a,d,c,f,e,h,g
                                    table[idx] = [b, a, d, c, f, e, h, g]
                                    idx += 1
    return table


def _generate_shifted_line_table(line_table):
    """Generate the shifted line table (pixels shifted left by one).

    The shifted index maps: for a given line index, compute the index of
    that same line but with pixels shifted left by one position.
    """
    shifted = np.zeros((6561, 8), dtype=np.uint8)

    # Build reverse lookup table matching the C precomputed table structure:
    # shifted_table[i] = line_table[shifted_index_of(i)]
    # where shifted_index_of permutes the base-3 digits
    # The C code generates this with nested loops matching the original Python TABLE_3
    reverse_table = np.zeros(6561, dtype=np.uint16)
    idx = 0
    for a in range(0, 2187, 729):
        for b in range(0, 729, 243):
            for c in range(0, 243, 81):
                for d in range(0, 81, 27):
                    for e in range(0, 27, 9):
                        for f in range(0, 9, 3):
                            for g in range(0, 3, 1):
                                for h in range(0, 6561, 2187):
                                    reverse_table[idx] = a + b + c + d + e + f + g + h
                                    idx += 1

    for i in range(6561):
        shifted[i] = line_table[reverse_table[i]]
    return shifted


# Module-level line tables (generated once on import)
LINE_TABLE = _generate_line_table()
LINE_TABLE_SHIFTED = _generate_shifted_line_table(LINE_TABLE)


# ---------------------------------------------------------------------------
# Tile position computation (matches kwz_compute_tile_positions in C)
# ---------------------------------------------------------------------------

def _compute_tile_positions():
    """Compute the 1200 tile positions in decode order (128x128 large tiles)."""
    positions = []
    for lty in range(0, KWZ_FRAME_HEIGHT, KWZ_LARGE_TILE):
        for ltx in range(0, KWZ_FRAME_WIDTH, KWZ_LARGE_TILE):
            for ty in range(0, KWZ_LARGE_TILE, KWZ_TILE_SIZE):
                y = lty + ty
                if y >= KWZ_FRAME_HEIGHT:
                    break
                for tx in range(0, KWZ_LARGE_TILE, KWZ_TILE_SIZE):
                    x = ltx + tx
                    if x >= KWZ_FRAME_WIDTH:
                        break
                    positions.append((x, y))
    return positions


TILE_POSITIONS = _compute_tile_positions()


# ---------------------------------------------------------------------------
# Bitpacked reader
# ---------------------------------------------------------------------------

class _BitReader:
    """Bitpacked reader matching the C kwz_bit_reader exactly."""

    __slots__ = ("data", "offset", "bit_value", "bit_index")

    def __init__(self, data, offset=0):
        self.data = data
        self.offset = offset
        self.bit_value = 0
        self.bit_index = 16  # Forces initial load on first read

    def read(self, num_bits):
        if self.bit_index + num_bits > 16:
            lo = self.data[self.offset]
            hi = self.data[self.offset + 1]
            next_val = lo | (hi << 8)
            self.offset += 2
            self.bit_value |= next_val << (16 - self.bit_index)
            self.bit_index -= 16

        mask = (1 << num_bits) - 1
        result = self.bit_value & mask
        self.bit_value >>= num_bits
        self.bit_index += num_bits
        return result


# ---------------------------------------------------------------------------
# Filename decoding
# ---------------------------------------------------------------------------

def _decode_filename(raw):
    """Decode a 28-byte KWZ filename, handling PPM-format fallback."""
    # Check for valid KWZ filename (chars 0-5, a-z)
    is_kwz = True
    for i in range(KWZ_FILENAME_LENGTH):
        c = raw[i]
        if not ((0x30 <= c <= 0x35) or (0x61 <= c <= 0x7A)):
            is_kwz = False
            break

    if is_kwz:
        return raw[:KWZ_FILENAME_LENGTH].decode("ascii").rstrip("\x00")

    # Try PPM format fallback
    try:
        mac = raw[0:3]
        ident = raw[3:16]
        edits = struct.unpack_from("<H", raw, 16)[0]
        # Validate: ident should be ASCII hex/alphanum, edits <= 999
        ident_str = ident.decode("ascii")
        if edits <= 999:
            mac_str = "".join("%02X" % c for c in mac)
            return "%s_%s_%03d" % (mac_str, ident_str, edits)
    except (UnicodeDecodeError, struct.error):
        pass

    # Fallback: hex representation
    return "0x" + raw[:KWZ_FILENAME_LENGTH].hex().upper()


# ---------------------------------------------------------------------------
# Layer decompression
# ---------------------------------------------------------------------------

def _decompress_layer(layer, prev_layer, data, size, is_diff):
    """Decompress a single layer from bitpacked tile data.

    Matches kwz_decompress_layer_v2 in kwz_video.c exactly.

    Args:
        layer: output numpy array (240, 320) uint8, modified in-place
        prev_layer: previous frame's layer data (240, 320) uint8, or None
        data: raw compressed bytes (memoryview or bytes)
        size: compressed data size in bytes
        is_diff: if True, this layer is a diff against the previous frame
    """
    if is_diff and prev_layer is not None:
        layer[:] = prev_layer
    else:
        layer[:] = 0

    if size == 0:
        return

    reader = _BitReader(data)
    t = 0

    while t < KWZ_TILE_COUNT:
        x, y = TILE_POSITIONS[t]
        tile_type = reader.read(3)

        if tile_type == 0:
            # Common index -> same line for all 8 rows
            line_idx = KWZ_COMMON_LINE_INDEX[reader.read(5)]
            line = LINE_TABLE[line_idx]
            for row in range(8):
                layer[y + row, x:x + 8] = line

        elif tile_type == 1:
            # Direct index -> same line for all 8 rows
            line_idx = reader.read(13)
            line = LINE_TABLE[line_idx]
            for row in range(8):
                layer[y + row, x:x + 8] = line

        elif tile_type == 2:
            # Common index -> alternating line_a (even) / line_b from shifted table (odd)
            idx = reader.read(5)
            line_idx_a = KWZ_COMMON_LINE_INDEX[idx]
            line_idx_b = KWZ_LINE_INDEX_SHIFTED[idx]
            line_a = LINE_TABLE[line_idx_a]
            line_b = LINE_TABLE_SHIFTED[line_idx_b]
            for row in range(8):
                if row & 1:
                    layer[y + row, x:x + 8] = line_b
                else:
                    layer[y + row, x:x + 8] = line_a

        elif tile_type == 3:
            # Direct index -> alternating with shifted table
            line_idx = reader.read(13)
            line_a = LINE_TABLE[line_idx]
            line_b = LINE_TABLE_SHIFTED[line_idx]
            for row in range(8):
                if row & 1:
                    layer[y + row, x:x + 8] = line_b
                else:
                    layer[y + row, x:x + 8] = line_a

        elif tile_type == 4:
            # Flags byte + per-row common(5)/direct(13) indices
            flags = reader.read(8)
            for row in range(8):
                if flags & (1 << row):
                    li = KWZ_COMMON_LINE_INDEX[reader.read(5)]
                else:
                    li = reader.read(13)
                layer[y + row, x:x + 8] = LINE_TABLE[li]

        elif tile_type == 5:
            # Skip: copy tiles from previous frame
            skip_count = reader.read(5)
            for s in range(skip_count + 1):
                if t >= KWZ_TILE_COUNT:
                    break
                sx, sy = TILE_POSITIONS[t]
                if prev_layer is not None:
                    for row in range(8):
                        layer[sy + row, sx:sx + 8] = prev_layer[sy + row, sx:sx + 8]
                if s < skip_count:
                    t += 1

        elif tile_type == 6:
            # No-op
            pass

        elif tile_type == 7:
            # Pattern + is_common + two line indices
            pattern = reader.read(2)
            is_common = reader.read(1)

            if is_common:
                line_idx_a = KWZ_COMMON_LINE_INDEX[reader.read(5)]
                line_idx_b = KWZ_COMMON_LINE_INDEX[reader.read(5)]
                pattern = (pattern + 1) % 4
            else:
                line_idx_a = reader.read(13)
                line_idx_b = reader.read(13)

            line_a = LINE_TABLE[line_idx_a]
            line_b = LINE_TABLE[line_idx_b]
            pat = TILE_TYPE_7_PATTERNS[pattern]

            for row in range(8):
                if pat[row]:
                    layer[y + row, x:x + 8] = line_b
                else:
                    layer[y + row, x:x + 8] = line_a

        t += 1


# ---------------------------------------------------------------------------
# Layer compositing
# ---------------------------------------------------------------------------

def _composite_frame(output, layer_a, layer_b, layer_c, flags):
    """Composite 3 layers into an RGB frame.

    Matches kwz_composite_frame in kwz_video.c exactly.

    Args:
        output: numpy array (240, 320, 3) uint8, modified in-place
        layer_a, layer_b, layer_c: numpy arrays (240, 320) uint8 with values 0-2
        flags: u32 frame flags from KMI entry
    """
    paper_idx = flags & 0x0F
    la_c1 = (flags >> 8) & 0x0F
    la_c2 = (flags >> 12) & 0x0F
    lb_c1 = (flags >> 16) & 0x0F
    lb_c2 = (flags >> 20) & 0x0F
    lc_c1 = (flags >> 24) & 0x0F
    lc_c2 = (flags >> 28) & 0x0F

    if paper_idx > 6:
        paper_idx = 0
    paper = PALETTE[paper_idx]

    # Fill with paper color
    output[:, :, 0] = paper[0]
    output[:, :, 1] = paper[1]
    output[:, :, 2] = paper[2]

    # Layer C (bottom)
    if lc_c1 < 7:
        mask = layer_c == 1
        color = PALETTE[lc_c1]
        output[mask, 0] = color[0]
        output[mask, 1] = color[1]
        output[mask, 2] = color[2]
    if lc_c2 < 7:
        mask = layer_c == 2
        color = PALETTE[lc_c2]
        output[mask, 0] = color[0]
        output[mask, 1] = color[1]
        output[mask, 2] = color[2]

    # Layer B (middle)
    if lb_c1 < 7:
        mask = layer_b == 1
        color = PALETTE[lb_c1]
        output[mask, 0] = color[0]
        output[mask, 1] = color[1]
        output[mask, 2] = color[2]
    if lb_c2 < 7:
        mask = layer_b == 2
        color = PALETTE[lb_c2]
        output[mask, 0] = color[0]
        output[mask, 1] = color[1]
        output[mask, 2] = color[2]

    # Layer A (top)
    if la_c1 < 7:
        mask = layer_a == 1
        color = PALETTE[la_c1]
        output[mask, 0] = color[0]
        output[mask, 1] = color[1]
        output[mask, 2] = color[2]
    if la_c2 < 7:
        mask = layer_a == 2
        color = PALETTE[la_c2]
        output[mask, 0] = color[0]
        output[mask, 1] = color[1]
        output[mask, 2] = color[2]


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------

def _decode_audio_track(data, step_index=0):
    """Decode variable-width 2/4-bit ADPCM audio.

    Matches kwz_decode_track in kwz_audio.c exactly.

    Args:
        data: raw audio bytes
        step_index: initial ADPCM step index (default 0)

    Returns:
        numpy array of int16 PCM samples
    """
    predictor = 0
    output = np.zeros(len(data) * 4, dtype=np.int16)  # Max 4 samples per byte
    output_pos = 0

    for byte in data:
        bit_pos = 0
        while bit_pos < 8:
            if step_index < KWZ_VARIABLE_THRESHOLD or bit_pos > 4:
                # 2-bit mode
                sample = byte & 0x3

                step = ADPCM_STEP_TABLE[step_index]
                diff = step >> 3

                if sample & 1:
                    diff += step
                if sample & 2:
                    diff = -diff

                predictor += diff
                step_index += ADPCM_INDEX_TABLE_2BIT[sample]

                byte >>= 2
                bit_pos += 2
            else:
                # 4-bit mode
                sample = byte & 0xF

                step = ADPCM_STEP_TABLE[step_index]
                diff = step >> 3

                if sample & 1:
                    diff += step >> 2
                if sample & 2:
                    diff += step >> 1
                if sample & 4:
                    diff += step
                if sample & 8:
                    diff = -diff

                predictor += diff
                step_index += ADPCM_INDEX_TABLE_4BIT[sample]

                byte >>= 4
                bit_pos += 4

            # Clamp
            if step_index < KWZ_STEP_INDEX_MIN:
                step_index = KWZ_STEP_INDEX_MIN
            elif step_index > KWZ_STEP_INDEX_MAX:
                step_index = KWZ_STEP_INDEX_MAX

            if predictor < KWZ_PREDICTOR_MIN:
                predictor = KWZ_PREDICTOR_MIN
            elif predictor > KWZ_PREDICTOR_MAX:
                predictor = KWZ_PREDICTOR_MAX

            output[output_pos] = np.int16(predictor * KWZ_SCALING_FACTOR)
            output_pos += 1

    return output[:output_pos]


# ---------------------------------------------------------------------------
# Parser class
# ---------------------------------------------------------------------------

class Parser:
    """KWZ file parser matching the libugomemo C implementation."""

    def __init__(self, buffer=None):
        self.buffer = None
        self.size = 0
        self.sections = {}
        self.is_folder_icon = False

        self._frame_count = 0
        self._frame_speed = 0
        self._framerate = 0.0
        self._thumb_index = 0
        self._layer_visibility = [False, False, False]

        self._frame_meta = []     # List of KMI entry tuples
        self._frame_offsets = []  # Byte offsets into KMC data per frame
        self._track_lengths = [0, 0, 0, 0, 0]

        self._kmc_data = None     # Raw KMC section data (after CRC32)

        # Layer buffers for frame decoding (persistent across frames for diffing)
        self._layer_a = np.zeros((KWZ_FRAME_HEIGHT, KWZ_FRAME_WIDTH), dtype=np.uint8)
        self._layer_b = np.zeros((KWZ_FRAME_HEIGHT, KWZ_FRAME_WIDTH), dtype=np.uint8)
        self._layer_c = np.zeros((KWZ_FRAME_HEIGHT, KWZ_FRAME_WIDTH), dtype=np.uint8)
        self._prev_layer_a = np.zeros((KWZ_FRAME_HEIGHT, KWZ_FRAME_WIDTH), dtype=np.uint8)
        self._prev_layer_b = np.zeros((KWZ_FRAME_HEIGHT, KWZ_FRAME_WIDTH), dtype=np.uint8)
        self._prev_layer_c = np.zeros((KWZ_FRAME_HEIGHT, KWZ_FRAME_WIDTH), dtype=np.uint8)
        self._prev_decoded_frame = -1

        self.meta = None

        # Native C acceleration handle
        self._native_ctx = None
        self._file_path = None

        if buffer is not None:
            self.load(buffer)

    @classmethod
    def open(cls, path):
        """Open a KWZ file from disk.

        Uses C acceleration via libugomemo if available for frame/audio decode.
        """
        instance = cls()
        instance._file_path = str(path)

        # Try native C backend
        instance._native_ctx = _native.native_kwz_open(instance._file_path)

        # Always parse in Python for metadata access
        with open(path, "rb") as f:
            instance.load(f)

        return instance

    def load(self, buffer):
        """Load and parse a KWZ file from a file-like object or bytes."""
        if isinstance(buffer, (bytes, bytearray, memoryview)):
            import io
            buffer = io.BytesIO(buffer)

        self.buffer = buffer

        # Get file size (excluding 256-byte signature)
        self.buffer.seek(0, 2)
        self.size = self.buffer.tell() - KWZ_SIGNATURE_SIZE
        self.buffer.seek(0, 0)

        # Parse section table
        offset = 0
        while offset < self.size:
            self.buffer.seek(offset)
            raw = self.buffer.read(8)
            if len(raw) < 8:
                break
            magic = raw[0:3]
            section_size = struct.unpack_from("<I", raw, 4)[0]
            self.sections[magic.decode("ascii")] = {
                "offset": offset,
                "length": section_size,
            }
            offset += section_size + 8

        # Parse KFH (header/meta) -- not present in folder icons
        if "KFH" in self.sections:
            self._decode_meta()
        else:
            self.is_folder_icon = True
            self._frame_count = 1

        # Parse KSN (sound) -- not present in comments or icons
        if "KSN" in self.sections:
            self._decode_ksn()

        # Parse KMI + compute frame offsets into KMC data
        if "KMI" in self.sections and "KMC" in self.sections:
            self._decode_kmi()
            # Cache KMC data (after the 4-byte CRC32)
            kmc_section = self.sections["KMC"]
            self.buffer.seek(kmc_section["offset"] + 12)  # 8 header + 4 CRC32
            self._kmc_data = self.buffer.read(kmc_section["length"] - 4)

    def unload(self):
        """Release resources."""
        if self._native_ctx is not None:
            _native.native_kwz_close(self._native_ctx)
            self._native_ctx = None

        if self.buffer is not None:
            try:
                self.buffer.close()
            except Exception:
                pass
            self.buffer = None

        self.size = 0
        self.sections = {}
        self._frame_count = 0
        self._thumb_index = 0
        self._frame_speed = 0
        self._framerate = 0.0
        self._layer_visibility = [False, False, False]
        self.is_folder_icon = False
        self._frame_meta = []
        self._frame_offsets = []
        self._track_lengths = [0, 0, 0, 0, 0]
        self._kmc_data = None
        self._prev_decoded_frame = -1

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def frame_speed(self):
        return self._frame_speed

    @property
    def framerate(self):
        return self._framerate

    # -----------------------------------------------------------------------
    # Public properties
    # -----------------------------------------------------------------------

    @property
    def thumb_index(self):
        return self._thumb_index

    @property
    def thumbnail_index(self):
        """Alias for thumb_index."""
        return self._thumb_index

    @property
    def lock(self):
        return self.meta.get('lock', 0) if self.meta else 0

    @property
    def loop(self):
        return self.meta.get('loop', 0) if self.meta else 0

    @property
    def creation_timestamp(self):
        return self.meta.get('creation_timestamp') if self.meta else None

    @property
    def modified_timestamp(self):
        return self.meta.get('modified_timestamp') if self.meta else None

    @property
    def app_version(self):
        return self.meta.get('app_version') if self.meta else None

    @property
    def current_author_name(self):
        return self.meta.get('current_username') if self.meta else None

    @property
    def parent_author_name(self):
        return self.meta.get('parent_username') if self.meta else None

    @property
    def root_author_name(self):
        return self.meta.get('root_username') if self.meta else None

    @property
    def current_author_id(self):
        return self.meta.get('current_fsid') if self.meta else None

    @property
    def parent_author_id(self):
        return self.meta.get('parent_fsid') if self.meta else None

    @property
    def root_author_id(self):
        return self.meta.get('root_fsid') if self.meta else None

    @property
    def current_filename(self):
        return self.meta.get('current_filename') if self.meta else None

    @property
    def parent_filename(self):
        return self.meta.get('parent_filename') if self.meta else None

    @property
    def root_filename(self):
        return self.meta.get('root_filename') if self.meta else None

    @property
    def track_sizes(self):
        """Track sizes list (matches PPM naming)."""
        return list(self._track_lengths)

    @property
    def layer_visibility(self):
        """Layer visibility as [layer_a, layer_b, layer_c]."""
        return list(self._layer_visibility)

    # -----------------------------------------------------------------------
    # Section parsers
    # -----------------------------------------------------------------------

    def _decode_meta(self):
        """Parse the KFH section. Matches kwz_process_kfh in kwz.c."""
        section = self.sections["KFH"]
        self.buffer.seek(section["offset"] + 12)  # 8 header + 4 CRC32

        creation_timestamp, modified_timestamp, app_version = struct.unpack("<III", self.buffer.read(12))

        root_author_id = self.buffer.read(KWZ_FSID_LENGTH)
        parent_author_id = self.buffer.read(KWZ_FSID_LENGTH)
        current_author_id = self.buffer.read(KWZ_FSID_LENGTH)

        root_author_name = self.buffer.read(22)
        parent_author_name = self.buffer.read(22)
        current_author_name = self.buffer.read(22)

        root_filename = self.buffer.read(KWZ_FILENAME_LENGTH)
        parent_filename = self.buffer.read(KWZ_FILENAME_LENGTH)
        current_filename = self.buffer.read(KWZ_FILENAME_LENGTH)

        frame_count, thumb_index, flags, speed, layer_flags = struct.unpack("<HHHBB", self.buffer.read(8))

        self._frame_count = frame_count
        self._thumb_index = thumb_index
        self._frame_speed = speed
        self._framerate = FRAMERATES[speed] if speed < len(FRAMERATES) else FRAMERATES[0]
        self._layer_visibility = [
            (layer_flags & 0x1) == 0,        # Layer A
            ((layer_flags >> 1) & 0x1) == 0,  # Layer B
            ((layer_flags >> 2) & 0x1) == 0,  # Layer C
        ]

        root_fsid_hex = root_author_id.hex()
        parent_fsid_hex = parent_author_id.hex()
        current_fsid_hex = current_author_id.hex()

        self.meta = {
            "lock": flags & 0x1,
            "loop": (flags >> 1) & 0x1,
            "flags": flags,
            "layer_flags": layer_flags,
            "app_version": app_version,
            "frame_count": frame_count,
            "frame_speed": speed,
            "thumb_index": thumb_index,
            "creation_timestamp": creation_timestamp + DSI_EPOCH,
            "modified_timestamp": modified_timestamp + DSI_EPOCH,
            "root_username": root_author_name.decode("utf-16-le").rstrip("\x00"),
            "root_fsid": root_fsid_hex,
            "root_fsid_ppm": convertKWZFSIDToPPM(root_fsid_hex),
            "root_filename": _decode_filename(root_filename),
            "parent_username": parent_author_name.decode("utf-16-le").rstrip("\x00"),
            "parent_fsid": parent_fsid_hex,
            "parent_fsid_ppm": convertKWZFSIDToPPM(parent_fsid_hex),
            "parent_filename": _decode_filename(parent_filename),
            "current_username": current_author_name.decode("utf-16-le").rstrip("\x00"),
            "current_fsid": current_fsid_hex,
            "current_fsid_ppm": convertKWZFSIDToPPM(current_fsid_hex),
            "current_filename": _decode_filename(current_filename),
        }

    def _decode_ksn(self):
        """Parse the KSN (sound) section. Matches kwz_process_ksn in kwz.c."""
        section = self.sections["KSN"]
        self.buffer.seek(section["offset"] + 8)

        recorded_speed = struct.unpack("<I", self.buffer.read(4))[0]
        bgm_size, se1_size, se2_size, se3_size, se4_size = struct.unpack("<IIIII", self.buffer.read(20))
        # 4 bytes CRC32 follows, then audio data

        self._track_lengths = [bgm_size, se1_size, se2_size, se3_size, se4_size]

        if self.meta is not None:
            self.meta.update({
                "bgm_used": bgm_size > 0,
                "se1_used": se1_size > 0,
                "se2_used": se2_size > 0,
                "se3_used": se3_size > 0,
                "se4_used": se4_size > 0,
                "bgm_digest": self._get_track_digest(0),
                "se1_digest": self._get_track_digest(1),
                "se2_digest": self._get_track_digest(2),
                "se3_digest": self._get_track_digest(3),
                "se4_digest": self._get_track_digest(4),
            })

    def _decode_kmi(self):
        """Parse KMI frame metadata entries. Matches kwz_decode_kmi in kwz.c."""
        self.buffer.seek(self.sections["KMI"]["offset"] + 8)

        # KMC data starts after 8-byte header + 4-byte CRC32
        data_offset = 0

        self._frame_meta = []
        self._frame_offsets = []

        for i in range(self._frame_count):
            raw = self.buffer.read(28)
            flags = struct.unpack_from("<I", raw, 0)[0]
            la_size = struct.unpack_from("<H", raw, 4)[0]
            lb_size = struct.unpack_from("<H", raw, 6)[0]
            lc_size = struct.unpack_from("<H", raw, 8)[0]
            # bytes 10-19: author FSID (10 bytes)
            la_depth = raw[20]
            lb_depth = raw[21]
            lc_depth = raw[22]
            sfx_flags = raw[23]
            unknown = struct.unpack_from("<H", raw, 24)[0]
            camera_flags = struct.unpack_from("<H", raw, 26)[0]

            self._frame_meta.append({
                "flags": flags,
                "layer_a_size": la_size,
                "layer_b_size": lb_size,
                "layer_c_size": lc_size,
                "layer_a_depth": la_depth,
                "layer_b_depth": lb_depth,
                "layer_c_depth": lc_depth,
                "sfx_flags": sfx_flags,
                "unknown": unknown,
                "camera_flags": camera_flags,
            })
            self._frame_offsets.append(data_offset)
            data_offset += la_size + lb_size + lc_size

    # -----------------------------------------------------------------------
    # Frame decoding
    # -----------------------------------------------------------------------

    def decode_frame(self, index):
        """Decode a frame to an RGB numpy array (240, 320, 3) uint8.

        Uses C acceleration if available, otherwise pure Python.
        Matches kwz_decode_frame / kwz_decode_frame_alloc in kwz_video.c.
        """
        if index < 0 or index >= self._frame_count:
            raise IndexError("Frame index %d out of range [0, %d)" % (index, self._frame_count))

        # Try native C decode
        if self._native_ctx is not None:
            result = _native.native_kwz_decode_frame(self._native_ctx, index)
            if result is not None:
                return result

        return self._decode_frame_python(index)

    def _decode_frame_python(self, index):
        """Pure Python frame decode. Decodes all frames from 0..index for correct diffing."""
        output = np.zeros((KWZ_FRAME_HEIGHT, KWZ_FRAME_WIDTH, 3), dtype=np.uint8)

        # Determine starting frame for sequential decode
        if self._prev_decoded_frame >= 0 and self._prev_decoded_frame < index:
            start = self._prev_decoded_frame + 1
        else:
            start = 0
            self._prev_layer_a[:] = 0
            self._prev_layer_b[:] = 0
            self._prev_layer_c[:] = 0

        for i in range(start, index + 1):
            entry = self._frame_meta[i]
            flags = entry["flags"]
            offset = self._frame_offsets[i]

            diff_a = not (flags & 0x10)
            diff_b = not (flags & 0x20)
            diff_c = not (flags & 0x40)

            la_size = entry["layer_a_size"]
            lb_size = entry["layer_b_size"]
            lc_size = entry["layer_c_size"]

            # Decompress each layer
            _decompress_layer(
                self._layer_a, self._prev_layer_a,
                self._kmc_data[offset:offset + la_size], la_size, diff_a
            )
            offset += la_size

            _decompress_layer(
                self._layer_b, self._prev_layer_b,
                self._kmc_data[offset:offset + lb_size], lb_size, diff_b
            )
            offset += lb_size

            _decompress_layer(
                self._layer_c, self._prev_layer_c,
                self._kmc_data[offset:offset + lc_size], lc_size, diff_c
            )

            # Save layers as previous for next frame
            if i < index:
                np.copyto(self._prev_layer_a, self._layer_a)
                np.copyto(self._prev_layer_b, self._layer_b)
                np.copyto(self._prev_layer_c, self._layer_c)

        # Composite final frame
        _composite_frame(output, self._layer_a, self._layer_b, self._layer_c,
                         self._frame_meta[index]["flags"])

        # Update prev layers for next call
        np.copyto(self._prev_layer_a, self._layer_a)
        np.copyto(self._prev_layer_b, self._layer_b)
        np.copyto(self._prev_layer_c, self._layer_c)
        self._prev_decoded_frame = index

        return output

    # -----------------------------------------------------------------------
    # Thumbnail
    # -----------------------------------------------------------------------

    def get_thumbnail(self):
        """Return raw thumbnail bytes from the KTN section."""
        section = self.sections["KTN"]
        self.buffer.seek(section["offset"] + 12)  # 8 header + 4 CRC32
        return self.buffer.read(section["length"] - 4)

    # -----------------------------------------------------------------------
    # Audio
    # -----------------------------------------------------------------------

    def has_audio_track(self, track):
        """Check if a given audio track (0=BGM, 1-4=SE1-SE4) has data."""
        if track < 0 or track > 4:
            return False
        return self._track_lengths[track] > 0

    def _get_audio_track_offset(self, track):
        """Calculate the file offset for a given audio track's raw data."""
        # Audio data starts after KSN header: 8 section header + 28 (speed + 5 sizes + crc32)
        offset = self.sections["KSN"]["offset"] + 36
        for i in range(track):
            offset += self._track_lengths[i]
        return offset

    def get_audio_track_raw(self, track):
        """Return the raw compressed audio bytes for a track."""
        if not self.has_audio_track(track):
            return b""
        size = self._track_lengths[track]
        self.buffer.seek(self._get_audio_track_offset(track))
        return self.buffer.read(size)

    def _get_track_digest(self, track):
        """Compute MD5 hex digest of raw audio track data."""
        if self.has_audio_track(track):
            return md5(self.get_audio_track_raw(track)).hexdigest()
        return None

    def decode_audio_track(self, track, step_index=0):
        """Decode an audio track to PCM int16 samples.

        Uses C acceleration if available, otherwise pure Python.
        Matches kwz_decode_track in kwz_audio.c.

        Args:
            track: track index (0=BGM, 1-4=SE1-SE4)
            step_index: initial ADPCM step index (default 0)

        Returns:
            numpy array of int16 PCM samples at 16364 Hz
        """
        if not self.has_audio_track(track):
            return np.array([], dtype=np.int16)

        # Try native C decode
        if self._native_ctx is not None:
            result = _native.native_kwz_decode_track(self._native_ctx, track, step_index)
            if result is not None:
                return result

        raw = self.get_audio_track_raw(track)
        return _decode_audio_track(raw, step_index)
