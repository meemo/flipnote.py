import struct
import numpy as np
from datetime import datetime, timezone

try:
    from flipnote._native import (
        NATIVE_AVAILABLE,
        native_ppm_open,
        native_ppm_close,
        native_ppm_decode_frame,
        native_ppm_decode_track,
    )
except ImportError:
    NATIVE_AVAILABLE = False

# -- Constants ----------------------------------------------------------------

PPM_FRAME_WIDTH = 256
PPM_FRAME_HEIGHT = 192
PPM_THUMBNAIL_WIDTH = 64
PPM_THUMBNAIL_HEIGHT = 48

DSI_EPOCH = 946706400  # Seconds since January 1 2000 00:00 UTC

# Framerates indexed from 0; speed = 8 - raw_speed
FRAMERATES = [0, 0.5, 1, 2, 4, 6, 12, 20, 30]

# -- Color tables -------------------------------------------------------------

THUMBNAIL_PALETTE = [
    (0xFF, 0xFF, 0xFF),
    (0x52, 0x52, 0x52),
    (0xFF, 0xFF, 0xFF),
    (0x9C, 0x9C, 0x9C),
    (0xFF, 0x48, 0x44),
    (0xC8, 0x51, 0x4F),
    (0xFF, 0xAD, 0xAC),
    (0x00, 0xFF, 0x00),
    (0x48, 0x40, 0xFF),
    (0x51, 0x4F, 0xB8),
    (0xAD, 0xAB, 0xFF),
    (0x00, 0xFF, 0x00),
    (0xB6, 0x57, 0xB7),
    (0x00, 0xFF, 0x00),
    (0x00, 0xFF, 0x00),
    (0x00, 0xFF, 0x00),
]

# Paper color by index (0 = black, 1 = white)
PAPER_COLORS = [
    (0x0E, 0x0E, 0x0E),
    (0xFF, 0xFF, 0xFF),
]

# Layer color by index (0 and 1 are replaced by inverse-of-paper at render time)
LAYER_COLORS = [
    (0x0E, 0x0E, 0x0E),  # placeholder
    (0x0E, 0x0E, 0x0E),  # placeholder
    (0xFF, 0x2A, 0x2A),  # red
    (0x0A, 0x39, 0xFF),  # blue
]

# Convenience aliases used by get_frame_palette
BLACK = (0x0E, 0x0E, 0x0E)
WHITE = (0xFF, 0xFF, 0xFF)
RED = (0xFF, 0x2A, 0x2A)
BLUE = (0x0A, 0x39, 0xFF)

# -- ADPCM tables -------------------------------------------------------------

ADPCM_STEP_TABLE = np.array([
        7,     8,     9,    10,    11,    12,
       13,    14,    16,    17,    19,    21,
       23,    25,    28,    31,    34,    37,
       41,    45,    50,    55,    60,    66,
       73,    80,    88,    97,   107,   118,
      130,   143,   157,   173,   190,   209,
      230,   253,   279,   307,   337,   371,
      408,   449,   494,   544,   598,   658,
      724,   796,   876,   963,  1060,  1166,
     1282,  1411,  1552,  1707,  1878,  2066,
     2272,  2499,  2749,  3024,  3327,  3660,
     4026,  4428,  4871,  5358,  5894,  6484,
     7132,  7845,  8630,  9493, 10442, 11487,
    12635, 13899, 15289, 16818, 18500, 20350,
    22385, 24623, 27086, 29794, 32767,
], dtype=np.int16)

ADPCM_INDEX_TABLE_4BIT = np.array([
    -1, -1, -1, -1,  2,  4,  6,  8,
    -1, -1, -1, -1,  2,  4,  6,  8,
], dtype=np.int8)

# -- Helpers ------------------------------------------------------------------


def _round_up_mult_4(n):
    return (n + 3) & ~3


def _decode_fsid(data):
    """8 bytes LE u64, rendered as reversed-byte uppercase hex (16 chars)."""
    return "".join("%02X" % b for b in reversed(data))


def _decode_filename(data):
    """18-byte filename: 3 MAC bytes + 13-char string + u16 LE edit count."""
    mac = data[:3]
    ident = data[3:16]
    edits = struct.unpack_from("<H", data, 16)[0]
    mac_str = "".join("%02X" % b for b in mac)
    ident_str = ""
    for b in ident:
        c = chr(b)
        if ("0" <= c <= "9") or ("A" <= c <= "F"):
            ident_str += c
        else:
            ident_str += "#"
    if edits > 999:
        edits = 999
    return "%s_%s_%03d" % (mac_str, ident_str, edits)


def _decode_filename_fragment(data):
    """8-byte root filename fragment: MAC(3)_first-5-bytes-as-hex."""
    parts = []
    for i, b in enumerate(data):
        if i == 3:
            parts.append("_")
        parts.append("%02X" % b)
    return "".join(parts)


def _unpack_line_encodings(data_48):
    """Unpack 48 bytes into 192 2-bit line encoding values."""
    enc = np.empty(192, dtype=np.uint8)
    for i in range(48):
        byte = data_48[i]
        for b in range(0, 8, 2):
            idx = i * 4 + b // 2
            if idx < 192:
                enc[idx] = (byte >> b) & 3
    return enc


def _decompress_layer(data, offset, line_encodings):
    """Decompress a single PPM layer from frame data. Returns (layer, new_offset)."""
    layer = np.zeros((PPM_FRAME_HEIGHT, PPM_FRAME_WIDTH), dtype=np.uint8)
    pos = offset

    for y in range(PPM_FRAME_HEIGHT):
        encoding = line_encodings[y]
        if encoding == 0:
            # Empty line -- already zero
            pass
        elif encoding == 1 or encoding == 2:
            if encoding == 2:
                layer[y, :] = 1
            chunk_flags = struct.unpack_from(">I", data, pos)[0]
            pos += 4
            pixel = 0
            for _ in range(32):
                if chunk_flags & 0x80000000:
                    chunk = data[pos]
                    pos += 1
                    for bit in range(8):
                        layer[y, pixel] = (chunk >> bit) & 1
                        pixel += 1
                else:
                    pixel += 8
                chunk_flags <<= 1
                chunk_flags &= 0xFFFFFFFF
        elif encoding == 3:
            pixel = 0
            while pixel < PPM_FRAME_WIDTH:
                chunk = data[pos]
                pos += 1
                for bit in range(8):
                    layer[y, pixel] = (chunk >> bit) & 1
                    pixel += 1

    return layer, pos


def _decode_adpcm(data, offset, length):
    """Decode IMA ADPCM audio with reversed nibbles. Returns numpy int16 array."""
    if length < 4:
        return np.array([], dtype=np.int16)

    # Header: i16 LE predictor, u8 step_index, u8 unknown
    predictor = struct.unpack_from("<h", data, offset)[0]
    step_index = data[offset + 2]
    # data[offset + 3] is unknown / unused

    step_index = max(0, min(88, step_index))
    predictor = int(predictor)

    buf_pos = offset + 4
    end_pos = offset + length
    samples = []
    low_nibble = True

    while buf_pos < end_pos:
        if low_nibble:
            sample = data[buf_pos] & 0xF
        else:
            sample = data[buf_pos] >> 4
            buf_pos += 1
        low_nibble = not low_nibble

        step = int(ADPCM_STEP_TABLE[step_index])
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
        if predictor < -32768:
            predictor = -32768
        elif predictor > 32767:
            predictor = 32767

        step_index += int(ADPCM_INDEX_TABLE_4BIT[sample])
        if step_index < 0:
            step_index = 0
        elif step_index > 88:
            step_index = 88

        samples.append(predictor)

    return np.array(samples, dtype=np.int16)


# -- Parser class -------------------------------------------------------------


class Parser:
    """PPM (Flipnote Studio DSi) file parser."""

    @classmethod
    def open(cls, path):
        """Open a .ppm file from a filesystem path."""
        instance = cls()
        instance._path = path
        instance.load(builtins_open(path, "rb"))
        return instance

    def __init__(self):
        self.stream = None
        self._path = None
        self._native_ctx = None

        # Metadata fields
        self.lock = None
        self.thumb_index = None
        self.current_author_name = None
        self.parent_author_name = None
        self.root_author_name = None
        self.current_author_id = None
        self.parent_author_id = None
        self.root_author_id = None
        self.current_filename = None
        self.parent_filename = None
        self.root_filename_fragment = None
        self.timestamp = None

        # Header fields
        self.animation_data_size = 0
        self.sound_data_size = 0
        self.frame_count = 0
        self.format_version = 0

        # Animation header
        self.offset_table = None
        self._frame_offset_table_size = 0
        self._anim_data_start = 0
        self.layer_1_visible = True
        self.layer_2_visible = True
        self.loop = False

        # Sound header
        self.frame_speed = 0
        self.bgm_speed = 0
        self.framerate = 0.0
        self.bgm_framerate = 0.0
        self._track_sizes = [0, 0, 0, 0]
        self._track_offsets = [0, 0, 0, 0]

        # Signature
        self.signature = None
        self.signature_padding = None

        # Frame state
        self.layers = None
        self.prev_layers = None
        self.prev_frame_index = -1

    def load(self, stream):
        """Load and parse a PPM file from an open binary stream."""
        self.stream = stream
        self._read_all_data()
        self._read_header()
        self._read_meta()
        self._read_animation_header()
        self._read_sound_header()
        self._read_signature()

        # Allocate frame buffers
        self.layers = np.zeros((2, PPM_FRAME_HEIGHT, PPM_FRAME_WIDTH), dtype=np.uint8)
        self.prev_layers = np.zeros((2, PPM_FRAME_HEIGHT, PPM_FRAME_WIDTH), dtype=np.uint8)
        self.prev_frame_index = -1

        # Try to open native context for C acceleration
        if NATIVE_AVAILABLE and self._path is not None:
            self._native_ctx = native_ppm_open(self._path)

    def _read_all_data(self):
        """Read the entire file into a bytes buffer for random access."""
        self.stream.seek(0)
        self._data = self.stream.read()

    def unload(self):
        """Close the stream and release native resources."""
        if self._native_ctx is not None:
            native_ppm_close(self._native_ctx)
            self._native_ctx = None
        if self.stream:
            self.stream.close()
            self.stream = None

    # -- Internal parsing -----------------------------------------------------

    def _read_header(self):
        """Parse the 16-byte file header."""
        d = self._data
        magic = d[0:4]
        if magic != b"PARA":
            raise ValueError("Invalid PPM magic: %r" % magic)
        self.animation_data_size = struct.unpack_from("<I", d, 4)[0]
        self.sound_data_size = struct.unpack_from("<I", d, 8)[0]
        fc = struct.unpack_from("<H", d, 12)[0]
        self.frame_count = fc + 1
        self.format_version = struct.unpack_from("<H", d, 14)[0]

    def _read_meta(self):
        """Parse the 0x90-byte metadata section at offset 0x10."""
        d = self._data
        off = 0x10

        self.lock = struct.unpack_from("<H", d, off)[0]
        off += 2
        self.thumb_index = struct.unpack_from("<H", d, off)[0]
        off += 2

        # Author names: 22 bytes each, UTF-16LE
        self.root_author_name = d[off:off + 22].decode("utf-16-le").rstrip("\x00")
        off += 22
        self.parent_author_name = d[off:off + 22].decode("utf-16-le").rstrip("\x00")
        off += 22
        self.current_author_name = d[off:off + 22].decode("utf-16-le").rstrip("\x00")
        off += 22

        # FSIDs: 8 bytes each, reversed hex
        self.parent_author_id = _decode_fsid(d[off:off + 8])
        off += 8
        self.current_author_id = _decode_fsid(d[off:off + 8])
        off += 8

        # Filenames: 18 bytes each
        self.parent_filename = _decode_filename(d[off:off + 18])
        off += 18
        self.current_filename = _decode_filename(d[off:off + 18])
        off += 18

        # Root FSID
        self.root_author_id = _decode_fsid(d[off:off + 8])
        off += 8

        # Root filename fragment (8 bytes)
        self.root_filename_fragment = _decode_filename_fragment(d[off:off + 8])
        off += 8

        # Timestamp: u32 LE seconds since Jan 1 2000
        raw_ts = struct.unpack_from("<I", d, off)[0]
        off += 4
        self.timestamp = datetime.fromtimestamp(raw_ts + DSI_EPOCH, tz=timezone.utc)

        # u16 padding
        off += 2

    def _read_animation_header(self):
        """Parse the animation header, offset table, and compute frame data start."""
        d = self._data
        off = 0x06A0

        table_size = struct.unpack_from("<H", d, off)[0]
        off += 2
        _unknown = struct.unpack_from("<I", d, off)[0]
        off += 4
        flags = struct.unpack_from("<H", d, off)[0]
        off += 2

        self.layer_1_visible = bool((flags >> 11) & 1)
        self.layer_2_visible = bool((flags >> 10) & 1)
        self.loop = bool((flags >> 1) & 1)

        self._frame_offset_table_size = table_size

        # Read offset table (array of u32 LE)
        num_offsets = table_size // 4
        self.offset_table = []
        for i in range(num_offsets):
            self.offset_table.append(struct.unpack_from("<I", d, off + i * 4)[0])

        # Animation data starts after the 8-byte header + offset table rounded up to mult of 4
        self._anim_data_start = 0x06A0 + 8 + _round_up_mult_4(table_size)

    def _read_sound_header(self):
        """Parse the sound header (0x20 bytes) and compute track offsets."""
        d = self._data

        # Sound header offset: animation header start + animation_data_size + SFX flags + padding
        sfx_flags_offset = 0x06A0 + self.animation_data_size
        sfx_flags_end = sfx_flags_offset + self.frame_count
        sound_header_offset = _round_up_mult_4(sfx_flags_end)

        off = sound_header_offset
        bgm_size = struct.unpack_from("<I", d, off)[0]
        se1_size = struct.unpack_from("<I", d, off + 4)[0]
        se2_size = struct.unpack_from("<I", d, off + 8)[0]
        se3_size = struct.unpack_from("<I", d, off + 12)[0]
        raw_frame_speed = d[off + 16]
        raw_bgm_speed = d[off + 17]

        self.frame_speed = 8 - raw_frame_speed
        self.bgm_speed = 8 - raw_bgm_speed
        self.framerate = FRAMERATES[self.frame_speed] if 0 <= self.frame_speed < len(FRAMERATES) else 0
        self.bgm_framerate = FRAMERATES[self.bgm_speed] if 0 <= self.bgm_speed < len(FRAMERATES) else 0

        self._track_sizes = [bgm_size, se1_size, se2_size, se3_size]

        # Sound data starts after the 0x20-byte sound header
        sound_data_start = sound_header_offset + 0x20
        self._track_offsets[0] = sound_data_start
        self._track_offsets[1] = sound_data_start + bgm_size
        self._track_offsets[2] = sound_data_start + bgm_size + se1_size
        self._track_offsets[3] = sound_data_start + bgm_size + se1_size + se2_size

        self._sfx_flags = d[sfx_flags_offset:sfx_flags_offset + self.frame_count]

    def _read_signature(self):
        """Read the 128-byte signature and 16-byte padding at the end of the file."""
        d = self._data
        # Signature is at the very end: last 144 bytes (128 sig + 16 padding)
        if len(d) >= 144:
            self.signature = d[-144:-16]
            self.signature_padding = d[-16:]

    # -- Public properties ----------------------------------------------------

    @property
    def track_sizes(self):
        """Track sizes as [bgm, se1, se2, se3]."""
        return list(self._track_sizes)

    @property
    def track_offsets(self):
        """Track byte offsets as [bgm, se1, se2, se3]."""
        return list(self._track_offsets)

    @property
    def sfx_flags(self):
        """Per-frame SFX flags byte array."""
        return self._sfx_flags

    @property
    def frame_data_size(self):
        """Size of compressed frame data (excluding offset table and 8-byte header)."""
        return self.animation_data_size - self._frame_offset_table_size - 8

    @property
    def thumbnail_index(self):
        """Alias for thumb_index."""
        return self.thumb_index

    # -- Thumbnail ------------------------------------------------------------

    def decode_thumbnail(self):
        """Decode the 64x48 thumbnail to an RGB numpy array (48, 64, 3) uint8.

        The thumbnail is stored as 8x8 tiles of 4-bit palette-indexed pixels,
        with the Y axis flipped (bottom-to-top).
        """
        d = self._data
        off = 0xA0
        output = np.zeros((PPM_THUMBNAIL_HEIGHT, PPM_THUMBNAIL_WIDTH, 3), dtype=np.uint8)

        for tile_y in range(0, PPM_THUMBNAIL_HEIGHT, 8):
            for tile_x in range(0, PPM_THUMBNAIL_WIDTH, 8):
                for line in range(8):
                    for pixel in range(0, 8, 2):
                        byte = d[off]
                        off += 1
                        x = tile_x + pixel
                        y = 47 - (tile_y + line)
                        lo = byte & 0xF
                        hi = byte >> 4
                        output[y, x] = THUMBNAIL_PALETTE[lo]
                        output[y, x + 1] = THUMBNAIL_PALETTE[hi]

        return output

    # -- Frame palette --------------------------------------------------------

    def get_frame_palette(self, index):
        """Return [paper_rgb, layer1_rgb, layer2_rgb] for a given frame index."""
        header = self._data[self._anim_data_start + self.offset_table[index]]
        paper_color = header & 1
        layer_1_color = (header >> 1) & 3
        layer_2_color = (header >> 3) & 3

        paper = PAPER_COLORS[paper_color]
        inverse_paper = PAPER_COLORS[paper_color ^ 1]

        pen = [inverse_paper, inverse_paper, RED, BLUE]
        return [paper, pen[layer_1_color], pen[layer_2_color]]

    # -- Frame decoding -------------------------------------------------------

    def _decode_frame_raw(self, index):
        """Decode a single frame's two layers, handling diffing. Updates internal state."""
        d = self._data

        # If this is not a keyframe and we haven't decoded the previous frame,
        # we need to decode all frames up to this one
        frame_pos = self._anim_data_start + self.offset_table[index]
        header = d[frame_pos]
        frame_type = (header >> 7) & 1

        if frame_type == 0 and index != 0 and self.prev_frame_index != index - 1:
            self._decode_frame_raw(index - 1)

        # Copy current layers to previous
        np.copyto(self.prev_layers, self.layers)
        self.prev_frame_index = index

        # Reset current layers
        self.layers.fill(0)

        pos = frame_pos
        header = d[pos]
        pos += 1

        frame_type = (header >> 7) & 1
        translate_flag = (header >> 5) & 3
        layer_2_color = (header >> 3) & 3
        layer_1_color = (header >> 1) & 3
        paper_color = header & 1

        translate_x = 0
        translate_y = 0
        if frame_type == 0 and translate_flag != 0:
            translate_x = struct.unpack_from("<b", d, pos)[0]
            translate_y = struct.unpack_from("<b", d, pos + 1)[0]
            pos += 2

        # Read line encodings (48 bytes per layer)
        line_enc_1 = _unpack_line_encodings(d[pos:pos + 48])
        pos += 48
        line_enc_2 = _unpack_line_encodings(d[pos:pos + 48])
        pos += 48

        # Decompress layers
        layer_1, pos = _decompress_layer(d, pos, line_enc_1)
        layer_2, pos = _decompress_layer(d, pos, line_enc_2)

        self.layers[0] = layer_1
        self.layers[1] = layer_2

        # Frame diffing: XOR with translated previous frame
        if frame_type == 0:
            for y in range(PPM_FRAME_HEIGHT):
                prev_y = y - translate_y
                if prev_y < 0 or prev_y >= PPM_FRAME_HEIGHT:
                    continue
                for x in range(PPM_FRAME_WIDTH):
                    prev_x = x - translate_x
                    if prev_x < 0 or prev_x >= PPM_FRAME_WIDTH:
                        continue
                    self.layers[0, y, x] ^= self.prev_layers[0, prev_y, prev_x]
                    self.layers[1, y, x] ^= self.prev_layers[1, prev_y, prev_x]

        return self.layers

    def get_frame_pixels(self, index):
        """Decode a frame and return a (192, 256) uint8 array with palette indices.

        0 = paper, 1 = layer 1, 2 = layer 2.
        """
        layers = self._decode_frame_raw(index)
        pixels = np.zeros((PPM_FRAME_HEIGHT, PPM_FRAME_WIDTH), dtype=np.uint8)
        # Layer 2 drawn on top of layer 1
        pixels[layers[0] > 0] = 1
        pixels[layers[1] > 0] = 2
        return pixels

    def decode_frame(self, index):
        """Decode a frame to an RGB numpy array (192, 256, 3) uint8.

        Uses C acceleration when available.
        """
        # Try native C decode first
        if self._native_ctx is not None:
            result = native_ppm_decode_frame(self._native_ctx, index)
            if result is not None:
                return result

        # Pure Python fallback
        layers = self._decode_frame_raw(index)

        frame_pos = self._anim_data_start + self.offset_table[index]
        header = self._data[frame_pos]
        paper_color = header & 1
        layer_1_color = (header >> 1) & 3
        layer_2_color = (header >> 3) & 3

        paper = PAPER_COLORS[paper_color]
        inverse_paper = PAPER_COLORS[paper_color ^ 1]

        output = np.zeros((PPM_FRAME_HEIGHT, PPM_FRAME_WIDTH, 3), dtype=np.uint8)

        # Fill with paper color
        output[:, :, 0] = paper[0]
        output[:, :, 1] = paper[1]
        output[:, :, 2] = paper[2]

        # Layer 1 drawn first
        mask1 = layers[0] > 0
        if layer_1_color <= 1:
            color = inverse_paper
        else:
            color = LAYER_COLORS[layer_1_color]
        output[mask1, 0] = color[0]
        output[mask1, 1] = color[1]
        output[mask1, 2] = color[2]

        # Layer 2 drawn on top
        mask2 = layers[1] > 0
        if layer_2_color <= 1:
            color = inverse_paper
        else:
            color = LAYER_COLORS[layer_2_color]
        output[mask2, 0] = color[0]
        output[mask2, 1] = color[1]
        output[mask2, 2] = color[2]

        return output

    # -- Audio decoding -------------------------------------------------------

    def decode_audio_track(self, track):
        """Decode an audio track (0=BGM, 1=SE1, 2=SE2, 3=SE3).

        Returns a numpy int16 array of PCM samples at 8192 Hz.
        Uses C acceleration when available.
        """
        if track < 0 or track > 3:
            raise ValueError("Track must be 0-3, got %d" % track)

        size = self._track_sizes[track]
        if size == 0:
            return np.array([], dtype=np.int16)

        # Try native C decode first
        if self._native_ctx is not None:
            result = native_ppm_decode_track(self._native_ctx, track)
            if result is not None:
                return result

        # Pure Python fallback
        offset = self._track_offsets[track]
        return _decode_adpcm(self._data, offset, size)


# Keep Python's built-in open accessible for the classmethod
builtins_open = open
