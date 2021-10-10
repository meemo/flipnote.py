import struct
import numpy as np
from hashlib import md5

from src.flipnote.schema import convertKWZFSIDToPPM

FRAMERATES = [0.2, 0.5, 1, 2, 4, 6, 8, 12, 20, 24, 30]

PALETTE = [(0xff, 0xff, 0xff),
           (0x10, 0x10, 0x10),
           (0xff, 0x10, 0x10),
           (0xff, 0xe7, 0x00),
           (0x00, 0x86, 0x31),
           (0x00, 0x38, 0xce),
           (0xff, 0xff, 0xff)]

# table1 - commonly occurring line offsets
TABLE_1 = np.array([0x0000, 0x0CD0, 0x19A0, 0x02D9, 0x088B, 0x0051, 0x00F3, 0x0009,
                    0x001B, 0x0001, 0x0003, 0x05B2, 0x1116, 0x00A2, 0x01E6, 0x0012,
                    0x0036, 0x0002, 0x0006, 0x0B64, 0x08DC, 0x0144, 0x00FC, 0x0024,
                    0x001C, 0x0004, 0x0334, 0x099C, 0x0668, 0x1338, 0x1004, 0x166C], dtype=np.uint16)

# table2 - commonly occurring line offsets, but the lines are shifted to the left by one pixel
TABLE_2 = np.array([0x0000, 0x0CD0, 0x19A0, 0x0003, 0x02D9, 0x088B, 0x0051, 0x00F3,
                    0x0009, 0x001B, 0x0001, 0x0006, 0x05B2, 0x1116, 0x00A2, 0x01E6,
                    0x0012, 0x0036, 0x0002, 0x02DC, 0x0B64, 0x08DC, 0x0144, 0x00FC,
                    0x0024, 0x001C, 0x099C, 0x0334, 0x1338, 0x0668, 0x166C, 0x1004], dtype=np.uint16)

ADPCM_STEP_TABLE = np.array([7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
                             19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
                             50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
                             130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
                             337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
                             876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
                             2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
                             5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
                             15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767], dtype=np.int16)

# index table for 2-bit samples
ADPCM_INDEX_TABLE_2 = np.array([-1, 2, -1, 2], dtype=np.int8)

# index table for 4-bit samples
ADPCM_INDEX_TABLE_4 = np.array([-1, -1, -1, -1, 2, 4, 6, 8,
                                -1, -1, -1, -1, 2, 4, 6, 8], dtype=np.int8)


class Parser:
    def __init__(self, buffer=None, processing_frames=True):
        # layer output buffers
        self.buffer = None
        self.size = 0

        self.sections = {}

        self.is_folder_icon = False

        self.frame_offsets = []
        self.frame_meta = []
        self.frame_count = 0

        self.track_frame_speed = 0

        self.layer_pixels = np.zeros((3, 240, 40), dtype="V8")

        self.thumb_index = 0
        self.frame_speed = 0
        self.layer_visibility = [False, False, False]

        self.track_lengths = None

        self.track_meta = None

        self.framerate = None

        self.meta = None

        self.prev_decoded_frame = 0

        self.LINE_TABLE = np.zeros(6561, dtype="V8")
        self.TABLE_3 = np.zeros(6561, dtype=np.uint16)

        # initial values for read_bits()
        self.bit_index = 16
        self.bit_value = 0

        if buffer:
            self.load(buffer, processing_frames)

    @classmethod
    def open(cls, path):
        with open(path, "rb") as buffer:
            return cls(buffer)

    def load(self, buffer, processing_frames):
        self.buffer = buffer

        # lazy way to get file length:
        # seek to the end (ignore signature), get the position, then seek back to the start
        self.buffer.seek(0, 2)
        self.size = self.buffer.tell() - 256
        self.buffer.seek(0, 0)

        # build list of section offsets + lengths
        offset = 0
        while offset < self.size:
            self.buffer.seek(offset)
            magic, length = struct.unpack("<3sxI", self.buffer.read(8))
            self.sections[str(magic, 'ascii')] = {"offset": offset, "length": length}
            offset += length + 8

        # read file header -- not present in folder icons
        if "KFH" in self.sections:
            self.decode_meta()
        else:
            self.is_folder_icon = True
            self.frame_count = 1

        # read sound data header -- not present in comments or icons
        if "KSN" in self.sections:
            self.buffer.seek(self.sections["KSN"]["offset"] + 8)
            self.track_frame_speed = struct.unpack("<I", self.buffer.read(4))
            self.track_lengths = struct.unpack("<IIIII", self.buffer.read(20))
            self.meta.update(self.get_track_meta())

        if processing_frames:
            self.buffer.seek(self.sections["KMI"]["offset"] + 8)
            offset = self.sections["KMC"]["offset"] + 12

            # parse each frame meta entry
            # https://github.com/Flipnote-Collective/flipnote-studio-3d-docs/wiki/kwz-format#kmi-frame-meta
            for i in range(self.frame_count):
                meta = struct.unpack("<IHHH10xBBBBI", self.buffer.read(28))

                self.frame_meta.append(meta)
                self.frame_offsets.append(offset)

                offset += meta[1] + meta[2] + meta[3]

                self.prev_decoded_frame = -1

    def unload(self):
        self.buffer.close()
        self.size = 0
        self.sections = {}
        self.frame_count = 0
        self.thumb_index = 0
        self.frame_speed = 0
        self.layer_visibility = [False, False, False]
        self.is_folder_icon = False
        self.frame_meta = []
        self.frame_offsets = []
        self.track_frame_speed = 0
        self.track_lengths = [0, 0, 0, 0, 0]
        self.prev_decoded_frame = -1

    def read_bits(self, num):
        if self.bit_index + num > 16:
            next_bits = int.from_bytes(self.buffer.read(2), byteorder="little")
            self.bit_value |= next_bits << (16 - self.bit_index)
            self.bit_index -= 16

        mask = (1 << num) - 1
        result = self.bit_value & mask

        self.bit_value >>= num
        self.bit_index += num

        return result

    def gen_line_tables(self):
        # table3 - line offsets, but the lines are shifted to the left by one pixel
        index = 0
        for a in range(0, 2187, 729):
            for b in range(0, 729, 243):
                for c in range(0, 243, 81):
                    for d in range(0, 81, 27):
                        for e in range(0, 27, 9):
                            for f in range(0, 9, 3):
                                for g in range(0, 3, 1):
                                    for h in range(0, 6561, 2187):
                                        self.TABLE_3[index] = a + b + c + d + e + f + g + h
                                        index += 1

        # linetable - contains every possible sequence of pixels for each tile line
        index = 0
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        for e in range(3):
                            for f in range(3):
                                for g in range(3):
                                    for h in range(3):
                                        self.LINE_TABLE[index] = bytes([b, a, d, c, f, e, h, g])
                                        index += 1

    @staticmethod
    def decode_filename(raw_filename):
        try:
            return raw_filename.decode("ascii")
        except UnicodeDecodeError:
            # in some DSi Gallery notes, Nintendo messed up and included the
            # PPM-format filename without converting it
            mac, ident, edits = struct.unpack("<3s13sH", raw_filename[0:18])
            return "{0}_{1}_{2:03d}".format("".join(["%02X" % c for c in mac]), ident.decode("ascii"), edits)

    def get_thumbnail(self):
        self.buffer.seek(self.sections["KTN"]["offset"] + 12)
        return self.buffer.read(self.sections["KTN"]["length"])

    def has_audio_track(self, track_index):
        return self.track_lengths[track_index] > 0

    def get_audio_track_offset(self, track_index):
        # offset starts after sound header
        offset = self.sections["KSN"]["offset"] + 36
        for i in range(track_index):
            offset += self.track_lengths[i]
        return offset

    def get_audio_track(self, track_index):
        size = self.track_lengths[track_index]

        self.buffer.seek(self.get_audio_track_offset(track_index))

        return self.buffer.read(size)

    def get_track_digest(self, track_index):
        if self.has_audio_track(track_index):
            return md5(self.get_audio_track(track_index)).hexdigest()
        else:
            return None

    def decode_audio_track(self, track_index, step_index=0):
        size = self.track_lengths[track_index]

        self.buffer.seek(self.get_audio_track_offset(track_index))

        # create an output buffer with enough space for 60 seconds of audio at 16364 Hz
        output = np.zeros(16364 * 60, dtype="<u2")
        output_offset = 0

        predictor = 0
        diff = 0

        for byte in self.buffer.read(size):
            bit_pos = 0
            while bit_pos < 8:
                if step_index < 18 or bit_pos > 4:
                    sample = byte & 0x3

                    step = ADPCM_STEP_TABLE[step_index]
                    diff = step >> 3

                    if sample & 1:
                        diff += step
                    if sample & 2:
                        diff = -diff

                    predictor += diff
                    step_index += ADPCM_INDEX_TABLE_2[sample]

                    byte >>= 2
                    bit_pos += 2
                else:
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
                    step_index += ADPCM_INDEX_TABLE_4[sample]

                    byte >>= 4
                    bit_pos += 4

                # clamp step index and diff
                step_index = max(0, min(step_index, 79))
                diff = max(-2048, min(diff, 2047))

                # add result to output buffer
                output[output_offset] = predictor * 16
                output_offset += 1

        # trim output array to the size of the track before returning
        return output[:output_offset]

    def decode_meta(self):
        self.buffer.seek(self.sections["KFH"]["offset"] + 12)
        creation_timestamp, modified_timestamp, app_version = struct.unpack("<III", self.buffer.read(12))
        root_author_id, parent_author_id, current_author_id = struct.unpack("<10s10s10s", self.buffer.read(30))
        root_author_name, parent_author_name, current_author_name = struct.unpack("<22s22s22s", self.buffer.read(66))
        root_filename, parent_filename, current_filename = struct.unpack("<28s28s28s", self.buffer.read(84))
        self.frame_count, self.thumb_index, flags, self.frame_speed, layer_flags = struct.unpack("<HHHBB",
                                                                                                 self.buffer.read(8))
        self.framerate = FRAMERATES[self.frame_speed]
        self.layer_visibility = [layer_flags & 0x1 == 0,  # Layer A
                                 (layer_flags >> 1) & 0x1 == 0,  # Layer B
                                 (layer_flags >> 2) & 0x1 == 0]  # Layer C

        self.meta = {
            "lock": flags & 0x1,
            "loop": (flags >> 1) & 0x01,
            "flags": flags,
            "layer_flags": layer_flags,
            "app_version": app_version,
            "frame_count": self.frame_count,
            "frame_speed": self.frame_speed,
            "thumb_index": self.thumb_index,
            "modified_timestamp": modified_timestamp + 946684800,
            "creation_timestamp": creation_timestamp + 946684800,
            "root_username": root_author_name.decode("utf-16").rstrip("\x00"),
            "root_fsid": root_author_id.hex(),
            "root_fsid_ppm": convertKWZFSIDToPPM(root_author_id.hex()),
            "root_filename": self.decode_filename(root_filename),
            "parent_username": parent_author_name.decode("utf-16").rstrip("\x00"),
            "parent_fsid": parent_author_id.hex(),
            "parent_fsid_ppm": convertKWZFSIDToPPM(root_author_id.hex()),
            "parent_filename": self.decode_filename(parent_filename),
            "current_username": current_author_name.decode("utf-16").rstrip("\x00"),
            "current_fsid": current_author_id.hex(),
            "current_fsid_ppm": convertKWZFSIDToPPM(current_author_id.hex()),
            "current_filename": self.decode_filename(current_filename)
        }

        return self.meta

    def get_track_meta(self):
        return {
            "bgm_used": self.has_audio_track(0),
            "se1_used": self.has_audio_track(1),
            "se2_used": self.has_audio_track(2),
            "se3_used": self.has_audio_track(3),
            "se4_used": self.has_audio_track(4),
            "bgm_digest": self.get_track_digest(0),
            "se1_digest": self.get_track_digest(1),
            "se2_digest": self.get_track_digest(2),
            "se3_digest": self.get_track_digest(3),
            "se4_digest": self.get_track_digest(4)
        }
