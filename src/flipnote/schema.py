from re import match
from textwrap import wrap

"""
This file contains functions relating to various schema within flipnotes.
"""


def verifyPPMFSID(input_fsid):
    """
    Verifies that a PPM format FSID is valid via regex
    """
    return match("[0159][0-9A-F]{15}", input_fsid) is not None


def verifyKWZFSID(input_fsid):
    """
    Verifies that a KWZ format FSID is valid via regex
    """
    return match("(00|10|12|14)[0-9A-F]{14}[0159][0-9A-F](00)?", input_fsid) is not None


def verifyKWZFilename(file_name):
    """
    Verifies that a KWZ format file name is valid via regex
    """
    return match(r"[a-z0-5]{28}(\.kwz)?", file_name) is not None


def verifyPPMFilename(name):
    """
    Verifies that a PPM format file name is valid via regex
    """
    return match(r"[0-9A-F]{6}_[0-9A-F]{13}_[0-9]{3}(\.ppm)?", name) is not None


def verifyPPMFilesystemFilename(name):
    """
    Verifies that a PPM format filesystem file name is valid via regex
    Filenames in the filesystem (outside of file meta) have the first character used as a checksum
    """
    return match(r"[0-9A-Z][0-9A-F]{5}_[0-9A-F]{13}_[0-9]{3}(\.ppm)?", name) is not None


def convertKWZFSIDToPPM(input_fsid):
    """
    Converts KWZ format FSIDs to PPM format.
    KWZ format FSIDs must be 18 or 20 characters.
    Any invalid input will be returned without modification.
    - e.g. if a PPM format FSID is used as the input or the length is invalid
    """
    output_fsid = input_fsid

    if verifyKWZFSID(input_fsid):
        # Trim the first byte of the FSID
        # FSIDs from KWZ files have an extra null(?) byte at the end, trim it if it exists
        if len(input_fsid) == 20:
            input_fsid = input_fsid[2:-2]
        else:
            input_fsid = input_fsid[2:]

        # Invert the FSID then split into byte sized chunks
        string_list = wrap(input_fsid[::-1], 2)

        # Invert each byte and append to the output string
        for i in range(len(string_list)):
            output_fsid += string_list[i][::-1]
    else:
        output_fsid = input_fsid

    return output_fsid.upper()


def convertPPMtoKWZ(input_fsid):
    """
    Converts PPM format FSIDs to KWZ format
    The first byte appears to be useless, as all versions (00, 10, 12, 14) refer to the same user
    This returns the FSID with 00 as the leading byte by default
    - Once/if the purpose of this byte is discovered, it will be modified
    The trailing null byte is also included
    """
    output_fsid = ""

    if verifyPPMFSID(input_fsid):
        # Invert the FSID then split into byte sized chunks
        string_list = wrap(input_fsid[::-1], 2)

        # Invert each byte and append to the output string
        for i in range(len(string_list)):
            output_fsid += string_list[i][::-1]

        output_fsid = "00" + output_fsid + "00"

    return output_fsid.upper()
