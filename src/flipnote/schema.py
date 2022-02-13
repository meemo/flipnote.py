from re import match
from textwrap import wrap
from base64 import b32decode

"""
This file contains functions relating to various schema within flipnotes.
"""

KWZ_FSID_trans = str.maketrans("CWMFJORDVEGBALKSNTHPYXQUIZ012345", "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")


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
    output_fsid = ""

    # Clean up input
    input_fsid = str(input_fsid).strip().upper()

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

    return output_fsid


def convertPPMtoKWZ(input_fsid):
    """
    Converts PPM format FSIDs to KWZ format
    The first byte appears to be useless, as all versions (00, 10, 12, 14) refer to the same user
    This returns the FSID with 00 as the leading byte by default
    - Once/if the purpose of this byte is discovered, it will be modified
    The trailing null byte is also included
    Any invalid input will be returned without modification.
    - e.g. if a PPM format FSID is used as the input or the length is invalid
    """

    # Clean up input
    output_fsid = str(input_fsid).strip().upper()

    if verifyPPMFSID(input_fsid):
        # Invert the FSID then split into byte sized chunks
        string_list = wrap(input_fsid[::-1], 2)

        # Invert each byte and append to the output string
        for i in range(len(string_list)):
            output_fsid += string_list[i][::-1]

        output_fsid = "00" + output_fsid + "00"

    return output_fsid


def unpackKWZFilename(input_filename):
    """
    Decodes a KWZ format filename to a string of its decoded hex bytes.
    Any invalid input that doesn't match the filename regex will be returned without modification.
    """
    # Clean up input
    output_filename = str(input_filename).strip().upper()

    if verifyKWZFilename(output_filename):
        # Convert custom base-32 encoded string to the standard base-32 alphabet
        str(output_filename).translate(KWZ_FSID_trans)

        # Add padding to allow for decoding
        input_filename += "===="

        output_filename = b32decode(input_filename).hex().upper()
    else:
        # Return without modification if verification fails
        output_filename = input_filename

    return output_filename

