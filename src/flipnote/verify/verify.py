from re import match


def ppmFSID(input_fsid):
    """
    Verifies that a PPM format FSID is valid via regex
    """
    return match("[0159][0-9A-F]{15}", input_fsid) is not None


def kwzFSID(input_fsid):
    """
    Verifies that a KWZ format FSID is valid via regex
    """
    return match("(00|10|12|14)[0-9A-F]{14}[0159][0-9A-F](00)?", input_fsid) is not None


def kwzFilename(file_name):
    """
    Verifies that a KWZ format file name is valid via regex
    """
    return match(r"[a-z0-5]{28}(\.kwz)?", file_name) is not None


def ppmFilename(name):
    """
    Verifies that a PPM format file name is valid via regex
    """
    return match(r"[0-9A-F]{6}_[0-9A-F]{13}_[0-9]{3}(\.ppm)?", name) is not None


def ppmFilesystemFilename(name):
    """
    Verifies that a PPM format filesystem file name is valid via regex
    Filenames in the filesystem (outside of file meta) have the first character used as a checksum
    """
    return match(r"[0-9A-Z][0-9A-F]{5}_[0-9A-F]{13}_[0-9]{3}(\.ppm)?", name) is not None
