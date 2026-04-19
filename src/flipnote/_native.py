"""
Optional ctypes wrapper for libugomemo C library.

If the shared library is available, NATIVE_AVAILABLE is True and the
native_* functions delegate to C for performance. Otherwise everything
falls back to pure Python.
"""

import ctypes
import ctypes.util
import os
import sys
import platform

NATIVE_AVAILABLE = False
_lib = None
_libc = None

def _find_library():
    """Try to locate libugomemo shared library."""
    candidates = []
    ext = "dylib" if platform.system() == "Darwin" else "so"
    name = f"libugomemo.{ext}"

    # 1. Bundled inside the installed package (pip install with C build)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(pkg_dir, name))

    # 2. Development: submodule build path
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.append(os.path.join(repo_dir, "lib", "libugomemo", "build", "shared", name))

    # 3. System-installed
    system = ctypes.util.find_library("ugomemo")
    if system:
        candidates.append(system)

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _init():
    global NATIVE_AVAILABLE, _lib, _libc

    path = _find_library()
    if path is None:
        return

    try:
        _lib = ctypes.CDLL(path)
        _libc = ctypes.CDLL(ctypes.util.find_library("c"))
    except OSError:
        _lib = None
        return

    # Set up function signatures

    # PPM
    _lib.ppm_open.restype = ctypes.c_void_p
    _lib.ppm_open.argtypes = [ctypes.c_char_p]
    _lib.ppm_close.restype = None
    _lib.ppm_close.argtypes = [ctypes.c_void_p]
    _lib.ppm_get_frame_count.restype = ctypes.c_uint16
    _lib.ppm_get_frame_count.argtypes = [ctypes.c_void_p]
    _lib.ppm_decode_frame_alloc.restype = ctypes.POINTER(ctypes.c_uint8)
    _lib.ppm_decode_frame_alloc.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    _lib.ppm_decode_track_alloc.restype = ctypes.POINTER(ctypes.c_int16)
    _lib.ppm_decode_track_alloc.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]

    # KWZ
    _lib.kwz_open.restype = ctypes.c_void_p
    _lib.kwz_open.argtypes = [ctypes.c_char_p]
    _lib.kwz_cleanup.restype = None
    _lib.kwz_cleanup.argtypes = [ctypes.c_void_p]
    _lib.kwz_get_frame_count.restype = ctypes.c_uint16
    _lib.kwz_get_frame_count.argtypes = [ctypes.c_void_p]
    _lib.kwz_decode_frame_alloc.restype = ctypes.POINTER(ctypes.c_uint8)
    _lib.kwz_decode_frame_alloc.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    _lib.kwz_decode_track_alloc.restype = ctypes.POINTER(ctypes.c_int16)
    _lib.kwz_decode_track_alloc.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_uint32)]

    NATIVE_AVAILABLE = True


# Convenience functions

def native_ppm_open(path):
    if not NATIVE_AVAILABLE:
        return None
    ctx = _lib.ppm_open(path.encode() if isinstance(path, str) else path)
    return ctx if ctx else None


def native_ppm_close(ctx):
    if ctx and NATIVE_AVAILABLE:
        _lib.ppm_close(ctx)


def native_ppm_decode_frame(ctx, index, width=256, height=192):
    """Decode a PPM frame via C. Returns numpy array (H, W, 3) uint8 or None."""
    if not NATIVE_AVAILABLE or not ctx:
        return None
    try:
        import numpy as np
        pixels = _lib.ppm_decode_frame_alloc(ctx, index)
        if not pixels:
            return None
        buf = (ctypes.c_uint8 * (width * height * 3)).from_address(ctypes.addressof(pixels.contents))
        result = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3)).copy()
        _libc.free(pixels)
        return result
    except Exception:
        return None


def native_ppm_decode_track(ctx, track):
    """Decode a PPM audio track via C. Returns numpy array of int16 or None."""
    if not NATIVE_AVAILABLE or not ctx:
        return None
    try:
        import numpy as np
        count = ctypes.c_uint32(0)
        samples = _lib.ppm_decode_track_alloc(ctx, track, ctypes.byref(count))
        if not samples:
            return None
        buf = (ctypes.c_int16 * count.value).from_address(ctypes.addressof(samples.contents))
        result = np.frombuffer(buf, dtype=np.int16).copy()
        _libc.free(samples)
        return result
    except Exception:
        return None


def native_kwz_open(path):
    if not NATIVE_AVAILABLE:
        return None
    ctx = _lib.kwz_open(path.encode() if isinstance(path, str) else path)
    return ctx if ctx else None


def native_kwz_close(ctx):
    if ctx and NATIVE_AVAILABLE:
        _lib.kwz_cleanup(ctx)


def native_kwz_decode_frame(ctx, index, width=320, height=240):
    """Decode a KWZ frame via C. Returns numpy array (H, W, 3) uint8 or None."""
    if not NATIVE_AVAILABLE or not ctx:
        return None
    try:
        import numpy as np
        pixels = _lib.kwz_decode_frame_alloc(ctx, index)
        if not pixels:
            return None
        buf = (ctypes.c_uint8 * (width * height * 3)).from_address(ctypes.addressof(pixels.contents))
        result = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3)).copy()
        _libc.free(pixels)
        return result
    except Exception:
        return None


def native_kwz_decode_track(ctx, track, step_index=-1):
    """Decode a KWZ audio track via C. Returns numpy array of int16 or None."""
    if not NATIVE_AVAILABLE or not ctx:
        return None
    try:
        import numpy as np
        count = ctypes.c_uint32(0)
        samples = _lib.kwz_decode_track_alloc(ctx, track, step_index, ctypes.byref(count))
        if not samples:
            return None
        buf = (ctypes.c_int16 * count.value).from_address(ctypes.addressof(samples.contents))
        result = np.frombuffer(buf, dtype=np.int16).copy()
        _libc.free(samples)
        return result
    except Exception:
        return None


# Initialize on import
_init()
