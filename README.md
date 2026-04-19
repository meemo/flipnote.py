# flipnote.py

Python library for parsing Nintendo Flipnote Studio animations.

Supports PPM (Flipnote Studio, DSi) and KWZ (Flipnote Studio 3D) formats.

Accelerated by [libugomemo](https://github.com/meemo/libugomemo) when build dependencies are available, otherwise falls back to pure Python+numpy.

## Installation

```sh
pip install flipnote
```

The install automatically compiles [libugomemo](https://github.com/meemo/libugomemo) (bundled C library) for ~15-40x faster frame and audio decoding. If the build dependencies aren't available, it falls back to pure Python.

### Build dependencies for C acceleration

A C compiler (clang or gcc) and development headers for OpenSSL, GMP, and zlib. If these aren't present, the install still succeeds and everything works -- just slower.

Debian/Ubuntu:
```sh
sudo apt install libssl-dev libgmp-dev zlib1g-dev
```

Fedora/RHEL:
```sh
sudo dnf install openssl-devel gmp-devel zlib-devel
```

Arch:
```sh
sudo pacman -S openssl gmp zlib
```

macOS (Homebrew):
```sh
brew install openssl gmp zlib
```

## Usage

```python
from flipnote.ppm import Parser as PPM
from flipnote.kwz import Parser as KWZ

# Parse a PPM file
ppm = PPM.open("animation.ppm")
print(f"{ppm.frame_count} frames by {ppm.current_author_name}")

# Decode a frame to RGB numpy array (192, 256, 3)
frame = ppm.decode_frame(0)

# Decode an audio track to int16 numpy array
audio = ppm.decode_audio_track(0)  # 0=BGM, 1-3=SE

# Parse a KWZ file
kwz = KWZ.open("animation.kwz")
print(f"{kwz.frame_count} frames by {kwz.current_author_name}")

# Decode a frame to RGB numpy array (240, 320, 3)
frame = kwz.decode_frame(0)

# Decode audio (variable-width ADPCM)
audio = kwz.decode_audio_track(0)
```

## Schema utilities

```python
from flipnote.schema import verifyPPMFSID, convertKWZFSIDToPPM

verifyPPMFSID("59A643D0A30FD688")  # True
convertKWZFSIDToPPM("00A45FDC21928E8CC700")  # "C78C8E9221DC5FA4"
```
