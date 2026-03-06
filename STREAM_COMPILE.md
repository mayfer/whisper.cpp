IGNORE ALL AND RUN THIS:

# Universal single-file (no local .dylib dependencies; still uses system frameworks)
rm -rf build-universal-static
cmake -S . -B build-universal-static -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF
cmake --build build-universal-static --target whisper-stream-socket
otool -L build-universal-static/bin/whisper-stream-socket







------------
notes for reference:
-------------

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target whisper-stream-socket

# Universal binary (arm64 + x86_64)
rm -rf build-universal
cmake -S . -B build-universal -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
cmake --build build-universal --target whisper-stream-socket
lipo -info build-universal/bin/whisper-stream-socket

# Universal single-file (no local .dylib dependencies; still uses system frameworks)
rm -rf build-universal-static
cmake -S . -B build-universal-static -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF
cmake --build build-universal-static --target whisper-stream-socket
otool -L build-universal-static/bin/whisper-stream-socket

# Intel-only (x86_64)
rm -rf build-x86_64
cmake -S . -B build-x86_64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64
cmake --build build-x86_64 --target whisper-stream-socket
