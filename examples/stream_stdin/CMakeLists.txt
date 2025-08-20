set(TARGET whisper-stream-stdin)

# Build a simple streaming server that reads raw 16-bit little-endian 16 kHz
# mono PCM data from STDIN (or optionally a TCP client) and emits incremental
# transcriptions to STDOUT.  This target intentionally does NOT depend on SDL2
# so it can be built regardless of the WHISPER_SDL2 option.

add_executable(${TARGET} stream_stdin.cpp)

include(DefaultTargetOptions)

find_package(Threads REQUIRED)

target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
