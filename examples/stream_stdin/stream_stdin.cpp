#include "common-whisper.h"
#include "whisper.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------------------------------------------------
// A minimal example that keeps a whisper.cpp context resident in memory and performs low-latency streaming transcription.
//
//  * The program expects raw 16-bit little-endian mono PCM sampled at 16 kHz on STDIN.
//    (You can convert from a WAV/AIFF/etc. source beforehand or simply strip the WAV header.)
//  * Partial / progressive transcripts are printed as newline-delimited JSON to STDOUT
//      {"type":"partial", "text":"hello worl"}
//  * After the sender closes STDIN (or when EOF is reached) the remaining audio is processed and a final line is printed
//      {"type":"final", "text":"hello world"}
//  * The process continues waiting for the next audio stream on STDIN unless --exit-on-eof is supplied.
//
// The protocol is intentionally trivial so that another process can `fork`/`exec` this binary and speak to it using two
// pipes.  For a network-based solution you could easily adapt the ReadThread() routine to read from a socket instead.
// ---------------------------------------------------------------------------------------------------------------------

#include "json.hpp" // single-header nlohmann/json that already exists in examples/

using json = nlohmann::json;

// command-line parameters (a trimmed subset of the ones used by examples/stream)
struct params_stream {
    // audio windowing
    int32_t step_ms   = 3000;   // inference step (new audio) [ms]
    int32_t length_ms = 10000;  // total audio per inference [ms]
    int32_t keep_ms   = 200;    // overlap with previous window [ms]

    // whisper parameters
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t beam_size  = -1;
    int32_t audio_ctx  = 0;
    bool     translate = false;
    bool     print_special = false;
    std::string language = "en";
    std::string model    = "models/ggml-base.en.bin";

    // misc
    bool exit_on_eof = false;   // quit after first stream
};

static void print_usage(char ** argv, const params_stream & p) {
    std::cerr << "usage: " << argv[0] << " [options]\n\n";
    std::cerr << "options:\n";
    std::cerr << "  -m FNAME, --model FNAME        model path (default " << p.model << ")\n";
    std::cerr << "  -t N,     --threads N          number of threads (default " << p.n_threads << ")\n";
    std::cerr << "  --step N                       step size in ms (default " << p.step_ms << ")\n";
    std::cerr << "  --length N                     window length in ms (default " << p.length_ms << ")\n";
    std::cerr << "  --keep N                       overlap with previous window in ms (default " << p.keep_ms << ")\n";
    std::cerr << "  --beam-size N                  beam size (default greedy)\n";
    std::cerr << "  -l LANG, --language LANG       language (default en)\n";
    std::cerr << "  -tr,     --translate           translate to english\n";
    std::cerr << "  --exit-on-eof                  terminate after first completed stream\n";
    std::cerr << "  -h,      --help                print this help and exit\n";
}

static bool parse_args(int argc, char ** argv, params_stream & p) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv, p);
            return false;
        } else if (arg == "-m" || arg == "--model") {
            p.model = argv[++i];
        } else if (arg == "-t" || arg == "--threads") {
            p.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--step") {
            p.step_ms = std::stoi(argv[++i]);
        } else if (arg == "--length") {
            p.length_ms = std::stoi(argv[++i]);
        } else if (arg == "--keep") {
            p.keep_ms = std::stoi(argv[++i]);
        } else if (arg == "--beam-size") {
            p.beam_size = std::stoi(argv[++i]);
        } else if (arg == "-l" || arg == "--language") {
            p.language = argv[++i];
        } else if (arg == "-tr" || arg == "--translate") {
            p.translate = true;
        } else if (arg == "--exit-on-eof") {
            p.exit_on_eof = true;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            print_usage(argv, p);
            return false;
        }
    }
    p.keep_ms   = std::min(p.keep_ms, p.step_ms);
    p.length_ms = std::max(p.length_ms, p.step_ms);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
//  Ring buffer that the reader thread writes PCM samples into while the main thread
//  consumes them.
// ---------------------------------------------------------------------------------------------------------------------

class pcm_ring_buffer {
public:
    void push(const float * data, size_t n) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buf.insert(m_buf.end(), data, data + n);
        m_cv.notify_all();
    }

    // Block until at least n samples are available, or until `finished`.
    // Returns the number of samples actually popped into dst (<= n)
    size_t pop(size_t n, std::vector<float> & dst) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [&]{ return m_buf.size() >= n || m_finished; });

        const size_t n_pop = std::min(n, m_buf.size());
        dst.assign(m_buf.begin(), m_buf.begin() + n_pop);
        m_buf.erase(m_buf.begin(), m_buf.begin() + n_pop);
        return n_pop;
    }

    void mark_finished() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_finished = true;
        m_cv.notify_all();
    }

    bool finished() const { std::lock_guard<std::mutex> lock(m_mutex); return m_finished && m_buf.empty(); }

    // Return current buffered samples (not thread-safe, call with lock if needed)
    size_t size() const { std::lock_guard<std::mutex> lock(m_mutex); return m_buf.size(); }

    // Non-blocking copy of up to n samples (used for leftover processing)
    void pop_all(std::vector<float> & dst) {
        std::lock_guard<std::mutex> lock(m_mutex);
        dst.assign(m_buf.begin(), m_buf.end());
        m_buf.clear();
    }

private:
    mutable std::mutex      m_mutex;
    std::condition_variable m_cv;
    std::vector<float>      m_buf;
    bool                    m_finished = false;
};

// ---------------------------------------------------------------------------------------------------------------------
//  Reader thread – reads raw 16-bit LE PCM from stdin and feeds it into the ring buffer
// ---------------------------------------------------------------------------------------------------------------------

void reader_thread_func(pcm_ring_buffer & rb) {
    constexpr size_t CHUNK_BYTES = 4096;
    std::vector<int16_t> chunk(CHUNK_BYTES / sizeof(int16_t));

    while (true) {
        std::cin.read(reinterpret_cast<char *>(chunk.data()), CHUNK_BYTES);
        std::streamsize got = std::cin.gcount();
        if (got <= 0) {
            break; // EOF or error
        }

        size_t n_samples = got / sizeof(int16_t);
        std::vector<float> f32(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            f32[i] = static_cast<float>(chunk[i]) / 32768.0f;
        }
        rb.push(f32.data(), f32.size());
    }

    rb.mark_finished();
}

// ---------------------------------------------------------------------------------------------------------------------
//  Helper to emit a JSON line to STDOUT and flush immediately (so the caller can
//  read it as soon as it is ready).
// ---------------------------------------------------------------------------------------------------------------------

static void emit_json(const std::string & type, const std::string & text) {
    json j;
    j["type"] = type;
    j["text"] = text;
    std::cout << j.dump() << std::endl;
}

// Concatenate all segments into a single text (without timestamps)
#ifdef __cplusplus
#define WHISPER_CTX whisper_context
#endif

static std::string collect_segments(struct whisper_context * ctx) {
    std::string res;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * txt = whisper_full_get_segment_text(ctx, i);
        if (i > 0) res += " ";
        res += txt;
    }
    return res;
}

// ---------------------------------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    params_stream params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    // -------------------------------------------------------------------------
    // initialise whisper context once and reuse forever
    // -------------------------------------------------------------------------

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; // rely on coreml/metal when built that way

    whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (!ctx) {
        std::cerr << "failed to load model from " << params.model << "\n";
        return 2;
    }

    const int n_samples_step = static_cast<int>(params.step_ms  * WHISPER_SAMPLE_RATE / 1000.0);
    const int n_samples_len  = static_cast<int>(params.length_ms* WHISPER_SAMPLE_RATE / 1000.0);
    const int n_samples_keep = static_cast<int>(params.keep_ms  * WHISPER_SAMPLE_RATE / 1000.0);

    // ring buffer for PCM coming from stdin
    pcm_ring_buffer rb;
    std::thread reader(reader_thread_func, std::ref(rb));

    std::vector<float> pcmf32_old; // kept audio from previous iteration
    int                iter = 0;

    while (true) {
        // ---------- pull new audio ---------------------------------------------------
        std::vector<float> pcmf32_new;
        size_t popped = rb.pop(n_samples_step, pcmf32_new);

        if (popped == 0) {
            if (rb.finished()) {
                break; // no more data coming – exit loop and perform final flush later
            } else {
                continue; // spurious, but keep waiting
            }
        }

        // build the working buffer: last part of previous + new samples
        const int n_samples_new  = pcmf32_new.size();
        const int n_samples_take = std::min((int)pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

        std::vector<float> pcmf32_cur(n_samples_new + n_samples_take);
        if (n_samples_take) {
            std::copy(pcmf32_old.end() - n_samples_take, pcmf32_old.end(), pcmf32_cur.begin());
        }
        std::copy(pcmf32_new.begin(), pcmf32_new.end(), pcmf32_cur.begin() + n_samples_take);

        pcmf32_old = pcmf32_cur; // save for next overlap

        // ---------- inference --------------------------------------------------------
        whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
        wparams.print_progress   = false;
        wparams.print_realtime   = false;
        wparams.print_timestamps = false;
        wparams.print_special    = params.print_special;
        wparams.translate        = params.translate;
        wparams.max_tokens       = 0; // let whisper decide
        wparams.n_threads        = params.n_threads;
        wparams.beam_search.beam_size = params.beam_size;
        wparams.audio_ctx        = params.audio_ctx;
        wparams.language         = params.language.c_str();

        if (whisper_full(ctx, wparams, pcmf32_cur.data(), pcmf32_cur.size()) != 0) {
            std::cerr << "whisper_full() failed" << std::endl;
            break;
        }

        emit_json("partial", collect_segments(ctx));

        ++iter;
    }

    // process any leftovers & emit final
    {
        std::vector<float> rest;
        rb.pop_all(rest);
    if (!rest.empty()) {
        // prepend the overlap from previous
            const int n_samples_new  = rest.size();
            const int n_samples_take = std::min((int)pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            std::vector<float> pcmf32_cur(n_samples_new + n_samples_take);
            if (n_samples_take) {
                std::copy(pcmf32_old.end() - n_samples_take, pcmf32_old.end(), pcmf32_cur.begin());
            }
            std::copy(rest.begin(), rest.end(), pcmf32_cur.begin() + n_samples_take);

        whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
            wparams.print_progress   = false;
            wparams.print_realtime   = false;
            wparams.print_timestamps = false;
            wparams.print_special    = params.print_special;
            wparams.translate        = params.translate;
            wparams.max_tokens       = 0;
            wparams.n_threads        = params.n_threads;
            wparams.beam_search.beam_size = params.beam_size;
            wparams.audio_ctx        = params.audio_ctx;
            wparams.language         = params.language.c_str();

            if (whisper_full(ctx, wparams, pcmf32_cur.data(), pcmf32_cur.size()) != 0) {
                std::cerr << "whisper_full() failed on final" << std::endl;
            }
        }
    }

    emit_json("final", collect_segments(ctx));

    // tidy up
    reader.join();

    if (!params.exit_on_eof) {
        // Restart reading loop for subsequent streams by spawning a fresh reader thread and resetting state
        while (true) {
            if (std::cin.eof()) {
                if (params.exit_on_eof) break;
                // clear eof flags and wait for new stdin data (e.g. when piping via named pipe)
                std::cin.clear();
            }

            // Block until there is something to read
            char c;
            std::cin.read(&c, 1);
            if (std::cin.gcount() == 0) {
                // still nothing, sleep a bit
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            // there is data – put back the byte and start again
            std::cin.putback(c);

            // reset buffers and spawn new reader
            pcmf32_old.clear();
            pcmf32_old.shrink_to_fit();

            pcm_ring_buffer rb2;
            std::thread reader2(reader_thread_func, std::ref(rb2));

            while (true) {
                std::vector<float> pcmf32_new;
                size_t popped = rb2.pop(n_samples_step, pcmf32_new);
                if (popped == 0) {
                    if (rb2.finished()) break;
                    continue;
                }

                const int n_samples_new  = pcmf32_new.size();
                const int n_samples_take = std::min((int)pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

                std::vector<float> pcmf32_cur(n_samples_new + n_samples_take);
                if (n_samples_take) {
                    std::copy(pcmf32_old.end() - n_samples_take, pcmf32_old.end(), pcmf32_cur.begin());
                }
                std::copy(pcmf32_new.begin(), pcmf32_new.end(), pcmf32_cur.begin() + n_samples_take);

                pcmf32_old = pcmf32_cur;

                whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
                wparams.print_progress   = false;
                wparams.print_realtime   = false;
                wparams.print_timestamps = false;
                wparams.print_special    = params.print_special;
                wparams.translate        = params.translate;
                wparams.max_tokens       = 0;
                wparams.n_threads        = params.n_threads;
                wparams.beam_search.beam_size = params.beam_size;
                wparams.audio_ctx        = params.audio_ctx;
                wparams.language         = params.language.c_str();

                if (whisper_full(ctx, wparams, pcmf32_cur.data(), pcmf32_cur.size()) != 0) {
                    std::cerr << "whisper_full() failed (loop)" << std::endl;
                    break;
                }

                emit_json("partial", collect_segments(ctx));
            }

            // flush leftovers for this stream
            std::vector<float> rest2; rb2.pop_all(rest2);
            if (!rest2.empty()) {
                whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
                wparams.print_progress   = false;
                wparams.print_realtime   = false;
                wparams.print_timestamps = false;
                wparams.print_special    = params.print_special;
                wparams.translate        = params.translate;
                wparams.max_tokens       = 0;
                wparams.n_threads        = params.n_threads;
                wparams.audio_ctx        = params.audio_ctx;
                wparams.language         = params.language.c_str();

                if (whisper_full(ctx, wparams, rest2.data(), rest2.size()) != 0) {
                    std::cerr << "whisper_full() failed (final loop)" << std::endl;
                }
            }

            emit_json("final", collect_segments(ctx));

            reader2.join();

            // after we closed this stream, continue outer while waiting for next
        }
    }

    whisper_free(ctx);
    return 0;
}
