#include "common-whisper.h"
#include "whisper.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <string>
#include <thread>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fcntl.h> // For fcntl
#include <errno.h> // For errno

// NOTE [2025-09-21]:
// The non-streaming mode previously lost audio older than ~30 s because we applied a
// ring-buffer "cap" (RING_CAP_MS = 30000) even when g_no_stream was enabled.
// In non-streaming flows the producer writes the entire audio quickly, the consumer
// pops in fixed steps, and the cap discards the oldest unconsumed samples. As a result
// only the most recent ~30 s survived to the final pass.
//
// Fix:
// - Do NOT apply any ring-buffer capping in non-streaming mode.
// - Add a --ring-cap <ms> CLI option (default 30000). Set to 0 to disable capping.
//   The cap is only honored when streaming is enabled. Non-streaming ignores it.
// - Clean up comments to reflect the actual default window sizes.

// ---- Control socket for app context updates ----
static const char * CTRL_SOCK_PATH = "/tmp/whisper_ctx.sock";
static std::string g_app_context;
static std::mutex  g_app_mtx;


// Very small helper: write all bytes, handling short writes
static bool write_all(int fd, const void * data, size_t len) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    while (len > 0) {
        ssize_t w = ::write(fd, p, len);
        if (w <= 0) return false;
        p += w; len -= w;
    }
    return true;
}

// ---------- Ring-buffer for inter-thread transfer ---------------------------------
class pcm_ring_buffer {
public:
    void push(const float * data, size_t n) {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_buf.insert(m_buf.end(), data, data + n);
        m_cv.notify_all();
    }

    // blocking pop of up to n samples – returns 0 if finished and nothing left
    size_t pop(size_t n, std::vector<float> & out) {
        std::unique_lock<std::mutex> lock(m_mtx);
        m_cv.wait(lock, [&]{ return m_finished || available_unlocked() >= n; });

        size_t avail = available_unlocked();
        size_t n_pop = std::min(n, avail);
        if (n_pop == 0) {
            return 0;
        }

        out.assign(m_buf.begin() + m_head, m_buf.begin() + m_head + n_pop);
        m_head += n_pop;
        compact_unlocked();
        return n_pop;
    }

    void pop_all(std::vector<float> & out) {
        std::lock_guard<std::mutex> lock(m_mtx);
        size_t avail = available_unlocked();
        out.assign(m_buf.begin() + m_head, m_buf.begin() + m_head + avail);
        reset_unlocked();
    }

    void mark_finished() {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_finished = true;
        m_cv.notify_all();
    }

    bool finished() const {
        std::lock_guard<std::mutex> lock(m_mtx);
        return m_finished && available_unlocked() == 0;
    }

    // Drop the first n samples (no-op if fewer are present)
    void drop(size_t n) {
        std::lock_guard<std::mutex> lock(m_mtx);
        size_t avail = available_unlocked();
        if (n >= avail) {
            reset_unlocked();
        } else {
            m_head += n;
            compact_unlocked();
        }
    }

    // Current buffered duration in milliseconds
    size_t duration_ms() const {
        std::lock_guard<std::mutex> lock(m_mtx);
        return (available_unlocked() * 1000) / WHISPER_SAMPLE_RATE;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mtx);
        return available_unlocked();
    }

private:
    size_t available_unlocked() const {
        return m_buf.size() - m_head;
    }

    void compact_unlocked() {
        if (m_head == 0) {
            return;
        }
        if (m_head >= m_buf.size()) {
            reset_unlocked();
            return;
        }
        // keep the buffer compact once head drifts far enough to matter
        if (m_head >= COMPACT_THRESHOLD) {
            m_buf.erase(m_buf.begin(), m_buf.begin() + m_head);
            m_head = 0;
        }
    }

    void reset_unlocked() {
        m_buf.clear();
        m_head = 0;
    }

    static constexpr size_t COMPACT_THRESHOLD = 8192;

    std::vector<float>       m_buf;
    size_t                   m_head     = 0;
    bool                     m_finished = false;
    mutable std::mutex       m_mtx;
    std::condition_variable  m_cv;
};

// Forward declaration of abort callback
static bool whisper_should_abort(void * user_data);

// ---------- Audio reader thread ----------------------------------------------------

void reader_thread(int client_fd, pcm_ring_buffer & rb, std::atomic<bool> * abort_flag) {
    constexpr size_t BUF_SZ = 4096;
    std::vector<int16_t> buf(BUF_SZ / sizeof(int16_t));

    while (true) {
        ssize_t r = ::read(client_fd, buf.data(), BUF_SZ);
        if (r <= 0) break; // EOF or error → finish

        size_t n_samples = r / sizeof(int16_t);
        std::vector<float> f32(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            f32[i] = static_cast<float>(buf[i]) / 32768.0f;
        }
        rb.push(f32.data(), f32.size());
    }

    // Signal main thread to cancel any ongoing whisper_full() calls
    if (abort_flag) {
        abort_flag->store(true);
    }

    rb.mark_finished();
}

// ---------- Helper: concatenate all segments --------------------------------------
static std::string collect_segments(struct whisper_context * ctx) {
    std::string out;
    const int n = whisper_full_n_segments(ctx);
    for (int i = 0; i < n; ++i) {
        const char * txt = whisper_full_get_segment_text(ctx, i);
        if (i) out += " ";
        out += txt;
    }
    return out;
}

static void append_with_overlap(std::string & accum, const std::string & part) {
    if (part.empty()) {
        return;
    }
    if (accum.empty()) {
        accum = part;
        return;
    }

    const size_t max_overlap = std::min(accum.size(), part.size());
    size_t overlap = 0;
    for (size_t len = max_overlap; len > 0; --len) {
        if (accum.compare(accum.size() - len, len, part, 0, len) == 0) {
            overlap = len;
            break;
        }
    }

    accum.append(part.substr(overlap));
}

// ---------- Helper: timestamped logging ---------------------------------------
static void log_ts(const std::string & msg) {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto now_c = system_clock::to_time_t(now);
    auto ms   = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm_now;
#if defined(_MSC_VER)
    localtime_s(&tm_now, &now_c);
#else
    localtime_r(&now_c, &tm_now);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count()
        << " " << msg;
    std::cerr << oss.str() << std::endl;
}

// ---------- Helper: create app-aware whisper prompt -----------------------------
static std::string create_whisper_prompt() {
    std::string current_app_copy;
    {
        std::lock_guard<std::mutex> lock(g_app_mtx);
        current_app_copy = g_app_context;
    }
    if (current_app_copy.empty()) {
        return "";
    }
    return "The user is currently using " + current_app_copy + " on macOS. Here is their voice transcribed command that will be parsed for either direct transcription, keyboard shortcuts, code, or terminal commands: ";
}

// -----------------------------------------------------------------------------
// Global config (tuned via CLI in main)
static int32_t g_step_ms   = 700;    // emit partials every 700 ms
static int32_t g_length_ms = 30000;  // 30-s rolling window fed to Whisper (streaming)
static int32_t g_keep_ms   = 200;    // overlap between windows

// When true, suppress incremental partial transcriptions and only run a final
// full-context whisper pass after the audio stream ends (set via --no-stream).
static std::atomic<bool> g_no_stream{false};

// Whisper parameters
static float   g_no_speech_thold = 0.7f;  // no speech threshold
static bool    g_enable_vad      = false; // enable voice activity detection
static std::string g_vad_model_path;      // optional path to VAD model
static bool    g_suppress_nst    = false; // suppress non-speech tokens

// Adaptive-scheduler & safety-net ---------------------------------------------
static const int32_t MIN_STEP_MS   = g_step_ms;   // lower bound for real-time feel
static const int32_t MAX_STEP_MS   = 10000;       // upper bound – keeps latency bounded
static const float   EWMA_ALPHA    = 0.30f;       // smoothing for running average
static const float   SAFETY_FACTOR = 1.10f;       // 10 % head-room between passes

// Ring-buffer cap (discard oldest audio when exceeded) – streaming only
static int32_t g_ring_cap_ms = 30000; // default 30 s, set --ring-cap 0 to disable

// ---------- Main per-connection handler -------------------------------------------

void process_connection(int client_fd, struct whisper_context * ctx) {
    // whisper params (could expose CLI later)
    // Abort flag shared between reader_thread and whisper abort callback
    std::atomic<bool> abort_requested(false);

    int32_t step_ms   = g_step_ms; // mutable for adaptive scheduling
    const int32_t length_ms = g_length_ms;
    const int32_t keep_ms   = g_keep_ms;

    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t beam_size = -1;

    // Adaptive scheduler state
    float   avg_ms   = (float)step_ms; // initialise EWMA

    auto configure_params = [&](whisper_full_params & wparams) {
        wparams.print_progress   = false;
        wparams.print_realtime   = false;
        wparams.print_timestamps = false;
        wparams.max_tokens       = 0;
        wparams.n_threads        = n_threads;
        wparams.beam_search.beam_size = beam_size;
        wparams.no_speech_thold  = g_no_speech_thold;
        wparams.suppress_nst     = g_suppress_nst;
        wparams.vad              = g_enable_vad;
        wparams.vad_model_path   = g_vad_model_path.empty() ? nullptr : g_vad_model_path.c_str();
        wparams.no_context       = true;
    };

    const int n_samples_len  = length_ms * WHISPER_SAMPLE_RATE / 1000;
    const int n_samples_keep = keep_ms   * WHISPER_SAMPLE_RATE / 1000;

    auto run_full_pass = [&](const std::vector<float> & audio) {
        std::string full_text;
        if (audio.empty()) {
            return full_text;
        }

        const int max_audio_ctx = whisper_model_n_audio_ctx(ctx);
        size_t max_chunk_samples = 0;
        if (max_audio_ctx > 0) {
            max_chunk_samples = size_t(max_audio_ctx) * 2 * WHISPER_HOP_LENGTH;
        }
        if (max_chunk_samples == 0) {
            max_chunk_samples = 30u * WHISPER_SAMPLE_RATE; // fall back to ~30 s chunks
        }

        const size_t keep_samples = std::max<int32_t>(g_keep_ms, 0) * WHISPER_SAMPLE_RATE / 1000;
        if (max_chunk_samples <= keep_samples) {
            max_chunk_samples = keep_samples + (WHISPER_SAMPLE_RATE / 2);
        }

        std::string prompt = create_whisper_prompt();
        const char * prompt_cstr = prompt.empty() ? nullptr : prompt.c_str();

        std::vector<float> overlap_tail;
        size_t offset = 0;
        int chunk_index = 0;

        while (offset < audio.size()) {
            size_t remaining = audio.size() - offset;
            size_t take = std::min(max_chunk_samples, remaining);

            std::vector<float> chunk;
            chunk.reserve(overlap_tail.size() + take);
            if (!overlap_tail.empty()) {
                chunk.insert(chunk.end(), overlap_tail.begin(), overlap_tail.end());
            }
            chunk.insert(chunk.end(), audio.begin() + offset, audio.begin() + offset + take);

            whisper_full_params wparams = whisper_full_default_params(beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
            configure_params(wparams);
            wparams.initial_prompt = (chunk_index == 0) ? prompt_cstr : nullptr;
            wparams.no_context = (chunk_index > 0);
            wparams.abort_callback = nullptr;
            wparams.abort_callback_user_data = nullptr;

            if (chunk_index == 0) {
                if (wparams.initial_prompt) {
                    std::cerr << "[whisper-socket] Final pass using prompt: \"" << wparams.initial_prompt << "\"" << std::endl;
                } else {
                    std::cerr << "[whisper-socket] Final pass using no initial prompt" << std::endl;
                }
            }

            auto t_chunk_start = std::chrono::steady_clock::now();
            int rc = whisper_full(ctx, wparams, chunk.data(), static_cast<int>(chunk.size()));
            auto t_chunk_end   = std::chrono::steady_clock::now();
            auto dur_ms        = std::chrono::duration_cast<std::chrono::milliseconds>(t_chunk_end - t_chunk_start).count();

            if (rc != 0) {
                std::cerr << "whisper_full() failed on chunk " << chunk_index << std::endl;
                break;
            }

            std::string part = collect_segments(ctx);
            append_with_overlap(full_text, part);
            log_ts("[FINAL] chunk " + std::to_string(chunk_index + 1) + " duration: " + std::to_string(dur_ms) + " ms");

            const size_t chunk_end = offset + take;
            const size_t tail_len = std::min(keep_samples, take);
            if (tail_len > 0) {
                overlap_tail.assign(audio.begin() + chunk_end - tail_len, audio.begin() + chunk_end);
            } else {
                overlap_tail.clear();
            }

            offset = chunk_end;
            ++chunk_index;
        }

        return full_text;
    };

    // This buffer must be created before the header read block, so we can
    // forward any audio data that might be received along with the header.
    pcm_ring_buffer rb;

    // No longer read headers from main socket - app context comes via control socket

    std::thread reader(reader_thread, client_fd, std::ref(rb), &abort_requested);

    std::vector<float> pcmf32_old;
    // Capture the *entire* audio stream so we can run a final full-context
    // transcription once the user stops speaking. This guarantees that the
    // final output covers the whole utterance instead of only what fits in
    // the rolling window.
    std::vector<float> pcmf32_all;
    // Accumulated transcript across all iterations
    std::string transcript_accum;

    auto send_json = [&](const std::string & type, const std::string & text){
        std::string line = std::string("{\"type\":\"") + type + "\",\"text\":\"" + text + "\"}\n";
        write_all(client_fd, line.data(), line.size());
        // Also print the transcribed text locally for both partial and final results
        if (!text.empty()) {
            log_ts("[" + type + "] " + text);
        }
    };

    log_ts("Mic started / connection opened");
    // processing loop
    while (true) {
        // Compute step size and pop corresponding samples from the ring buffer
        const int n_samples_step = step_ms * WHISPER_SAMPLE_RATE / 1000;

        if (!g_no_stream) {
            const size_t backlog_samples = rb.size();
            const size_t max_backlog_samples = (size_t)std::max(0, n_samples_len + n_samples_keep);
            if (backlog_samples > max_backlog_samples) {
                rb.drop(backlog_samples - max_backlog_samples);
            }
        }

        std::vector<float> pcmf32_new;
        size_t popped = rb.pop(n_samples_step, pcmf32_new);
        if (popped == 0) {
            if (rb.finished()) break;
            continue;
        }

        // If the audio stream has finished, skip further partial inference and jump to final pass.
        if (rb.finished()) {
            // Still append the last batch of samples to the cumulative buffer so the final pass sees them.
            pcmf32_all.insert(pcmf32_all.end(), pcmf32_new.begin(), pcmf32_new.end());
            break;
        }

        const int n_samples_new  = pcmf32_new.size();
        const int n_samples_take = std::min((int)pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

        std::vector<float> pcmf32_cur(n_samples_new + n_samples_take);
        if (n_samples_take) std::copy(pcmf32_old.end()-n_samples_take, pcmf32_old.end(), pcmf32_cur.begin());
        std::copy(pcmf32_new.begin(), pcmf32_new.end(), pcmf32_cur.begin() + n_samples_take);
        pcmf32_old = pcmf32_cur;

        // ------------------------------------------------------------------
        // Append *new* samples (no overlap) to the cumulative buffer so we
        // have the raw audio for a high-fidelity final pass later.
        // ------------------------------------------------------------------
        pcmf32_all.insert(pcmf32_all.end(), pcmf32_new.begin(), pcmf32_new.end());

        if (!g_no_stream) {
            whisper_full_params wparams = whisper_full_default_params(beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
            configure_params(wparams);

            // Set app-aware initial prompt for better transcription - get fresh value each time
            std::string prompt = create_whisper_prompt();
            wparams.initial_prompt = prompt.empty() ? nullptr : prompt.c_str();
            
            if (wparams.initial_prompt) {
                std::cerr << "[whisper-socket] Using prompt: \"" << wparams.initial_prompt << "\"" << std::endl;
            } else {
                std::cerr << "[whisper-socket] Using no initial prompt" << std::endl;
            }

            // Set abort callback so we can cancel this inference if recording stops
            wparams.abort_callback = whisper_should_abort;
            wparams.abort_callback_user_data = &abort_requested;
            abort_requested.store(false); // reset for this inference

            auto t_start = std::chrono::steady_clock::now();
            whisper_full(ctx, wparams, pcmf32_cur.data(), pcmf32_cur.size());
            auto t_end   = std::chrono::steady_clock::now();
            auto dur_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
            std::string part = collect_segments(ctx);
            append_with_overlap(transcript_accum, part);
            send_json("partial", part);
            log_ts(std::string("[PART] transcription time: ") + std::to_string(dur_ms) + " ms");

            // ------------------------------------------------------------------
            // Adaptive step: update EWMA and derive next step_ms within bounds
            // ------------------------------------------------------------------
            avg_ms = (1.0f - EWMA_ALPHA) * avg_ms + EWMA_ALPHA * (float)dur_ms;

            // Manual clamp (C++11 compatible) – avoids std::clamp dependency
            int32_t new_step = int(avg_ms * SAFETY_FACTOR);
            if (new_step < MIN_STEP_MS) new_step = MIN_STEP_MS;
            if (new_step > MAX_STEP_MS) new_step = MAX_STEP_MS;
            step_ms = new_step;
        }

        // ------------------------------------------------------------------
        // Ring-buffer cap – discard oldest audio if backlog exceeds threshold.
        // IMPORTANT: This is ONLY applied in streaming mode. In non-streaming
        // mode we must not drop any audio so the final pass sees the full input.
        // ------------------------------------------------------------------
        if (!g_no_stream && g_ring_cap_ms > 0) {
            while (rb.duration_ms() > g_ring_cap_ms) {
                rb.drop(n_samples_step);
            }
        }
    }

    // flush leftovers
    {
        std::vector<float> pcmf32_new;
        rb.pop_all(pcmf32_new);

        if (!pcmf32_new.empty()) {
            // Just add the leftover samples to the cumulative buffer for the final pass.
            // Skip the redundant whisper processing since the final pass will handle everything.
            pcmf32_all.insert(pcmf32_all.end(), pcmf32_new.begin(), pcmf32_new.end());
        }
    }

    // ------------------------------------------------------------------
    // FINAL PASS – transcribe the *full* audio for maximum accuracy
    // ------------------------------------------------------------------

    std::string final_transcript;
    if (!pcmf32_all.empty()) {
        abort_requested.store(false); // ensure final pass is not aborted
        auto t_start = std::chrono::steady_clock::now();
        final_transcript = run_full_pass(pcmf32_all);
        if (final_transcript.empty()) {
            final_transcript = transcript_accum;
        }
        auto t_end   = std::chrono::steady_clock::now();
        auto dur_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        log_ts(std::string("[FINAL] total duration: ") + std::to_string(dur_ms) + " ms");
    } else {
        // Fallback to whatever we accumulated during streaming (should not happen)
        final_transcript = transcript_accum;
    }

    send_json("final", final_transcript);

    // Small delay to ensure final JSON is fully transmitted before closing connection
    // This prevents race condition where connection closes before interpreter reads final data
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    log_ts("Mic ended / connection closed");

    reader.join();
    ::shutdown(client_fd, SHUT_RDWR);
    ::close(client_fd);
}

// ---------- ggml abort callback -----------------------------------------------------
static bool whisper_should_abort(void * user_data) {
    const std::atomic<bool> * flag = static_cast<const std::atomic<bool> *>(user_data);
    return flag && flag->load();
}

// ---------- Main ------------------------------------------------------------------

struct whisper_context * g_ctx = nullptr;

void cleanup(int sig) {
    if (g_ctx) {
        whisper_free(g_ctx);
        g_ctx = nullptr;
    }
    exit(0);
}

int main(int argc, char ** argv) {
    signal(SIGPIPE, SIG_IGN);
    const char * sock_path = "/tmp/whisper_stream.sock";

    // Print help/usage if requested or no args
    if (argc == 1 || (argc > 1 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0))) {
        std::cerr << "Usage: " << argv[0] << " [options]\n"
                  << "  --socket PATH                 Path to UNIX socket (default: /tmp/whisper_stream.sock)\n"
                  << "  --step N                      Emit partials every N ms (default: 700)\n"
                  << "  --length N                    Rolling window length in ms (streaming; default: 30000)\n"
                  << "  --keep N                      Overlap between windows in ms (default: 200)\n"
                  << "  -nth N, --no-speech-thold N   No speech probability threshold (default: 0.7)\n"
                  << "  --vad                         Enable voice activity detection (VAD) before transcription\n"
                  << "  --vad-model PATH, --vad-path PATH  Path to custom VAD model (.bin)\n"
                  << "  --no-stream                   Only run final full-context pass after stream ends\n"
                  << "  --ring-cap MS                 Max backlog to keep in ring buffer during streaming (0 = unlimited, default: 30000)\n"
                  << "  -sns, --suppress-nst          Suppress non-speech tokens\n"
                  << "  -h, --help                    Show this help message\n";
        return 0;
    }

    if (argc > 1 && std::strcmp(argv[1], "--socket") == 0 && argc > 2) {
        sock_path = argv[2];
    }

    // Optional tuning parameters
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--step") == 0) {
            g_step_ms = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--length") == 0) {
            g_length_ms = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--keep") == 0) {
            g_keep_ms = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "-nth") == 0 || std::strcmp(argv[i], "--no-speech-thold") == 0) {
            g_no_speech_thold = std::atof(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--vad-model") == 0 || std::strcmp(argv[i], "--vad-path") == 0) {
            g_vad_model_path = argv[i + 1];
        } else if (std::strcmp(argv[i], "--ring-cap") == 0) {
            g_ring_cap_ms = std::atoi(argv[i + 1]);
            if (g_ring_cap_ms < 0) g_ring_cap_ms = 0;
        }
    }

    // Warn if VAD requested but no model path provided
    if (g_enable_vad && g_vad_model_path.empty()) {
        std::cerr << "[whisper-socket] --vad specified but no --vad-model/--vad-path given. Disabling VAD." << std::endl;
        g_enable_vad = false;
    }

    // Independent scan for flag-style options (no additional parameter)
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-stream") == 0) {
            g_no_stream = true;
        } else if (std::strcmp(argv[i], "-sns") == 0 || std::strcmp(argv[i], "--suppress-nst") == 0 || std::strcmp(argv[i], "--suppress_nst") == 0) {
            g_suppress_nst = true;
        } else if (std::strcmp(argv[i], "--vad") == 0) {
            g_enable_vad = true;
        }
    }

    ::unlink(sock_path); // remove previous

    int srv_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv_fd < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_un addr; std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path)-1);

    if (::bind(srv_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }

    if (::listen(srv_fd, 4) < 0) {
        perror("listen");
        return 1;
    }

    std::cerr << "[whisper-socket] listening on " << sock_path << std::endl;

    // ---- start control socket thread to receive app context asynchronously ----
    std::thread ctrl_thread([](){
        ::unlink(CTRL_SOCK_PATH);
        int ctrl_fd = ::socket(AF_UNIX, SOCK_DGRAM, 0);
        if (ctrl_fd < 0) {
            perror("ctrl socket");
            return;
        }
        struct sockaddr_un caddr; std::memset(&caddr, 0, sizeof(caddr));
        caddr.sun_family = AF_UNIX;
        std::strncpy(caddr.sun_path, CTRL_SOCK_PATH, sizeof(caddr.sun_path)-1);
        if (::bind(ctrl_fd, (struct sockaddr*)&caddr, sizeof(caddr)) < 0) {
            perror("ctrl bind");
            return;
        }
        char buf[256];
        while (true) {
            ssize_t n = ::recvfrom(ctrl_fd, buf, sizeof(buf)-1, 0, nullptr, nullptr);
            if (n <= 0) continue;
            buf[n] = 0;
            std::string line(buf, n);
            if (line.find("\"type\":\"app_context\"") != std::string::npos) {
                size_t pos = line.find("\"app\":\"");
                if (pos != std::string::npos) {
                    pos += 7;
                    size_t end = line.find('"', pos);
                    if (end != std::string::npos) {
                        std::string app = line.substr(pos, end - pos);
                        {
                            std::lock_guard<std::mutex> lock(g_app_mtx);
                            g_app_context = app;
                        }
                        std::cerr << "[whisper-socket] Updated app context: " << app << std::endl;
                    }
                }
            } else if (line.find("\"type\":\"stream\"") != std::string::npos) {
                bool on = line.find("\"enabled\":true") != std::string::npos;
                g_no_stream = !on;
                std::cerr << "[whisper-socket] Streaming " << (on ? "ENABLED" : "DISABLED") << std::endl;
            }
        }
    });
    ctrl_thread.detach();

    ggml_backend_load_all();

    // ---------------------------------------------------------------------
    // Model path – can be passed via --model <file>, otherwise env WHISPER_MODEL or default path
    const char * model_path = getenv("WHISPER_MODEL");
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            model_path = argv[i + 1];
            break;
        }
    }
    if (!model_path) {
        model_path = "models/ggml-base.en.bin"; // relative to repo root
    }

    std::cerr << "[whisper-socket] loading model " << model_path << " …" << std::endl;

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu  = true;
    cparams.flash_attn = true;
    struct whisper_context * ctx = whisper_init_from_file_with_params(model_path, cparams);
    g_ctx = ctx;
    if (!g_ctx) {
        std::cerr << "failed to load model" << std::endl;
        return 2;
    }
    signal(SIGINT, cleanup);
    signal(SIGTERM, cleanup);

    while (true) {
        int client_fd = ::accept(srv_fd, nullptr, nullptr);
        if (client_fd < 0) {
            perror("accept");
            continue;
        }
        std::cerr << "[whisper-socket] client connected" << std::endl;
        process_connection(client_fd, ctx);
        std::cerr << "[whisper-socket] client done" << std::endl;
    }

    return 0;
}
