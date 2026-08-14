
#pragma once
#include "gguf.h"
#include "unicode_categories.h"

#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

static void build_byte_encoder(std::string byte2str[256]) {

    int bs[256], cs[256], n = 0, total = 0;

    for (int b = '!'; b <= '~'; b++) {
        bs[total] = b;
        cs[total] = b;
        total++;
    }
    for (int b = 0xA1; b <= 0xAC; b++) {
        bs[total] = b;
        cs[total] = b;
        total++;
    }
    for (int b = 0xAE; b <= 0xFF; b++) {
        bs[total] = b;
        cs[total] = b;
        total++;
    }

    bool used[256] = {};
    for (int i = 0; i < total; i++) {
        used[bs[i]] = true;
    }
    for (int b = 0; b < 256; b++) {
        if (!used[b]) {
            bs[total] = b;
            cs[total] = 256 + n;
            n++;
            total++;
        }
    }
    assert(total == 256);

    for (int i = 0; i < 256; i++) {
        int  cp = cs[i];
        char buf[4];
        int  len;
        if (cp < 0x80) {
            buf[0] = (char) cp;
            len    = 1;
        } else if (cp < 0x800) {
            buf[0] = (char) (0xC0 | (cp >> 6));
            buf[1] = (char) (0x80 | (cp & 0x3F));
            len    = 2;
        } else {
            buf[0] = (char) (0xE0 | (cp >> 12));
            buf[1] = (char) (0x80 | ((cp >> 6) & 0x3F));
            buf[2] = (char) (0x80 | (cp & 0x3F));
            len    = 3;
        }
        byte2str[bs[i]] = std::string(buf, len);
    }
}

static bool utf8_is_continuation(unsigned char value) {
    return (value & 0xC0) == 0x80;
}

static int utf8_codepoint(const char * s, int remaining, int * advance) {
    if (remaining <= 0) {
        *advance = 0;
        return -1;
    }
    const unsigned char c = static_cast<unsigned char>(s[0]);
    if (c < 0x80) {
        *advance = 1;
        return c;
    }
    if (remaining >= 2 && c >= 0xC2 && c <= 0xDF &&
        utf8_is_continuation(static_cast<unsigned char>(s[1]))) {
        *advance = 2;
        return ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[1]) & 0x3F);
    }
    if (remaining >= 3 && c >= 0xE0 && c <= 0xEF &&
        utf8_is_continuation(static_cast<unsigned char>(s[1])) &&
        utf8_is_continuation(static_cast<unsigned char>(s[2])) &&
        !(c == 0xE0 && static_cast<unsigned char>(s[1]) < 0xA0) &&
        !(c == 0xED && static_cast<unsigned char>(s[1]) >= 0xA0)) {
        *advance = 3;
        return ((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[1]) & 0x3F) << 6) |
               (static_cast<unsigned char>(s[2]) & 0x3F);
    }
    if (remaining >= 4 && c >= 0xF0 && c <= 0xF4 &&
        utf8_is_continuation(static_cast<unsigned char>(s[1])) &&
        utf8_is_continuation(static_cast<unsigned char>(s[2])) &&
        utf8_is_continuation(static_cast<unsigned char>(s[3])) &&
        !(c == 0xF0 && static_cast<unsigned char>(s[1]) < 0x90) &&
        !(c == 0xF4 && static_cast<unsigned char>(s[1]) >= 0x90)) {
        *advance = 4;
        return ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[1]) & 0x3F) << 12) |
               ((static_cast<unsigned char>(s[2]) & 0x3F) << 6) |
               (static_cast<unsigned char>(s[3]) & 0x3F);
    }
    *advance = 1;
    return c;
}

static bool is_letter(int cp) {
    return cp >= 0 && mm3_unicode_is_letter((uint32_t) cp);
}

static bool is_digit(int cp) {
    return cp >= 0 && mm3_unicode_is_number((uint32_t) cp);
}

static bool is_whitespace(int cp) {
    return cp >= 0 && mm3_unicode_is_whitespace((uint32_t) cp);
}

static bool is_newline(int cp) {
    return cp == '\n' || cp == '\r';
}

static std::vector<std::string> gpt2_pre_tokenize(const std::string & text) {
    std::vector<std::string> chunks;
    const char *             s   = text.c_str();
    int                      len = (int) text.size();
    int                      i   = 0;

    while (i < len) {
        int adv;
        int cp = utf8_codepoint(s + i, len - i, &adv);

        if ((cp == '\'' || cp == 0x2019) && i + adv < len) {
            const char * rest      = s + i + adv;
            int          rlen      = len - i - adv;
            auto         try_match = [&](const char * suffix, int slen) -> bool {
                if (rlen >= slen) {

                    for (int k = 0; k < slen; k++) {
                        char c1 = rest[k], c2 = suffix[k];
                        if (c1 >= 'A' && c1 <= 'Z') {
                            c1 = (char) (c1 + 32);
                        }
                        if (c1 != c2) {
                            return false;
                        }
                    }

                    if (rlen > slen) {
                        int a2;
                        int cp2 = utf8_codepoint(rest + slen, rlen - slen, &a2);
                        if (is_letter(cp2)) {
                            return false;
                        }
                    }
                    chunks.push_back(std::string(s + i, adv + slen));
                    i += adv + slen;
                    return true;
                }
                return false;
            };
            if (try_match("ll", 2)) {
                continue;
            }
            if (try_match("re", 2)) {
                continue;
            }
            if (try_match("ve", 2)) {
                continue;
            }
            if (try_match("s", 1)) {
                continue;
            }
            if (try_match("t", 1)) {
                continue;
            }
            if (try_match("m", 1)) {
                continue;
            }
            if (try_match("d", 1)) {
                continue;
            }
        }

        if (is_letter(cp)) {
            int start = i;
            i += adv;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, len - i, &a2);
                if (!is_letter(cp2)) {
                    break;
                }
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }
        if (!is_newline(cp) && !is_letter(cp) && !is_digit(cp) && !is_whitespace(cp)) {

            int start = i;
            int after = i + adv;
            if (after < len) {
                int a2;
                int cp2 = utf8_codepoint(s + after, len - after, &a2);
                if (is_letter(cp2)) {
                    i = after + a2;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, len - i, &a3);
                        if (!is_letter(cp3)) {
                            break;
                        }
                        i += a3;
                    }
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
            }
        }

        if (is_digit(cp)) {
            const int start = i;
            i += adv;
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        if (is_newline(cp)) {
            int start = i;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, len - i, &a2);
                if (!is_newline(cp2)) {
                    break;
                }
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        if (is_whitespace(cp)) {
            int start  = i;

            int ws_end = i;
            int last_start = i;
            int ws_count = 0;
            while (ws_end < len) {
                int a2;
                int cp2 = utf8_codepoint(s + ws_end, len - ws_end, &a2);
                if (!is_whitespace(cp2) || is_newline(cp2)) {
                    break;
                }
                last_start = ws_end;
                ws_end += a2;
                ws_count++;
            }

            bool followed_by_non_ws = false;
            if (ws_end < len) {
                int a2;
                int cp2 = utf8_codepoint(s + ws_end, len - ws_end, &a2);
                followed_by_non_ws = !is_whitespace(cp2) && !is_newline(cp2);
            }
            if (followed_by_non_ws && ws_count > 1) {

                chunks.push_back(std::string(s + start, last_start - start));
                i = last_start;
                continue;
            }

            i = start + adv;
            if (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, len - i, &a2);
                if (is_letter(cp2)) {
                    i += a2;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, len - i, &a3);
                        if (!is_letter(cp3)) {
                            break;
                        }
                        i += a3;
                    }
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
                if (is_digit(cp2)) {
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
                if (!is_whitespace(cp2) && !is_newline(cp2)) {
                    int pstart = start;
                    while (i < len) {
                        int a3;
                        int cp3 = utf8_codepoint(s + i, len - i, &a3);
                        if (is_whitespace(cp3) || is_letter(cp3) || is_digit(cp3)) {
                            break;
                        }
                        i += a3;
                    }
                    while (i < len) {
                        int a4;
                        int cp4 = utf8_codepoint(s + i, len - i, &a4);
                        if (!is_newline(cp4)) {
                            break;
                        }
                        i += a4;
                    }
                    chunks.push_back(std::string(s + pstart, i - pstart));
                    continue;
                }
            }

            i = ws_end;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, len - i, &a2);
                if (!is_whitespace(cp2)) {
                    break;
                }
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        {
            int start = i;
            i += adv;
            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, len - i, &a2);
                if (is_whitespace(cp2) || is_letter(cp2) || is_digit(cp2) || is_newline(cp2)) {
                    break;
                }
                i += a2;
            }

            while (i < len) {
                int a2;
                int cp2 = utf8_codepoint(s + i, len - i, &a2);
                if (!is_newline(cp2)) {
                    break;
                }
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
        }
    }
    return chunks;
}

struct BPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<std::string, int> merges;
    std::string                          byte2str[256];
    int                                  eos_id;
    int                                  n_vocab;
    std::vector<std::string>             id_to_str;
};

static bool load_bpe_from_gguf(BPETokenizer * tok, const char * gguf_path) {
    build_byte_encoder(tok->byte2str);

    struct gguf_init_params gp  = { true, NULL };
    struct gguf_context *   ctx = gguf_init_from_file(gguf_path, gp);
    if (!ctx) {
        fprintf(stderr, "[BPE] Failed to open %s\n", gguf_path);
        return false;
    }

    int64_t tok_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    int64_t mrg_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (tok_key < 0 || mrg_key < 0) {
        fprintf(stderr, "[BPE] Tokenizer not found in %s\n", gguf_path);
        gguf_free(ctx);
        return false;
    }

    int n_tokens = (int) gguf_get_arr_n(ctx, tok_key);
    int n_merges = (int) gguf_get_arr_n(ctx, mrg_key);

    for (int i = 0; i < n_tokens; i++) {
        const char * s             = gguf_get_arr_str(ctx, tok_key, (size_t) i);
        tok->vocab[std::string(s)] = i;
    }

    for (int i = 0; i < n_merges; i++) {
        const char * s              = gguf_get_arr_str(ctx, mrg_key, (size_t) i);
        tok->merges[std::string(s)] = i;
    }

    gguf_free(ctx);

    tok->n_vocab = (int) tok->vocab.size();
    tok->eos_id  = 151643;

    tok->id_to_str.resize(tok->n_vocab);
    for (auto & kv : tok->vocab) {
        if (kv.second >= 0 && kv.second < tok->n_vocab) {
            tok->id_to_str[kv.second] = kv.first;
        }
    }

    fprintf(stderr, "[BPE] Loaded from GGUF: %d vocab, %d merges\n", tok->n_vocab, n_merges);
    return true;
}

static std::string byte_level_encode(const BPETokenizer * tok, const std::string & text) {
    std::string out;
    for (unsigned char c : text) {
        out += tok->byte2str[c];
    }
    return out;
}

static std::vector<std::string> bpe_merge(const std::unordered_map<std::string, int> & merge_rank,
                                          const std::vector<std::string> &             symbols) {
    if (symbols.size() <= 1) {
        return symbols;
    }

    std::vector<std::string> work = symbols;

    while (work.size() > 1) {

        int best_rank = INT_MAX;
        int best_pos  = -1;
        for (int i = 0; i < (int) work.size() - 1; i++) {
            std::string key = work[i] + " " + work[i + 1];
            auto        it  = merge_rank.find(key);
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = i;
            }
        }
        if (best_pos < 0) {
            break;
        }

        std::string merged = work[best_pos] + work[best_pos + 1];
        work[best_pos]     = merged;
        work.erase(work.begin() + best_pos + 1);
    }
    return work;
}

static void encode_chunk(const BPETokenizer * tok, const std::string & chunk, std::vector<int> & ids) {

    std::string encoded = byte_level_encode(tok, chunk);

    std::vector<std::string> symbols;
    const char *             s   = encoded.c_str();
    int                      len = (int) encoded.size();
    int                      i   = 0;
    while (i < len) {
        int adv;
        utf8_codepoint(s + i, len - i, &adv);
        symbols.push_back(std::string(s + i, adv));
        i += adv;
    }

    std::vector<std::string> merged = bpe_merge(tok->merges, symbols);

    for (const auto & piece : merged) {
        auto it = tok->vocab.find(piece);
        if (it != tok->vocab.end()) {
            ids.push_back(it->second);
        } else {

            fprintf(stderr, "[BPE] WARNING: unknown token '%s'\n", piece.c_str());
            for (unsigned char c : piece) {
                auto it2 = tok->vocab.find(std::string(1, c));
                if (it2 != tok->vocab.end()) {
                    ids.push_back(it2->second);
                }
            }
        }
    }
}

static std::vector<int> bpe_encode(const BPETokenizer * tok, const std::string & text, bool add_eos = true) {
    std::vector<int>  ids;
    const std::string special = "<|endoftext|>";

    size_t pos = 0;
    while (pos < text.size()) {
        size_t      found   = text.find(special, pos);
        std::string segment = (found == std::string::npos) ? text.substr(pos) : text.substr(pos, found - pos);

        if (!segment.empty()) {
            auto chunks = gpt2_pre_tokenize(segment);
            for (const auto & chunk : chunks) {
                encode_chunk(tok, chunk, ids);
            }
        }

        if (found == std::string::npos) {
            break;
        }
        ids.push_back(tok->eos_id);
        pos = found + special.size();
    }

    if (add_eos) {
        ids.push_back(tok->eos_id);
    }
    return ids;
}
