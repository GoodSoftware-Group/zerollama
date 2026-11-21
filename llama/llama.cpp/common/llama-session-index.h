#pragma once

// Longest earlier copy of the text so far, across every sequence this
// server has seen. A match proposes the tokens that followed it. Adding a
// token is amortized constant time. Past `capacity` tokens the older half
// is forgotten and the index is rebuilt, which keeps it near a hundred
// bytes per token held.

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

struct llama_session_index {
    static constexpr uint32_t NONE = UINT32_MAX;
    // Deeper states stand for short suffixes and keep an older occurrence.
    static constexpr size_t RECENCY_DEPTH = 64;

    struct node {
        uint32_t len       = 0;
        uint32_t link      = NONE;
        uint32_t first_out = NONE;
        uint32_t recent    = NONE;
        uint32_t earlier   = NONE;
    };

    std::vector<uint32_t> tokens;
    std::vector<node> nodes;
    std::unordered_map<uint64_t, uint32_t> edges;
    // Each state's tokens as a linked list, so a state can be cloned.
    std::vector<std::pair<uint32_t, uint32_t>> out;
    uint32_t last = 0;
    size_t capacity = 1u << 18;
    // Shortest match, counting the token about to run, worth drafting from.
    size_t min_match = 3;

    explicit llama_session_index(size_t capacity_in = 1u << 18) : capacity(capacity_in < 2 ? 2 : capacity_in) {
        clear();
    }

    size_t len() const { return tokens.size(); }

    void clear() {
        tokens.clear();
        nodes.clear();
        edges.clear();
        out.clear();
        nodes.push_back(node{});
        last = 0;
    }

    void extend(const uint32_t * data, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            if (tokens.size() == capacity) {
                const size_t mid = capacity / 2;
                std::vector<uint32_t> keep(tokens.begin() + mid, tokens.end());
                clear();
                for (uint32_t t : keep) {
                    push(t);
                }
            }
            push(data[i]);
        }
    }

    // Up to `limit` tokens that followed the latest earlier occurrence of
    // the longest suffix of the history that continues with `next`.
    std::vector<uint32_t> draft(uint32_t next, size_t limit) const {
        uint32_t v = last;
        while (v != NONE && edges.find(key(v, next)) == edges.end()) {
            v = nodes[v].link;
        }
        if (v == NONE || (size_t) nodes[v].len < (min_match > 0 ? min_match - 1 : 0)) {
            return {};
        }
        const node & found = nodes[edges.at(key(v, next))];
        const size_t n = tokens.size();
        const uint32_t ends[2] = { found.recent, found.earlier };
        for (uint32_t end : ends) {
            if (end == NONE) {
                continue;
            }
            const size_t start = (size_t) end + 1;
            if (start >= n) {
                continue;
            }
            const size_t stop = start + limit < n ? start + limit : n;
            return std::vector<uint32_t>(tokens.begin() + start, tokens.begin() + stop);
        }
        return {};
    }

private:
    static uint64_t key(uint32_t from, uint32_t token) {
        return ((uint64_t) from << 32) | token;
    }

    uint32_t new_node(uint32_t len, uint32_t link, uint32_t recent, uint32_t earlier) {
        nodes.push_back(node{ len, link, NONE, recent, earlier });
        return (uint32_t) (nodes.size() - 1);
    }

    void add_edge(uint32_t from, uint32_t token, uint32_t to) {
        edges.insert_or_assign(key(from, token), to);
        out.push_back({ token, nodes[from].first_out });
        nodes[from].first_out = (uint32_t) (out.size() - 1);
    }

    void push(uint32_t c) {
        const uint32_t end = (uint32_t) tokens.size();
        tokens.push_back(c);
        const uint32_t cur = new_node(nodes[last].len + 1, NONE, end, NONE);
        uint32_t p = last;
        while (p != NONE && edges.find(key(p, c)) == edges.end()) {
            add_edge(p, c, cur);
            p = nodes[p].link;
        }
        const uint32_t link = p == NONE ? 0 : link_of(p, c);
        nodes[cur].link = link;
        last = cur;

        uint32_t s = link;
        for (size_t i = 0; i < RECENCY_DEPTH; ++i) {
            if (s == 0) {
                break;
            }
            node & nd = nodes[s];
            nd.earlier = nd.recent;
            nd.recent = end;
            s = nd.link;
        }
    }

    // The state for the strings of `q` no longer than len(p) + 1, which now
    // also end at the new token `c`.
    uint32_t link_of(uint32_t p, uint32_t c) {
        const uint32_t q = edges.at(key(p, c));
        if (nodes[p].len + 1 == nodes[q].len) {
            return q;
        }
        const uint32_t link = nodes[q].link;
        const uint32_t recent = nodes[q].recent;
        const uint32_t earlier = nodes[q].earlier;
        const uint32_t clone = new_node(nodes[p].len + 1, link, recent, earlier);
        uint32_t e = nodes[q].first_out;
        while (e != NONE) {
            const auto edge = out[e];
            add_edge(clone, edge.first, edges.at(key(q, edge.first)));
            e = edge.second;
        }
        while (p != NONE && edges.find(key(p, c)) != edges.end() && edges.at(key(p, c)) == q) {
            edges.insert_or_assign(key(p, c), clone);
            p = nodes[p].link;
        }
        nodes[q].link = clone;
        return clone;
    }
};

#ifdef LLAMA_SESSION_INDEX_TEST
#include <cstdio>
#include <cstdlib>

static std::vector<uint32_t> session_expected(const std::vector<uint32_t> & history, uint32_t next, size_t limit, size_t min_match) {
    std::vector<uint32_t> text = history;
    text.push_back(next);
    const size_t n = history.size();
    auto occurs = [&](size_t len) {
        std::vector<size_t> starts;
        if (len > n) {
            return starts;
        }
        const uint32_t * suffix = text.data() + (text.size() - len);
        for (size_t i = n - len + 1; i-- > 0; ) {
            bool same = true;
            for (size_t k = 0; k < len; ++k) {
                if (history[i + k] != suffix[k]) {
                    same = false;
                    break;
                }
            }
            if (same) {
                starts.push_back(i + len);
            }
            if (i == 0) {
                break;
            }
        }
        return starts;
    };
    size_t found = 0;
    bool any = false;
    for (size_t len = n; len >= 1; --len) {
        if (!occurs(len).empty()) {
            found = len;
            any = true;
            break;
        }
        if (len == 1) {
            break;
        }
    }
    if (!any || found < min_match) {
        return {};
    }
    for (size_t start : occurs(found)) {
        if (start < n) {
            const size_t stop = start + limit < n ? start + limit : n;
            return std::vector<uint32_t>(history.begin() + start, history.begin() + stop);
        }
    }
    return {};
}

static bool session_eq(const std::vector<uint32_t> & a, const std::vector<uint32_t> & b) {
    return a == b;
}

int main() {
    {
        llama_session_index index(64);
        index.min_match = 2;
        const uint32_t hist[] = { 1, 2, 3, 4, 9, 9, 2, 3, 5, 7, 1, 2 };
        index.extend(hist, sizeof(hist) / sizeof(hist[0]));
        if (!session_eq(index.draft(3, 3), { 4, 9, 9 })) {
            std::fputs("longest latest match should continue 1,2,3 with 4,9,9\n", stderr);
            return 1;
        }
        const uint32_t more[] = { 3, 8, 8, 2 };
        index.extend(more, 4);
        if (!session_eq(index.draft(3, 2), { 8, 8 })) {
            std::fputs("a newer 2,3 should continue with 8,8\n", stderr);
            return 1;
        }
    }
    {
        llama_session_index index(8);
        index.min_match = 2;
        const uint32_t hist[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        index.extend(hist, 9);
        if (index.len() != 5) {
            std::fputs("a full index should keep the newer half\n", stderr);
            return 1;
        }
        if (!index.draft(2, 2).empty()) {
            std::fputs("the forgotten half should not draft\n", stderr);
            return 1;
        }
        const uint32_t five = 5;
        index.extend(&five, 1);
        if (!session_eq(index.draft(6, 2), { 7, 8 })) {
            std::fputs("the kept half should still draft\n", stderr);
            return 1;
        }
    }
    {
        uint64_t rng = 12345;
        auto random = [&](uint64_t m) {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            return (uint32_t) ((rng >> 33) % m);
        };
        for (uint64_t alphabet : { 2ull, 3ull, 5ull }) {
            llama_session_index index(1u << 12);
            index.min_match = 2;
            std::vector<uint32_t> history;
            for (int step = 0; step < 400; ++step) {
                const uint32_t t = random(alphabet);
                index.extend(&t, 1);
                history.push_back(t);
                const uint32_t next = random(alphabet);
                if (!session_eq(index.draft(next, 6), session_expected(history, next, 6, 2))) {
                    std::fprintf(stderr, "draft disagreed with brute force at step %d\n", step);
                    return 1;
                }
            }
        }
    }
    std::puts("session ok");
    return 0;
}
#endif
