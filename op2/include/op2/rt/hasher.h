#pragma once

// A rapidhash over words added one at a time: a call's shape, a parameter
// set's key, a plan's key.  The words are hashed a block of Block at a time,
// each block seeded with the hash so far.

#include <extern/rapidhash.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace op::rt {

template<typename Word, std::size_t Block>
class Hasher {
private:
    std::array<Word, Block> m_words;
    std::size_t m_count = 0;
    std::uint64_t m_hash;

public:
    explicit Hasher(std::uint64_t seed = RAPID_SEED) : m_hash{seed} {}

    Hasher& add(Word word) {
        if (m_count == Block) {
            m_hash = rapidhash_withSeed(m_words.data(), Block * sizeof(Word), m_hash);
            m_count = 0;
        }
        m_words[m_count++] = word;
        return *this;
    }

    // The hash of the words added so far.
    std::uint64_t value() const {
        return rapidhash_withSeed(m_words.data(), m_count * sizeof(Word), m_hash);
    }
};

} // namespace op::rt
