// Session-only record exclusions. Never encoded as classification bytes.
#pragma once
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>
namespace lnt {
using DeletionMask = std::unordered_map<uint64_t, std::vector<uint32_t>>;
using SharedDeletionMask = std::shared_ptr<const DeletionMask>;
inline bool record_deleted(const SharedDeletionMask& mask, uint64_t node, uint32_t local) {
    if (!mask) return false;
    auto it = mask->find(node);
    return it != mask->end() && std::binary_search(it->second.begin(), it->second.end(), local);
}
}
