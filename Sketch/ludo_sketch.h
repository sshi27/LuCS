/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/

#pragma once

#include "../common.h"
#include "../Ludo/ludo.h"

// Class for efficiently storing key->value mappings when the size is
// known in advance and the keys are pre-hashed into uint64s.
// Keys should have "good enough" randomness (be spread across the
// entire 64 bit space).
//
// Important:  Clients wishing to use deterministic keys must
// ensure that their keys fall in the range 0 .. (uint64max-1);
// the table uses 2^64-1 as the "not occupied" flag.
//
// Inserted k must be unique, and there are no update
// or delete functions (until some subsequent use of this table
// requires them).
//
// Threads must synchronize their access to a PresizedHeadlessCuckoo.
//
// The cuckoo hash table is 4-way associative (each "bucket" has 4
// "slots" for key/value entries).  Uses breadth-first-search to find
// a good cuckoo path with less data movement (see
// http://www.cs.cmu.edu/~dga/papers/cuckoo-eurosys14.pdf )

struct LS_PathEntry {
  uint32_t bid: 30;
  uint8_t sid : 2;
  uint8_t newSeed;
  uint8_t s0:2, s1:2, s2:2, s3:2;
  vector<uint32_t> locatorCC;
  // locatorCC contains one end of the modified key, while overflowCC contains both ends
};

template<class Key, class Value, uint8_t VL = sizeof(Value) * 8, uint8_t DL = 0>
class DataPlaneLudoSketch {
  static const uint8_t kSlotsPerBucket = 4;   // modification to this value leads to undefined behavior
  static const uint8_t bucketLength = LocatorSeedLength + kSlotsPerBucket * (VL + DL);

  static const uint64_t ValueMask = (1ULL << VL) - 1;
  static const uint64_t DigestMask = ((1ULL << DL) - 1) << VL;
  static const uint64_t VDMask = (1ULL << (VL + DL)) - 1;

public:
  FastHasher64<Key> h;
  uint32_t num_buckets_;

  std::vector<uint64_t> memory;
  FastHasher64<Key> digestH;
  DataPlaneOthello<Key, uint8_t, 1> locator;
  CuckooHashTable<uint32_t, uint8_t> overflow;

  struct Bucket {  // only as parameters and return values for easy access. the storage is compact.
    uint8_t seed;
    Value values[kSlotsPerBucket];

    bool operator==(const Bucket &other) const {
      if (seed != other.seed) return false;

      for (char s = 0; s < kSlotsPerBucket; s++) {
        if ((values[s] & ValueMask) != (other.values[s] & ValueMask)) return false;
      }

      return true;
    }

    bool operator!=(const Bucket &other) const {
      return !(*this == other);
    }
  };

  explicit DataPlaneLudoSketch(const ControlPlaneLudo<Key, Value, VL, DL> &cp)
    : num_buckets_(cp.buckets_.size()), h(cp.h), locator(cp.locator), overflow(cp.entryCount * 0.012),
      digestH(cp.digestH) {

    resetMemory();

    for (uint32_t bktIdx = 0; bktIdx < num_buckets_; ++bktIdx) {
      const typename ControlPlaneLudo<Key, Value, VL>::Bucket &cpBucket = cp.buckets_[bktIdx];
      Bucket dpBucket;
      dpBucket.seed = cpBucket.seed > MaxArrangementSeed ? MaxArrangementSeed : cpBucket.seed;
      memset(dpBucket.values, 0, kSlotsPerBucket * sizeof(Value));

      if (cpBucket.seed >= MaxArrangementSeed) {
        overflow.insert(bktIdx, cpBucket.seed);
      }

      const FastHasher64<Key> locateHash(cpBucket.seed);

      for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
        if (cpBucket.occupiedMask & (1U << slot)) {
          const Key &k = cpBucket.keys[slot];
          dpBucket.values[locateHash(k) >> 62] = cpBucket.values[slot];
        }
      }

      writeBucket(dpBucket, bktIdx);
    }
  }

  template<class V2>
  DataPlaneLudoSketch(const ControlPlaneLudo<Key, V2, VL, DL> &cp, unordered_map<V2, Value> m)
    : num_buckets_(cp.buckets_.size()), h(cp.h), locator(cp.locator),
      overflow(cp.entryCount * 0.012), digestH(cp.digestH) {

    resetMemory();

    for (uint32_t bktIdx = 0; bktIdx < num_buckets_; ++bktIdx) {
      const typename ControlPlaneLudo<Key, V2, VL>::Bucket &cpBucket = cp.buckets_[bktIdx];
      Bucket dpBucket;
      dpBucket.seed = cpBucket.seed >= MaxArrangementSeed ? MaxArrangementSeed : cpBucket.seed;
      memset(dpBucket.values, 0, kSlotsPerBucket * sizeof(Value));

      if (cpBucket.seed >= MaxArrangementSeed) {
        overflow.insert(bktIdx, cpBucket.seed);
      }

      const FastHasher64<Key> locateHash(cpBucket.seed);

      for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
        if (cpBucket.occupiedMask & (1U << slot)) {
          const Key &k = cpBucket.keys[slot];
          dpBucket.values[locateHash(k) >> 62] = m[cpBucket.values[slot]];
        }
      }

      writeBucket(dpBucket, bktIdx);
    }
  }

  inline void resetMemory() {
    memory.resize(((uint64_t) num_buckets_ * bucketLength + 63) / 64);
  }

  template<char length>
  inline void writeMem(uint64_t start, char offset, uint64_t v) {
    assert(v < (1ULL << length));

    // [offset, offset + length) should be 0 and others are 1
    uint64_t mask = ~uint64_t(((1ULL << length) - 1) << offset);
    char overflow = char(offset + length - 64);

    memory[start] &= mask;
    memory[start] |= (uint64_t) v << offset;

    if (overflow > 0) {
      mask = uint64_t(-1) << overflow;     // lower "overflow" bits should be 0, and others are 1
      memory[start + 1] &= mask;
      memory[start + 1] |= v >> (length - overflow);
    }

    assert(v == readMem<length>(start, offset));
  }

  template<char length>
  inline uint64_t readMem(uint64_t start, char offset) const {
    char left = char(offset + length - 64);
    left = char(left < 0 ? 0 : left);

    uint64_t mask = ~(uint64_t(-1)
      << (length - left));   // lower length-left bits should be 1, and others are 0
    uint64_t result = (memory[start] >> offset) & mask;

    if (left > 0) {
      mask = ~(uint64_t(-1) << left);     // lower left bits should be 1, and others are 0
      result |= (memory[start + 1] & mask) << (length - left);
    }

    return result;
  }

  vector<uint8_t> lock = vector<uint8_t>(8192, 0);

  // Only call it during update!
  inline void writeBucket(Bucket &bucket, uint32_t index) {
    uint64_t i1 = (uint64_t) index * bucketLength;
    uint64_t start = i1 / 64;
    char offset = char(i1 % 64);

    assert(bucket.seed <= MaxArrangementSeed);
    writeMem<LocatorSeedLength>(start, offset, bucket.seed);

    for (char i = 0; i < kSlotsPerBucket; ++i) {
      uint64_t offsetFromBeginning = i1 + LocatorSeedLength + i * VL;
      start = offsetFromBeginning / 64;
      offset = char(offsetFromBeginning % 64);

      writeMem<VL>(start, offset, bucket.values[i]);
    }

#ifndef NDEBUG
    Bucket retrieved = readBucket(index);
    retrieved.seed = min(MaxArrangementSeed, retrieved.seed);
    assert(retrieved == bucket);
#endif
  }

  inline Bucket readBucket(uint32_t index) const {
    Bucket bucket;

    uint64_t i1 = (uint64_t) index * bucketLength;
    uint64_t start = i1 / 64;
    char offset = char(i1 % 64);

    bucket.seed = readMem<LocatorSeedLength>(start, offset);
    if (bucket.seed == MaxArrangementSeed) {
      overflow.lookUp(index, bucket.seed);
    }

    for (char i = 0; i < kSlotsPerBucket; ++i) {
      uint64_t offsetFromBeginning = i1 + LocatorSeedLength + i * VL;
      start = offsetFromBeginning / 64;
      offset = char(offsetFromBeginning % 64);

      bucket.values[i] = readMem<VL>(start, offset);
    }

    return bucket;
  }

  inline void writeSlot(uint32_t bid, char sid, Value val) {
    uint64_t offsetFromBeginning = uint64_t(bid) * bucketLength + LocatorSeedLength + sid * VL;
    uint64_t start = offsetFromBeginning / 64;
    char offset = char(offsetFromBeginning % 64);

    writeMem<VL>(start, offset, val);
  }

  inline Value readSlot(uint32_t bid, char sid) {
    uint64_t offsetFromBeginning = uint64_t(bid) * bucketLength + LocatorSeedLength + sid * VL;
    uint64_t start = offsetFromBeginning / 64;
    char offset = char(offsetFromBeginning % 64);

    return readMem<VL>(start, offset);
  }

  unordered_map<Key, Value> fallback;

  // Returns true if found.  Sets *out = value.
  inline bool lookUp(const Key &k, Value &out) {
    if (!fallback.empty()) {
      auto it = fallback.find(k);
      if (it != fallback.end())
        return it->second;
    }
    uint32_t buckets[2];
    fast_map_to_buckets(h(k), buckets);

    while (true) {
      uint8_t va1 = lock[buckets[0] & 8191], vb1 = lock[buckets[1] & 8191];
      COMPILER_BARRIER();
      if (va1 % 2 == 1 || vb1 % 2 == 1) continue;

      Bucket bucket = readBucket(buckets[locator.lookUp(k)]);

      COMPILER_BARRIER();
      uint8_t va2 = lock[buckets[0] & 8191], vb2 = lock[buckets[1] & 8191];

      if (va1 != va2 || vb1 != vb2) continue;

      uint64_t i = FastHasher64<Key>(bucket.seed)(k) >> 62;
      Value result = bucket.values[i];

      if ((result & DigestMask) == ((digestH(k) << VL) & DigestMask)) {
        out = result & ValueMask;
        return true;
      } else { return false; }
    }
  }

  inline void applyInsert(const vector<LS_PathEntry> &path, Value value) {
    for (int i = 0; i < path.size(); ++i) {
      LS_PathEntry entry = path[i];
      Bucket bucket = readBucket(entry.bid);
      bucket.seed = min(entry.newSeed, MaxArrangementSeed);

      uint8_t toSlots[] = {entry.s0, entry.s1, entry.s2, entry.s3};

      Value buffer[4];       // solve the permutation is slow. just copy the 4 elements
      for (char s = 0; s < 4; ++s) {
        buffer[s] = bucket.values[s];
      }

      for (char s = 0; s < 4; ++s) {
        bucket.values[toSlots[s]] = buffer[s];
      }

      if (i + 1 == path.size()) {  // put the new value
        bucket.values[entry.sid] = value;
      } else {  // move key from another bucket and slot to this bucket and slot
        LS_PathEntry from = path[i + 1];
        uint8_t tmp[4] = {from.s0, from.s1, from.s2, from.s3};
        uint8_t sid;
        for (uint8_t ii = 0; ii < 4; ++ii) {
          if (tmp[ii] == from.sid) {
            sid = ii;
            break;
          }
        }
        bucket.values[entry.sid] = readSlot(from.bid, sid);
      }

      lock[entry.bid & 8191]++;
      COMPILER_BARRIER();

      if (entry.locatorCC.size()) {
        locator.fixHalfTreeByConnectedComponent(entry.locatorCC, 1);
      }

      if (bucket.seed == MaxArrangementSeed) {
        overflow.insert(entry.bid, entry.newSeed, true);
      }
      writeBucket(bucket, entry.bid);

      COMPILER_BARRIER();
      lock[entry.bid & 8191]++;
    }
  }

  inline void applyUpdate(uint32_t bs, Value val) {
    uint32_t bid = bs >> 2;
    uint8_t sid = bs & 3;

    lock[bid & 8191]++;
    COMPILER_BARRIER();

    writeSlot(bid, sid, val & ValueMask);

    COMPILER_BARRIER();
    lock[bid & 8191]++;
  }

  inline uint64_t getMemoryCost() const {
    return memory.size() * 8;
  }

  // Utility function to compute (x * y) >> 64, or "multiply high".
  // On x86-64, this is a single instruction, but not all platforms
  // support the __uint128_t type, so we provide a generic
  // implementation as well.
  inline uint32_t multiply_high_u32(uint32_t x, uint32_t y) const {
    return (uint32_t) (((uint64_t) x * (uint64_t) y) >> 32);
  }

  inline void fast_map_to_buckets(uint64_t x, uint32_t *twoBuckets) const {
    // Map x (uniform in 2^64) to the range [0, num_buckets_ -1]
    // using Lemire's alternative to modulo reduction:
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    // Instead of x % N, use (x * N) >> 64.
    twoBuckets[0] = multiply_high_u32(x, num_buckets_);
    twoBuckets[1] = multiply_high_u32(x >> 32, num_buckets_);
  }
};

