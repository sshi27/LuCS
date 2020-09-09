#include <FilterCascading/FilterCascading.h>
#include "common.h"
#include "cstdlib"
#include "SetSep/setsep.h"
#include "BloomFilter/bloom_flitable.h"
#include "CuckooPresized/cuckoo_map.h"
#include "CuckooPresized/cuckoo_ht.h"
#include "CuckooPresized/cuckoo_filter_control_plane.h"
#include "CuckooPresized/cuckoo_filtable.h"
#include "Othello/data_plane_othello.h"
#include "Ludo/ludo.h"
#include "DPH/dph.h"

int version = 12;

template<class K>
struct OthelloChange {
  int8_t type;
  vector<uint32_t> cc;
  uint64_t xorTemplate;
  int marks[2];
};


template<int VL, class V>
void seedLength() {
  typedef uint32_t K;
  
  for (int repeat = 0; repeat < 10; ++repeat)
    for (uint64_t nn = 1048576; nn <= 16 * 1048576; nn *= 2)
      try {
        LFSRGen<K> keyGen(0x1234567801234567ULL, max((uint64_t) 1E8, nn), 0);
        LFSRGen<V> valueGen(0x1234567887654321ULL, nn, 0);
        
        vector<K> keys(nn);
        vector<V> values(nn);
        
        for (uint64_t i = 0; i < nn; i++) {
          keyGen.gen(&keys[i]);
          V v;
          valueGen.gen(&v);
          values[i] = v & (uint(-1) >> (32 - VL));
        }
        
        ControlPlaneLudo<K, V, VL> cp(nn);
        for (int i = 0; i < nn * 2 / 3; ++i) {
          vector<MPC_PathEntry> path;
          cp.insert(keys[i], values[i], &path);
        }
        
        cp.prepareToExport();
        
        vector<uint> seedLengths;  // 需要多长的seed, 才不会overflow
        for (ControlPlaneLudo<uint32_t, uint8_t, 4, 0>::Bucket b: cp.buckets_) {
          seedLengths.push_back((uint8_t) ceil(log2(b.seed + 2)));
        }
        
        std::sort(seedLengths.begin(), seedLengths.end());
        
        unsigned long e = seedLengths.size() - 1;
        cout << nn << ": ";
        for (int i = 0; i <= 1000; ++i) {
          cout << seedLengths[e * i / 1000] << " ";
        }
        cout << endl;
        
        vector<uint> pathLengths;
        vector<uint> perEntryCCLengths;
        vector<uint> singleCCLengths;
        Clocker gen("MPC generate 1E6 updates");
        // prepare many updates. modification : insertion : deletion = 1:1:1
        for (int i = 0; i < 1E5; ++i) {
          if ((i & 1) == 0) {  // delete
            K k;
            while (true) {
              k = keys[rand() % keys.size()];
              V tmp;
              if (cp.lookUp(k, tmp)) {
                break;
              }
            }
            cp.remove(k);
          } else { // insert
            K k;
            while (true) {
              k = keys[rand() % keys.size()];
              
              V tmp;
              if (!cp.lookUp(k, tmp)) {
                break;
              }
            }
            V v = rand();
            vector<MPC_PathEntry> path;
            cp.insert(k, v, &path);
            pathLengths.push_back(path.size());
            
            uint32_t s = 0;
            for (auto e: path) {
              s += e.locatorCC.size() * 4;
              singleCCLengths.push_back(e.locatorCC.size());
            }
            
            perEntryCCLengths.push_back(s);
          }
        }
        
        {
          cout << "Path length samples (1001 points): " << endl;
          std::sort(pathLengths.begin(), pathLengths.end());
          unsigned long e = pathLengths.size() - 1;
          for (int i = 0; i <= 1000; ++i) {
            cout << pathLengths[e * i / 1000] << " ";
          }
          cout << endl;
        }
        
        {
          cout << "CC size samples (1001 points): " << endl;
          std::sort(singleCCLengths.begin(), singleCCLengths.end());
          unsigned long e = singleCCLengths.size() - 1;
          for (int i = 0; i <= 1000; ++i) {
            cout << singleCCLengths[e * i / 1000] << " ";
          }
          cout << endl;
        }
        
        {
          cout << "Entry CC size samples (1001 points): " << endl;
          std::sort(perEntryCCLengths.begin(), perEntryCCLengths.end());
          unsigned long e = perEntryCCLengths.size() - 1;
          for (int i = 0; i <= 1000; ++i) {
            cout << perEntryCCLengths[e * i / 1000] << " ";
          }
          cout << endl;
        }
        gen.stop();
      } catch (exception &e) {
        cerr << e.what() << endl;
      }
}

template<int VL, class V>
void pathLength() {
  typedef uint32_t K;
  
  for (int repeat = 0; repeat < 10; ++repeat)
    for (uint64_t nn = 1048576; nn <= 16 * 1048576; nn *= 2)
      try {
        LFSRGen<K> keyGen(0x1234567801234567ULL, max((uint64_t) 1E8, nn), 0);
        LFSRGen<V> valueGen(0x1234567887654321ULL, nn, 0);
        
        vector<K> keys(nn);
        vector<V> values(nn);
        
        for (uint64_t i = 0; i < nn; i++) {
          keyGen.gen(&keys[i]);
          V v;
          valueGen.gen(&v);
          values[i] = v & (uint(-1) >> (32 - VL));
        }
        
        vector<uint> pathLengths;
        
        ControlPlaneCuckooMap<K, V, uint8_t, false> cp(nn);
        
        for (int i = 0; i < nn; ++i) {
          vector<CuckooMove> path;
          cp.template insert<true>(keys[i], values[i], &path);
          pathLengths.push_back(path.size());
        }
        
        {
          cout << nn << ": ";
          std::sort(pathLengths.begin(), pathLengths.end());
          unsigned long e = pathLengths.size() - 1;
          for (int i = 0; i <= 1000; ++i) {
            cout << pathLengths[e * i / 1000] << " ";
          }
          cout << endl;
        }
        
      } catch (exception &e) {
        cerr << e.what() << endl;
      }
}

template<class K>
void lookupThroughputBinary() {
  const uint32_t lookupCnt = 1U << 25;
  
  for (int repeat = 0; repeat < 10; ++repeat)
    for (uint64_t nn = 128 * 1024; nn <= (1024ULL * 1024 * 1024); nn *= 2) {
      try {
        ostringstream oss;
        
        oss << "binary" << ", key set size " << nn << ", repeat#" << repeat
            << ", version#" << version;
        
        string logName = "../dist/logs/" + oss.str() + ".log";
        
        ifstream testLog(logName);
        string lastLine, tmp;
        
        while (getline(testLog, tmp)) {
          if (tmp.size() && tmp[0] == '|') lastLine = tmp;
        }
        testLog.close();
        
        if (lastLine.size() >= 3 && lastLine[2] == '-') continue;
        
        TeeOstream tos(logName);
        Clocker clocker(oss.str(), &tos);
        
        LFSRGen<K> keyGen(0x1234567801234567ULL, max((uint64_t) lookupCnt, nn), 0);
        LFSRGen<uint8_t> valueGen(0x1234567887654321ULL, nn, 0);
        
        vector<K> keys(max((uint64_t) lookupCnt, nn));
        vector<uint8_t> values(nn);
        
        double ratio = 0.2825;
        uint nPos = nn * ratio;
        uint nNeg = nn - nPos;
        vector<K> posKeys;
        posKeys.reserve(nPos);
        vector<K> negKeys;
        negKeys.reserve(nNeg);
        
        uint ip = 0, in = 0;
        for (uint64_t i = 0; i < nn; i++) {
          uint8_t v = (ip == nPos) ? 0 : (in == nNeg) ? 1 : (rand() / (double) RAND_MAX) < ratio;
          values[i] = v;
          keyGen.gen(&keys[i]);
          if (v) {
            ip++;
            posKeys.push_back(keys[i]);
          } else {
            in++;
            negKeys.push_back(keys[i]);
          }
        }
        
        for (uint64_t i = nn; i < max((uint64_t) lookupCnt, nn); i++) {
          keys[i] = keys[i % nn];
        }
        
        vector<K> zipfianKeys;
        zipfianKeys.reserve(lookupCnt);
        
        for (int i = 0; i < lookupCnt; ++i) {
          InputBase::distribution = exponential;
          InputBase::bound = nn;
          
          uint seed = Hasher32<string>()(logName);
          InputBase::setSeed(seed);
          
          uint32_t idx = InputBase::rand();
          zipfianKeys.push_back(keys[idx]);
        }
        
        typedef uint8_t V;
        const uint VL = 1;
        
        {
          Clocker c("FilterCascades");
          
          Clocker construction("CP construction");
          ControlPlaneFilterCascades<K, V> cp(nPos, nNeg);
          cp.batch_insert(posKeys, negKeys);
          construction.stop();
          
          Clocker exp("export");
          DataPlaneFilterCascades<K, V> dp(cp);
          exp.stop();
          
          uint8_t stupid;
          
          {
            Clocker lz("lookup Zipfian");
            
            for (int ii = 0; ii < lookupCnt; ++ii) {
              const K &k = zipfianKeys.at(ii);
              uint8_t val = dp.query(k);
              stupid += val;
            }
          }
          
          {
            Clocker lz("lookup uniform");
            for (int ii = 0; ii < lookupCnt; ++ii) {
              const K &k = keys[ii];
              uint8_t val = dp.query(k);
              stupid += val;
            }
          }
          
          printf("%d\b", stupid & 7);
        }
        
        {
          Clocker c("Othello");
          
          Clocker construction("CP construction");
          ControlPlaneOthello<K, V, VL> cp(nn, true, keys, values);
          construction.stop();
          
          Clocker exp("export");
          DataPlaneOthello<K, V, VL> dp(cp);
          exp.stop();
          
          uint8_t stupid;
          
          {
            Clocker lz("lookup Zipfian");
            
            for (int ii = 0; ii < lookupCnt; ++ii) {
              const K &k = zipfianKeys.at(ii);
              uint8_t val;
              dp.lookUp(k, val);
              stupid += val;
            }
          }
          
          {
            Clocker lz("lookup uniform");
            for (int ii = 0; ii < lookupCnt; ++ii) {
              const K &k = keys[ii];
              uint8_t val;
              dp.lookUp(k, val);
              stupid += val;
            }
          }
          
          printf("%d\b", stupid & 7);
        }
        
        {
          Clocker c("SetSep");
          
          Clocker construction("CP construction");
          SetSep<K, V, VL> cp(nn, true, keys, values);
          construction.stop();
          
          uint8_t stupid;
          
          {
            Clocker lz("lookup Zipfian");
            
            for (int ii = 0; ii < lookupCnt; ++ii) {
              const K &k = zipfianKeys.at(ii);
              uint8_t val;
              cp.lookUp(k, val);
              stupid += val;
            }
          }
          
          {
            Clocker lz("lookup uniform");
            for (int ii = 0; ii < lookupCnt; ++ii) {
              const K &k = keys[ii];
              uint8_t val;
              cp.lookUp(k, val);
              stupid += val;
            }
          }
          
          printf("%d\b", stupid & 7);
        }
      } catch (exception &e) {
        cerr << e.what() << endl;
      }
    }
}


int main() {
  commonInit();

//  seedLength<4, uint8_t>();
//  pathLength<4, uint8_t>();
  
  lookupThroughputBinary<uint32_t>();
  
  return 0;
}
