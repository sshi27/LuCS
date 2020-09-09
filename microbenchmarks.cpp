#include <Sketch/ludo_sketch.h>
#include "common.h"
#include "Ludo/ludo.h"

int version = 12;
int cores = min(20U, std::thread::hardware_concurrency());

typedef uint32_t Key;
const uint32_t lookupCnt = 1 << 25;

template<int VL, class Val>
void testLudoAndSketch(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys, int upToThreads) {
  Clocker construction("Ludo construction");
  
  Clocker cpBuild("CP build");
  ControlPlaneLudo<Key, Val, VL> cp(nn);
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  cpBuild.stop();
  
  Clocker cpPrepare("CP prepare for DP");
  cp.prepareToExport();
  cpPrepare.stop();
  construction.stop();
  
  Clocker exp("Ludo export");
  DataPlaneLudo<Key, Val, VL> dp(cp);
  exp.stop();
  
  Hasher32<Key> h[3];
  for (int i = 0; i < 3; ++i) h[i].setSeed(rand());
  
  vector<uint16_t> a[3];
  a[0].resize(dp.locator.ma);
  a[1].resize(dp.locator.mb);
  a[2].resize(dp.num_buckets_);
  
  for (int threadCnt = 1; threadCnt <= upToThreads; ++threadCnt) {
    thread threads[threadCnt];
    uint32_t start[threadCnt];
    
    for (int i = 0; i < threadCnt; ++i) {
      start[i] = i / threadCnt * zipfianKeys.size();
    }
    
    for (Distribution distribution: {uniform, exponential}) {
      ostringstream oss;
      oss << "Ludo parallel lookup " << threadCnt << " threads " << lookupCnt << " keys "
          << (distribution == exponential ? "Zipfian" : "uniform");
      Clocker plookup(oss.str());
      
      for (int i = 0; i < threadCnt; ++i) {
        threads[i] = std::thread([](DataPlaneLudo<Key, Val, VL> *dp, uint32_t start,
                                    const vector<Key> *zipfianKeys, uint32_t lookupCnt, vector<uint16_t> *a, Hasher32<Key> *h) {
          int stupid = 0;
          
          int ii = start;
          do {
            const Key &k = zipfianKeys->at(ii);
            Val val;
            dp->lookUp(k, val);
            stupid += val;
            
            for (int i = 0; i < 3; ++i) {
              unsigned short &i1 = a[i][h[i](k) % a[i].size()];
              
              stupid += i1++;
            }
            
            if (ii == lookupCnt - 1) ii = -1;
            ++ii;
          } while (ii != start);
          printf("%d\b", stupid & 7);
        }, &dp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt, a, h);
      }
      
      for (int i = 0; i < threadCnt; ++i) {
        threads[i].join();
      }
    }
  }
}

template<int VL, class Val>
void testLudoSketch(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys, uint upToThreads) {
  Clocker construction("LudoSketch construction");
  
  Clocker cpBuild("CP build");
  ControlPlaneLudo<Key, Val, VL> cp(nn);
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  cpBuild.stop();
  
  Clocker cpPrepare("CP prepare for DP");
  cp.prepareToExport();
  cpPrepare.stop();
  construction.stop();
  
  Clocker exp("LudoSketch export");
  DataPlaneLudoSketch<Key, Val, VL> dp(cp);
  exp.stop();
  
  for (int threadCnt = 1; threadCnt <= upToThreads; ++threadCnt) {
    thread threads[threadCnt];
    uint32_t start[threadCnt];
    
    for (int i = 0; i < threadCnt; ++i) {
      start[i] = i / threadCnt * zipfianKeys.size();
    }
    
    for (Distribution distribution: {uniform, exponential}) {
      ostringstream oss;
      oss << "LudoSketch parallel lookup " << threadCnt << " threads " << lookupCnt << " keys "
          << (distribution == exponential ? "Zipfian" : "uniform");
      Clocker plookup(oss.str());
      
      for (int i = 0; i < threadCnt; ++i) {
        threads[i] = std::thread([](DataPlaneLudoSketch<Key, Val, VL> *dp, uint32_t start,
                                    const vector<Key> *zipfianKeys, uint32_t lookupCnt) {
          int stupid = 0;
          
          int ii = start;
          do {
            const Key &k = zipfianKeys->at(ii);
            Val val;
            dp->lookUp(k, val);
            stupid += val;
            
            if (ii == lookupCnt - 1) ii = -1;
            ++ii;
          } while (ii != start);
          printf("%d\b", stupid & 7);
        }, &dp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
      }
      
      for (int i = 0; i < threadCnt; ++i) {
        threads[i].join();
      }
    }
  }
}

template<int VL, class Val>
void test() {
  for (int repeat = 0; repeat < 10; ++repeat)
    for (uint64_t nn = 131072; nn <= (1U << 30); nn *= 2)
      try {
        ostringstream oss;
        
        oss << "value length " << VL << ", key set size " << nn << ", repeat#" << repeat
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
        
        LFSRGen<Key> keyGen(0x1234567801234567ULL, max((uint64_t) lookupCnt, nn), 0);
        LFSRGen<Val> valueGen(0x1234567887654321ULL, nn, 0);
        
        vector<Key> keys(max((uint64_t) lookupCnt, nn));
        vector<Val> values(nn);
        
        for (uint64_t i = 0; i < nn; i++) {
          keyGen.gen(&keys[i]);
          Val v;
          valueGen.gen(&v);
          values[i] = v & (uint(-1) >> (32 - VL));
        }
        
        for (uint64_t i = nn; i < max((uint64_t) lookupCnt, nn); i++) {
          keys[i] = keys[i % nn];
        }
        
        vector<Key> zipfianKeys;
        zipfianKeys.reserve(lookupCnt);
        
        for (int i = 0; i < lookupCnt; ++i) {
          InputBase::distribution = exponential;
          InputBase::bound = nn;
          
          uint seed = Hasher32<string>()(logName);
          InputBase::setSeed(seed);
          
          uint32_t idx = InputBase::rand();
          zipfianKeys.push_back(keys[idx]);
        }
        
        
        try {
          testLudoSketch<VL, Val>(keys, values, nn, zipfianKeys, repeat <= 1 ? cores : 1);
        } catch (exception &e) {
          cout << e.what() << endl;
          break;
        }
        
        try {
          testLudoAndSketch<VL, Val>(keys, values, nn, zipfianKeys, repeat <= 1 ? cores : 1);
        } catch (exception &e) {
          cout << e.what() << endl;
          break;
        }
        return;
      } catch (exception &e) {
        cerr << e.what() << endl;
      }
}

int main(int argc, char **argv) {
  commonInit();
  
  if (argc == 1) {
    for (int i = 0; i < 100; ++i) test<4, uint8_t>();
    for (int i = 0; i < 100; ++i) test<8, uint8_t>();
    for (int i = 0; i < 100; ++i) test<12, uint16_t>();
    for (int i = 0; i < 100; ++i) test<16, uint16_t>();
    for (int i = 0; i < 100; ++i) test<20, uint32_t>();
//    test<32>();
  }
  
  switch (atoi(argv[1])) {
    case 4:
      for (int i = 0; i < 100; ++i) test<4, uint8_t>();
      break;
    case 8:
      for (int i = 0; i < 100; ++i) test<8, uint8_t>();
      break;
    case 12:
      for (int i = 0; i < 100; ++i) test<12, uint16_t>();
      break;
    case 16:
      for (int i = 0; i < 100; ++i) test<16, uint16_t>();
      break;
    case 20:
      for (int i = 0; i < 100; ++i) test<20, uint32_t>();
      break;
  }
  
  return 0;
}
