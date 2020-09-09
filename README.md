# Repo for *Don't Work on Individual Data Plane Algorithms. Put Them Together!*

## Abstract

Algorithms and data structures for data plane network functions have been extensively studied in the literature. Recently programmable network techniques have been used to design new methods that achieve less memory cost and higher throughput. However, most of these studies only focus on individual network functions, such as packet forwarding, trafﬁc measurement, and load balancing. To our knowledge no study has been conducted to design compact data structures and algorithms for multiple and co-existed network functions. We argue that there is a huge space of optimization if we design algorithms and data structures considering multiple co-exited network functions, compared to designing them individually. It is because many of them share similar design goals and building blocks. We use two recently published methods in this year as examples and present a new memorycompact design that serves both FIB and trafﬁc measurement functions by a novel integration of the two methods. The preliminary results show that the new design can achieve almost 2x throughput compared to running them individually while achieving higher accuracy of measurement in theory. In addition, we will discuss potential research directions and challenges.

Our paper will appear on the 19th ACM Workshop on Hot Topics in Networks (HotNets 2020).  

**Paper Link:** https://users.soe.ucsc.edu/~qian/papers/LuCS-hotnets.pdf

## Repository Structure

- Ludo/ Ludo hashing
- BloomFilter/ Bloom filter
- CuckooPresized/ Cuckoo hashing
- Othello/ Othello hashing
- Sketch/ Sketch


## Run demo

```sh
# 0. setup 
sudo apt-get install google-perftools libgoogle-perftools-dev cmake build-essential pkgconf

# 1. build
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -G "CodeBlocks - Unix Makefiles" ..
make microbenchmarks -j4

# 2. run tests
./microbenchmarks
```

For more details, check out `microbenchmarks.cpp`. It contains methods on parallel lookup and dynamic updates.

## Authors

- Chen Qian(cqian12@ucsc.edu)
- Shouqian Shi(sshi27@ucsc.edu)
- Xiaofeng Shi(xshi24@ucsc.edu)
- Minmei Wang(mwang107@ucsc.edu)