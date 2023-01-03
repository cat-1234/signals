# signals

Signals generator that matches price update in 4.3-5.3ns against indicator patterns (up to 64 patterns)

tested on 4x 2GHz intel vm with linux x86_64 and using google benchmark

to run:
(requires linux x86_64 and google benchmark installed in location findable by cmake find_package)
```
git clone https://github.com/cat-1234/signals.git
cd signals
cmake -B build
cmake --build build
./build/signals
```


in global scope form in main.cpp for easier reading, and in object oriented form in _encapsulated_design.hpp (same performance for both)

(to test encapsulated_design, include it at the top of main.cpp comment out the rest of main.cpp)
