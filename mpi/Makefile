CXX = mpicxx
CXXFLAGS = -O2 -Wall -c -std=c++11
LDFLAGS = -O2 -Wall
OBJECTS = main.o process.o

all: main

main: $(OBJECTS)
	$(CXX) -o main $(OBJECTS) $(LDFLAGS)

%.o: %.cc Makefile
	$(CXX) $(CXXFLAGS) $<

