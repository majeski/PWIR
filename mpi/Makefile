CXX = mpicxx
CXXFLAGS = -O2 -Wall -c -std=c++11
LDFLAGS = -O2 -Wall
OBJECTS = process.o process_1.o process_2.o
ALL = collisions-1 collisions-2

all: $(ALL)

collisions-1: $(OBJECTS) main_1.o
	$(CXX) -o collisions-1 $(LDFLAGS) $(OBJECTS) main_1.o

collisions-2: $(OBJECTS) main_2.o
	$(CXX) -o collisions-2 $(LDFLAGS) $(OBJECTS) main_2.o

main_1.o: main.cc Makefile
	$(CXX) $(CXXFLAGS) -DV1 main.cc -o main_1.o

main_2.o: main.cc Makefile
	$(CXX) $(CXXFLAGS) -DV2 main.cc -o main_2.o

%.o: %.cc Makefile
	$(CXX) $(CXXFLAGS) $<

clean:
	rm -rf $(OBJECTS) main_1.o main_2.o $(ALL)

