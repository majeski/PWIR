#ifndef PROCESS__H
#define PROCESS__H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <iostream>
#include <tuple>
#include <vector>

#include <mpi.h>

using lld = long long int;
#define MPI_LLD MPI_LONG_LONG_INT

struct GalInfo {
  float vX;
  float vY;
  float mass;
};

struct SpaceInfo {
  float x;
  float y;
  float cellWidth;
  float cellHeight;
};

struct Star {
  long long int id;
  float x;
  float y;

  inline bool operator<(const Star &other) const {
    if (id < 0 && other.id < 0) {
      return id > other.id;
    }
    return id < other.id;
  }
};

class AbstractProcess {
 public:
  int rank;
  int numProcesses;

  int hor;
  int ver;

  AbstractProcess();
  virtual ~AbstractProcess() = default;

  void readGal(const std::string &filename, int galNum,
               std::vector<Star> *stars);
  void calcSpace(const std::vector<Star> &stars);

  void exchangeSpaceInfo();
  void exchangeGalaxiesInfo();

  void distributeInitialStars(const std::vector<Star> &stars);
  void recvInitialStars();

  void step(float delta);

  void printStars(std::string gal1, std::string gal2);

 protected:
  std::vector<lld> ids;
  std::vector<float> coords;
  std::vector<float> speeds;
  std::vector<float> accs;
  std::vector<float> masses;

  GalInfo gal1;
  GalInfo gal2;
  SpaceInfo space;

  virtual void getOtherStars(std::vector<float> *coords,
                             std::vector<float> *masses);
  virtual std::vector<int> otherStarsExchangeOrder() const = 0;

  std::vector<int> allToAllOrder() const;
  int exchangeCount(int otherRank, int toSend) const;
  void exchangeOtherStars(int otherRank, std::vector<float> &coords,
                          std::vector<float> &masses, int toRecv,
                          float *otherCoords, float *otherMasses) const;

 private:
  bool firstStep;

  void sendSpaceInfo() const;
  void recvSpaceInfo();

  void sendGalaxiesInfo() const;
  void recvGalaxiesInfo();

  std::vector<Star> receiveAllStarsForPrint();
  void sendAllStarsForPrint();

  void setInitialSpeed();
  void updateMasses();

  int starCell(float x, float y) const;

  std::vector<lld> updateAccs(const std::vector<float> &otherCoords,
                              const std::vector<float> &otherMasses);
  void updateCoords(double delta, const std::vector<lld> &toSkip);
  void updateSpeeds(const std::vector<float> &oldAccs, float delta,
                    const std::vector<lld> &toSkip);
  void fixCoords();

  void exchangeStars();
  void doExchangeStars(std::vector<lld> *ids, std::vector<float> *coords,
                       std::vector<float> *speeds, std::vector<float> *accs);
  void sendStarsTo(int otherRank, std::vector<lld> &ids,
                   std::vector<float> &coords, std::vector<float> &speeds,
                   std::vector<float> &accs,
                   std::vector<MPI_Request> *requests);
  void recvStarsFrom(int otherRank, int count, lld *ids, float *coords,
                     float *speeds, float *accs,
                     std::vector<MPI_Request> *requests) const;
};

#endif
