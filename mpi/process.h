#ifndef PROCESS__H
#define PROCESS__H

#include <vector>

using lld = long long int;

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

class Process {
 public:
  int rank;
  int numProcesses;

  int hor;
  int ver;

  Process();

  void readGal(const std::string &filename, int galNum,
               std::vector<Star> *stars);
  void calcSpace(const std::vector<Star> &stars);

  void exchangeSpaceInfo();
  void exchangeGalaxiesInfo();

  void distributeInitialStars(const std::vector<Star> &stars);
  void recvInitialStars();

  void step(float delta);

  void printStars(std::string gal1, std::string gal2) const;

 private:
  std::vector<lld> ids;
  std::vector<float> coords;
  std::vector<float> speeds;
  std::vector<float> accs;
  std::vector<float> masses;

  GalInfo gal1;
  GalInfo gal2;
  SpaceInfo space;

  bool firstStep;

  void sendSpaceInfo() const;
  void recvSpaceInfo();

  void sendGalaxiesInfo() const;
  void recvGalaxiesInfo();

  std::vector<Star> receiveAllStarsForPrint() const;
  void sendAllStarsForPrint() const;

  void setInitialSpeed();
  void updateMasses();

  inline int starCell(float x, float y) const {
    int cellX = (x - space.x) / space.cellWidth;
    int cellY = (y - space.y) / space.cellHeight;
    return cellY * ver + cellX;
  }

  void getOtherStars(std::vector<float> *coords, std::vector<float> *masses,
                     std::vector<lld> *count);
  void updateAccs(const std::vector<float> &otherCoords,
                  const std::vector<float> &otherMasses);
  void updateCoords(float delta);
  void updateSpeeds(const std::vector<float> &oldAccs, float delta);

  void exchangeStars();
  void doExchangeStars(std::vector<lld> *ids, std::vector<float> *coords,
                       std::vector<float> *speeds, std::vector<float> *accs);
  std::vector<int> exchangeStarsPartners() const;
  void sendStarsTo(int otherRank, const std::vector<lld> &ids,
                   const std::vector<float> &coords,
                   const std::vector<float> &speeds,
                   const std::vector<float> &accs) const;
  void recvStarsFrom(int otherRank, lld count, lld *ids, float *coords,
                     float *speeds, float *accs) const;
};

#endif
