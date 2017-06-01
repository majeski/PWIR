#ifndef PROCESS_2__H
#define PROCESS_2__H

#include "process.h"

class Process2 : public AbstractProcess {
 protected:
  virtual void getOtherStars(std::vector<float> *otherCoords,
                             std::vector<float> *otherMasses) override;
  std::vector<int> exchangeOrder() const;

  lld exchangeCount(int otherRank);
  void exchangeOtherStars(int otherRank, lld toRecv, float *coords,
                          float *masses);
};

#endif
