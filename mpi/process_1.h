#ifndef PROCESS_1__H
#define PROCESS_1__H

#include "process.h"

class Process1 : public AbstractProcess {
 protected:
  virtual void getOtherStars(std::vector<float> *otherCoords,
                             std::vector<float> *otherMasses) override;
};

#endif
