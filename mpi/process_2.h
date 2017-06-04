#ifndef PROCESS_2__H
#define PROCESS_2__H

#include "process.h"

class Process2 : public AbstractProcess {
 protected:
  virtual std::vector<int> otherStarsExchangeOrder() const override;
};

#endif
