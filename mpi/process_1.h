#ifndef PROCESS_1__H
#define PROCESS_1__H

#include "process.h"

class Process1 : public AbstractProcess {
 protected:
  virtual std::vector<int> otherStarsExchangeOrder() const override;
};

#endif
