#ifndef PROCESS_3__H
#define PROCESS_3__H

#include "process_2.h"

class Process3 : public Process2 {
 protected:
  virtual void getOtherStars(std::vector<float> *coords,
                             std::vector<float> *masses) override;

 private:
  void getSuperStar(float *x, float *y, float *mass) const;
};

#endif
