#include "process_3.h"

void Process3::getOtherStars(std::vector<float> *otherCoords,
                             std::vector<float> *otherMasses) {
  std::vector<int> toRecv;
  lld countSum = 0;
  auto order = otherStarsExchangeOrder();
  for (int otherRank : order) {
    toRecv.push_back(exchangeCount(otherRank, ids.size() > 0));
    countSum += toRecv.back();
  }

  otherCoords->resize(countSum * 2);
  otherMasses->resize(countSum);

  float *coordsPtr = otherCoords->data();
  float *massesPtr = otherMasses->data();

  std::vector<float> coords;
  std::vector<float> masses;
  if (ids.size() > 0) {
    coords.resize(2);
    masses.resize(1);
    getSuperStar(coords.data(), coords.data() + 1, masses.data());
  }

  for (int i = 0; i < (int)order.size(); i++) {
    int otherRank = order[i];
    int count = toRecv[i];
    exchangeOtherStars(otherRank, coords, masses, count, coordsPtr, massesPtr);
    coordsPtr += count * 2;
    massesPtr += count;
  }
}

void Process3::getSuperStar(float *x, float *y, float *mass) const {
  double massSum = 0;
  for (float m : masses) {
    massSum += m;
  }

  double xSum = 0;
  double ySum = 0;
  for (lld i = 0; i < (lld)ids.size(); i++) {
    xSum += double(coords[i * 2]) * masses[i];
    ySum += double(coords[i * 2 + 1]) * masses[i];
  }

  *x = xSum / massSum;
  *y = ySum / massSum;
  *mass = massSum;
}
