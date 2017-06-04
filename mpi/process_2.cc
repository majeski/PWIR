#include "process_2.h"

std::vector<int> Process2::otherStarsExchangeOrder() const {
  std::vector<int> partners;
  const int rankX = rank % ver;
  const int rankY = rank / ver;
  auto addPartner = [&](int x, int y) {
    int rank = y * ver + x;
    if (x >= 0 && y >= 0 && x < ver && y < hor) {
      partners.push_back(rank);
    }
  };

  // A B A
  {
    const bool firstInPair = rankX % 2 == 0;
    if (firstInPair) {
      addPartner(rankX + 1, rankY);
      addPartner(rankX - 1, rankY);
    } else {
      addPartner(rankX - 1, rankY);
      addPartner(rankX + 1, rankY);
    }
  }
  // A
  // B
  // A
  {
    const bool firstInPair = rankY % 2 == 0;
    if (firstInPair) {
      addPartner(rankX, rankY + 1);
      addPartner(rankX, rankY - 1);
    } else {
      addPartner(rankX, rankY - 1);
      addPartner(rankX, rankY + 1);
    }
  }
  // A
  //  B
  //   A
  {
    const bool firstInPair = std::min(rankY, rankX) % 2 == 0;
    if (firstInPair) {
      addPartner(rankX + 1, rankY + 1);
      addPartner(rankX - 1, rankY - 1);
    } else {
      addPartner(rankX - 1, rankY - 1);
      addPartner(rankX + 1, rankY + 1);
    }
  }
  //   A
  //  B
  // A
  {
    const bool firstInPair = std::min(rankY, ver - rankX - 1) % 2 == 0;
    if (firstInPair) {
      addPartner(rankX - 1, rankY + 1);
      addPartner(rankX + 1, rankY - 1);
    } else {
      addPartner(rankX + 1, rankY - 1);
      addPartner(rankX - 1, rankY + 1);
    }
  }

  return partners;
}
