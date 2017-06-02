#include "process_2.h"

void Process2::getOtherStars(std::vector<float> *otherCoords,
                             std::vector<float> *otherMasses) {
  std::vector<int> toRecv;
  lld countSum = 0;
  auto order = exchangeOrder();
  for (int otherRank : order) {
    toRecv.push_back(exchangeCount(otherRank));
    countSum += toRecv.back();
  }

  otherCoords->resize(countSum * 2);
  otherMasses->resize(countSum);

  float *coordsPtr = otherCoords->data();
  float *massesPtr = otherMasses->data();

  for (int i = 0; i < order.size(); i++) {
    int otherRank = order[i];
    int count = toRecv[i];
    exchangeOtherStars(otherRank, count, coordsPtr, massesPtr);
    coordsPtr += count * 2;
    massesPtr += count;
  }
}

std::vector<int> Process2::exchangeOrder() const {
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

int Process2::exchangeCount(int otherRank) {
  int toSend = ids.size();
  int toRecv;

  auto send = [&]() {
    MPI_Send(&toSend, 1, MPI_INTEGER, otherRank, 601, MPI_COMM_WORLD);
  };
  auto recv = [&]() {
    MPI_Recv(&toRecv, 1, MPI_INTEGER, otherRank, 601, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  };

  if (rank < otherRank) {
    send();
    recv();
  } else {
    recv();
    send();
  }
  return toRecv;
}

void Process2::exchangeOtherStars(int otherRank, int toRecv, float *otherCoords,
                                  float *otherMasses) {
  int toSend = ids.size();

  auto send = [&]() {
    if (toSend > 0) {
      MPI_Send(coords.data(), toSend * 2, MPI_FLOAT, otherRank, 602,
               MPI_COMM_WORLD);
      MPI_Send(masses.data(), toSend, MPI_FLOAT, otherRank, 603,
               MPI_COMM_WORLD);
    }
  };
  auto recv = [&]() {
    if (toRecv > 0) {
      MPI_Recv(otherCoords, toRecv * 2, MPI_FLOAT, otherRank, 602,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(otherMasses, toRecv, MPI_FLOAT, otherRank, 603, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  };

  if (rank < otherRank) {
    send();
    recv();
  } else {
    recv();
    send();
  }
}
