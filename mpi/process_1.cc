#include "process_1.h"

void Process1::getOtherStars(std::vector<float> *otherCoords,
                             std::vector<float> *otherMasses) {
  std::vector<lld> count;
  count.resize(numProcesses);

  count[rank] = ids.size();

  lld starsCount = -count[rank];
  for (int i = 0; i < numProcesses; i++) {
    MPI_Bcast(count.data() + i, 1, MPI_LLD, i, MPI_COMM_WORLD);
    starsCount += count[i];
  }

  otherCoords->resize(starsCount * 2);
  otherMasses->resize(starsCount);

  float *otherCoordsPtr = otherCoords->data();
  float *otherMassesPtr = otherMasses->data();

  for (int i = 0; i < numProcesses; i++) {
    if (count[i] == 0) {
      continue;
    }

    void *whereCoords;
    void *whereMasses;
    if (i == rank) {
      whereCoords = this->coords.data();
      whereMasses = this->masses.data();
    } else {
      whereCoords = otherCoordsPtr;
      otherCoordsPtr += count[i] * 2;
      whereMasses = otherMassesPtr;
      otherMassesPtr += count[i];
    }

    MPI_Bcast(whereCoords, count[i] * 2, MPI_FLOAT, i, MPI_COMM_WORLD);
    MPI_Bcast(whereMasses, count[i], MPI_FLOAT, i, MPI_COMM_WORLD);
  }
}
