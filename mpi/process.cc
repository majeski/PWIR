#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <mpi.h>

#include "process.h"

#define MPI_LLD MPI_LONG_LONG_INT

Process::Process() : firstStep(true) {}

void Process::readGal(const std::string &filename, int galNum,
                      std::vector<Star> *stars) {
  FILE *f = fopen(filename.c_str(), "r");
  if (f == NULL) {
    // TODO
  }

  GalInfo *gal = galNum == 1 ? &gal1 : &gal2;

  lld n;
  fscanf(f, "%lld", &n);
  fscanf(f, "%f %f", &(gal->vX), &(gal->vY));
  fscanf(f, "%f", &(gal->mass));
  for (lld i = 1; i <= n; i++) {
    Star star;
    star.id = (gal == &gal1 ? 1 : -1) * i;
    fscanf(f, "%f %f", &star.x, &star.y);
    stars->push_back(star);
  }

  fclose(f);
}

void Process::calcSpace(const std::vector<Star> &stars) {
  float minX, maxX, minY, maxY, distX, distY;

  minX = stars[0].x;
  maxX = stars[0].x;
  minY = stars[0].y;
  maxY = stars[0].y;
  for (const auto &star : stars) {
    minX = std::min(star.x, minX);
    maxX = std::max(star.x, maxX);
    minY = std::min(star.y, minY);
    maxY = std::max(star.y, maxY);
  }

  distX = maxX - minX;
  distY = maxY - minY;
  minX = minX - distX / 2;
  maxX = maxX + distX / 2;
  minY = minY - distY / 2;
  maxY = maxY + distY / 2;

  space.x = minX;
  space.y = minY;
  space.cellWidth = (maxX - minX) / ver;
  space.cellHeight = (maxY - minY) / hor;
}

void Process::exchangeSpaceInfo() {
  if (rank == 0) {
    sendSpaceInfo();
  } else {
    recvSpaceInfo();
  }
}

void Process::sendSpaceInfo() const {
  float spaceInfo[4] = {space.x, space.y, space.cellWidth, space.cellHeight};

  int err = MPI_Bcast(spaceInfo, 4, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if (err) {
    // TODO
  }
}

void Process::recvSpaceInfo() {
  float spaceInfo[4];

  int err = MPI_Bcast(spaceInfo, 4, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if (err) {
    // TODO
  }

  space.x = spaceInfo[0];
  space.y = spaceInfo[1];
  space.cellWidth = spaceInfo[2];
  space.cellHeight = spaceInfo[3];
}

void Process::exchangeGalaxiesInfo() {
  if (rank == 0) {
    sendGalaxiesInfo();
  } else {
    recvGalaxiesInfo();
  }
}

void Process::sendGalaxiesInfo() const {
  float speedAndMass[6] = {gal1.vX, gal1.vY, gal1.mass,
                           gal2.vX, gal2.vY, gal2.mass};

  int err = MPI_Bcast(speedAndMass, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if (err) {
    // TODO
  }
}

void Process::recvGalaxiesInfo() {
  float speedAndMass[6];

  int err = MPI_Bcast(speedAndMass, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if (err) {
    // TODO
  }

  gal1.vX = speedAndMass[0];
  gal1.vY = speedAndMass[1];
  gal1.mass = speedAndMass[2];
  gal2.vX = speedAndMass[3];
  gal2.vY = speedAndMass[4];
  gal2.mass = speedAndMass[5];
}

void Process::distributeInitialStars(const std::vector<Star> &allStars) {
  std::vector<lld> ids[numProcesses];
  std::vector<float> coords[numProcesses];
  lld starsCount[numProcesses];

  for (const auto &star : allStars) {
    int cell = starCell(star.x, star.y);
    ids[cell].push_back(star.id);
    coords[cell].push_back(star.x);
    coords[cell].push_back(star.y);
  }
  for (int i = 0; i < numProcesses; i++) {
    starsCount[i] = ids[i].size();
  }

  lld count;
  int err = MPI_Scatter(starsCount, 1, MPI_LLD, &count, 1, MPI_LLD, 0,
                        MPI_COMM_WORLD);
  if (err) {
    // TODO
  }

  this->ids = std::move(ids[0]);
  this->coords = std::move(coords[0]);
  for (int i = 1; i < numProcesses; i++) {
    if (starsCount[i] > 0) {
      err =
          MPI_Send(ids[i].data(), starsCount[i], MPI_LLD, i, 0, MPI_COMM_WORLD);
      if (err) {
        // TODO
      }
      err = MPI_Send(coords[i].data(), starsCount[i] * 2, MPI_FLOAT, i, 1,
                     MPI_COMM_WORLD);
      if (err) {
        // TODO
      }
    }
  }

  setInitialSpeed();
  updateMasses();
}

void Process::recvInitialStars() {
  lld count;

  int err =
      MPI_Scatter(NULL, 0, MPI_LLD, &count, 1, MPI_LLD, 0, MPI_COMM_WORLD);
  if (err) {
    // TODO
  }

  if (count > 0) {
    ids.resize(count);
    err = MPI_Recv(ids.data(), count, MPI_LLD, 0, 0, MPI_COMM_WORLD, NULL);
    if (err) {
      // TODO
    }

    coords.resize(count * 2);
    err =
        MPI_Recv(coords.data(), count * 2, MPI_LLD, 0, 1, MPI_COMM_WORLD, NULL);
    if (err) {
      // TODO
    }
  }

  setInitialSpeed();
  updateMasses();
}

void Process::setInitialSpeed() {
  for (lld id : this->ids) {
    GalInfo *gal = id < 0 ? &gal2 : &gal1;
    this->speeds.push_back(gal->vX);
    this->speeds.push_back(gal->vY);
  }
}

void Process::updateMasses() {
  masses.resize(ids.size());
  for (lld i = 0; i < ids.size(); i++) {
    masses[i] = ids[i] < 0 ? gal2.mass : gal1.mass;
  }
}

inline std::pair<float, float> calcF(float star1X, float star1Y, float mass1,
                                     float star2X, float star2Y, float mass2) {
  float G = 155893.597;
  float dX = star1X - star2X;
  float dY = star1Y - star2Y;
  float norm = sqrt(dX * dX + dY * dY);
  return {(G * mass1 * mass2 * (star1X - star2X)) / (norm * norm * norm),
          (G * mass1 * mass2 * (star1Y - star2Y)) / (norm * norm * norm)};
}

void Process::step(float delta) {
  std::vector<float> otherCoords;
  std::vector<float> otherMasses;
  std::vector<float> oldAccs;
  std::vector<lld> count;

  if (!firstStep) {
    oldAccs = accs;
  }

  getOtherStars(&otherCoords, &otherMasses, &count);
  updateAccs(otherCoords, otherMasses);

  if (!firstStep) {
    updateSpeeds(oldAccs, delta);
  }

  updateCoords(delta);
  firstStep = false;
}

void Process::updateAccs(const std::vector<float> &otherCoords,
                         const std::vector<float> &otherMasses) {
  lld otherCount = otherCoords.size() / 2;
  accs.resize(coords.size());

  for (lld i = 0; i < ids.size(); i++) {
    lld xIdx = i * 2;
    lld yIdx = i * 2 + 1;
    accs[xIdx] = 0;
    accs[yIdx] = 0;

    auto nextStar = [&](float otherX, float otherY, float otherMass) {
      float x = coords[xIdx];
      float y = coords[yIdx];
      float mass = masses[i];
      float dX, dY;
      std::tie(dX, dY) = calcF(x, y, mass, otherX, otherY, otherMass);
      accs[xIdx] += dX;
      accs[yIdx] += dY;
    };

    for (lld j = 0; j < ids.size(); j++) {
      if (i == j) {
        continue;
      }
      nextStar(coords[j * 2], coords[j * 2 + 1], masses[j]);
    }

    for (lld j = 0; j < otherCount; j++) {
      nextStar(otherCoords[j * 2], otherCoords[j * 2 + 1], otherMasses[j]);
    }

    accs[xIdx] /= -masses[i];
    accs[yIdx] /= -masses[i];
  }
}

void Process::updateCoords(float delta) {
  for (lld i = 0; i < coords.size(); i++) {
    coords[i] += speeds[i] * delta + 0.5 * accs[i] * delta * delta;
  }

  for (lld i = 0; i < coords.size(); i += 2) {
    float &x = coords[i];
    const float spaceWidth = space.cellWidth * ver;

    if (x < space.x) {
      float distX = space.x - x;
      distX -= floor(distX / spaceWidth) * spaceWidth;
      x = space.x + spaceWidth - distX;
    } else if (x > space.x + spaceWidth) {
      float distX = x - (space.x + spaceWidth);
      distX -= floor(distX / spaceWidth) * spaceWidth;
      x = space.x + distX;
    }

    float &y = coords[i + 1];
    const float spaceHeight = space.cellHeight * hor;
    if (y < space.y) {
      float distY = space.y - y;
      distY -= floor(distY / spaceHeight) * spaceHeight;
      y = space.y + spaceHeight - distY;
    } else if (x > space.x + spaceWidth) {
      float distY = y - (space.y + spaceHeight);
      distY -= floor(distY / spaceHeight) * spaceHeight;
      y = space.y + distY;
    }
  }
}

void Process::updateSpeeds(const std::vector<float> &oldAccs, float delta) {
  for (lld i = 0; i < speeds.size(); i++) {
    speeds[i] += 0.5 * (oldAccs[i] + accs[i]) * delta;
  }
}

void Process::getOtherStars(std::vector<float> *otherCoords,
                            std::vector<float> *otherMasses,
                            std::vector<lld> *count) {
  int err;
  count->resize(numProcesses);

  (*count)[rank] = ids.size();

  lld starsCount = -(*count)[rank];
  for (int i = 0; i < numProcesses; i++) {
    err = MPI_Bcast(count->data() + i, 1, MPI_LLD, i, MPI_COMM_WORLD);
    if (err) {
      // TODO
    }
    starsCount += (*count)[i];
  }

  otherCoords->resize(starsCount * 2);
  otherMasses->resize(starsCount);

  float *otherCoordsPtr = otherCoords->data();
  float *otherMassesPtr = otherMasses->data();

  for (int i = 0; i < numProcesses; i++) {
    if ((*count)[i] == 0) {
      continue;
    }

    void *whereCoords;
    void *whereMasses;
    if (i == rank) {
      whereCoords = this->coords.data();
      whereMasses = this->masses.data();
    } else {
      whereCoords = otherCoordsPtr;
      otherCoordsPtr += (*count)[i] * 2;
      whereMasses = otherMassesPtr;
      otherMassesPtr += (*count)[i];
    }

    err = MPI_Bcast(whereCoords, (*count)[i] * 2, MPI_FLOAT, i, MPI_COMM_WORLD);
    if (err) {
      // TODO
    }
    err = MPI_Bcast(whereMasses, (*count)[i], MPI_FLOAT, i, MPI_COMM_WORLD);
    if (err) {
      // TODO
    }
  }
}

void Process::printStars(std::string gal1, std::string gal2) const {
  if (rank == 0) {
    auto stars = receiveAllStarsForPrint();
    std::sort(stars.begin(), stars.end());

    FILE *out1 = fopen(gal1.c_str(), "w");
    FILE *out2 = fopen(gal2.c_str(), "w");
    if (!out1 || !out2) {
      // TODO
    }

    for (const auto &star : stars) {
      FILE *out = star.id < 0 ? out2 : out1;
      fprintf(out, "%0.1f %0.1f\n", star.x, star.y);
    }

    fclose(out1);
    fclose(out2);
  } else {
    sendAllStarsForPrint();
  }
}

std::vector<Star> Process::receiveAllStarsForPrint() const {
  lld count = ids.size();
  lld starsCount[numProcesses];

  int err =
      MPI_Gather(&count, 1, MPI_LLD, starsCount, 1, MPI_LLD, 0, MPI_COMM_WORLD);
  if (err) {
    // TODO
  }

  std::vector<Star> stars;
  std::vector<lld> tmpIds;
  std::vector<float> tmpCoords;
  for (int proc = 0; proc < numProcesses; proc++) {
    if (proc == 0) {
      tmpIds = ids;
      tmpCoords = this->coords;
    } else if (starsCount[proc]) {
      tmpIds.resize(starsCount[proc]);
      tmpCoords.resize(starsCount[proc] * 2);

      // TODO status
      err = MPI_Recv(tmpIds.data(), starsCount[proc], MPI_LLD, proc, 100,
                     MPI_COMM_WORLD, NULL);
      if (err) {
        // TODO
      }
      err = MPI_Recv(tmpCoords.data(), starsCount[proc] * 2, MPI_FLOAT, proc,
                     101, MPI_COMM_WORLD, NULL);
      if (err) {
        // TODO
      }
    } else {
      continue;
    }

    for (int i = 0; i < tmpIds.size(); i++) {
      Star star;
      star.id = tmpIds[i];
      star.x = tmpCoords[i * 2];
      star.y = tmpCoords[i * 2 + 1];
      stars.push_back(star);
    }
  }

  return stars;
}

void Process::sendAllStarsForPrint() const {
  lld count = ids.size();
  int err;

  err = MPI_Gather(&count, 1, MPI_LLD, NULL, 0, MPI_LONG_LONG_INT, 0,
                   MPI_COMM_WORLD);
  if (err) {
    // TODO
  }
  if (count) {
    err = MPI_Send(ids.data(), count, MPI_LLD, 0, 100, MPI_COMM_WORLD);
    if (err) {
      // TODO
    }
    err = MPI_Send(coords.data(), count * 2, MPI_FLOAT, 0, 101, MPI_COMM_WORLD);
    if (err) {
      // TODO
    }
  }
}
