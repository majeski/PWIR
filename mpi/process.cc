#include "process.h"

AbstractProcess::AbstractProcess() : firstStep(true) {}

void AbstractProcess::readGal(const std::string &filename, int galNum,
                              std::vector<Star> *stars) {
  FILE *f = fopen(filename.c_str(), "r");
  if (f == NULL) {
    throw std::runtime_error("Unable to open file " + filename);
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

void AbstractProcess::calcSpace(const std::vector<Star> &stars) {
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

void AbstractProcess::exchangeSpaceInfo() {
  if (rank == 0) {
    sendSpaceInfo();
  } else {
    recvSpaceInfo();
  }
}

void AbstractProcess::sendSpaceInfo() const {
  float spaceInfo[4] = {space.x, space.y, space.cellWidth, space.cellHeight};

  MPI_Bcast(spaceInfo, 4, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void AbstractProcess::recvSpaceInfo() {
  float spaceInfo[4];

  MPI_Bcast(spaceInfo, 4, MPI_FLOAT, 0, MPI_COMM_WORLD);

  space.x = spaceInfo[0];
  space.y = spaceInfo[1];
  space.cellWidth = spaceInfo[2];
  space.cellHeight = spaceInfo[3];
}

void AbstractProcess::exchangeGalaxiesInfo() {
  if (rank == 0) {
    sendGalaxiesInfo();
  } else {
    recvGalaxiesInfo();
  }
}

void AbstractProcess::sendGalaxiesInfo() const {
  float speedAndMass[6] = {gal1.vX, gal1.vY, gal1.mass,
                           gal2.vX, gal2.vY, gal2.mass};

  MPI_Bcast(speedAndMass, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void AbstractProcess::recvGalaxiesInfo() {
  float speedAndMass[6];

  MPI_Bcast(speedAndMass, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);

  gal1.vX = speedAndMass[0];
  gal1.vY = speedAndMass[1];
  gal1.mass = speedAndMass[2];
  gal2.vX = speedAndMass[3];
  gal2.vY = speedAndMass[4];
  gal2.mass = speedAndMass[5];
}

void AbstractProcess::distributeInitialStars(
    const std::vector<Star> &allStars) {
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
  MPI_Scatter(starsCount, 1, MPI_LLD, &count, 1, MPI_LLD, 0, MPI_COMM_WORLD);

  this->ids = std::move(ids[0]);
  this->coords = std::move(coords[0]);
  for (int i = 1; i < numProcesses; i++) {
    if (starsCount[i] > 0) {
      MPI_Send(ids[i].data(), starsCount[i], MPI_LLD, i, 0, MPI_COMM_WORLD);
      MPI_Send(coords[i].data(), starsCount[i] * 2, MPI_FLOAT, i, 1,
               MPI_COMM_WORLD);
    }
  }

  setInitialSpeed();
  updateMasses();
}

int AbstractProcess::starCell(float x, float y) const {
  int cellX = std::min(int((x - space.x) / space.cellWidth), ver - 1);
  int cellY = std::min(int((y - space.y) / space.cellHeight), hor - 1);
  return cellY * ver + cellX;
}

void AbstractProcess::recvInitialStars() {
  lld count;

  MPI_Scatter(nullptr, 0, MPI_LLD, &count, 1, MPI_LLD, 0, MPI_COMM_WORLD);

  if (count > 0) {
    ids.resize(count);
    MPI_Recv(ids.data(), count, MPI_LLD, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    coords.resize(count * 2);
    MPI_Recv(coords.data(), count * 2, MPI_LLD, 0, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  setInitialSpeed();
  updateMasses();
}

void AbstractProcess::setInitialSpeed() {
  for (lld id : this->ids) {
    GalInfo *gal = id < 0 ? &gal2 : &gal1;
    this->speeds.push_back(gal->vX);
    this->speeds.push_back(gal->vY);
  }
}

void AbstractProcess::updateMasses() {
  masses.resize(ids.size());
  for (lld i = 0; i < ids.size(); i++) {
    masses[i] = ids[i] < 0 ? gal2.mass : gal1.mass;
  }
}

inline std::pair<float, float> calcF(float star1X, float star1Y, float mass1,
                                     float star2X, float star2Y, float mass2) {
  static const float G = 155893.597;
  const float dX = star1X - star2X;
  const float dY = star1Y - star2Y;
  const float norm = sqrt(dX * dX + dY * dY);
  return {(G * mass1 * mass2 * (star1X - star2X)) / (norm * norm * norm),
          (G * mass1 * mass2 * (star1Y - star2Y)) / (norm * norm * norm)};
}

void AbstractProcess::step(float delta) {
  std::vector<float> otherCoords;
  std::vector<float> otherMasses;
  std::vector<float> oldAccs;

  if (!firstStep) {
    oldAccs = accs;
  }

  getOtherStars(&otherCoords, &otherMasses);
  const auto toSkip = updateAccs(otherCoords, otherMasses);

  if (!firstStep) {
    updateSpeeds(oldAccs, delta, toSkip);
  }

  updateCoords(delta, toSkip);
  firstStep = false;
  exchangeStars();
  updateMasses();
}

std::vector<lld> AbstractProcess::updateAccs(
    const std::vector<float> &otherCoords,
    const std::vector<float> &otherMasses) {
  lld otherCount = otherCoords.size() / 2;
  std::vector<lld> toSkip;
  accs.resize(coords.size());

  for (lld i = 0; i < ids.size(); i++) {
    const lld xIdx = i * 2;
    const lld yIdx = i * 2 + 1;
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

    if (-accs[xIdx] > FLT_MAX / 2 || -accs[yIdx] > FLT_MAX / 2) {
      toSkip.push_back(xIdx / 2);
    }
    accs[xIdx] /= -masses[i];
    accs[yIdx] /= -masses[i];
  }

  return toSkip;
}

void AbstractProcess::updateSpeeds(const std::vector<float> &oldAccs,
                                   float delta,
                                   const std::vector<lld> &toSkip) {
  auto curToSkip = toSkip.begin();
  for (lld i = 0; i < speeds.size(); i++) {
    if (curToSkip != toSkip.end() && *curToSkip == i / 2) {
      ++curToSkip;
      i++;
      continue;
    }
    speeds[i] += 0.5 * (oldAccs[i] + accs[i]) * delta;
  }
}

void AbstractProcess::updateCoords(float delta,
                                   const std::vector<lld> &toSkip) {
  auto curToSkip = toSkip.begin();
  for (lld i = 0; i < coords.size(); i++) {
    if (curToSkip != toSkip.end() && *curToSkip == i / 2) {
      ++curToSkip;
      i++;
      continue;
    }
    coords[i] += speeds[i] * delta + 0.5 * accs[i] * delta * delta;
  }

  fixCoords();
}

void AbstractProcess::fixCoords() {
  for (lld i = 0; i < coords.size(); i += 2) {
    float &x = coords[i];
    float &y = coords[i + 1];
    const float spaceWidth = space.cellWidth * ver;
    const float spaceHeight = space.cellHeight * hor;

    if (x < space.x) {
      float distX = space.x - x;
      distX -= floor(distX / spaceWidth) * spaceWidth;
      x = space.x + spaceWidth - distX;
    } else if (x > space.x + spaceWidth) {
      float distX = x - (space.x + spaceWidth);
      distX -= floor(distX / spaceWidth) * spaceWidth;
      x = space.x + distX;
    }

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

void AbstractProcess::exchangeStars() {
  std::vector<lld> ids[numProcesses];
  std::vector<float> coords[numProcesses];
  std::vector<float> speeds[numProcesses];
  std::vector<float> accs[numProcesses];

  for (lld i = 0; i < this->ids.size(); i++) {
    const float xIdx = i * 2;
    const float yIdx = i * 2 + 1;
    const int cell = starCell(this->coords[xIdx], this->coords[yIdx]);
    ids[cell].push_back(this->ids[i]);
    coords[cell].push_back(this->coords[xIdx]);
    coords[cell].push_back(this->coords[yIdx]);
    speeds[cell].push_back(this->speeds[xIdx]);
    speeds[cell].push_back(this->speeds[yIdx]);
    accs[cell].push_back(this->accs[xIdx]);
    accs[cell].push_back(this->accs[yIdx]);
  }

  doExchangeStars(ids, coords, speeds, accs);

  this->ids = ids[rank];
  this->coords = coords[rank];
  this->speeds = speeds[rank];
  this->accs = accs[rank];
}

void AbstractProcess::doExchangeStars(std::vector<lld> *ids,
                                      std::vector<float> *coords,
                                      std::vector<float> *speeds,
                                      std::vector<float> *accs) {
  std::vector<lld> amountToSend;
  std::vector<lld> amountToRecv;

  amountToSend.resize(numProcesses);
  amountToRecv.resize(numProcesses);
  for (int i = 0; i < numProcesses; i++) {
    amountToSend[i] = ids[i].size();
  }

  lld oldData = ids[rank].size();
  lld countToRecv = 0;
  for (int i = 0; i < numProcesses; i++) {
    MPI_Scatter(amountToSend.data(), 1, MPI_LLD, amountToRecv.data() + i, 1,
                MPI_LLD, i, MPI_COMM_WORLD);
    if (i != rank) {
      countToRecv += amountToRecv[i];
    }
  }

  ids[rank].resize(oldData + countToRecv);
  coords[rank].resize((oldData + countToRecv) * 2);
  speeds[rank].resize((oldData + countToRecv) * 2);
  accs[rank].resize((oldData + countToRecv) * 2);

  lld *idsPtr = ids[rank].data() + oldData;
  float *coordsPtr = coords[rank].data() + (oldData * 2);
  float *speedsPtr = speeds[rank].data() + (oldData * 2);
  float *accsPtr = accs[rank].data() + (oldData * 2);

  auto send = [&](int otherRank) {
    if (amountToSend[otherRank]) {
      sendStarsTo(otherRank, ids[otherRank], coords[otherRank],
                  speeds[otherRank], accs[otherRank]);
    }
  };
  auto recv = [&](int otherRank) {
    if (amountToRecv[otherRank]) {
      lld count = amountToRecv[otherRank];
      recvStarsFrom(otherRank, count, idsPtr, coordsPtr, speedsPtr, accsPtr);
      idsPtr += count;
      coordsPtr += count * 2;
      speedsPtr += count * 2;
      accsPtr += count * 2;
    }
  };

  for (int otherRank : exchangeStarsOrder()) {
    if (rank < otherRank) {
      send(otherRank);
      recv(otherRank);
    } else {
      recv(otherRank);
      send(otherRank);
    }
  }
}

std::vector<int> AbstractProcess::exchangeStarsOrder() const {
  std::vector<int> partners;
  for (int step = 1; step <= std::max(rank, numProcesses - rank - 1); step++) {
    const int groupSize = step * 2;
    const bool isLeftSide = rank % groupSize < groupSize / 2;
    int newPartner1 = rank + (isLeftSide ? step : -step);
    int newPartner2 = rank + (isLeftSide ? -step : step);
    for (int newPartner : {newPartner1, newPartner2}) {
      if (newPartner >= 0 && newPartner < numProcesses) {
        partners.push_back(newPartner);
      }
    }
  }

  return partners;
}

void AbstractProcess::sendStarsTo(int otherRank, const std::vector<lld> &ids,
                                  const std::vector<float> &coords,
                                  const std::vector<float> &speeds,
                                  const std::vector<float> &accs) const {
  lld count = ids.size();
  MPI_Send(ids.data(), count, MPI_LLD, otherRank, 1001, MPI_COMM_WORLD);
  MPI_Send(coords.data(), count * 2, MPI_FLOAT, otherRank, 1002,
           MPI_COMM_WORLD);
  MPI_Send(speeds.data(), count * 2, MPI_FLOAT, otherRank, 1003,
           MPI_COMM_WORLD);
  MPI_Send(accs.data(), count * 2, MPI_FLOAT, otherRank, 1004, MPI_COMM_WORLD);
}

void AbstractProcess::recvStarsFrom(int otherRank, lld count, lld *ids,
                                    float *coords, float *speeds,
                                    float *accs) const {
  MPI_Recv(ids, count, MPI_LLD, otherRank, 1001, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(coords, count * 2, MPI_FLOAT, otherRank, 1002, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(speeds, count * 2, MPI_FLOAT, otherRank, 1003, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(accs, count * 2, MPI_FLOAT, otherRank, 1004, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
}

void AbstractProcess::printStars(std::string gal1, std::string gal2) const {
  if (rank == 0) {
    auto stars = receiveAllStarsForPrint();
    std::sort(stars.begin(), stars.end());

    FILE *out1 = fopen(gal1.c_str(), "w");
    if (!out1) {
      throw std::runtime_error("Unable to open file " + gal1 + " for writing");
    }
    FILE *out2 = fopen(gal2.c_str(), "w");
    if (!out2) {
      throw std::runtime_error("Unable to open file " + gal2 + " for writing");
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

std::vector<Star> AbstractProcess::receiveAllStarsForPrint() const {
  lld count = ids.size();
  lld starsCount[numProcesses];
  std::vector<MPI_Request> requests;

  MPI_Gather(&count, 1, MPI_LLD, starsCount, 1, MPI_LLD, 0, MPI_COMM_WORLD);
  lld countSum = 0;
  for (int i = 1; i < numProcesses; i++) {
    countSum += starsCount[i];
    if (starsCount[i] > 0) {
      requests.resize(requests.size() + 2);
    }
  }

  std::vector<lld> tmpIds;
  std::vector<float> tmpCoords;
  tmpIds.resize(countSum);
  tmpCoords.resize(countSum * 2);
  int lastRequest = 0;

  lld *idsPtr = tmpIds.data();
  float *coordsPtr = tmpCoords.data();
  for (int proc = 1; proc < numProcesses; proc++) {
    lld count = starsCount[proc];
    if (count) {
      MPI_Irecv(idsPtr, count, MPI_LLD, proc, 100, MPI_COMM_WORLD,
               &(requests[lastRequest++]));
      MPI_Irecv(coordsPtr, count * 2, MPI_FLOAT, proc, 101, MPI_COMM_WORLD,
               &(requests[lastRequest++]));
      idsPtr += count;
      coordsPtr += count * 2;
    }
  }

  std::vector<Star> stars;
  stars.reserve(ids.size() + countSum);
  for (lld i = 0; i < ids.size(); i++) {
    Star star;
    star.id = ids[i];
    star.x = coords[i * 2];
    star.y = coords[i * 2 + 1];
    stars.push_back(star);
  }

  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  for (lld i = 0; i < tmpIds.size(); i++) {
    Star star;
    star.id = tmpIds[i];
    star.x = tmpCoords[i * 2];
    star.y = tmpCoords[i * 2 + 1];
    stars.push_back(star);
  }
  return stars;
}

void AbstractProcess::sendAllStarsForPrint() const {
  lld count = ids.size();

  MPI_Gather(&count, 1, MPI_LLD, NULL, 0, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
  if (count) {
    MPI_Send(ids.data(), count, MPI_LLD, 0, 100, MPI_COMM_WORLD);
    MPI_Send(coords.data(), count * 2, MPI_FLOAT, 0, 101, MPI_COMM_WORLD);
  }
}
