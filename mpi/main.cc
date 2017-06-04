#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include "process_1.h"
#include "process_2.h"
#include "process_3.h"

std::unordered_map<std::string, std::string> parseArgs(int argc, char **argv) {
  std::unordered_map<std::string, std::string> args;
  for (int i = 1; i < argc; i++) {
    std::string str = argv[i];
    if (str == "--v" || str == "-v" || str == "v") {
      args.insert(std::make_pair("v", ""));
    } else {
      if (i + 1 < argc) {
        args.insert(
            std::make_pair(std::string(argv[i]), std::string(argv[i + 1])));
        i++;
      }
    }
  }

  return args;
}

bool readParams(std::unordered_map<std::string, std::string> args,
                std::unique_ptr<AbstractProcess> &p, bool *verbose,
                std::string *gal1, std::string *gal2, float *delta,
                float *total) {
  for (auto key : {"hor", "ver", "gal1", "gal2", "delta", "total"}) {
    std::string key2 = key;
    if (args.count(key2)) {
      continue;
    }

    key2 = "-" + key2;
    if (args.count(key2)) {
      args[key] = args[key2];
      continue;
    }

    key2 = "-" + key2;
    if (args.count(key2)) {
      args[key] = args[key2];
      continue;
    }

    std::cerr << "Missing option: " << key << "\n";
    return false;
  }

  try {
    *verbose = args.count("v") > 0;
    p->hor = std::stoi(args["hor"]);
    p->ver = std::stoi(args["ver"]);
    *gal1 = args["gal1"];
    *gal2 = args["gal2"];
    *delta = std::stof(args["delta"]);
    *total = std::stof(args["total"]);
  } catch (const std::exception &e) {
    std::cerr << "Argument format error: " << e.what() << "\n";
    return false;
  }

  return true;
}

bool checkParams(int hor, int ver, float delta, float total, int rank,
                 int numProcesses) {
  const double eps = 10e-9;
  auto err = [&](const std::string &msg) {
    if (rank == 0) {
      std::cerr << msg << std::endl;
    }
  };

  if (hor < 0 || ver < 0 || numProcesses < 0 ||
      (lld)hor * ver != numProcesses) {
    err("incorrect number of processes");
    return false;
  }

  if (delta < 0.1 - eps || delta > 1.0 + eps) {
    err("incorrect time step");
    return false;
  }
  if (total < 1.0 - eps || total > 100.0 + eps) {
    err("incorrect total duration");
    return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  std::unique_ptr<AbstractProcess> p;
#ifdef V1
  p.reset(new Process1());
#elif V2
  p.reset(new Process2());
#elif V3
  p.reset(new Process3());
#else
  assert(false && "No version choosen");
#endif

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p->numProcesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &p->rank);

  auto args = parseArgs(argc, argv);
  std::string gal1Filename, gal2Filename;
  float delta, total;
  bool verbose;
  if (!readParams(args, p, &verbose, &gal1Filename, &gal2Filename, &delta,
                  &total)) {
    MPI_Finalize();
    return 0;
  }

  if (!checkParams(p->hor, p->ver, delta, total, p->rank, p->numProcesses)) {
    MPI_Finalize();
    return 0;
  }

  try {
    std::vector<Star> stars;
    if (p->rank == 0) {
      p->readGal(std::move(gal1Filename), 1, &stars);
      p->readGal(std::move(gal2Filename), 2, &stars);
      p->calcSpace(stars);
    }

    p->exchangeSpaceInfo();
    p->exchangeGalaxiesInfo();

    if (p->rank == 0) {
      p->distributeInitialStars(stars);
      stars.clear();
    } else {
      p->recvInitialStars();
    }

    for (int i = 0; i < round(total / delta); i++) {
      if (p->rank == 0) {
        printf("step %d\n", i);
      }
      if (verbose) {
        p->printStars("res1_" + std::to_string(i) + ".txt",
                      "res2_" + std::to_string(i) + ".txt");
      }

      p->step(delta);
    }
    p->printStars("res1.txt", "res2.txt");
  } catch (std::exception &e) {
    std::cerr << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_IO);
    return 0;
  }

  MPI_Finalize();
}
