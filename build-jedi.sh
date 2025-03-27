module --force purge
module load Stages/2025
module load GCC
module load ParaStationMPI
module load CMake
module load CUDA

rm -rf build
mkdir build
# -DCMAKE_PREFIX_PATH=/p/project1/jureap1/sewell1/ascent/scripts/build_ascent/install \
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_HOPPER90=ON \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DENABLE_FFT=ON \
  -DENABLE_SOLVERS=ON \
  -DENABLE_ALPINE=True \
  -DENABLE_TESTS=OFF \
  -DUSE_ALTERNATIVE_VARIANT=ON \
  -DIPPL_PLATFORMS=CUDA \
  -DAscent_DIR=/p/project1/jureap1/sewell1/ascent/scripts/build_ascent/install/ascent-checkout/lib/cmake/ascent


cmake --build build -j8
