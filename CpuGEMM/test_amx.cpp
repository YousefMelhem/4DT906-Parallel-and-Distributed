#include "amx.hpp"
#include <concepts>
#include <gtest/gtest.h>
#include <vecLib/vecLib.h>
#include <iostream>
#include <iomanip>

#define barrier __asm volatile("dsb ish" : : : "memory")

static uint64_t now(clockid_t which) {
  barrier;
  uint64_t result = clock_gettime_nsec_np(which);
  barrier;
  return result;
}

template <size_t Repeat>
void benchmark(const size_t flops, std::function<void(void)> func,
               std::string unit = "Gflop/s") {
  auto start_time = now(CLOCK_THREAD_CPUTIME_ID);
  for (int i = 0; i < Repeat; ++i)
    func();
  auto end_time = now(CLOCK_THREAD_CPUTIME_ID);
  double ns = end_time - start_time;
  auto gflops = (flops * Repeat) / ns;
  std::cout << std::setw(20) << " " << unit << ": " << gflops << std::endl;
}

TEST(test_amx, test_ldxy) {
  float a[8 * 16];
  float b[8 * 16];
  for (size_t i = 0; i < 8 * 16; i++) {
    a[i] = (float)i;
  }
  constexpr auto ldop = ldxy{}.multiple().multiple_four();
  constexpr auto stop = stxy{}.multiple();
  // clang-format off
  AMX_SET();
  AMX_LDX(ldop.register_index(0).bind(a)); /* a[0:4] => x[0:4] */
  AMX_LDX(ldop.register_index(4).bind(a + 16 * 4));  /* a[4:8] => x[4:8] */
  AMX_STX(stop.register_index(0).bind(b));
  AMX_STX(stop.register_index(2).bind(b + 16 * 2));
  AMX_STX(stop.register_index(4).bind(b + 16 * 4));
  AMX_STX(stop.register_index(6).bind(b + 16 * 6));
  AMX_CLR();
  // clang-format on
  for (size_t i = 0; i < 8 * 16; i++) {
    GTEST_ASSERT_EQ(a[i], b[i]);
  }
}

TEST(test_amx, test_ldstz) {
  constexpr auto total = 16 * 8 * 8;
  float a[total];
  float b[total];
  float c[total];
  for (size_t i = 0; i < total; i++) {
    a[i] = (float)i;
  }
  constexpr auto ldop = ldstz{}; //.multiple();
  AMX_SET();
  for (size_t i = 0; i < 64; i++) {
    AMX_LDZ(ldop.row_index(i).bind(a + i * 16)); /* a[0:16] => z[_,0:16] */
    AMX_STZ(ldop.row_index(i).bind(b + i * 16));
  }
  for (size_t i = 0; i < 32; i++) {
    AMX_STZ(ldop.multiple().row_index(i * 2).bind(
        c + i * 16 * 2)); /*  z[i:i+2,0:16] => c[i:i+32] */
  }
  AMX_CLR();
  for (size_t i = 0; i < total; i++) {
    GTEST_ASSERT_EQ(a[i], b[i]);
    GTEST_ASSERT_EQ(a[i], c[i]);
  }
}

TEST(test_amx, test_ldstzi) {
  constexpr auto total = 16 * 8 * 8;
  float a[total];
  float b[total];
  float c[total];
  for (size_t i = 0; i < total; i++) {
    a[i] = (float)i;
  }
  AMX_SET();
  int i = 0;
  AMX_LDZI(ldstzi{}.right().row_index(i).bind(
      a + i * 16)); /* a[0:16] => z[_,0:16] */
  AMX_STZ(ldstz().row_index(i).bind(b + i * 16));
  AMX_STZ(ldstz().row_index((i + 1)).bind(b + (i + 1) * 16));
  AMX_CLR();
  for (size_t i = 0; i < 8; i++) {
    GTEST_ASSERT_EQ(a[i * 2], b[8 + i]);
  }
  for (size_t i = 0; i < 8; i++) {
    GTEST_ASSERT_EQ(a[i * 2 + 1], b[16 + 8 + i]);
  }
}

TEST(test_amx, test_fma) {
  float x[8][16];
  float y[8][16];
  float z[16][4][16];
  for (size_t i = 0; i < 8; i++) {
    for (size_t v = 0; v < 16; v++) {
      x[i][v] = (float)(i * 8 + v);
      y[i][v] = x[i][v] + 1;
    }
  }
  AMX_SET();
  AMX_LDX(ldxy().bind(x));
  AMX_LDY(ldxy().bind(y));
  AMX_FMA32(fma32().vector_mode().skip_z()); // 0 += x * y
  AMX_STZ(ldstz().bind(z));
  for (size_t i = 0; i < 16; i++) {
    GTEST_ASSERT_EQ(z[0][0][i], x[0][i] * y[0][i]);
  }

  AMX_LDX(ldxy().multiple().multiple_four().bind(x));
  AMX_LDY(ldxy().multiple().multiple_four().bind(y));
  AMX_LDX(ldxy().multiple().multiple_four().register_index(4).bind(x[4]));
  AMX_LDY(ldxy().multiple().multiple_four().register_index(4).bind(y[4]));
  AMX_FMA32(fma32().skip_z().z_row(0)); // mat mode
  for (size_t j = 0; j < 16; j++) {
    for (size_t z_row = 0; z_row < 4; z_row++) {
      AMX_STZ(ldstz().row_index(j * 4 + z_row).bind(z[j][z_row]));
    }
  }

  for (size_t z_row = 0; z_row < 4; z_row++) {
    for (size_t i = 0; i < 8; i++) {
      for (size_t j = 0; j < 8; j++) {
        if (z_row == 0) {
          GTEST_ASSERT_EQ(z[j][z_row][i], x[0][i] * y[0][j]);
        } else {
          GTEST_ASSERT_EQ(z[j][z_row][i], 0.0f);
        }
      }
    }
  }
  AMX_CLR();
}

TEST(test_amx, test_matfp) {
  float x[8][16];
  float y[8][16];
  float z[16][4][16];
  for (size_t i = 0; i < 8; i++) {
    for (size_t v = 0; v < 16; v++) {
      x[i][v] = (float)(i * 8 + v);
      y[i][v] = x[i][v] + 1;
    }
  }
  AMX_SET();
  AMX_LDX(ldxy().multiple().multiple_four().bind(x));
  AMX_LDY(ldxy().multiple().multiple_four().bind(y));
  AMX_MATFP(matfp().dtype_mode(matfp_dtype_t::f32f32).z_row(1)); // 0 += x * y
  AMX_STZ(ldstz().bind(z));
  for (size_t j = 0; j < 16; j++) {
    for (size_t z_row = 0; z_row < 4; z_row++) {
      AMX_STZ(ldstz().row_index(j * 4 + z_row).bind(z[j][z_row]));
    }
  }

  for (size_t z_row = 0; z_row < 4; z_row++) {
    for (size_t i = 0; i < 8; i++) {
      for (size_t j = 0; j < 8; j++) {
        if (z_row == 1) {
          GTEST_ASSERT_EQ(z[j][z_row][i], x[0][i] * y[0][j]);
        } else {
          GTEST_ASSERT_EQ(z[j][z_row][i], 0.0f);
        }
      }
    }
  }
  AMX_CLR();
}
/**
 * @brief
 * z_nums 1, 410.158 Gflop/s
 * z_nums 2, 824.180 Gflop/s
 * z_nums 3, 1244.750 Gflop/s
 * z_nums 4, 1665.007 Gflop/s
 */
TEST(test_amx, test_matfp_flops) {
  std::function<void(void)> perf_func;
  constexpr size_t infer_times = 754974720;
  for (size_t z_nums = 1; z_nums <= 4; z_nums++) {
    perf_func = [&z_nums]() {
      constexpr uint64_t a = matfp().dtype_mode(matfp_dtype_t::f32f32).z_row(0);
      constexpr uint64_t b = matfp().dtype_mode(matfp_dtype_t::f32f32).z_row(1);
      constexpr uint64_t c = matfp().dtype_mode(matfp_dtype_t::f32f32).z_row(2);
      constexpr uint64_t d = matfp().dtype_mode(matfp_dtype_t::f32f32).z_row(3);
      AMX_MATFP(a);
      if (z_nums > 1)
        AMX_MATFP(b);
      if (z_nums > 2)
        AMX_MATFP(c);
      if (z_nums > 3)
        AMX_MATFP(d);
    };
    auto start_time = now(CLOCK_THREAD_CPUTIME_ID);
    AMX_SET();
    for (int i = 0; i < infer_times; ++i)
      perf_func();
    AMX_CLR();
    auto end_time = now(CLOCK_THREAD_CPUTIME_ID);
    double ns = end_time - start_time;
    auto gflops = (16 * 16 * 2 * z_nums * infer_times) / ns;
    printf("z_nums %ld, %.3f Gflop/s\n", z_nums, gflops);
  }
}

TEST(test_amx, test_load_bandwith) {
  constexpr size_t K = (65536 / 4 / (16 * 4)) * 4;
  float M[K * 2][16 * 4]{};
  float N[K * 2][16 * 4]{};

  ldxy args[2];
  constexpr size_t infer_times = 75497;
  std::function<void()> perf_func = nullptr;
  for (size_t reg_num = 1; reg_num <= 2; reg_num++) {
    for (bool near : {true, false}) {
      for (auto &&x_width : {1, 2, 4}) {
        for (auto &&y_width : {0, 1, 2, 4}) {
          if ((reg_num == 1 && y_width > 0) || (reg_num == 2 && y_width == 0)) {
            continue;
          }
          perf_func = [&M, &N, &near, &reg_num, &x_width, &y_width]() {
            auto ldx = ldxy().register_index(0);
            auto ldy = ldxy().register_index(0);
            if (x_width >= 2)
              ldx = ldx.multiple();
            if (x_width >= 4)
              ldx = ldx.multiple_four();
            if (reg_num > 1) {
              if (y_width >= 2)
                ldy = ldy.multiple();
              if (y_width >= 4)
                ldy = ldy.multiple_four();
            }

            if (near) {
              for (size_t i = 0; i < K; i++) {
                AMX_LDX(ldx.bind(M[i]));
                if (reg_num > 1) {
                  AMX_LDY(ldy.bind(N[i]));
                }
              }
            } else {
              for (size_t i = 0; i < K / 2; i++) {
                for (size_t j = 0; j < 2; j++) {
                  AMX_LDX(ldx.bind(M[j * K + i]));
                  if (reg_num > 1) {
                    AMX_LDY(ldy.bind(N[j * K + i]));
                  }
                }
              }
            }
          };
          printf("reg_num %ld, near %d, x_width %d, y_width %d", reg_num, near,
                 x_width, y_width);
          AMX_SET();
          benchmark<infer_times>(16 * (x_width + y_width) * K * sizeof(float),
                                 perf_func, "GB/s");
          AMX_CLR();
        }
      }
    }
  }
}

TEST(test_amx, test_store_bandwith) {
  constexpr size_t K = (65536 / 4 / (16 * 4)) * 2;
  float CNear[16][16 * 4]{};
  float C[16][K]{};

  constexpr size_t infer_times = 7549720;
  std::function<void()> perf_func = nullptr;

  for (size_t z_num : {1, 2, 3, 4}) {
    for (bool near : {true, false}) {
      for (size_t width : {1, 2}) {
        if (width * z_num > 4) {
          continue;
        }
        perf_func = [&C, &CNear, &near, &z_num, &width]() {
          auto ldst = width == 2 ? ldstz().multiple() : ldstz();
          for (size_t z = 0; z < z_num; z++) {
            for (size_t m = 0; m < 16; m++) {
              AMX_STZ(ldst.row_index(m * 4 + z * width)
                          .bind(near ? CNear[m] + 16 * z * width
                                     : C[m] + 16 * z * width));
            }
          }
        };
        printf("znum : %ld, near: %d, width : %ld ,", z_num, near, width);
        AMX_SET();
        benchmark<infer_times>(16 * 16 * 4 * z_num * width, perf_func, "GB/s");
        AMX_CLR();
      }
    }
  }
}

/**
 * @brief 验证load过数据之后是否会存储在cache中
 *  K factor:    1        4        8
 *      GB/s: 17.7216  27.3286   31.463
 *      GB/s: 176.291  242.449   250.904
 *      GB/s: 146.449  18.2435   9.70864
 *   note load 同一地址比load连续地址更慢, 这一点无法理解.
 *   load 的过程中应该会加入l1 cache, 当factor为1时,C被load完刚好存满l1 cache,
 * 所以他的速度也比较快.
 *   note factor 大于4时, M理论上也无法完成cache, 但是速度更快, 这一点无法理解.
 */
TEST(test_amx, test_load_and_cache) {
  constexpr size_t K = (65536 / 4 / (16 * 2)) * 8; /* 65536 是cache size */
  float N[1][16 * 2]{};
  float M[K][16 * 2]{};
  float *C = (float *)malloc(K * 65536);
  AMX_SET();
  benchmark<10000>(
      32 * K * 4,
      [&N]() {
        for (size_t i = 0; i < K; i++) {
          AMX_LDX(ldxy().multiple().register_index(0).bind((void *)N[0]));
        }
      },
      "GB/s");
  AMX_CLR();
  AMX_SET();
  benchmark<10000>(
      32 * K * 4,
      [&M]() {
        for (size_t i = 0; i < K; i++) {
          AMX_LDX(ldxy().multiple().register_index(0).bind((void *)M[i]));
        }
      },
      "GB/s");
  AMX_CLR();
  AMX_SET();
  benchmark<10000>(
      32 * K * 4,
      [&C]() {
        for (size_t i = 0; i < K; i++) {
          AMX_LDX(
              ldxy().multiple().register_index(0).bind((void *)(C + i * K)));
        }
      },
      "GB/s");
  AMX_CLR();
  free(C);
}

TEST(test_amx, test_genlut_generate) {
  float table[1 * 16];
  for (size_t i = 0; i < 16; i++) {
    table[i] = i;
  }
  float source[1 * 16];
  for (size_t i = 0; i < 16; i++) {
    source[i] = i;
  }

  /*
  在生成模式中, 会从register_table中加载DT[LC]向量, 然后此时还需要有source 向量
  然后生成IT[LC]向量将被写到x或y寄存器, 剩下的x或y寄存器将会被置0.

  然后遍历所有的source, 找到第一个table[v] > source[i], 此时IT[i]  = v - 1;
  假设table =[0,1,2,3], source =[0,1,2,3]
  那么此时index = [0,1,2,3],
  然后将index按u4的形式从0开始按右向左pack得到最终结果.
  */
  ic(table);
  ic(source);
  AMX_SET();
  AMX_LDX(ldxy().register_index(0).bind(table));
  AMX_LDX(ldxy().register_index(1).bind(source));
  AMX_GENLUT(genlut()
                 .mode(genlut_mode::g_f32_u4_16)
                 .register_table(0)
                 .src_offset(16 * 4));
  AMX_STX(stxy().bind(table));
  AMX_CLR();
  uint32_t index[16];
  auto bits = *reinterpret_cast<uint64_t *>(&table);
  for (size_t i = 0; i < 16; i++) {
    int s = i * 4;
    index[i] = (bits & (0b1111ul << s)) >> s;
  }
  ic(index);
  for (size_t i = 0; i < 16; i++) {
    GTEST_ASSERT_EQ(source[i], index[i]);
  }
}

TEST(test_amx, test_genlut_lookup) {
  // 假设我要生成一个倒序加载
  float table[1 * 16];
  float source[1 * 16];
  float output[1 * 16];
  for (size_t i = 0; i < 16; i++) {
    table[i] = i;
    source[i] = 15 - i;
  }

  ic(table);
  ic(source);
  AMX_SET();
  AMX_LDX(ldxy().register_index(0).bind(table));
  AMX_LDX(ldxy().register_index(1).bind(source));
  // generate
  AMX_GENLUT(genlut()
                 .mode(genlut_mode::g_f32_u4_16)
                 .register_table(0)
                 .src_offset(16 * 4));
  // look up时, source是上一步生成的index, table是实际的数据
  for (size_t i = 0; i < 16; i++) {
    source[i] = 16 + i;
  }
  AMX_LDX(ldxy().register_index(1).bind(source));
  AMX_GENLUT(genlut()
                 .mode(genlut_mode::lc_a32_u4_16)
                 .register_table(1)
                 .src_offset(0)
                 .dest_y());
  AMX_STY(stxy().bind(output));
  AMX_CLR();
  for (size_t i = 0; i < 16; i++) {
    GTEST_ASSERT_EQ(output[i], source[15 - i]);
  }
}

template <bool LoadC, bool StoreC, size_t KTile, size_t M, size_t N>
void gepdot(size_t curM, size_t curK, size_t curN,
            const float packedA[KTile][32], const float packedB[KTile][32],
            float C[M][N]) {
  static_assert(KTile % 4 == 0, "not support k%4!=0");

  if constexpr (LoadC) {
    // load acc value.
    for (size_t om = 0; om < 2; om++) {
      for (size_t m = 0; m < 16; m++) {
        AMX_STZ(
            ldstz().row_index(m * 4 + om * 2).multiple().bind(C[om * 16 + m]));
      }
    }
  }

  for (size_t k = curK; k < curK + KTile; k += 4) {
    for (size_t ok = k; ok < k + 4; ok += 2) {
      // load [m0,k0], [m1,k0], [m0,k1], [m1,k1]
      AMX_LDY(ldxy().register_index(0).multiple().multiple_four().bind(
          (void *)packedA[ok]));
      // load [n0,k0], [n1,k0], [n0,k1], [n1,k1]
      AMX_LDX(ldxy().register_index(0).multiple().multiple_four().bind(
          (void *)packedB[ok]));
      // compute 8 times.
      // [[m0,n0],[m0,n1],
      //  [m1,n0],[m1,n1]]
      for (size_t ik = ok; ik < ok + 2; ik++) {
        auto fma = ik == 0 ? fma32().skip_z() : fma32();
        for (size_t m = 0; m < 2; m++) {
          for (size_t n = 0; n < 2; n++) {
            AMX_FMA32(fma.z_row(m * 2 + n)
                          .y_offset((ik - ok) * 2 * 16 * 4 + m * 16 * 4)
                          .x_offset((ik - ok) * 2 * 16 * 4 + n * 16 * 4));
          }
        }
      }
    }
  }

  // last time need store C.
  if constexpr (StoreC) {
    for (size_t om = 0; om < 2; om++) {
      for (size_t m = 0; m < 16; m++) {
        AMX_STZ(
            ldstz().row_index(m * 4 + om * 2).multiple().bind(C[om * 16 + m]));
      }
    }
  }
}

/**
 * @brief k很大可以流水时, 1652.19 gflops. 已经达到峰值算力的90%了
 *
 */
TEST(test_amx, test_gepdot) {
  constexpr size_t M = 16 * 2, K = 8192, N = 16 * 2;
  float A[M][K];
  float B[K][N];
  float C[M][N];
  float RefC[M][N];
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      A[i][j] = i * K + j;
    }
  }
  for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < N; j++) {
      B[i][j] = i * K + j;
    }
  }

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
              (const float *)A, K, (const float *)B, N, 0.0f, (float *)RefC, N);

  float PackedA[K][M];
  for (size_t k = 0; k < K; k++) {
    for (size_t m = 0; m < M; m++) {
      PackedA[k][m] = A[m][k];
    }
  }

  auto func1 = [&]() -> void {
    gepdot<false, true, K, M, N>(0, 0, 0, PackedA, B, C);
  };

  AMX_SET();
  benchmark<100000>(2 * M * K * N, func1);
  AMX_CLR();
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      GTEST_ASSERT_EQ(C[i][j], RefC[i][j]);
    }
  }
}

template <size_t M, size_t N>
void store_mn(size_t z_row, size_t m, size_t n, float C[M][N]) {
  for (size_t i = 0; i < 16; i++) {
    AMX_STZ(ldstz().row_index(i * 4 + z_row).bind(C[m + i] + n));
  }
}

template <size_t M, size_t N>
void load_mn(size_t z_row, size_t m, size_t n, float C[M][N]) {
  for (size_t i = 0; i < 16; i++) {
    AMX_LDZ(ldstz().row_index(i * 4 + z_row).bind(C[m + i] + n));
  }
}

template <size_t MTile, size_t KTile, size_t NTile, size_t M, size_t K,
          size_t N>
void gebp(size_t curM, size_t curK, size_t curN,
          const float packedA[KTile][MTile], const float B[K][N],
          float C[M][N]) {
  static_assert(KTile <= 8, "KTile <= 8");
  static_assert(MTile == 16, "MTile == 16");
  static_assert(NTile % 64 == 0, "NTile % 64 == 0");
  // cache packed A
  for (size_t k = curK; k < curK + KTile; k++) {
    AMX_LDY(ldxy().register_index(k).bind((void *)(packedA[k])));
  }
  for (size_t on = curN; on < curN + NTile; on += 64) {
    for (size_t k = curK; k < curK + KTile; k++) {
      // load 4 n, compute 4 time.
      AMX_LDX(ldxy().register_index(0).multiple().multiple_four().bind(
          (void *)(B[k] + on)));
      for (size_t z_row = 0; z_row < 4; z_row++) {
        size_t noffset = on + z_row * 16;
        auto fma = fma32()
                       .y_offset(k * 16 * 4 /* bytes */)
                       .x_offset(z_row * 16 * 4 /* bytes */)
                       .z_row(z_row);
        if (k == 0) {
          fma = fma.skip_z();
        } else if (k == curK) {
          load_mn<M, N>(z_row, curM, noffset, C);
        }
        AMX_FMA32(fma);
        if (k == curK + KTile - 1) {
          store_mn<M, N>(z_row, curM, noffset, C);
        }
      }
    }
  }
}

TEST(test_amx, test_gebp) {
  constexpr size_t M = 16, K = 8, N = 1024;
  float A[M][K];
  float B[K][N];
  float C[M][N];
  float RefC[M][N];
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      A[i][j] = (i * K + j) / (float)K;
    }
  }
  for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < N; j++) {
      B[i][j] = (i * K + j) / (float)K;
    }
  }

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
              (const float *)A, K, (const float *)B, N, 0.0f, (float *)RefC, N);

  float PackedA[K][M];
  for (size_t m = 0; m < M; m++) {
    for (size_t k = 0; k < K; k++) {
      PackedA[k][m] = A[m][k];
    }
  }

  auto func1 = [&]() -> void {
    gebp<M, K, N, M, K, N>(0, 0, 0, PackedA, B, C);
  };

  AMX_SET();
  benchmark<1000000>(2 * M * K * N, func1);
  AMX_CLR();
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      GTEST_ASSERT_EQ(C[i][j], RefC[i][j]);
    }
  }
}

/**
 * @brief 这个版本不限制B的形状
 *
 */
template <size_t KTile, size_t M, size_t K, size_t N>
void gepdot_general(bool loadC, bool storeC, size_t curM, size_t curK,
                    size_t curN, const float packedA[KTile][32],
                    const float B[K][N], float C[M][N]) {
  static_assert(KTile >= 4 && KTile % 4 == 0, "not support k%4!=0");
  static_assert(N > 16, "N > 16");

  if (loadC) {
    // load acc value.
    for (size_t om = 0; om < 2; om++) {
      for (size_t m = 0; m < 16; m++) {
        AMX_STZ(ldstz()
                    .row_index(m * 4 + om * 2)
                    .multiple()
                    .bind(C[curM + om * 16 + m] + curN));
      }
    }
  }

  for (size_t k = curK; k < curK + KTile; k += 4) {
    for (size_t ok = k, yi = 0; ok < k + 4; ok += 2, yi += 4) {
      // load [m0,k0], [m1,k0], [m0,k1], [m1,k1] to y[yi:yi+4]
      AMX_LDY(ldxy().register_index(yi).multiple().multiple_four().bind(
          (void *)packedA[ok - curK]));

      for (size_t ik = ok, xi = 0; ik < ok + 2; ik++, xi += 2) {
        // load [n0,k0], [n1,k0] to x[yi+xi:yi+xi+2]
        AMX_LDX(ldxy().register_index(yi + xi).multiple().bind(
            (void *)(B[ik] + curN)));
        // compute 4 times.
        // [[m0,n0](k0),[m0,n1](k0),
        //  [m1,n0](k0),[m1,n1](k0)]
        auto fma = ik == 0 ? fma32().skip_z() : fma32();
        for (size_t m = 0; m < 2; m++) {
          for (size_t n = 0; n < 2; n++) {
            constexpr size_t elem = 16 * 4;
            AMX_FMA32(fma.z_row(m * 2 + n)
                          .y_offset(yi * elem + xi * elem + m * elem)
                          .x_offset(yi * elem + xi * elem + n * elem));
          }
        }
      }
    }
  }

  // last time need store C.
  if (storeC) {
    for (size_t om = 0; om < 2; om++) {
      for (size_t m = 0; m < 16; m++) {
        AMX_STZ(ldstz()
                    .row_index(m * 4 + om * 2)
                    .multiple()
                    .bind(C[curM + om * 16 + m] + curN));
      }
    }
  }
}

TEST(test_amx, test_gepdot_general) {
  constexpr size_t M = 32 * 4, K = 256, N = 32 * 16;
  float(*A)[K] = (float(*)[K])malloc(M * K * 4);
  float(*B)[N] = (float(*)[N])malloc(K * N * 4);
  float(*C)[N] = (float(*)[N])malloc(M * N * 4);
  float(*RefC)[N] = (float(*)[N])malloc(M * N * 4);
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      A[i][j] = (i * K + j) / (float)(K * M);
    }
  }
  for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < N; j++) {
      B[i][j] = (i * K + j) / (float)(K * N);
    }
  }

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
              (const float *)A, K, (const float *)B, N, 0.0f, (float *)RefC, N);

  auto func1 = [&]() -> void {
    constexpr size_t KTile = 32, MTile = 32, NTile = 32;
    for (size_t mo = 0; mo < M; mo += MTile) {
      float PackedA[K][MTile];
      for (size_t no = 0; no < N; no += NTile) {
        for (size_t ko = 0; ko < K; ko += KTile) {
          // each time pack local innertile.
          if (no == 0) {
            for (size_t mi = 0; mi < MTile; mi++) {
              for (size_t i = ko; i < ko + KTile; i++) {
                PackedA[i][mi] = A[mo + mi][i];
              }
            }
          }
          gepdot_general<KTile, M, K, N>(false, (ko + KTile == K), mo, ko, no,
                                         PackedA + ko, B, C);
        }
      }
    }
  };

  printf("gepdot general");
  AMX_SET();
  benchmark<10000>(2 * M * K * N, func1);
  AMX_CLR();
  printf("cblas_sgemm");
  benchmark<10000>(2 * M * K * N, [&]() {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                (const float *)A, K, (const float *)B, N, 0.0f, (float *)RefC,
                N);
  });
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      GTEST_ASSERT_EQ(C[i][j], RefC[i][j]);
    }
  }
  free(A);
  free(B);
  free(C);
  free(RefC);
}