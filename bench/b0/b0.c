/* B0 rows (STATUS.md): the same Level 3 shapes the roofline scores,
 * timed through a CBLAS library compiled to WebAssembly and run in
 * Node on the same runner as this module, single-threaded. Linked
 * once against OpenBLAS and once against BLIS by bench/b0/build-wasm.sh;
 * prints `library,op,n,ms,gflops` per row, min over the iterations.
 * The matrices are LCG-filled like the bench's. */
#include <cblas.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static unsigned long long st = 0x9E3779B97F4A7C15ull;
static double next(void) {
	st = st * 6364136223846793005ull + 1442695040888963407ull;
	return ((double)(st >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}
static void fill(double *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = next(); }
static void fillf(float *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = (float)next(); }

#define ROW(lib, op, n, body, flops) do { \
	double best = 1e300; int iters = (n) >= 1024 ? 5 : 9; \
	body; /* warm */ \
	for (int r = 0; r < iters; r++) { double t0 = now_ms(); body; double t = now_ms() - t0; if (t < best) best = t; } \
	printf("%s,%s,%d,%.3f,%.2f\n", lib, op, (int)(n), best, (double)(flops) / (best / 1e3) / 1e9); \
} while (0)

int main(int argc, char **argv) {
	const char *lib = argc > 1 ? argv[1] : "cblas";
	int sizes[] = {512, 1024};
	for (int si = 0; si < 2; si++) {
		int n = sizes[si];
		size_t nn = (size_t)n * n;
		double *a = malloc(nn * 8), *b = malloc(nn * 8), *c = malloc(nn * 8);
		float *af = malloc(nn * 4), *bf = malloc(nn * 4), *cf = malloc(nn * 4);
		double *az = malloc(nn * 16), *bz = malloc(nn * 16), *cz = malloc(nn * 16);
		float *ac = malloc(nn * 8), *bc = malloc(nn * 8), *cc = malloc(nn * 8);
		fill(a, nn); fill(b, nn); fill(c, nn); fillf(af, nn); fillf(bf, nn); fillf(cf, nn);
		fill(az, 2 * nn); fill(bz, 2 * nn); fill(cz, 2 * nn); fillf(ac, 2 * nn); fillf(bc, 2 * nn); fillf(cc, 2 * nn);
		for (int i = 0; i < n; i++) a[(size_t)i * n + i] = 2.0 * n + 1.0; /* diagonally dominant for the solves */
		double alz[2] = {0.001, 0.0}, bez[2] = {0.5, 0.0};
		float alc[2] = {0.001f, 0.0f}, bec[2] = {0.5f, 0.0f};
		double flops3 = 2.0 * n * (double)n * n;
		ROW(lib, "dgemm", n, cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 0.001, a, n, b, n, 0.5, c, n), flops3);
		ROW(lib, "sgemm", n, cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 0.001f, af, n, bf, n, 0.5f, cf, n), flops3);
		ROW(lib, "zgemm", n, cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alz, az, n, bz, n, bez, cz, n), 4.0 * flops3);
		ROW(lib, "cgemm", n, cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alc, ac, n, bc, n, bec, cc, n), 4.0 * flops3);
		if (n == 512) {
			ROW(lib, "dsymm_left", n, cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, n, n, 0.001, a, n, b, n, 0.5, c, n), flops3);
			ROW(lib, "dsyrk", n, cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, n, 0.001, a, n, 0.5, c, n), flops3 / 2.0);
			ROW(lib, "dtrmm_left", n, cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n, n, 1.0, a, n, b, n), flops3 / 2.0);
			ROW(lib, "dtrsm_left", n, cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n, n, 1.0, a, n, b, n), flops3 / 2.0);
		}
		free(a); free(b); free(c); free(af); free(bf); free(cf); free(az); free(bz); free(cz); free(ac); free(bc); free(cc);
	}
	return 0;
}
