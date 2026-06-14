//go:build !amd64 && !riscv64 && !ppc64le && !s390x && !(arm64 && go1.27)

package adler32

// blockGeneric is the block width of the portable fallback. Architectures
// without a SIMD kernel still fold the bulk of each chunk through chunkSIMD so
// the same updateChunk control flow (block fold + scalar tail) is exercised
// everywhere; the fold itself is a plain scalar loop here.
const blockGeneric = 8

// simdBlockBytes returns the number of leading bytes of a len-l chunk the
// portable fallback folds in chunkSIMD: the largest multiple of blockGeneric.
// The remaining bytes (l mod blockGeneric) go to updateChunk's scalar tail.
func simdBlockBytes(l int) int {
	if l >= blockGeneric {
		return l &^ (blockGeneric - 1)
	}
	return 0
}

// chunkSIMD folds p (a whole multiple of blockGeneric, len <= nmax) into
// (s1, s2) with a portable scalar loop. It performs no modular reduction; the
// caller keeps each call within nmax so the 32-bit accumulators cannot overflow.
// The arithmetic is identical to hash/adler32's inner loop, so the running
// totals match the standard library exactly.
func chunkSIMD(s1, s2 uint32, p []byte) (uint32, uint32) {
	for _, x := range p {
		s1 += uint32(x)
		s2 += s1
	}
	return s1, s2
}
