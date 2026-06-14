//go:build ppc64le

package adler32

import (
	stdadler "hash/adler32"
	"testing"
)

// TestPPC64Dispatch drives both ppc64le dispatch branches. With VSX present (the
// production case and the QEMU power9 target) simdBlockBytes folds whole 16-byte
// blocks through the VSX kernel and updateChunk's scalar tail handles only the
// 0..15-byte remainder; forcing hasVSX low exercises the no-SIMD fallback
// (simdBlockBytes returns 0, so the scalar tail folds every byte). Each variant
// is checked bit-for-bit against hash/adler32, including the positional-weight
// patterns (ascending bytes) that would expose a wrong weight order.
func TestPPC64Dispatch(t *testing.T) {
	check := func(tag string) {
		for _, n := range sizes {
			for _, gen := range []func(int) []byte{
				func(n int) []byte { return randBytes(n, int64(n)*17+1) },
				func(n int) []byte { return bytesAsc(n) },
				func(n int) []byte { return bytesFill(n, 0xff) },
			} {
				src := gen(n)
				if got, want := Checksum(src), stdadler.Checksum(src); got != want {
					t.Fatalf("%s n=%d: got=%#08x want=%#08x", tag, n, got, want)
				}
			}
		}
	}

	// VSX present (baseline ppc64le / QEMU power9): the vector kernel runs.
	check("hasVSX")

	// Force the no-SIMD fallback: simdBlockBytes returns 0 and updateChunk's
	// scalar tail does all the work.
	saved := hasVSX
	defer func() { hasVSX = saved }()
	hasVSX = false
	check("noVSX")
}
