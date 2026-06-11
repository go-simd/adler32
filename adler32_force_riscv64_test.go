//go:build riscv64

package adler32

import (
	stdadler "hash/adler32"
	"testing"
)

// TestRISCVDispatch drives both riscv64 dispatch branches. With the V extension
// present (the common QEMU/hardware case) simdBlockBytes folds the whole chunk
// through the RVV kernel and updateChunk's scalar tail never runs; forcing hasV
// low exercises the no-V fallback (simdBlockBytes returns 0, so the scalar tail
// in updateChunk folds every byte) so both paths are covered on a single host.
// Each variant is checked bit-for-bit against hash/adler32.
func TestRISCVDispatch(t *testing.T) {
	check := func(tag string) {
		for _, n := range sizes {
			for _, gen := range []func(int) []byte{
				func(n int) []byte { return randBytes(n, int64(n)*17+1) },
				func(n int) []byte { return bytesAsc(n) },
			} {
				src := gen(n)
				if got, want := Checksum(src), stdadler.Checksum(src); got != want {
					t.Fatalf("%s n=%d: got=%#08x want=%#08x", tag, n, got, want)
				}
			}
		}
	}

	// Real CPU flag (V present under the QEMU rv64,v=true target): RVV kernel.
	check("hasV")

	// Force the no-V fallback: simdBlockBytes returns 0 and updateChunk's scalar
	// tail does all the work.
	saved := hasV
	defer func() { hasV = saved }()
	hasV = false
	check("noV")
}
