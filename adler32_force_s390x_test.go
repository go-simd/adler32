//go:build s390x

package adler32

import (
	stdadler "hash/adler32"
	"testing"
)

// TestS390XDispatch drives both s390x dispatch branches. With the vector
// facility present (the QEMU default and any z13+ host) simdBlockBytes folds
// whole 16-byte blocks through the vector kernel and updateChunk's scalar tail
// handles only the 0..15-byte remainder; forcing hasVX low exercises the scalar
// fallback (simdBlockBytes returns 0). Each variant is checked bit-for-bit
// against hash/adler32, including the ascending-byte pattern that would expose a
// wrong positional weight order under big-endian lane numbering.
func TestS390XDispatch(t *testing.T) {
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

	// Vector facility present (QEMU default / z13+): the vector kernel runs.
	check("hasVX")

	// Force the scalar fallback: simdBlockBytes returns 0 and updateChunk's
	// scalar tail does all the work.
	saved := hasVX
	defer func() { hasVX = saved }()
	hasVX = false
	check("noVX")
}
