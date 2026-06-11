//go:build amd64

package adler32

import (
	stdadler "hash/adler32"
	"testing"
)

// checksumForce computes Adler-32 driving a chosen kernel (SSE or AVX2)
// directly, mirroring update/updateChunk but pinning the block path so both
// amd64 kernels are validated even when the runtime CPU (or Rosetta) would not
// dispatch to one of them.
func checksumForce(data []byte, avx2 bool) uint32 {
	block := blockSSE
	if avx2 {
		block = blockAVX2
	}
	s1, s2 := uint32(1), uint32(0)
	p := data
	for len(p) > 0 {
		var q []byte
		if len(p) > nmax {
			p, q = p[:nmax], p[nmax:]
		}
		if n := len(p) &^ (block - 1); n >= block {
			if avx2 {
				s1, s2 = adlerAVX2(s1, s2, p[:n], n/blockAVX2)
			} else {
				s1, s2 = adlerSSE(s1, s2, p[:n], n/blockSSE)
			}
			p = p[n:]
		}
		for _, x := range p {
			s1 += uint32(x)
			s2 += s1
		}
		s1 %= mod
		s2 %= mod
		p = q
	}
	return s2<<16 | s1
}

// TestForceKernels validates both the SSE and AVX2 kernels against
// hash/adler32 across many sizes and patterns, independent of CPU dispatch.
// AVX2 is only exercised when the CPU supports it (the instructions would #UD
// otherwise).
func TestForceKernels(t *testing.T) {
	patterns := []struct {
		name string
		gen  func(n int) []byte
	}{
		{"rand", func(n int) []byte { return randBytes(n, int64(n)*17+1) }},
		{"zero", func(n int) []byte { return make([]byte, n) }},
		{"ff", func(n int) []byte { return bytesFill(n, 0xff) }},
		{"asc", bytesAsc},
	}
	for _, avx2 := range []bool{false, true} {
		if avx2 && !hasAVX2 {
			continue
		}
		for _, p := range patterns {
			for _, n := range sizes {
				src := p.gen(n)
				if got, want := checksumForce(src, avx2), stdadler.Checksum(src); got != want {
					t.Fatalf("avx2=%v %s n=%d: got=%#08x want=%#08x", avx2, p.name, n, got, want)
				}
			}
		}
	}
}

func benchForce(b *testing.B, avx2 bool) {
	if avx2 && !hasAVX2 {
		b.Skip("no AVX2")
	}
	src := randBytes(1<<20, 2)
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		checksumForce(src, avx2)
	}
}

func BenchmarkChecksumForceSSE(b *testing.B)  { benchForce(b, false) }
func BenchmarkChecksumForceAVX2(b *testing.B) { benchForce(b, true) }
