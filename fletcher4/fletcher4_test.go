package fletcher4

import (
	"math/rand/v2"
	"testing"
)

// refScalar is OpenZFS fletcher_4_scalar_native (module/zcommon/zfs_fletcher.c,
// zfs-2.3.0, lines 317-336) transcribed word for word: the exact oracle.
func refScalar(buf []byte) [4]uint64 {
	var a, b, c, d uint64
	n := len(buf) / 4
	for i := 0; i < n; i++ {
		ip := uint32(buf[4*i]) | uint32(buf[4*i+1])<<8 | uint32(buf[4*i+2])<<16 | uint32(buf[4*i+3])<<24
		a += uint64(ip)
		b += a
		c += b
		d += c
	}
	return [4]uint64{a, b, c, d}
}

func randBuf(r *rand.Rand, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(r.Uint32())
	}
	return b
}

func TestScalarEqualsOpenZFSScalar(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for n := 0; n < 600; n++ {
		b := randBuf(r, n)
		if got, want := SumScalar(b), refScalar(b); got != want {
			t.Fatalf("len %d: %x != %x", n, got, want)
		}
	}
}

// Sum (whatever kernel this CPU picks) equals the scalar oracle on every length,
// including partial words, and on misaligned starts.
func TestSumEqualsOracle(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for n := 0; n <= 700; n++ {
		raw := randBuf(r, n+3)
		for off := 0; off < 3; off++ {
			b := raw[off : off+n]
			if got, want := Sum(b), refScalar(b); got != want {
				t.Fatalf("%s len %d off %d: %x != %x", Implementation(), n, off, got, want)
			}
		}
	}
	for _, n := range []int{4096, 65536, 1 << 17, 1<<20 + 36} {
		b := randBuf(r, n)
		if got, want := Sum(b), refScalar(b); got != want {
			t.Fatalf("%s len %d: %x != %x", Implementation(), n, got, want)
		}
	}
}

// All-ones words push every accumulator through many 2^64 wraps.
func TestSumWrap(t *testing.T) {
	b := make([]byte, 1<<22)
	for i := range b {
		b[i] = 0xFF
	}
	if got, want := Sum(b), refScalar(b); got != want {
		t.Fatalf("%s: %x != %x", Implementation(), got, want)
	}
}

// Each kernel on its own, over whole blocks with a scalar tail, against the oracle.
func TestEveryKernel(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	ks := kernels()
	if len(ks) == 0 {
		t.Skip("no SIMD kernel on this CPU")
	}
	for _, k := range ks {
		for _, n := range []int{2 * k.blockBytes, 2*k.blockBytes + 4, 5*k.blockBytes + 28, 1000, 1 << 16, 1<<16 + 60} {
			b := randBuf(r, n)
			b = b[:len(b)&^3]
			if got, want := sumWith(b, k.lanes, k.blockBytes, k.run), refScalar(b); got != want {
				t.Fatalf("%s len %d: %x != %x", k.name, n, got, want)
			}
		}
	}
}

// The fold must equal OpenZFS's own fini formulas, transcribed verbatim, for
// arbitrary lane accumulators (not only those a real run can produce).
func TestFoldEqualsOpenZFSFini(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for it := 0; it < 2000; it++ {
		var out [32]uint64
		for i := range out {
			out[i] = r.Uint64()
		}

		// zfs_fletcher_intel.c fletcher_4_avx2_fini (zfs-2.3.0, lines 57-82):
		// avx[g].v[j] is out[g*4+j].
		v := func(g, j int) uint64 { return out[g*4+j] }
		A := v(0, 0) + v(0, 1) + v(0, 2) + v(0, 3)
		B := 0 - v(0, 1) - 2*v(0, 2) - 3*v(0, 3) +
			4*v(1, 0) + 4*v(1, 1) + 4*v(1, 2) + 4*v(1, 3)
		C := v(0, 2) + 3*v(0, 3) - 6*v(1, 0) - 10*v(1, 1) - 14*v(1, 2) - 18*v(1, 3) +
			16*v(2, 0) + 16*v(2, 1) + 16*v(2, 2) + 16*v(2, 3)
		D := 0 - v(0, 3) + 4*v(1, 0) + 10*v(1, 1) + 20*v(1, 2) + 34*v(1, 3) -
			48*v(2, 0) - 64*v(2, 1) - 80*v(2, 2) - 96*v(2, 3) +
			64*v(3, 0) + 64*v(3, 1) + 64*v(3, 2) + 64*v(3, 3)
		if got := fold(4, &out); got != [4]uint64{A, B, C, D} {
			t.Fatalf("L=4 (avx2 fini): %x != %x", got, [4]uint64{A, B, C, D})
		}

		// zfs_fletcher_avx512.c fletcher_4_avx512f_fini (lines 45-73).
		CcA := [8]uint64{0, 0, 1, 3, 6, 10, 15, 21}
		CcB := [8]uint64{28, 36, 44, 52, 60, 68, 76, 84}
		DcA := [8]uint64{0, 0, 0, 1, 4, 10, 20, 35}
		DcB := [8]uint64{56, 84, 120, 164, 216, 276, 344, 420}
		DcC := [8]uint64{448, 512, 576, 640, 704, 768, 832, 896}
		w := func(g, j int) uint64 { return out[g*8+j] }
		A = w(0, 0)
		B = 8 * w(1, 0)
		C = 64*w(2, 0) - CcB[0]*w(1, 0)
		D = 512*w(3, 0) - DcC[0]*w(2, 0) + DcB[0]*w(1, 0)
		for i := 1; i < 8; i++ {
			ui := uint64(i)
			A += w(0, i)
			B += 8*w(1, i) - ui*w(0, i)
			C += 64*w(2, i) - CcB[i]*w(1, i) + CcA[i]*w(0, i)
			D += 512*w(3, i) - DcC[i]*w(2, i) + DcB[i]*w(1, i) - DcA[i]*w(0, i)
		}
		if got := fold(8, &out); got != [4]uint64{A, B, C, D} {
			t.Fatalf("L=8 (avx512f fini): %x != %x", got, [4]uint64{A, B, C, D})
		}

		// zfs_fletcher_aarch64_neon.c fletcher_4_aarch64_neon_fini (lines 58-70).
		n := func(g, j int) uint64 { return out[g*2+j] }
		A = n(0, 0) + n(0, 1)
		B = 2*n(1, 0) + 2*n(1, 1) - n(0, 1)
		C = 4*n(2, 0) - n(1, 0) + 4*n(2, 1) - 3*n(1, 1)
		D = 8*n(3, 0) - 4*n(2, 0) + 8*n(3, 1) - 8*n(2, 1) + n(1, 1)
		if got := fold(2, &out); got != [4]uint64{A, B, C, D} {
			t.Fatalf("L=2 (neon fini): %x != %x", got, [4]uint64{A, B, C, D})
		}
	}
}

func FuzzSum(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"))
	f.Fuzz(func(t *testing.T, b []byte) {
		if got, want := Sum(b), refScalar(b); got != want {
			t.Fatalf("%s len %d: %x != %x", Implementation(), len(b), got, want)
		}
	})
}

var sink [4]uint64

func benchSum(b *testing.B, n int, f func([]byte) [4]uint64) {
	buf := randBuf(rand.New(rand.NewPCG(9, 9)), n)
	b.SetBytes(int64(n))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = f(buf)
	}
}

func BenchmarkScalar4K(b *testing.B)   { benchSum(b, 4<<10, SumScalar) }
func BenchmarkSum4K(b *testing.B)      { benchSum(b, 4<<10, Sum) }
func BenchmarkScalar128K(b *testing.B) { benchSum(b, 128<<10, SumScalar) }
func BenchmarkSum128K(b *testing.B)    { benchSum(b, 128<<10, Sum) }
func BenchmarkSum1M(b *testing.B)      { benchSum(b, 1<<20, Sum) }
