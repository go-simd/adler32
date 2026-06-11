package adler32

import (
	stdadler "hash/adler32"
	"math/rand"
	"testing"

	mhr3 "github.com/mhr3/adler32-simd"
)

var sizes = []int{
	0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65,
	1000, 1024, 4096,
	nmax - 1, nmax, nmax + 1, 2*nmax - 1, 2 * nmax, 2*nmax + 1,
	1 << 16, 1 << 20,
}

func randBytes(n int, seed int64) []byte {
	b := make([]byte, n)
	rand.New(rand.NewSource(seed)).Read(b)
	return b
}

// TestChecksumMatchesStdlib checks Checksum is bit-for-bit identical to
// hash/adler32.Checksum across edge sizes, including the nmax (5552) chunk
// boundary and multi-chunk inputs where the modular reduction kicks in.
func TestChecksumMatchesStdlib(t *testing.T) {
	for _, n := range sizes {
		for _, seed := range []int64{1, 2, 7} {
			src := randBytes(n, int64(n)*31+seed)
			if got, want := Checksum(src), stdadler.Checksum(src); got != want {
				t.Fatalf("Checksum n=%d seed=%d: got=%#08x want=%#08x", n, seed, got, want)
			}
		}
	}
}

// TestChecksumPatterns covers all-zero, all-0xff and ascending byte patterns
// (max-weight / max-byte stressors for the weighted SIMD sum) at several sizes.
func TestChecksumPatterns(t *testing.T) {
	for _, n := range []int{16, 32, 64, 5552, 5553, 11104, 100000} {
		patterns := map[string][]byte{
			"zero": make([]byte, n),
			"ff":   bytesFill(n, 0xff),
			"asc":  bytesAsc(n),
		}
		for name, src := range patterns {
			if got, want := Checksum(src), stdadler.Checksum(src); got != want {
				t.Fatalf("Checksum %s n=%d: got=%#08x want=%#08x", name, n, got, want)
			}
		}
	}
}

func bytesFill(n int, v byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = v
	}
	return b
}

func bytesAsc(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i)
	}
	return b
}

// TestNewMatchesStdlib checks the streaming hash.Hash32 (multiple Writes, then
// Sum32 and Sum) is identical to hash/adler32.New().
func TestNewMatchesStdlib(t *testing.T) {
	got := New()
	want := stdadler.New()
	chunks := [][]byte{
		randBytes(3, 1), randBytes(16, 2), randBytes(5552, 3),
		randBytes(5553, 4), randBytes(0, 5), randBytes(70000, 6),
	}
	for _, c := range chunks {
		got.Write(c)
		want.Write(c)
		if got.Sum32() != want.Sum32() {
			t.Fatalf("Sum32 after %d-byte write: got=%#08x want=%#08x", len(c), got.Sum32(), want.Sum32())
		}
	}
	if string(got.Sum(nil)) != string(want.Sum(nil)) {
		t.Fatalf("Sum mismatch")
	}
	got.Reset()
	want.Reset()
	if got.Sum32() != want.Sum32() {
		t.Fatalf("Sum32 after Reset: got=%#08x want=%#08x", got.Sum32(), want.Sum32())
	}
	if got.Size() != want.Size() || got.BlockSize() != want.BlockSize() {
		t.Fatalf("Size/BlockSize mismatch")
	}
}

// TestMhr3MatchesStdlib confirms the competitor is also identical, so the bench
// compares like for like.
func TestMhr3MatchesStdlib(t *testing.T) {
	for _, n := range sizes {
		src := randBytes(n, int64(n)+11)
		if got, want := mhr3.Checksum(src), stdadler.Checksum(src); got != want {
			t.Fatalf("mhr3 n=%d: got=%#08x want=%#08x", n, got, want)
		}
	}
}

// FuzzChecksum compares Checksum to hash/adler32.Checksum on arbitrary input.
func FuzzChecksum(f *testing.F) {
	f.Add([]byte(""))
	f.Add([]byte("hello world"))
	f.Add(randBytes(16, 1))
	f.Add(randBytes(33, 2))
	f.Add(randBytes(5552, 3))
	f.Add(randBytes(5553, 4))
	f.Add(randBytes(70000, 5))
	f.Fuzz(func(t *testing.T, data []byte) {
		if got, want := Checksum(data), stdadler.Checksum(data); got != want {
			t.Fatalf("Checksum(%d bytes): got=%#08x want=%#08x", len(data), got, want)
		}
	})
}

func benchData() []byte { return randBytes(1<<20, 2) }

func BenchmarkChecksum(b *testing.B) {
	src := benchData()
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Checksum(src)
	}
}

func BenchmarkChecksumStdlib(b *testing.B) {
	src := benchData()
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stdadler.Checksum(src)
	}
}

// BenchmarkChecksumMhr3 benchmarks github.com/mhr3/adler32-simd, a pure-Go SIMD
// Adler-32 (amd64 SSE3 + arm64 NEON) transpiled from Chromium/zlib via gocc.
func BenchmarkChecksumMhr3(b *testing.B) {
	src := benchData()
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mhr3.Checksum(src)
	}
}
