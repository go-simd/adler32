package fletcher64

import (
	"encoding/binary"
	"testing"
)

// ref is the straightforward form of the algorithm in the Apple File System
// Reference ("Fletcher 64 Checksum Algorithm"): reduce modulo 2^32-1 after every
// word. It is the oracle for Sum, which defers the reduction.
func ref(p []byte) uint64 {
	const m = uint64(0xFFFFFFFF)
	var s1, s2 uint64
	for i := 0; i+4 <= len(p); i += 4 {
		s1 = (s1 + uint64(binary.LittleEndian.Uint32(p[i:]))) % m
		s2 = (s2 + s1) % m
	}
	c1 := m - (s1+s2)%m
	c2 := m - (s1+c1)%m
	return c2<<32 | c1
}

func pattern(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7 + 3)
	}
	return b
}

// TestKnownAnswers pins Sum to the values the go-filesystems/apfs v0.7.0 local
// fletcher64 (the optimised kernel this one was moved from) returned on the
// same inputs.
func TestKnownAnswers(t *testing.T) {
	for _, tc := range []struct {
		n    int
		want uint64
	}{
		{0, 0xffffffffffffffff},
		{3, 0xffffffffffffffff},
		{4, 0x18110a03cfddebf9},
		{8, 0x644f3a254f7295b8},
		{12, 0x00d6ac8362a1e11f},
		{4088, 0x1c193445c3b48774},
		{4096, 0x1b275151e6d6a8b4},
		{16384, 0x9d2cb5146acb3303},
		{65536 + 4 + 2, 0x75db3af891f35b64},
	} {
		if got := Sum(pattern(tc.n)); got != tc.want {
			t.Errorf("Sum(pattern(%d)) = %#016x, want %#016x", tc.n, got, tc.want)
		}
	}
	ff := make([]byte, 4088)
	for i := range ff {
		ff[i] = 0xff
	}
	if got := Sum(ff); got != 0xffffffffffffffff {
		t.Errorf("Sum(all 0xff) = %#016x", got)
	}
}

// TestMatchesReference covers every length around the unroll (128), the
// reduction run (4096) and the word size, and several runs of the deferred
// reduction, with all-ones words that make the sums as large as they can get.
func TestMatchesReference(t *testing.T) {
	ones := make([]byte, 5*4096+7)
	for i := range ones {
		ones[i] = 0xff
	}
	src := pattern(len(ones))
	for n := 0; n <= 600; n++ {
		if got, want := Sum(src[:n]), ref(src[:n]); got != want {
			t.Fatalf("n=%d: %#x != %#x", n, got, want)
		}
	}
	for _, n := range []int{4095, 4096, 4097, 4100, 8191, 8192, 8193, 12288 + 4, len(ones)} {
		if got, want := Sum(src[:n]), ref(src[:n]); got != want {
			t.Fatalf("pattern n=%d: %#x != %#x", n, got, want)
		}
		if got, want := Sum(ones[:n]), ref(ones[:n]); got != want {
			t.Fatalf("ones n=%d: %#x != %#x", n, got, want)
		}
	}
}

// TestSealVerifies is the defining property: appending the checksum's two
// words to the data makes the checksum of the whole come out as "zero".
func TestSealVerifies(t *testing.T) {
	data := pattern(4088)
	c := Sum(data)
	full := append(append([]byte{}, data...), make([]byte, 8)...)
	binary.LittleEndian.PutUint64(full[len(data):], c)
	const m = uint64(0xFFFFFFFF)
	var s1, s2 uint64
	for i := 0; i+4 <= len(full); i += 4 {
		s1 = (s1 + uint64(binary.LittleEndian.Uint32(full[i:]))) % m
		s2 = (s2 + s1) % m
	}
	if s1 != 0 || s2 != 0 {
		t.Fatalf("sums after appending the checksum = %d, %d; want 0, 0", s1, s2)
	}
}

func TestNoAllocs(t *testing.T) {
	b := pattern(4088)
	if a := testing.AllocsPerRun(100, func() { _ = Sum(b) }); a != 0 {
		t.Fatalf("Sum allocates %v times", a)
	}
}

func FuzzSum(f *testing.F) {
	f.Add([]byte{})
	f.Add(pattern(4088))
	f.Add(pattern(4099))
	f.Add(make([]byte, 128))
	f.Fuzz(func(t *testing.T, p []byte) {
		if got, want := Sum(p), ref(p); got != want {
			t.Fatalf("len %d: Sum %#x != reference %#x", len(p), got, want)
		}
	})
}

func BenchmarkSum(b *testing.B) {
	for _, n := range []int{4088, 65536} {
		buf := pattern(n)
		b.Run(sizeName(n), func(b *testing.B) {
			b.SetBytes(int64(n))
			b.ReportAllocs()
			for b.Loop() {
				_ = Sum(buf)
			}
		})
	}
}

func BenchmarkReference(b *testing.B) {
	buf := pattern(4088)
	b.SetBytes(int64(len(buf)))
	b.ReportAllocs()
	for b.Loop() {
		_ = ref(buf)
	}
}

func sizeName(n int) string {
	if n >= 1024 && n%1024 == 0 {
		return itoa(n/1024) + "KiB"
	}
	return itoa(n) + "B"
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for ; n > 0; n /= 10 {
		d = append([]byte{byte('0' + n%10)}, d...)
	}
	return string(d)
}
