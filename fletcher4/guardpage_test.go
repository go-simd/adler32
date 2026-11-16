//go:build unix

package fletcher4

import (
	"math/rand/v2"
	"syscall"
	"testing"
)

// A kernel that reads one byte past its input faults here: the buffer ends flush
// against a PROT_NONE page, for every length around the block sizes.
func TestKernelsStayInsideTheBuffer(t *testing.T) {
	page := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Skip("mmap:", err)
	}
	defer syscall.Munmap(mem)
	if err := syscall.Mprotect(mem[page:], syscall.PROT_NONE); err != nil {
		t.Skip("mprotect:", err)
	}
	r := rand.New(rand.NewPCG(11, 12))
	for n := 0; n <= 520; n += 4 {
		buf := mem[page-n : page : page]
		for i := range buf {
			buf[i] = byte(r.Uint32())
		}
		if got, want := Sum(buf), refScalar(buf); got != want {
			t.Fatalf("%s len %d: %x != %x", Implementation(), n, got, want)
		}
		for _, k := range kernels() {
			if n < 2*k.blockBytes {
				continue
			}
			if got, want := sumWith(buf, k.lanes, k.blockBytes, k.run), refScalar(buf); got != want {
				t.Fatalf("%s len %d: %x != %x", k.name, n, got, want)
			}
		}
	}
}
