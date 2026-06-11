//go:build !amd64 && !riscv64 && !(arm64 && go1.27)

package adler32

// simdBlockBytes reports how many leading bytes of a chunk the SIMD kernel
// handles; on architectures without a kernel that is none, so the whole chunk
// goes to the scalar tail in updateChunk.
func simdBlockBytes(int) int { return 0 }

// chunkSIMD is never reached on this architecture (simdBlockBytes returns 0).
func chunkSIMD(s1, s2 uint32, _ []byte) (uint32, uint32) { return s1, s2 }
