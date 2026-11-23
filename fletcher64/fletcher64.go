// Package fletcher64 is the Fletcher-64 object checksum of Apple's APFS
// (obj_phys_t.o_cksum), also used by HFS+/APFS keybags and other Apple
// on-disk objects.
//
// The input is read as little-endian 32-bit words w. Two running sums are kept
// modulo 2^32-1:
//
//	s1 += w; s2 += s1
//
// and the stored checksum is the pair (c1, c2) that makes a further appended
// pair of words (c1, c2) leave both sums at zero:
//
//	c1 = (2^32-1) - ((s1 + s2) mod (2^32-1))
//	c2 = (2^32-1) - ((s1 + c1) mod (2^32-1))
//	checksum = c2<<32 | c1
//
// (Apple File System Reference, "Fletcher 64 Checksum Algorithm"). For an APFS
// object the checksum covers the block WITHOUT its first 8 bytes, which hold
// the checksum itself: pass block[8:].
//
// This is a different algorithm from the ZFS fletcher4 in the sibling package
// fletcher4 (four sums mod 2^64, no modular reduction).
//
// The kernel defers the modular reduction: s1 and s2 are 64-bit and only
// reduced every 4096 bytes, so the hot loop is additions only. A trailing
// partial word (len(p) % 4 != 0) is ignored.
package fletcher64

import "encoding/binary"

const mod = uint64(0xFFFFFFFF)

// Sum returns the Fletcher-64 checksum of p as APFS stores it in o_cksum, where
// p is the object without its first 8 bytes. A trailing partial 32-bit word is
// ignored.
//
// It is a pure function of its argument that touches no shared state and is
// meant for per-block hot paths, so it is excluded from race instrumentation
// (//go:norace): instrumenting every word load made a race-detector test suite
// 2.4x slower once every object read verifies its checksum.
//
//go:norace
func Sum(p []byte) uint64 {
	// The reduction modulo 2^32-1 is deferred to every 1024 words: s1 then
	// stays below 2^42 and s2 below 2^52, so nothing overflows and the result
	// equals reducing after every word.
	const run = 1024 * 4
	var s1, s2 uint64
	for len(p) >= 4 {
		n := len(p) &^ 3
		if n > run {
			n = run
		}
		i := 0
		// 128 bytes (32 words) per iteration. Coverage instrumentation
		// (-cover, atomic under -race) adds a counter update per loop
		// iteration, so the iteration count, not the word count, sets the cost
		// on CI: 32 iterations per 4 KiB block instead of 512.
		for ; i+128 <= n; i += 128 {
			b := p[i : i+128 : i+128]
			v0 := binary.LittleEndian.Uint64(b[0:8])
			v1 := binary.LittleEndian.Uint64(b[8:16])
			v2 := binary.LittleEndian.Uint64(b[16:24])
			v3 := binary.LittleEndian.Uint64(b[24:32])
			v4 := binary.LittleEndian.Uint64(b[32:40])
			v5 := binary.LittleEndian.Uint64(b[40:48])
			v6 := binary.LittleEndian.Uint64(b[48:56])
			v7 := binary.LittleEndian.Uint64(b[56:64])
			v8 := binary.LittleEndian.Uint64(b[64:72])
			v9 := binary.LittleEndian.Uint64(b[72:80])
			v10 := binary.LittleEndian.Uint64(b[80:88])
			v11 := binary.LittleEndian.Uint64(b[88:96])
			v12 := binary.LittleEndian.Uint64(b[96:104])
			v13 := binary.LittleEndian.Uint64(b[104:112])
			v14 := binary.LittleEndian.Uint64(b[112:120])
			v15 := binary.LittleEndian.Uint64(b[120:128])
			s1 += v0 & 0xFFFFFFFF
			s2 += s1
			s1 += v0 >> 32
			s2 += s1
			s1 += v1 & 0xFFFFFFFF
			s2 += s1
			s1 += v1 >> 32
			s2 += s1
			s1 += v2 & 0xFFFFFFFF
			s2 += s1
			s1 += v2 >> 32
			s2 += s1
			s1 += v3 & 0xFFFFFFFF
			s2 += s1
			s1 += v3 >> 32
			s2 += s1
			s1 += v4 & 0xFFFFFFFF
			s2 += s1
			s1 += v4 >> 32
			s2 += s1
			s1 += v5 & 0xFFFFFFFF
			s2 += s1
			s1 += v5 >> 32
			s2 += s1
			s1 += v6 & 0xFFFFFFFF
			s2 += s1
			s1 += v6 >> 32
			s2 += s1
			s1 += v7 & 0xFFFFFFFF
			s2 += s1
			s1 += v7 >> 32
			s2 += s1
			s1 += v8 & 0xFFFFFFFF
			s2 += s1
			s1 += v8 >> 32
			s2 += s1
			s1 += v9 & 0xFFFFFFFF
			s2 += s1
			s1 += v9 >> 32
			s2 += s1
			s1 += v10 & 0xFFFFFFFF
			s2 += s1
			s1 += v10 >> 32
			s2 += s1
			s1 += v11 & 0xFFFFFFFF
			s2 += s1
			s1 += v11 >> 32
			s2 += s1
			s1 += v12 & 0xFFFFFFFF
			s2 += s1
			s1 += v12 >> 32
			s2 += s1
			s1 += v13 & 0xFFFFFFFF
			s2 += s1
			s1 += v13 >> 32
			s2 += s1
			s1 += v14 & 0xFFFFFFFF
			s2 += s1
			s1 += v14 >> 32
			s2 += s1
			s1 += v15 & 0xFFFFFFFF
			s2 += s1
			s1 += v15 >> 32
			s2 += s1
		}
		// Two words per 8-byte load: low word first (little-endian).
		for ; i+8 <= n; i += 8 {
			v := binary.LittleEndian.Uint64(p[i : i+8])
			s1 += v & 0xFFFFFFFF
			s2 += s1
			s1 += v >> 32
			s2 += s1
		}
		for ; i < n; i += 4 {
			s1 += uint64(binary.LittleEndian.Uint32(p[i : i+4]))
			s2 += s1
		}
		s1 %= mod
		s2 %= mod
		p = p[n:]
	}
	c1 := mod - ((s1 + s2) % mod)
	c2 := mod - ((s1 + c1) % mod)
	return (c2 << 32) | c1
}
