/*
 * Copyright 2017 Dgraph Labs, Inc. and Contributors
 * Modifications copyright (C) 2017 Andy Kimball and Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package arenaskl

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble/internal/invariants"
)

// Arena is lock-free.
type Arena struct {
	n   atomic.Uint64
	buf []byte
}

const nodeAlignment = 4

var (
	// ErrArenaFull indicates that the arena is full and cannot perform any more
	// allocations.
	ErrArenaFull = errors.New("allocation failed because arena is full")
)

const MaxArenaSize = min(math.MaxUint32, math.MaxInt)

// freedArenaStacks stores the stack traces of freed/recycled arenas.
// Only populated in invariant builds. When a use-after-free is detected,
// the stored stack identifies who freed/recycled the arena.
//
// Diagnostic scaffolding for cockroach#166984: suspected use-after-free
// in the memtable skiplist arena. This infrastructure converts a silent
// memory corruption (stale iterator reading freed arena) into a
// deterministic panic with both the read-site and free-site stack traces.
// TODO(cockroach#166984): remove once root cause is identified and fixed.
var freedArenaStacks sync.Map // uintptr(*Arena) → []uintptr

func recordFreedStack(a *Arena) {
	var callers [8]uintptr
	n := runtime.Callers(2, callers[:])
	freedArenaStacks.Store(uintptr(unsafe.Pointer(a)), callers[:n])
}

func lookupFreedStack(a *Arena) string {
	v, ok := freedArenaStacks.Load(uintptr(unsafe.Pointer(a)))
	if !ok {
		return "(no stack captured)"
	}
	frames := runtime.CallersFrames(v.([]uintptr))
	var s string
	for {
		frame, more := frames.Next()
		s += fmt.Sprintf("  %s\n    %s:%d\n", frame.Function, frame.File, frame.Line)
		if !more {
			break
		}
	}
	return s
}

// NewArena allocates a new arena using the specified buffer as the backing
// store.
func NewArena(buf []byte) *Arena {
	if len(buf) > MaxArenaSize {
		if invariants.Enabled {
			panic(errors.AssertionFailedf("attempting to create arena of size %d", errors.Safe(len(buf))))
		}
		buf = buf[:MaxArenaSize]
	}
	a := &Arena{
		buf: buf,
	}
	// We don't store data at position 0 in order to reserve offset=0 as a kind of
	// nil pointer.
	a.n.Store(1)
	return a
}

// Size returns the number of bytes allocated by the arena.
func (a *Arena) Size() uint32 {
	s := a.n.Load()
	if s > MaxArenaSize {
		// The last failed allocation can push the size higher than len(a.buf).
		// Saturate at the maximum representable offset.
		return MaxArenaSize
	}
	return uint32(s)
}

// Capacity returns the capacity of the arena.
func (a *Arena) Capacity() uint32 {
	return uint32(len(a.buf))
}

// alloc allocates a buffer of the given size and with the given alignment
// (which must be a power of 2).
//
// If overflow is not 0, it also ensures that many bytes after the buffer are
// inside the arena (this is used for structures that are larger than the
// requested size but don't use those extra bytes).
func (a *Arena) alloc(size, alignment, overflow uint32) (uint32, error) {
	if invariants.Enabled && (alignment&(alignment-1)) != 0 {
		panic(errors.AssertionFailedf("invalid alignment %d", errors.Safe(alignment)))
	}
	// Verify that the arena isn't already full.
	origSize := a.n.Load()
	if int(origSize) > len(a.buf) {
		return 0, ErrArenaFull
	}

	// Pad the allocation with enough bytes to ensure the requested alignment.
	padded := uint64(size) + uint64(alignment) - 1

	newSize := a.n.Add(padded)
	if newSize+uint64(overflow) > uint64(len(a.buf)) {
		return 0, ErrArenaFull
	}

	// Return the aligned offset.
	offset := (uint32(newSize) - size) & ^(alignment - 1)
	return offset, nil
}

func (a *Arena) getBytes(offset uint32, size uint32) []byte {
	if offset == 0 {
		return nil
	}
	if invariants.Enabled && a.buf == nil {
		panic(errors.AssertionFailedf(
			"use-after-free: arena access after free/recycle.\nFreed at:\n%s", lookupFreedStack(a)))
	}
	return a.buf[offset : offset+size : offset+size]
}

func (a *Arena) getPointer(offset uint32) unsafe.Pointer {
	if offset == 0 {
		return nil
	}
	if invariants.Enabled && a.buf == nil {
		panic(errors.AssertionFailedf(
			"use-after-free: arena access after free/recycle.\nFreed at:\n%s", lookupFreedStack(a)))
	}
	return unsafe.Pointer(&a.buf[offset])
}

func (a *Arena) getPointerOffset(ptr unsafe.Pointer) uint32 {
	if ptr == nil {
		return 0
	}
	return uint32(uintptr(ptr) - uintptr(unsafe.Pointer(&a.buf[0])))
}

// Poison invalidates the arena so that any subsequent access panics on
// the nil buf slice. In invariant builds, the buffer is first filled
// with 0xCC and the caller stack is recorded for diagnostic purposes.
//
// Called from memTable.free() when the arena buffer is returned to the
// C allocator. After this call, any stale iterator accessing this arena
// will panic immediately instead of silently reading freed memory.
//
// TODO(cockroach#166984): diagnostic scaffolding for suspected
// use-after-free. Remove once root cause is identified and fixed.
func (a *Arena) Poison() {
	if invariants.Enabled {
		recordFreedStack(a)
	}
	invariants.Mangle(a.buf)
	a.buf = nil
}

// Invalidate disconnects this Arena from its backing buffer by nil-ing
// the buf slice, without modifying the buffer contents. Use instead of
// Poison when the buffer will be reused (e.g., memtable recycling). In
// invariant builds, the caller stack is recorded for diagnostic purposes.
//
// Called from releaseMemAccounting before stashing the memtable for
// recycling. The old Arena object becomes disconnected from the buffer,
// which will be reused by a new Arena in the next memTable.init() call.
//
// TODO(cockroach#166984): diagnostic scaffolding for suspected
// use-after-free. Remove once root cause is identified and fixed.
func (a *Arena) Invalidate() {
	if invariants.Enabled {
		recordFreedStack(a)
	}
	a.buf = nil
}
