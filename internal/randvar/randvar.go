// Copyright 2019 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package randvar

import "math/rand/v2"

// Static models a random variable that pulls from a distribution with static
// bounds
type Static interface {
	Uint64(rng *rand.Rand) uint64
}

// Dynamic models a random variable that pulls from a distribution with an
// upper bound that can change dynamically using the IncMax method.
type Dynamic interface {
	Static

	// Increment the max value the variable will return.
	IncMax(delta uint64)

	// Read the current max value the variable will return.
	Max() uint64
}
