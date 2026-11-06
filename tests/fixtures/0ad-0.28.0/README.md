# Pristine 0 A.D. 0.28.0 sources the engine patches touch

`source/main.cpp`, `source/rlinterface/RLInterface.cpp` and `source/rlinterface/RLInterface.h`, byte for byte as in the
`0ad_0.28.0` PPA source package (identical to the upstream `v0.28.0` tag). `tests/test_engine_idle_patch.py` pins their
sha256 and checks that `patches/rl-realtime.patch` then `patches/rl-idle-wait.patch` apply to them with
`patch -p1 -F0`, so a patch that has drifted from the real source fails here and not in the 15-minute image build.

License: © Wildfire Games, GPL-2.0-or-later, like `patches/` (see `patches/COPYING`). These files keep that license.
