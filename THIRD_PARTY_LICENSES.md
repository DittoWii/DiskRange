# Third-Party Licenses

DiskRange as a whole is distributed under the **GNU Affero General Public
License v3.0** (see `LICENSE`). The project bundles the third-party components
listed below; each retains its own license, and the AGPL-3.0 component among
them (`svs`) is the reason the combined work is released under AGPL-3.0 rather
than a more permissive license.

First-party code authored for this project (everything outside `third/`, plus
`third/symqglib` itself) is published as part of the AGPL-3.0 combined work.

## Top-level vendored dependencies (`third/`)

| Component | License | License file |
| :--- | :--- | :--- |
| RaBitQ-Library | Apache-2.0 | `third/RaBitQ-Library/LICENSE` |
| Eigen | MPL-2.0<br>(primarily MPL-2.0; some<br>modules BSD / Apache-2.0) | `third/eigen/COPYING.MPL2`<br>`third/eigen/COPYRIGHT.Debian` |
| nlohmann/json | MIT | SPDX header in<br>`third/json/single_include/nlohmann/json.hpp` |
| liburing | MIT<br>(also available under LGPL-2.1) | `third/liburing/LICENSE`<br>`third/liburing/COPYING`<br>`third/liburing/COPYING.GPL` |
| symqglib | First-party<br>(authored for this project;<br>bundles the nested deps below) | — |

## Nested dependencies vendored inside `third/symqglib/third/`

| Component | License | License file |
| :--- | :--- | :--- |
| ffht | MIT | `third/symqglib/third/ffht/LICENSE` |
| ngt | Apache-2.0 | `third/symqglib/third/ngt/LICENSE` |
| svs | **AGPL-3.0**<br>(derived from Intel<br>Scalable Vector Search) | `third/symqglib/third/svs/LICENSE` |
| Eigen (nested copy) | MPL-2.0<br>(see note for top-level Eigen) | `third/symqglib/third/Eigen/LICENSE` |

## Note on the AGPL-3.0 obligation

`third/symqglib/third/svs/array.hpp` is derived from Intel Scalable Vector
Search and is licensed under AGPL-3.0. It is included transitively
(`symqglib/qg/qg.hpp` and `symqglib/utils/rotator.hpp` include it), so it is
compiled into the DiskRange binaries. Under AGPL-3.0 this makes the combined
work subject to AGPL-3.0, including its network-use (Section 13) source-offer
requirement. Redistribute accordingly.
