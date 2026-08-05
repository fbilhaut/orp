# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.0] - 2026-08-05

### Changed

* `Pipeline::expected_inputs` and `Pipeline::expected_outputs` now take the parameters as arguments (in case the expected tensors depend on them) and return iterators for greater flexibility.
* Add `inspect` binary that can be used to display various model properties from metadata and input/output tensors
* Switch to v1.0.0 of `composable`.


### Fixed

* Fixed error message in case of expected/actual tensors mismatch.
* Pin the `rc.9` version of `ort-sys` (`ort` was not doing so before `rc.10`, which would break builds of crates that depend on earlier versions, see [pykeio/ort#399](https://github.com/pykeio/ort/issues/399)).

## [0.9.2] - 2025-03-23

### Added

* Add the ability to load a model from memory, see `Model::new_from_bytes` ([PR#1](https://github.com/fbilhaut/orp/pull/1)).
* Add the possibility for a pipeline to expose which input/output tensors are required in the ONNX model schema (see `Pipeline::expected_inputs` and `Pipeline::expected_outputs`).


## [0.9.1] - 2025-01-30

### Fixed 

* Fix documentation URL

## [0.9.0] - 2025-01-26

### Added

* Initial release (externalized from [`gline-rs`](https://github.com/fbilhaut/gline-rs))
