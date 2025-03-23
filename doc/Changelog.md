# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.9.3] - UNRELEASED

...


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
