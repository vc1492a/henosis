# Changelog
All notable changes to Henosis will be documented in this Changelog.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## 0.0.6 - 2018-03-15
### Fixed
- An issue where all X and y data (not only X_train and y_train) were resampled when upsampling or downsampling, causing
the test or validation set information to bleed into the training data.

## 0.0.5 - 2018-03-01
### Fixed
- A KeyError issue that prevented request logs to be stored in certain scenarios.

## 0.0.4 - 2018-02-27
### Configured
- Cross-origin request configuration and cleanup of code base.

## 0.0.3 - 2018-02-26
### Configured
- Cross-origin request configuration.

## 0.0.2 - 2018-02-26
### Added
- Addition of Flask-CORS to allow for cross-origin requests.

## 0.0.1 - 2018-02-23
### Added
- Initial release of Henosis codebase on PyPi.