# python-scaffold

Describe your project here.

## Installation

### Rye

Package manager for Python.

```bash
brew install rye
```

### Taplo

Formatter for TOML.

```bash
brew install taplo
```

## Setup

### First Sync

```bash
rye sync
```

### Pre-commit

```bash
rye run pre-commit install
```

## Change Python's version

To change your python's version into *3.x*

```bash
rye pin <3.x>
rye sync
```

## Add package

To add *python-package*

```bash
rye add <python-package>
rye sync
```

## Enable logger

To enable logger, place `.env` on project root

```properties:.env
LOG_LEVEL=INFO
LOG_DIR=/path/to/log/dir
TZ=Asia/Tokyo
```
