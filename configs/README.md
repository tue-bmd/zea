
---
pretty_name: "zea configs"
tags:
  - ultrasound
  - configuration
  - zea
---

# zea Configuration Files

This repository contains configuration files for [zea](https://github.com/tue-bmd/zea), a toolbox for cognitive ultrasound imaging.

## Synchronization

Configuration files are automatically synchronized from the main zea repository:

- **Main branch**: Latest config files from the `main` branch
- **Release tags**: Config files compatible with specific zea releases (e.g., `v0.0.10`, `v0.0.11`)

## Usage

```python
import zea

# Load a specific config file
config = zea.Config.from_path("hf://zeahub/configs/config_picmus_rf.yaml")

# Load from a specific release
config = zea.Config.from_path("hf://zeahub/configs/config_picmus_rf.yaml", revision="v0.0.11")
```

## Documentation

For detailed documentation and usage examples, visit:
- 📚 [zea.readthedocs.io](https://zea.readthedocs.io)
- 🔬 [Examples & Tutorials](https://zea.readthedocs.io/en/latest/examples.html)

## Source

Source repository: [github.com/tue-bmd/zea](https://github.com/tue-bmd/zea)
