
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
- **Release tags**: Config files compatible with specific zea releases (e.g., `v0.0.1`, `v0.0.2`)

## Usage

```python
import zea

# Load a specific config file
config = zea.Config.from_hf("zeahub/configs", "config_picmus_rf.yaml", repo_type="dataset")

# Load from a specific release
config = zea.Config.from_hf("zeahub/configs", "config_camus.yaml", repo_type="dataset", revision="v0.0.2")
```

## Documentation

For detailed documentation and usage examples, visit:
- 📚 [zea.readthedocs.io](https://zea.readthedocs.io)
- 🔬 [Examples & Tutorials](https://zea.readthedocs.io/en/latest/examples.html)

## Source

Source repository: [github.com/tue-bmd/zea](https://github.com/tue-bmd/zea)
