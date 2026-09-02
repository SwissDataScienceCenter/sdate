# Security Policy

## Supported Versions

We actively support the following versions of ASTRA-Torch:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in ASTRA-Torch, please report it by emailing [security@chip-project.org](mailto:security@chip-project.org).

**Please do not report security vulnerabilities through public GitHub issues.**

When reporting a vulnerability, please provide:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any relevant code snippets or proof-of-concept
- Your contact information for follow-up

## Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Investigation**: We will investigate and assess the issue within 5 business days
- **Resolution**: Critical vulnerabilities will be patched within 14 days
- **Disclosure**: We will coordinate with you on responsible disclosure timing

## Security Best Practices

When using ASTRA-Torch:

1. **Keep Dependencies Updated**: Regularly update ASTRA-Torch and its dependencies
2. **Input Validation**: Validate all input data, especially when loading from external sources
3. **Resource Limits**: Be aware of memory usage when processing large datasets
4. **Access Control**: Ensure appropriate file system permissions for data files
5. **Network Security**: Use secure protocols when downloading datasets or models

## Known Security Considerations

### Data Loading
- HDF5 files can contain malicious content - only load trusted datasets
- Large datasets may cause memory exhaustion - implement appropriate limits

### GPU Usage
- CUDA operations may have specific security considerations
- Ensure proper GPU resource management in multi-user environments

### Dependencies
- ASTRA Toolbox: Follow ASTRA's security guidelines
- PyTorch: Keep PyTorch updated for security patches
- NumPy/SciPy: Monitor for security updates

## Vulnerability Disclosure Policy

We follow responsible disclosure practices:

1. **Private Reporting**: Vulnerabilities should be reported privately first
2. **Coordinated Disclosure**: We will work with reporters to determine appropriate disclosure timing
3. **Credit**: Security researchers will be credited in our security advisories (if desired)
4. **Public Disclosure**: After fixes are available, we will publish security advisories

## Security Updates

Security updates will be:
- Released as patch versions (e.g., 0.1.1 → 0.1.2)
- Announced in our changelog and GitHub releases
- Documented with affected versions and mitigation steps

## Contact

For security-related questions or concerns:
- Email: [security@chip-project.org](mailto:security@chip-project.org)
- For non-security issues, please use our [GitHub Issues](https://github.com/chip-project/astra-torch/issues)
