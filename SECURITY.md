# Security Policy

## Supported Versions

Security fixes are applied to the latest release on the `master` branch.

## Reporting a Vulnerability

If you discover a security vulnerability in nm, please report it privately:

- Open a [GitHub Security Advisory](https://github.com/tragisch/14_NN/security/advisories/new)
  (preferred), or
- Contact the maintainers via GitHub.

Please do not open a public issue for security vulnerabilities.

We will acknowledge your report within 72 hours and aim to provide a fix or
mitigation within 30 days, depending on severity and complexity.

## Scope

nm is a local neural-network inference/training library. It processes data
in-process and does not expose network services by default. Security concerns
include: memory safety in C parsing code (e.g. GraphIR import, `.girw` weight
loading), integer overflow in tensor shape calculations, and unsafe deserialization.
