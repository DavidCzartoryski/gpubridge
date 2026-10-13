# Security policy

## Reporting a vulnerability

Please report security issues privately, not in a public issue. Use GitHub's
private vulnerability reporting: open the repository's **Security** tab and
choose **Report a vulnerability**. Include what you found, how to reproduce
it, and the gpubridge and PyTorch versions.

If that option isn't available, open an issue that asks for a private
contact and leave out the details.

## Scope

gpubridge is an alpha, and only the latest release and `main` get fixes. It
assumes every rank of a job is trusted: like `torch.distributed`, its process
groups and bridge transports don't authenticate peers or encrypt traffic. Run
jobs on networks you trust.
