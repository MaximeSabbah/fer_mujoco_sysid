"""Errors raised while loading and validating identification artifacts."""


class ArtifactValidationError(ValueError):
    """An artifact is malformed, unsafe, or violates its semantic contract."""
