"""Preferred command-line entry point for the XRD dashboard."""

if __package__:
    from .app import main
else:
    from app import main


if __name__ == "__main__":
    main()
