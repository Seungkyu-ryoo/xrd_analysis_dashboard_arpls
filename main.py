"""Preferred command-line entry point for the XRD dashboard."""

if __package__:
    from .xrd_dashboard.ui.app import main
else:
    from xrd_dashboard.ui.app import main


if __name__ == "__main__":
    main()
