"""Compatibility launcher for the dashboard's original filename."""

if __package__:
    from .main_optimized import *  # noqa: F401,F403
else:
    from main_optimized import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
