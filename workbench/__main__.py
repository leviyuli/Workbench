"""Entry point: ``python -m workbench``."""

# The import lives inside the guard so that multiprocessing 'spawn' child
# processes (used to parallelize EIS fitting) can re-import this module as
# ``__mp_main__`` without pulling in Qt or launching a second GUI.
if __name__ == "__main__":
    from workbench.app import main
    main()
